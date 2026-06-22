"""SSL for the discrete-flow backbone over fragment-shuffle views.

Each batch is a list of fragmented-SMILES strings. Per molecule we tokenize once
and produce two views by shuffling the fragment order at the token level (split
on the separator id, shuffle, rejoin). Both views go through the *same* encoder,
so the (optionally learnable) encode-time is shared across them by construction.
The adapter, the encode-time parameter (when learnable), and the DDiT backbone
(when ``encoder.freeze_backbone`` is false) all train — whatever has
``requires_grad`` is optimized.

Losses (``ssl_loss``):
  * ``ntxent`` (default) — symmetric NT-Xent / InfoNCE on projection head outputs.
  * ``lejepa`` — invariance loss + SIGReg on unnormalized pooled latents: every
    view (intact shuffles + masked) is pulled directly toward the molecule's
    intact center (no predictor), while SIGReg keeps the target batch isotropic.
    Needs >= 1 global (intact) view; masked local views supply the augmentation.
  * ``hybrid`` — NT-Xent on the first two global views' projections, linearly
    annealed (1.0 -> 0.0 over ``hybrid_anneal_steps``) in favor of the LeJEPA
    loss. Motivation: LeJEPA alone reaches near-full numerical rank but very
    low effective rank, because nothing in its objective pushes different
    molecules apart between-sample — NT-Xent's explicit pairwise repulsion
    supplies that directly while it's annealed in.
  * ``ijepa`` — I-JEPA: one masked encoder pass (``<UNK>`` holes), visible adapter
    reps + learnable mask-token predictor vs EMA intact targets at hole indices.
    One intact view per molecule plus ``lejepa_n_local_views`` masked locals.
    Masking is fragment-, span-, or mixed-mode (``ijepa_mask_mode``).
"""

from __future__ import annotations

import copy
import logging
import random

import lightning as L
import numpy as np
import torch

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder, pad_batch
from lattice_lab.backbone.discrete_flow import resolve_mask_token_id
from lattice_lab.data.fragment_views import (
    mask_fragment_ids,
    mask_local_ids,
    shuffle_fragment_ids,
)
from lattice_lab.models.schedules import cosine_with_warmup
from lattice_lab.training.ssl_loss import (
    IJEPALoss,
    LeJEPALoss,
    NTXentLoss,
    _FingerprintCache,
    lejepa_retrieval_acc1,
    similarity_distillation_loss,
    tanimoto_target_matrix,
    top1_paired_accuracy,
)
from lattice_lab.training.ssl_val_probes import (
    SSLValProbes,
    embedding_batch_collapse_diag,
    embedding_covariance_rank,
)

logger = logging.getLogger(__name__)


class DiscreteFlowSSLModule(L.LightningModule):
    def __init__(
        self,
        encoder: DiscreteFlowEncoder,
        *,
        frag_sep_id: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 30_000,
        temperature: float = 0.1,
        fp_weight: float = 0.0,
        fp_radius: int = 2,
        fp_bits: int = 2048,
        ssl_loss: str = "ntxent",
        lejepa_lambda: float = 0.05,
        lejepa_n_global_views: int = 2,
        lejepa_n_local_views: int = 2,
        lejepa_mask_frac: float = 0.5,
        lejepa_mask_frac_max: float | None = None,
        lejepa_mask_token_id: int | None = None,
        ijepa_mask_mode: str = "fragment",
        ijepa_ema_decay: float = 0.996,
        # When true, visible context reps cannot attend to <UNK> hole keys in DDiT
        # + adapter (paper-faithful context-only encoding on one masked pass).
        ijepa_block_hole_attn: bool = False,
        ijepa_predictor_layers: int = 1,
        ijepa_predictor_heads: int = 2,
        ijepa_predictor_ff_mult: int = 2,
        ijepa_context_sigreg_lambda: float = 0.0,
        ijepa_collapse_diag_every_n_steps: int = 50,
        rank_subsample: int = 256,
        sigreg_num_projections: int = 256,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_eps: float = 1e-8,
        # hybrid only: linear anneal of the ntxent/lejepa mix weight, alpha =
        # max(0, 1 - global_step/hybrid_anneal_steps) -- 1.0 (pure ntxent) at
        # step 0 down to 0.0 (pure lejepa) by this step, then held at 0.
        hybrid_anneal_steps: int = 2000,
        # Covariance-rank of the teacher (I-JEPA EMA) or online global pooled z_m,
        # logged every N train steps and on every val batch. 0 disables.
        train_rank_every_n_steps: int = 50,
        # Diagnostic: per-loss-term gradient L2 norm w.r.t. trainable params,
        # via a separate torch.autograd.grad call per term (retain_graph=True,
        # doesn't disturb Lightning's own backward on the summed loss).
        # Settles "which term is actually driving updates" directly instead of
        # inferring it from (easily misleading) loss *values*. One extra
        # backward-equivalent pass per term every training step -- off by
        # default, only meant for short diagnostic runs.
        log_grad_norms: bool = False,
        # ijepa only: warn when the predictor ignores context (see on_validation_epoch_end).
        condition_margin: float = 0.01,
        val_probe_n_molecules: int = 2000,
        val_probe_every_n_epochs: int = 1,
        val_probe_encode_batch_size: int = 128,
        val_probe_ridge_alpha: float = 1.0,
        val_probe_test_size: float = 0.2,
        val_probe_tsne_perplexity: float | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder
        ssl_loss = ssl_loss.lower()
        if ssl_loss not in {"ntxent", "lejepa", "hybrid", "ijepa"}:
            raise ValueError(
                f"ssl_loss must be 'ntxent', 'lejepa', 'hybrid', or 'ijepa', "
                f"got {ssl_loss!r}"
            )
        self.ssl_loss = ssl_loss
        # "hybrid" reuses lejepa's global+local view construction (it needs the
        # masked local views for SIGReg too), just also takes a projection for
        # the annealed ntxent term.
        uses_lejepa_views = ssl_loss in ("lejepa", "hybrid")
        # ijepa doesn't use the global/local pooled-view stack, but it does use
        # fragment masking (mask token + mask fraction) to build its hole.
        uses_masking = uses_lejepa_views or ssl_loss == "ijepa"
        # Morgan-fingerprint similarity distillation: aligns the cosine geometry
        # of z_m with Tanimoto similarity, injecting the chemical-similarity
        # structure plain instance-discrimination / invariance never learns.
        # This is the ingredient behind the LATTICE baseline adapter
        # (fp_weight=2.0); dropping it collapses LIT-PCBA retrieval. The
        # distillation always runs on an L2-normalized *copy* of z_m, so it works
        # for both ntxent (already normalized) and lejepa (unnormalized latents
        # kept raw for SIGReg) without disturbing the primary objective's space.
        self.fp_weight = float(fp_weight)
        if self.fp_weight < 0:
            raise ValueError(f"fp_weight must be >= 0, got {fp_weight}")
        self._fp_cache = (
            _FingerprintCache(radius=fp_radius, n_bits=fp_bits)
            if self.fp_weight > 0
            else None
        )
        # hybrid needs both loss modules; ntxent/lejepa each need only their own.
        self.ntxent_loss_fn = (
            NTXentLoss(temperature=temperature) if ssl_loss in ("ntxent", "hybrid") else None
        )
        self.lejepa_loss_fn = (
            LeJEPALoss(
                lejepa_lambda=lejepa_lambda,
                sigreg_num_projections=sigreg_num_projections,
                sigreg_knots=sigreg_knots,
                sigreg_t_max=sigreg_t_max,
                sigreg_eps=sigreg_eps,
            )
            if uses_lejepa_views
            else None
        )
        self.ijepa_loss_fn = (
            IJEPALoss(
                dim=encoder.adapter.d_adapter,
                lejepa_lambda=lejepa_lambda,
                context_sigreg_lambda=ijepa_context_sigreg_lambda,
                sigreg_num_projections=sigreg_num_projections,
                sigreg_knots=sigreg_knots,
                sigreg_t_max=sigreg_t_max,
                sigreg_eps=sigreg_eps,
                ijepa_predictor_layers=ijepa_predictor_layers,
                ijepa_predictor_heads=ijepa_predictor_heads,
                ijepa_predictor_ff_mult=ijepa_predictor_ff_mult,
            )
            if ssl_loss == "ijepa"
            else None
        )
        self.encoder_ema: DiscreteFlowEncoder | None = None
        if ssl_loss == "ijepa":
            if not (0.0 <= float(ijepa_ema_decay) < 1.0):
                raise ValueError(
                    f"ijepa_ema_decay must be in [0, 1), got {ijepa_ema_decay}"
                )
            self.encoder_ema = copy.deepcopy(encoder)
            self.encoder_ema.eval()
            for p in self.encoder_ema.parameters():
                p.requires_grad = False
        self._rng = random.Random(seed)
        self._val_probes = SSLValProbes(
            n_molecules=val_probe_n_molecules,
            seed=seed,
            every_n_epochs=val_probe_every_n_epochs,
            encode_batch_size=val_probe_encode_batch_size,
            ridge_alpha=val_probe_ridge_alpha,
            probe_test_size=val_probe_test_size,
            tsne_perplexity=val_probe_tsne_perplexity,
        )
        self._val: dict[str, list[float]] = {
            "loss": [], "acc": [], "inv": [], "inv_rel": [], "sigreg": [],
            "predict": [], "sigreg_context": [], "view_diversity": [], "fp": [], "ntxent": [],
            "cond_true": [], "cond_shuf": [], "cond_zero": [],
            "cond_gap_zero": [], "cond_gap_shuf": [],
            "rank_effective": [], "rank_numerical": [],
        }
        self._mask_token_id = (
            resolve_mask_token_id(
                encoder.bundle.tokenizer, override=lejepa_mask_token_id,
            )
            if uses_masking
            else None
        )
        if uses_lejepa_views and int(lejepa_n_global_views) < 1:
            raise ValueError(
                f"lejepa_n_global_views must be >= 1, got {lejepa_n_global_views}"
            )
        if uses_masking:
            frac_lo = float(lejepa_mask_frac)
            frac_hi = float(lejepa_mask_frac_max) if lejepa_mask_frac_max is not None else frac_lo
            if not (0.0 < frac_lo <= 1.0):
                raise ValueError(
                    f"lejepa_mask_frac must be in (0, 1], got {lejepa_mask_frac}"
                )
            if not (0.0 < frac_hi <= 1.0):
                raise ValueError(
                    f"lejepa_mask_frac_max must be in (0, 1], got {lejepa_mask_frac_max}"
                )
            if frac_lo > frac_hi:
                raise ValueError(
                    f"lejepa_mask_frac ({frac_lo}) must be <= lejepa_mask_frac_max ({frac_hi})"
                )
        if ssl_loss == "hybrid" and int(lejepa_n_global_views) < 2:
            raise ValueError(
                "hybrid loss pairs the first two global views for ntxent; "
                f"need lejepa_n_global_views >= 2, got {lejepa_n_global_views}"
            )
        if ssl_loss == "ijepa":
            if int(lejepa_n_local_views) < 1:
                raise ValueError(
                    f"lejepa_n_local_views must be >= 1 for ijepa, "
                    f"got {lejepa_n_local_views}"
                )
            mode = ijepa_mask_mode.lower()
            if mode not in {"fragment", "span", "mixed"}:
                raise ValueError(
                    f"ijepa_mask_mode must be fragment, span, or mixed, got {ijepa_mask_mode!r}"
                )
        logger.info(
            "discrete-flow SSL: ssl_loss=%s lejepa_global=%d lejepa_local=%d "
            "ijepa_mask=%s block_hole_attn=%s pred_layers=%s mask_id=%s "
            "hybrid_anneal_steps=%s learnable_time=%s init_time=%.3f frag_sep_id=%d",
            ssl_loss,
            lejepa_n_global_views if uses_lejepa_views else 0,
            lejepa_n_local_views if uses_lejepa_views or ssl_loss == "ijepa" else 0,
            ijepa_mask_mode if ssl_loss == "ijepa" else "n/a",
            ijepa_block_hole_attn if ssl_loss == "ijepa" else "n/a",
            ijepa_predictor_layers if ssl_loss == "ijepa" else "n/a",
            self._mask_token_id if uses_masking else "n/a",
            hybrid_anneal_steps if ssl_loss == "hybrid" else "n/a",
            encoder.learnable_time,
            self.encoder.encode_time_value,
            frag_sep_id,
        )

    # -- fragment-shuffle augmentation -------------------------------------- #
    def _sample_mask_frac(self) -> float:
        """Mask fraction for one view. Fixed when ``lejepa_mask_frac_max`` is unset;
        otherwise uniform in ``[lejepa_mask_frac, lejepa_mask_frac_max]``."""
        lo = float(self.hparams.lejepa_mask_frac)
        hi = self.hparams.lejepa_mask_frac_max
        if hi is None:
            return lo
        hi = float(hi)
        return lo if hi <= lo else self._rng.uniform(lo, hi)

    @staticmethod
    def _body_ids(item: str | list[int], tokenizer) -> list[int]:
        if isinstance(item, list):
            return item
        return tokenizer.encode(item, add_special_tokens=False)

    @staticmethod
    def _subsample_rows(z: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or z.size(0) <= k:
            return z
        idx = torch.randperm(z.size(0), device=z.device)[:k]
        return z[idx]

    def _two_views(self, view_strings: list[str] | list[list[int]]):
        """Tokenize each view, build two fragment-shuffled token sequences,
        return two padded ``(ids, mask)`` batches on the module device."""
        b = self.encoder.bundle
        sep = int(self.hparams.frag_sep_id)
        seqs_a: list[list[int]] = []
        seqs_b: list[list[int]] = []
        for s in view_strings:
            body = self._body_ids(s, b.tokenizer)
            sa = shuffle_fragment_ids(body, sep, self._rng)
            sb = shuffle_fragment_ids(body, sep, self._rng)
            seqs_a.append([b.bos_id, *sa, b.eos_id])
            seqs_b.append([b.bos_id, *sb, b.eos_id])
        ids_a, mask_a = pad_batch(seqs_a, pad_id=b.pad_id)
        ids_b, mask_b = pad_batch(seqs_b, pad_id=b.pad_id)
        dev = self.device
        return (ids_a.to(dev), mask_a.to(dev)), (ids_b.to(dev), mask_b.to(dev))

    @staticmethod
    def _split_batch(batch) -> tuple[list[str] | list[list[int]], list[str] | None]:
        """Accept view strings, pretokenized body ids, or ``(views, smiles)``."""
        if isinstance(batch, tuple) and len(batch) == 2:
            views, smiles = batch
            return list(views), list(smiles)
        return list(batch), None

    def _encode_ntxent_views(self, views):
        """Return ``(z_a_proj, z_b_proj, z_a_pooled)``.

        ``z_*_proj`` are the SimCLR projection outputs (NT-Xent loss);
        ``z_a_pooled`` is the L2-normalized z_m of view a, used by the optional
        Tanimoto similarity-distillation loss.
        """
        (ids_a, mask_a), (ids_b, mask_b) = self._two_views(views)
        z_a_pooled, z_a = self.encoder.encode_token_ids(ids_a, mask_a, return_projection=True)
        _, z_b = self.encoder.encode_token_ids(ids_b, mask_b, return_projection=True)
        return z_a, z_b, z_a_pooled

    def _fp_distillation(self, z_pooled, smiles):
        """MSE(cos(z_pooled), Tanimoto(smiles)); ``None`` if disabled.

        Distillation runs on an L2-normalized copy of ``z_pooled`` so the cosine
        geometry is well defined regardless of the primary loss: ntxent latents
        are already unit-norm (no-op), while lejepa latents stay unnormalized for
        SIGReg and are only normalized here for the cosine target.
        """
        if self.fp_weight <= 0 or self._fp_cache is None:
            return None
        if smiles is None:
            raise ValueError(
                "fp_weight > 0 but batch has no SMILES; set data.return_smiles=true"
            )
        import numpy as np
        bits_np = self._fp_cache.bits(smiles)
        bits = torch.from_numpy(np.ascontiguousarray(bits_np)).to(z_pooled.device)
        target = tanimoto_target_matrix(bits).to(z_pooled.dtype)
        z_unit = torch.nn.functional.normalize(z_pooled, dim=-1)
        return similarity_distillation_loss(z_unit, target)

    def _encode_lejepa_views(self, batch, *, with_projection: bool = False):
        """Return global ``[B,Vg,D]``, all ``[B,Vg+Vl,D]`` raw pooled latents +
        diversity. When ``with_projection`` is set (hybrid loss), also returns
        the L2-normalized projection-head output for the global views
        ``[B,Vg,P]`` — derived from the same forward pass, no extra encode.
        """
        n_global = int(self.hparams.lejepa_n_global_views)
        n_local = int(self.hparams.lejepa_n_local_views)
        n_all = n_global + n_local
        b = self.encoder.bundle
        sep = int(self.hparams.frag_sep_id)
        mask_id = int(self._mask_token_id)
        seqs: list[list[int]] = []
        n_changed = 0
        for s in batch:
            body = self._body_ids(s, b.tokenizer)
            first_global: list[int] | None = None
            for _ in range(n_global):
                shuffled = shuffle_fragment_ids(body, sep, self._rng)
                seq = [b.bos_id, *shuffled, b.eos_id]
                if first_global is None:
                    first_global = seq
                elif seq != first_global:
                    n_changed += 1
                seqs.append(seq)
            for _ in range(n_local):
                masked = mask_fragment_ids(
                    body, sep, mask_id, self._rng, frac=self._sample_mask_frac(),
                )
                seqs.append([b.bos_id, *masked, b.eos_id])
        ids, mask = pad_batch(seqs, pad_id=b.pad_id)
        dev = self.device
        out = self.encoder.encode_token_ids(
            ids.to(dev), mask.to(dev), return_projection=with_projection, normalize=False
        )
        z, proj = out if with_projection else (out, None)
        z_all = z.view(len(batch), n_all, -1)
        z_global = z_all[:, :n_global, :]
        diversity = n_changed / max(len(batch) * max(n_global - 1, 1), 1)
        if not with_projection:
            return z_global, z_all, diversity
        proj_global = torch.nn.functional.normalize(
            proj.view(len(batch), n_all, -1)[:, :n_global], dim=-1
        )
        return z_global, z_all, diversity, proj_global

    @staticmethod
    def _gather_hole_tokens(tok: torch.Tensor, hole: torch.Tensor) -> torch.Tensor:
        """Stack adapter reps at hole positions: ``[B,T,D]`` + ``[B,T]`` → ``[N,D]``."""
        if tok.shape[:2] != hole.shape:
            raise ValueError(f"tok/hole shape mismatch: {tuple(tok.shape)} vs {tuple(hole.shape)}")
        return tok[hole.bool()]

    @staticmethod
    def _gather_visible_tokens(
        tok: torch.Tensor, hole: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Stack visible (non-hole) adapter reps: ``[B,T,D]`` → ``[N_vis,D]``."""
        return tok[(valid & ~hole).bool()]

    def _encode_ijepa(self, batch):
        """I-JEPA views: one masked encode + intact targets at hole indices.

        Returns ``(tok_masked, hole, valid, target_ema, target_online, z_pooled,
        z_teacher_global)``.

        * ``tok_masked``: online encoder on masked locals, per-token adapter reps ``[B,T,D]``.
          Predictor reads **visible** rows only; ``<UNK>`` rows supply sequence length /
          optional attention context unless ``ijepa_block_hole_attn`` blocks hole keys.
        * ``hole`` / ``valid``: boolean masks aligned with ``tok_masked`` (post-BOS frame).
        * ``target_ema``: EMA encoder, intact view at hole columns — stop-grad endpoint.
        * ``target_online``: online encoder, intact view at hole columns — SIGReg.
        * ``z_pooled``: ``[B_mol, D]`` mean-pooled intact online ``z_m``.
        * ``z_teacher_global``: ``[B_mol, D]`` intact-view pooled latents from the EMA teacher.

        ponytail: assumes ``<UNK>`` (mask_id) does not occur naturally in the body.
        """
        b = self.encoder.bundle
        sep = int(self.hparams.frag_sep_id)
        mask_id = int(self._mask_token_id)
        n_local = int(self.hparams.lejepa_n_local_views)
        mask_mode = str(self.hparams.ijepa_mask_mode).lower()
        intact_seqs: list[list[int]] = []
        masked_seqs: list[list[int]] = []
        for s in batch:
            body = self._body_ids(s, b.tokenizer)
            ordered = shuffle_fragment_ids(body, sep, self._rng)
            intact_seqs.append([b.bos_id, *ordered, b.eos_id])
            for _ in range(n_local):
                masked = mask_local_ids(
                    ordered,
                    sep,
                    mask_id,
                    self._rng,
                    frac=self._sample_mask_frac(),
                    mode=mask_mode,
                )
                masked_seqs.append([b.bos_id, *masked, b.eos_id])
        intact_ids, intact_mask = pad_batch(intact_seqs, pad_id=b.pad_id)
        masked_ids, masked_mask = pad_batch(masked_seqs, pad_id=b.pad_id)
        dev = self.device
        intact_ids, intact_mask = intact_ids.to(dev), intact_mask.to(dev)
        masked_ids, masked_mask = masked_ids.to(dev), masked_mask.to(dev)
        hole = (masked_ids == mask_id)[:, 1:]
        valid = masked_mask[:, 1:].bool()
        hole_attn = hole if bool(self.hparams.ijepa_block_hole_attn) else None
        _, tok_masked = self.encoder.encode_token_ids(
            masked_ids,
            masked_mask,
            normalize=False,
            return_tokens=True,
            hole_mask=hole_attn,
        )
        with torch.no_grad():
            z_teacher_global, tok_ema = self.encoder_ema.encode_token_ids(
                intact_ids, intact_mask, normalize=False, return_tokens=True
            )
        z_pooled, tok_online = self.encoder.encode_token_ids(
            intact_ids, intact_mask, normalize=False, return_tokens=True
        )
        tok_ema = tok_ema.repeat_interleave(n_local, dim=0)
        tok_online = tok_online.repeat_interleave(n_local, dim=0)
        target_ema = self._gather_hole_tokens(tok_ema, hole)
        target_online = self._gather_hole_tokens(tok_online, hole)
        if target_ema.numel() == 0:
            raise RuntimeError("ijepa batch has no hole tokens — check masking")
        return tok_masked, hole, valid, target_ema, target_online, z_pooled, z_teacher_global

    @staticmethod
    def _update_ema(online: torch.nn.Module, ema: torch.nn.Module, decay: float) -> None:
        with torch.no_grad():
            for p_online, p_ema in zip(online.parameters(), ema.parameters()):
                p_ema.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    def _update_ijepa_ema(self) -> None:
        assert self.encoder_ema is not None
        self._update_ema(
            self.encoder, self.encoder_ema, float(self.hparams.ijepa_ema_decay)
        )

    def _hybrid_alpha(self) -> float:
        """Linear anneal: 1.0 (pure ntxent) at step 0 -> 0.0 (pure lejepa) by
        ``hybrid_anneal_steps``, held at 0 after."""
        anneal = int(self.hparams.hybrid_anneal_steps)
        if anneal <= 0:
            return 0.0
        return max(0.0, 1.0 - self.global_step / anneal)

    def _compute_loss(
        self,
        *,
        z_a: torch.Tensor | None = None,
        z_b: torch.Tensor | None = None,
        z_global: torch.Tensor | None = None,
        z_all: torch.Tensor | None = None,
        proj_global: torch.Tensor | None = None,
    ):
        if self.ssl_loss == "ntxent":
            assert z_a is not None and z_b is not None
            loss = self.ntxent_loss_fn(z_a, z_b)
            return loss, {
                "inv": None, "inv_rel": None, "sigreg": None,
                "ntxent": None, "alpha": None,
            }
        assert z_global is not None and z_all is not None
        terms = self.lejepa_loss_fn(z_global, z_all)
        if self.ssl_loss != "hybrid":
            if self.hparams.log_grad_norms and torch.is_grad_enabled():
                self._log_grad_norms(inv=terms.inv, sigreg=terms.sigreg)
            return terms.total, {
                "inv": terms.inv.detach(), "inv_rel": terms.inv_rel,
                "sigreg": terms.sigreg.detach(), "ntxent": None, "alpha": None,
            }
        assert proj_global is not None
        ntxent_loss = self.ntxent_loss_fn(proj_global[:, 0], proj_global[:, 1])
        alpha = self._hybrid_alpha()
        total = alpha * ntxent_loss + (1.0 - alpha) * terms.total
        if self.hparams.log_grad_norms and torch.is_grad_enabled():
            self._log_grad_norms(ntxent=ntxent_loss, inv=terms.inv, sigreg=terms.sigreg)
        return total, {
            "inv": terms.inv.detach(),
            "inv_rel": terms.inv_rel,
            "sigreg": terms.sigreg.detach(),
            "ntxent": ntxent_loss.detach(),
            "alpha": alpha,
        }

    def _log_grad_norms(self, **terms: torch.Tensor) -> None:
        """Diagnostic: L2 norm of each loss term's gradient w.r.t. the
        trainable params, via a separate ``autograd.grad`` per term
        (``retain_graph=True`` so Lightning's own backward on the *summed*
        loss afterward is unaffected). Settles which term actually drives
        updates, rather than inferring it from loss values.
        """
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        logs: dict[str, float] = {}
        for name, term in terms.items():
            grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            sq_sum = sum(g.float().pow(2).sum() for g in grads if g is not None)
            logs[f"train/grad_norm_{name}"] = float(sq_sum ** 0.5)
        self.log_dict(logs, on_step=True, batch_size=1)

    def _log_step(
        self,
        prefix: str,
        *,
        loss,
        z_a,
        z_b,
        extras,
        batch_size: int,
        prog_bar: bool,
        view_diversity: float | None = None,
        z_all: torch.Tensor | None = None,
    ) -> None:
        logs = {
            f"{prefix}/encode_time": self.encoder.encode_time_value,
        }
        if z_b is not None:
            logs[f"{prefix}/acc@1"] = top1_paired_accuracy(z_a.detach(), z_b.detach())
        elif z_all is not None and z_a is not None:
            acc = lejepa_retrieval_acc1(z_a.detach(), z_all.detach())
            if acc is not None:
                logs[f"{prefix}/acc@1"] = acc
        if extras["inv"] is not None:
            logs[f"{prefix}/inv"] = extras["inv"]
            logs[f"{prefix}/inv_rel"] = extras["inv_rel"]
            logs[f"{prefix}/sigreg"] = extras["sigreg"]
        if extras["ntxent"] is not None:
            logs[f"{prefix}/ntxent"] = extras["ntxent"]
            logs[f"{prefix}/hybrid_alpha"] = extras["alpha"]
        if view_diversity is not None:
            logs[f"{prefix}/view_diversity"] = view_diversity
        on_step, on_epoch = prefix == "train", prefix != "train"
        self.log(f"{prefix}/loss", loss.detach(), on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=batch_size)
        self.log_dict(logs, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size)

    def _rank_due(self) -> bool:
        every = int(self.hparams.train_rank_every_n_steps)
        return every > 0 and self.global_step % every == 0

    def _global_rank_metrics(self, z: torch.Tensor) -> tuple[float, float]:
        """Covariance rank of ``[B, D]`` global pooled latents."""
        flat = self._subsample_rows(
            z.detach().float(), int(self.hparams.rank_subsample),
        ).cpu().numpy()
        return embedding_covariance_rank(flat)

    def _log_global_rank(
        self, z: torch.Tensor, *, prefix: str, batch_size: int, on_step: bool,
    ) -> None:
        eff, num = self._global_rank_metrics(z)
        self.log_dict(
            {f"{prefix}/rank_effective": eff, f"{prefix}/rank_numerical": num},
            on_step=on_step,
            on_epoch=not on_step,
            batch_size=batch_size,
        )

    def _append_val_rank(self, z: torch.Tensor) -> None:
        eff, num = self._global_rank_metrics(z)
        self._val["rank_effective"].append(eff)
        self._val["rank_numerical"].append(num)

    def _online_global_z(self, z_global: torch.Tensor) -> torch.Tensor:
        """First intact global view per molecule: ``[B, Vg, D]`` → ``[B, D]``."""
        return z_global[:, 0]

    def _collapse_diag_due(self) -> bool:
        every = int(self.hparams.ijepa_collapse_diag_every_n_steps)
        return every > 0 and self.global_step % every == 0

    def _ntxent_global_z(self, views) -> torch.Tensor:
        """Unnormalized pooled z_m of view-a (online; no teacher in NT-Xent)."""
        (ids_a, mask_a), _ = self._two_views(views)
        with torch.no_grad():
            return self.encoder.encode_token_ids(ids_a, mask_a, normalize=False)

    def _log_ijepa_collapse_diag(
        self,
        tok_masked: torch.Tensor,
        hole: torch.Tensor,
        valid: torch.Tensor,
        target_ema: torch.Tensor,
        target_online: torch.Tensor,
        *,
        spectrum: bool,
        batch_size: int,
    ) -> None:
        """EMA-target collapse checks + visible-context / pred alignment."""
        assert self.ijepa_loss_fn is not None
        visible = self._gather_visible_tokens(tok_masked, hole, valid).detach().float()
        ema = target_ema.detach().float()
        online = target_online.detach().float()
        ema_norm = torch.nn.functional.normalize(ema, dim=-1)
        online_norm = torch.nn.functional.normalize(online, dim=-1)
        pred = self.ijepa_loss_fn._predict(tok_masked, hole, valid=valid).detach().float()
        pred_norm = torch.nn.functional.normalize(pred, dim=-1)
        logs: dict[str, float] = {
            "diagnostics/target_std": float(ema.std(dim=0).mean()),
            "diagnostics/target_online_std": float(online.std(dim=0).mean()),
            "diagnostics/context_std": float(visible.std(dim=0).mean()),
            "diagnostics/n_hole_tokens": float(ema.shape[0]),
            "diagnostics/cos_online_ema": float((online_norm * ema_norm).sum(-1).mean()),
            "diagnostics/cos_pred_ema": float((pred_norm * ema_norm).sum(-1).mean()),
        }
        if spectrum:
            k = int(self.hparams.rank_subsample)
            ema_s = self._subsample_rows(ema, k).cpu().numpy()
            _, top = embedding_batch_collapse_diag(ema_s, top_k=5)
            for i, v in enumerate(top, start=1):
                logs[f"diagnostics/target_eig_{i}"] = v
        self.log_dict(logs, on_step=True, batch_size=batch_size)

    # -- lifecycle ---------------------------------------------------------- #
    def training_step(self, batch, batch_idx):
        views, smiles = self._split_batch(batch)
        bs = len(views)
        log_rank = self._rank_due()
        log_collapse = self._collapse_diag_due()
        z_rank = None
        if self.ssl_loss == "ntxent":
            z_a, z_b, z_a_pooled = self._encode_ntxent_views(views)
            loss, extras = self._compute_loss(z_a=z_a, z_b=z_b)
            fp_loss = self._fp_distillation(z_a_pooled, smiles)
            if fp_loss is not None:
                self.log("train/fp", fp_loss.detach(), on_step=True, batch_size=bs)
                loss = loss + self.fp_weight * fp_loss
            self._log_step(
                "train", loss=loss, z_a=z_a, z_b=z_b, extras=extras,
                batch_size=bs, prog_bar=True,
            )
            if log_rank:
                z_rank = self._ntxent_global_z(views)
        elif self.ssl_loss == "ijepa":
            (
                tok_masked, hole, valid, target_ema, target_online, z_pooled,
                z_teacher_global,
            ) = self._encode_ijepa(views)
            terms = self.ijepa_loss_fn(
                tok_masked, hole, target_ema, valid=valid, target_sigreg=target_online,
            )
            loss = terms.total
            fp_loss = self._fp_distillation(z_pooled, smiles)
            if fp_loss is not None:
                self.log("train/fp", fp_loss.detach(), on_step=True, batch_size=bs)
                loss = loss + self.fp_weight * fp_loss
            if self.hparams.log_grad_norms and torch.is_grad_enabled():
                self._log_grad_norms(predict=terms.predict, sigreg=terms.sigreg)
            self.log("train/loss", loss.detach(), on_step=True, prog_bar=True, batch_size=bs)
            self.log_dict(
                {
                    "train/predict": terms.predict.detach(),
                    "train/sigreg": terms.sigreg.detach(),
                    "train/sigreg_context": terms.sigreg_context.detach(),
                    "train/encode_time": self.encoder.encode_time_value,
                },
                on_step=True, batch_size=bs,
            )
            if log_collapse:
                self._log_ijepa_collapse_diag(
                    tok_masked, hole, valid, target_ema, target_online,
                    spectrum=True, batch_size=bs,
                )
            if log_rank:
                z_rank = z_teacher_global
        else:
            with_proj = self.ssl_loss == "hybrid"
            out = self._encode_lejepa_views(views, with_projection=with_proj)
            z_global, z_all, diversity = out[0], out[1], out[2]
            proj_global = out[3] if with_proj else None
            loss, extras = self._compute_loss(
                z_global=z_global, z_all=z_all, proj_global=proj_global,
            )
            fp_loss = self._fp_distillation(z_global[:, 0], smiles)
            if fp_loss is not None:
                self.log("train/fp", fp_loss.detach(), on_step=True, batch_size=bs)
                loss = loss + self.fp_weight * fp_loss
            self._log_step(
                "train", loss=loss, z_a=z_global, z_b=None, extras=extras,
                batch_size=bs, prog_bar=True, view_diversity=diversity, z_all=z_all,
            )
            if log_rank:
                z_rank = self._online_global_z(z_global)
        if log_rank and z_rank is not None:
            self._log_global_rank(z_rank, prefix="train", batch_size=bs, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        views, smiles = self._split_batch(batch)
        z_rank = None
        rank_on = int(self.hparams.train_rank_every_n_steps) > 0
        if self.ssl_loss == "ntxent":
            z_a, z_b, z_a_pooled = self._encode_ntxent_views(views)
            loss, extras = self._compute_loss(z_a=z_a, z_b=z_b)
            self._val["acc"].append(top1_paired_accuracy(z_a, z_b))
            fp_loss = self._fp_distillation(z_a_pooled, smiles)
            if fp_loss is not None:
                self._val["fp"].append(float(fp_loss))
                loss = loss + self.fp_weight * fp_loss
            if rank_on:
                z_rank = self._ntxent_global_z(views)
        elif self.ssl_loss == "ijepa":
            (
                tok_masked, hole, valid, target_ema, target_online, z_pooled,
                z_teacher_global,
            ) = self._encode_ijepa(views)
            terms = self.ijepa_loss_fn(
                tok_masked, hole, target_ema, valid=valid, target_sigreg=target_online,
            )
            loss = terms.total
            fp_loss = self._fp_distillation(z_pooled, smiles)
            if fp_loss is not None:
                self._val["fp"].append(float(fp_loss))
                loss = loss + self.fp_weight * fp_loss
            extras = {"inv": None, "ntxent": None}
            self._val["predict"].append(float(terms.predict))
            self._val["sigreg"].append(float(terms.sigreg))
            self._val["sigreg_context"].append(float(terms.sigreg_context))
            gap = self.ijepa_loss_fn.condition_bypass_gap(
                tok_masked, hole, target_ema, valid=valid,
            )
            self._val["cond_true"].append(gap["predict_true"])
            self._val["cond_shuf"].append(gap["predict_shuf"])
            self._val["cond_zero"].append(gap["predict_zero"])
            self._val["cond_gap_zero"].append(gap["gap_zero"])
            self._val["cond_gap_shuf"].append(gap["gap_shuf"])
            if rank_on:
                z_rank = z_teacher_global
        else:
            with_proj = self.ssl_loss == "hybrid"
            out = self._encode_lejepa_views(views, with_projection=with_proj)
            z_global, z_all, diversity = out[0], out[1], out[2]
            proj_global = out[3] if with_proj else None
            loss, extras = self._compute_loss(
                z_global=z_global, z_all=z_all, proj_global=proj_global,
            )
            acc = lejepa_retrieval_acc1(z_global, z_all)
            if acc is not None:
                self._val["acc"].append(acc)
            self._val["view_diversity"].append(diversity)
            fp_loss = self._fp_distillation(z_global[:, 0], smiles)
            if fp_loss is not None:
                self._val["fp"].append(float(fp_loss))
                loss = loss + self.fp_weight * fp_loss
            if rank_on:
                z_rank = self._online_global_z(z_global)
        if rank_on and z_rank is not None:
            self._append_val_rank(z_rank)
        self._val["loss"].append(loss.item())
        if extras["inv"] is not None:
            self._val["inv"].append(float(extras["inv"]))
            self._val["inv_rel"].append(float(extras["inv_rel"]))
            self._val["sigreg"].append(float(extras["sigreg"]))
        if extras["ntxent"] is not None:
            self._val["ntxent"].append(float(extras["ntxent"]))

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.ssl_loss == "ijepa":
            self._update_ijepa_ema()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Embed the encoder skeleton config so the ckpt is self-describing.

        Downstream (Stage-5 EBM and every eval/precompute CLI) rebuilds the
        encoder via :func:`load_encoder_from_ckpt` straight from this file —
        including the DDiT hook layer range, which is not recoverable from the
        weights and would otherwise have to be re-specified (and could drift).
        """
        cfg = getattr(self.encoder, "build_config", None)
        if cfg is not None:
            checkpoint["encoder_config"] = dict(cfg)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Backfill ``encoder_ema`` when loading a pre-EMA checkpoint."""
        if self.encoder_ema is None:
            return
        state = checkpoint.get("state_dict", {})
        if any(k.startswith("encoder_ema.") for k in state):
            return
        self.encoder_ema.load_state_dict(self.encoder.state_dict())

    def on_fit_start(self) -> None:
        trainer = getattr(self, "trainer", None)
        dm = getattr(trainer, "datamodule", None) if trainer is not None else None
        if dm is None:
            return
        self._val_probes.val_ratio = float(dm.val_ratio)
        self._val_probes.test_ratio = float(dm.test_ratio)
        self._val_probes.split_seed = int(dm.split_seed)
        self._val_probes.prepare(dm.shard_dir)

    def on_validation_epoch_end(self) -> None:
        out = {
            "val/loss": float(np.mean(self._val["loss"])) if self._val["loss"] else float("nan"),
            "val/acc@1": float(np.mean(self._val["acc"])) if self._val["acc"] else 0.0,
            "val/encode_time": self.encoder.encode_time_value,
        }
        if self._val["inv"]:
            out["val/inv"] = float(np.mean(self._val["inv"]))
            out["val/inv_rel"] = float(np.mean(self._val["inv_rel"]))
            out["val/sigreg"] = float(np.mean(self._val["sigreg"]))
        if self._val["predict"]:
            out["val/predict"] = float(np.mean(self._val["predict"]))
            out["val/sigreg"] = float(np.mean(self._val["sigreg"]))
            out["val/sigreg_context"] = float(np.mean(self._val["sigreg_context"]))
        if self._val["cond_true"]:
            out["val/cond_true"] = float(np.mean(self._val["cond_true"]))
            out["val/cond_shuf"] = float(np.mean(self._val["cond_shuf"]))
            out["val/cond_zero"] = float(np.mean(self._val["cond_zero"]))
            out["val/cond_gap_zero"] = float(np.mean(self._val["cond_gap_zero"]))
            out["val/cond_gap_shuf"] = float(np.mean(self._val["cond_gap_shuf"]))
        if self._val["view_diversity"]:
            out["val/view_diversity"] = float(np.mean(self._val["view_diversity"]))
        if self._val["fp"]:
            out["val/fp"] = float(np.mean(self._val["fp"]))
        if self._val["ntxent"]:
            out["val/ntxent"] = float(np.mean(self._val["ntxent"]))
            out["val/hybrid_alpha"] = self._hybrid_alpha()
        if self._val["rank_effective"]:
            out["val/rank_effective"] = float(np.mean(self._val["rank_effective"]))
            out["val/rank_numerical"] = float(np.mean(self._val["rank_numerical"]))
        out.update(self._val_probes.maybe_run(self))
        self.log_dict(out, prog_bar=True, sync_dist=True)
        if self._val["cond_gap_zero"]:
            margin = float(self.hparams.condition_margin)
            gap_z = float(np.mean(self._val["cond_gap_zero"]))
            gap_s = float(np.mean(self._val["cond_gap_shuf"]))
            if gap_z < margin and gap_s < margin:
                logger.warning(
                    "predictor may be ignoring context: "
                    "gap_zero=%.4f gap_shuf=%.4f < margin=%.4f "
                    "(predict_true=%.4f predict_zero=%.4f predict_shuf=%.4f)",
                    gap_z, gap_s, margin,
                    float(np.mean(self._val["cond_true"])),
                    float(np.mean(self._val["cond_zero"])),
                    float(np.mean(self._val["cond_shuf"])),
                )
        for k in self._val:
            self._val[k].clear()

    def configure_optimizers(self):
        hp = self.hparams
        # Trainable encoder params = adapter, the DDiT backbone (when
        # encoder.freeze_backbone is false), and the encode-time parameter (when
        # learnable). Whatever has requires_grad goes in. time_logit is a scalar
        # gate — no weight decay.
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, p in self.encoder.named_parameters():
            if not p.requires_grad:
                continue
            if name == "time_logit":
                no_decay.append(p)
            else:
                decay.append(p)
        # ijepa's flow head lives on the loss module (not the encoder); it must
        # train too. (lejepa/ntxent loss modules have no trainable params.)
        if self.ijepa_loss_fn is not None:
            decay.extend(p for p in self.ijepa_loss_fn.parameters() if p.requires_grad)
        groups: list[dict] = []
        if decay:
            groups.append({"params": decay, "weight_decay": hp.weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        optim = torch.optim.AdamW(groups, lr=hp.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: cosine_with_warmup(s, hp.warmup_steps, hp.total_steps)
        )
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
