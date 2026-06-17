"""SSL for the discrete-flow backbone over fragment-shuffle views.

Each batch is a list of fragmented-SMILES strings. Per molecule we tokenize once
and produce two views by shuffling the fragment order at the token level (split
on the separator id, shuffle, rejoin). Both views go through the *same* encoder,
so the (optionally learnable) encode-time is shared across them by construction.
Only the adapter and — when enabled — the encode-time parameter train; the DDiT
backbone stays frozen.

Losses (``ssl_loss``):
  * ``ntxent`` (default) — symmetric NT-Xent / InfoNCE on projection head outputs.
  * ``lejepa`` — LeJEPA with global (fragment-shuffle) + local (one masked
    fragment per view) on unnormalized pooled latents.
  * ``hybrid`` — NT-Xent on the first two global views' projections, linearly
    annealed (1.0 -> 0.0 over ``hybrid_anneal_steps``) in favor of the LeJEPA
    loss on all views. Motivation: LeJEPA alone reaches near-full numerical
    rank but very low effective rank, because nothing in its objective pushes
    different molecules apart between-sample — NT-Xent's explicit pairwise
    repulsion supplies that directly while it's annealed in.
"""

from __future__ import annotations

import logging
import random

import lightning as L
import numpy as np
import torch

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder, pad_batch
from lattice_lab.backbone.discrete_flow import resolve_mask_token_id
from lattice_lab.data.fragment_views import mask_fragment_ids, shuffle_fragment_ids
from lattice_lab.models.schedules import cosine_with_warmup
from lattice_lab.training.ssl_loss import (
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
    _l2_normalize_rows,
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
        lejepa_mask_token_id: int | None = None,
        sigreg_num_projections: int = 256,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_eps: float = 1e-8,
        # hybrid only: linear anneal of the ntxent/lejepa mix weight, alpha =
        # max(0, 1 - global_step/hybrid_anneal_steps) -- 1.0 (pure ntxent) at
        # step 0 down to 0.0 (pure lejepa) by this step, then held at 0.
        hybrid_anneal_steps: int = 2000,
        # Cheap covariance-rank diagnostic logged every N training steps from
        # the current batch's raw pooled latents (both methods, same metric) —
        # far finer-grained than the val probe's val_check_interval, so the
        # early-training rank trajectory (e.g. LeJEPA's invariance-vs-SIGReg
        # tension in the first ~1-2k steps) is actually visible. 0 disables.
        train_rank_every_n_steps: int = 50,
        # Diagnostic: per-loss-term gradient L2 norm w.r.t. trainable params,
        # via a separate torch.autograd.grad call per term (retain_graph=True,
        # doesn't disturb Lightning's own backward on the summed loss).
        # Settles "which term is actually driving updates" directly instead of
        # inferring it from (easily misleading) loss *values*. One extra
        # backward-equivalent pass per term every training step -- off by
        # default, only meant for short diagnostic runs.
        log_grad_norms: bool = False,
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
        if ssl_loss not in {"ntxent", "lejepa", "hybrid"}:
            raise ValueError(
                f"ssl_loss must be 'ntxent', 'lejepa', or 'hybrid', got {ssl_loss!r}"
            )
        self.ssl_loss = ssl_loss
        # "hybrid" reuses lejepa's global+local view construction (it needs the
        # masked local views for SIGReg too), just also takes a projection for
        # the annealed ntxent term.
        uses_lejepa_views = ssl_loss in ("lejepa", "hybrid")
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
            "loss": [], "acc": [], "inv": [], "sigreg": [], "view_diversity": [], "fp": [],
            "ntxent": [],
        }
        self._mask_token_id = (
            resolve_mask_token_id(
                encoder.bundle.tokenizer, override=lejepa_mask_token_id,
            )
            if uses_lejepa_views
            else None
        )
        if uses_lejepa_views and int(lejepa_n_global_views) < 1:
            raise ValueError(
                f"lejepa_n_global_views must be >= 1, got {lejepa_n_global_views}"
            )
        if ssl_loss == "hybrid" and int(lejepa_n_global_views) < 2:
            raise ValueError(
                "hybrid loss pairs the first two global views for ntxent; "
                f"need lejepa_n_global_views >= 2, got {lejepa_n_global_views}"
            )
        logger.info(
            "discrete-flow SSL: ssl_loss=%s lejepa_global=%d lejepa_local=%d "
            "mask_id=%s hybrid_anneal_steps=%s learnable_time=%s init_time=%.3f "
            "frag_sep_id=%d",
            ssl_loss,
            lejepa_n_global_views if uses_lejepa_views else 0,
            lejepa_n_local_views if uses_lejepa_views else 0,
            self._mask_token_id if uses_lejepa_views else "n/a",
            hybrid_anneal_steps if ssl_loss == "hybrid" else "n/a",
            encoder.learnable_time,
            self.encoder.encode_time_value,
            frag_sep_id,
        )

    # -- fragment-shuffle augmentation -------------------------------------- #
    def _two_views(self, view_strings: list[str]):
        """Tokenize each view, build two fragment-shuffled token sequences,
        return two padded ``(ids, mask)`` batches on the module device."""
        b = self.encoder.bundle
        sep = int(self.hparams.frag_sep_id)
        seqs_a: list[list[int]] = []
        seqs_b: list[list[int]] = []
        for s in view_strings:
            body = b.tokenizer.encode(s, add_special_tokens=False)
            sa = shuffle_fragment_ids(body, sep, self._rng)
            sb = shuffle_fragment_ids(body, sep, self._rng)
            seqs_a.append([b.bos_id, *sa, b.eos_id])
            seqs_b.append([b.bos_id, *sb, b.eos_id])
        ids_a, mask_a = pad_batch(seqs_a, pad_id=b.pad_id)
        ids_b, mask_b = pad_batch(seqs_b, pad_id=b.pad_id)
        dev = self.device
        return (ids_a.to(dev), mask_a.to(dev)), (ids_b.to(dev), mask_b.to(dev))

    @staticmethod
    def _split_batch(batch) -> tuple[list[str], list[str] | None]:
        """Accept either ``list[str]`` views or ``(views, smiles)`` tuples."""
        if isinstance(batch, tuple) and len(batch) == 2:
            views, smiles = batch
            return list(views), list(smiles)
        return list(batch), None

    def _encode_ntxent_views(self, views, *, with_raw_pooled: bool = False):
        """Return ``(z_a_proj, z_b_proj, z_a_pooled, raw_pooled)``.

        ``z_*_proj`` are the SimCLR projection outputs (NT-Xent loss);
        ``z_a_pooled`` is the L2-normalized z_m of view a, used by the optional
        Tanimoto similarity-distillation loss. When ``with_raw_pooled`` is set,
        also runs two extra no-grad encodes (same shuffled tokens, ``normalize=
        False``) to get both views' *raw* pooled latents stacked ``[B, 2, D]``
        for the train-rank diagnostic — ``None`` otherwise.
        """
        (ids_a, mask_a), (ids_b, mask_b) = self._two_views(views)
        z_a_pooled, z_a = self.encoder.encode_token_ids(ids_a, mask_a, return_projection=True)
        _, z_b = self.encoder.encode_token_ids(ids_b, mask_b, return_projection=True)
        raw_pooled = None
        if with_raw_pooled:
            with torch.no_grad():
                raw_a = self.encoder.encode_token_ids(ids_a, mask_a, normalize=False)
                raw_b = self.encoder.encode_token_ids(ids_b, mask_b, normalize=False)
            raw_pooled = torch.stack([raw_a, raw_b], dim=1)
        return z_a, z_b, z_a_pooled, raw_pooled

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
            body = b.tokenizer.encode(s, add_special_tokens=False)
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
                masked = mask_fragment_ids(body, sep, mask_id, self._rng)
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
            return loss, {"inv": None, "sigreg": None, "ntxent": None, "alpha": None}
        assert z_global is not None and z_all is not None
        terms = self.lejepa_loss_fn(z_global, z_all)
        if self.ssl_loss != "hybrid":
            if self.hparams.log_grad_norms and torch.is_grad_enabled():
                self._log_grad_norms(inv=terms.inv, sigreg=terms.sigreg)
            return terms.total, {
                "inv": terms.inv.detach(), "sigreg": terms.sigreg.detach(),
                "ntxent": None, "alpha": None,
            }
        assert proj_global is not None
        ntxent_loss = self.ntxent_loss_fn(proj_global[:, 0], proj_global[:, 1])
        alpha = self._hybrid_alpha()
        total = alpha * ntxent_loss + (1.0 - alpha) * terms.total
        if self.hparams.log_grad_norms and torch.is_grad_enabled():
            self._log_grad_norms(ntxent=ntxent_loss, inv=terms.inv, sigreg=terms.sigreg)
        return total, {
            "inv": terms.inv.detach(),
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
            logs[f"{prefix}/sigreg"] = extras["sigreg"]
        if extras["ntxent"] is not None:
            logs[f"{prefix}/ntxent"] = extras["ntxent"]
            logs[f"{prefix}/hybrid_alpha"] = extras["alpha"]
        if view_diversity is not None:
            logs[f"{prefix}/view_diversity"] = view_diversity
        on_step, on_epoch = prefix == "train", prefix != "train"
        self.log(f"{prefix}/loss", loss.detach(), on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=batch_size)
        self.log_dict(logs, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size)

    def _train_rank_due(self) -> bool:
        every = int(self.hparams.train_rank_every_n_steps)
        return every > 0 and self.global_step % every == 0

    def _log_train_rank(self, z_multiview: torch.Tensor) -> None:
        """Covariance-rank diagnostic from the current batch's raw pooled
        latents, both as-is and L2-normalized (mirrors ``embedding_covariance_
        rank`` usage in the val probe) — same metric for ntxent and lejepa so
        the early-training trajectory is directly comparable.
        """
        flat_raw = (
            z_multiview.detach().float().reshape(-1, z_multiview.shape[-1]).cpu().numpy()
        )
        eff_raw, num_raw = embedding_covariance_rank(flat_raw)
        eff_norm, num_norm = embedding_covariance_rank(_l2_normalize_rows(flat_raw))
        self.log_dict(
            {
                "train/rank_effective_raw": eff_raw,
                "train/rank_numerical_raw": num_raw,
                "train/rank_effective_norm": eff_norm,
                "train/rank_numerical_norm": num_norm,
            },
            on_step=True,
            batch_size=z_multiview.shape[0],
        )

    # -- lifecycle ---------------------------------------------------------- #
    def training_step(self, batch, batch_idx):
        views, smiles = self._split_batch(batch)
        bs = len(views)
        log_rank = self._train_rank_due()
        if self.ssl_loss == "ntxent":
            z_a, z_b, z_a_pooled, raw_pooled = self._encode_ntxent_views(
                views, with_raw_pooled=log_rank,
            )
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
                self._log_train_rank(raw_pooled)
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
                self._log_train_rank(z_all)
        return loss

    def validation_step(self, batch, batch_idx):
        views, smiles = self._split_batch(batch)
        if self.ssl_loss == "ntxent":
            z_a, z_b, z_a_pooled, _ = self._encode_ntxent_views(views)
            loss, extras = self._compute_loss(z_a=z_a, z_b=z_b)
            self._val["acc"].append(top1_paired_accuracy(z_a, z_b))
            fp_loss = self._fp_distillation(z_a_pooled, smiles)
            if fp_loss is not None:
                self._val["fp"].append(float(fp_loss))
                loss = loss + self.fp_weight * fp_loss
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
        self._val["loss"].append(loss.item())
        if extras["inv"] is not None:
            self._val["inv"].append(float(extras["inv"]))
            self._val["sigreg"].append(float(extras["sigreg"]))
        if extras["ntxent"] is not None:
            self._val["ntxent"].append(float(extras["ntxent"]))

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
            out["val/sigreg"] = float(np.mean(self._val["sigreg"]))
        if self._val["view_diversity"]:
            out["val/view_diversity"] = float(np.mean(self._val["view_diversity"]))
        if self._val["fp"]:
            out["val/fp"] = float(np.mean(self._val["fp"]))
        if self._val["ntxent"]:
            out["val/ntxent"] = float(np.mean(self._val["ntxent"]))
            out["val/hybrid_alpha"] = self._hybrid_alpha()
        out.update(self._val_probes.maybe_run(self))
        self.log_dict(out, prog_bar=True, sync_dist=True)
        for k in self._val:
            self._val[k].clear()

    def configure_optimizers(self):
        hp = self.hparams
        # Trainable encoder params = adapter (+ encode-time parameter if learnable);
        # the DDiT backbone is frozen. time_logit is a scalar gate — no weight decay.
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, p in self.encoder.named_parameters():
            if not p.requires_grad:
                continue
            if name == "time_logit":
                no_decay.append(p)
            else:
                decay.append(p)
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
