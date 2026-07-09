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
  * ``siglip`` — same two-view setup as ntxent, but a sigmoid pairwise loss
    (SigLIP) with a learnable temperature + bias instead of softmax InfoNCE.
  * ``lejepa`` — invariance loss + SIGReg on unnormalized pooled latents: every
    view (intact shuffles + masked) is pulled directly toward the molecule's
    intact center (no predictor), while SIGReg keeps the target batch isotropic.
    Needs >= 1 global (intact) view; masked local views supply the augmentation.
  * ``hybrid`` — NT-Xent on the first two global views' projections, linearly
    annealed (1.0 -> 0.0 over ``hybrid_anneal_steps``) in favor of the LeJEPA
    loss.
  * ``ijepa`` — I-JEPA: one corrupted encoder pass (selected positions replaced
    with uniform random body tokens), visible adapter reps + learnable mask-token
    predictor vs EMA intact targets at hole indices.
    One intact view per molecule plus ``lejepa_n_local_views`` masked locals.
    Masking is fragment-, span-, or mixed-mode (``ijepa_mask_mode``). With
    ``ijepa_gram_weight > 0``, adds DINOv3 Gram anchoring: the online intact
    per-token (patch) Gram matrix is pulled toward the stop-grad EMA teacher's,
    constraining second-order patch statistics to stabilize dense features.
"""

from __future__ import annotations

import copy
import logging
import math
import random

import lightning as L
import numpy as np
import torch

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder, ijepa_noise_token_pool, pad_batch
from lattice_lab.backbone.discrete_flow import resolve_mask_token_id
from lattice_lab.data.fragment_views import (
    join_fragment_ids,
    mask_frags,
    mask_local_frags,
    noise_local_frags,
    shuffle_frags,
    split_fragment_ids,
)
from lattice_lab.models.schedules import cosine_ema_decay, cosine_with_warmup
from lattice_lab.training.ssl_loss import (
    DINOHead,
    DINOLoss,
    IJEPALoss,
    LeJEPALoss,
    NTXentLoss,
    PooledEmbeddingPredictor,
    SigLIPLoss,
    VISReg,
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


def _adamw_no_decay(name: str, p: torch.nn.Parameter) -> bool:
    """AdamW convention: decay weight matrices only, not bias/norm/scalars."""
    return (
        p.ndim < 2
        or name.endswith(".bias")
        or "norm" in name.lower()
        or name == "time_logit"
    )


class DiscreteFlowSSLModule(L.LightningModule):
    def __init__(
        self,
        encoder: DiscreteFlowEncoder,
        *,
        frag_sep_id: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        temperature: float = 0.1,
        fp_weight: float = 0.0,
        fp_radius: int = 2,
        fp_bits: int = 2048,
        # Linear anneal of fp_weight to 0 over this many optimizer steps; 0 = constant.
        fp_anneal_steps: int = 0,
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
        # + adapter on the masked encode pass.
        ijepa_block_hole_attn: bool = False,
        ijepa_predictor_layers: int = 1,
        ijepa_predictor_heads: int = 2,
        ijepa_predictor_ff_mult: int = 2,
        # Token-level masked prediction (the core I-JEPA objective). Set 0 to drop it
        # — with glob/gram also off, locals become independent fragment shuffles and
        # match the teacher purely via the pooled noise-invariance term.
        ijepa_predict_weight: float = 1.0,
        ijepa_glob_weight: float = 1.0,
        ijepa_inv_weight: float = 0.1,
        # Noise invariance: whole-molecule mean-pool of the uniform-noise-corrupted
        # view pulled toward the stop-grad EMA-teacher clean global. 0 disables.
        ijepa_noise_inv_weight: float = 0.0,
        # DINOv3 Gram anchoring: match online intact patch-Gram to the EMA teacher's
        # (stop-grad). 0 disables. Stabilizes dense features over long schedules.
        ijepa_gram_weight: float = 0.0,
        # DINO prototype head: student (online noised locals) matches a centered,
        # sharpened stop-grad EMA-teacher prototype distribution of the clean global.
        # Coexists additively with the other terms. 0 disables (no head is built).
        ijepa_dino_weight: float = 0.0,
        ijepa_dino_out_dim: int = 4096,
        ijepa_dino_hidden: int = 2048,
        ijepa_dino_bottleneck: int = 256,
        ijepa_dino_teacher_temp: float = 0.04,
        ijepa_dino_student_temp: float = 0.1,
        ijepa_dino_center_momentum: float = 0.9,
        # DINO student input: ``pool`` = noised mean-pool → prototypes directly;
        # ``predict`` = mean-pool → MLP predictor → prototypes (asymmetric views).
        ijepa_dino_student: str = "predict",
        # Replace SIGReg with VISReg (variance + sliced-Wasserstein sketching) as
        # the I-JEPA pooled-z_m anti-collapse regularizer (arXiv:2606.02572).
        ijepa_use_visreg: bool = False,
        ijepa_visreg_gamma: float = 1.0,
        ijepa_visreg_shape_coeff: float = 1.0,
        ijepa_visreg_num_projections: int = 4096,
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
        # Per-loss-term gradient L2 norm w.r.t. trainable params (separate
        # autograd.grad per term, retain_graph=True). One extra backward-equivalent
        # pass per term every training step; off by default.
        log_grad_norms: bool = False,
        # ijepa only: warn when the predictor ignores context (see on_validation_epoch_end).
        condition_margin: float = 0.01,
        val_probe_n_molecules: int = 2000,
        val_probe_every_n_epochs: int = 1,
        val_probe_encode_batch_size: int = 128,
        val_probe_ridge_alpha: float = 1.0,
        val_probe_test_size: float = 0.2,
        val_probe_tsne_perplexity: float | None = None,
        # Cross-modal 3D view: a Uni-Mol point-cloud co-encoder whose pooled
        # embedding a predictor regresses from the 2D pooled z_m. LeJEPA-style —
        # symmetric alignment (no EMA/stop-grad), collapse prevented by VISReg on
        # each modality. 0 disables. ``encoder_3d`` is Hydra-instantiated.
        encoder_3d: torch.nn.Module | None = None,
        view3d_weight: float = 0.0,
        # Cross-modal alignment metric: "l2" (LeJEPA MSE on raw embeddings; scale
        # kept safe by VISReg on each modality) or "cosine".
        view3d_loss: str = "l2",
        # Also apply VISReg to the 1D (2D adapter) pooled z_m in the cross-modal
        # term, not just the 3D modality. The intended pairing is
        # ``ssl_loss=ntxent`` (NT-Xent as the within-1D-modality objective) with
        # the LeJEPA cross-modal predictor (1D -> 3D): NT-Xent stops 1D collapse,
        # VISReg on 1D *and* 3D keeps both encoders isotropic under the symmetric
        # (no stop-grad) alignment. Redundant for ssl_loss=lejepa/ijepa, which
        # already regularize the 1D z_m.
        view3d_visreg_1d: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "encoder_3d"])
        self.encoder = encoder
        ssl_loss = ssl_loss.lower()
        if ssl_loss not in {"ntxent", "siglip", "lejepa", "hybrid", "ijepa"}:
            raise ValueError(
                f"ssl_loss must be 'ntxent', 'siglip', 'lejepa', 'hybrid', or "
                f"'ijepa', got {ssl_loss!r}"
            )
        self.ssl_loss = ssl_loss
        # ntxent and siglip share the same two-view (projection) contrastive path;
        # they differ only in the loss applied to the paired projections.
        self._contrastive = ssl_loss in ("ntxent", "siglip")
        # "hybrid" reuses lejepa's global+local view construction (it needs the
        # masked local views for SIGReg too), just also takes a projection for
        # the annealed ntxent term.
        uses_lejepa_views = ssl_loss in ("lejepa", "hybrid")
        # ijepa doesn't use the global/local pooled-view stack, but it does use
        # fragment masking (mask token + mask fraction) to build its hole.
        uses_masking = uses_lejepa_views or ssl_loss == "ijepa"
        # Optional Morgan-fingerprint similarity distillation on an L2-normalized
        # copy of z_m (orthogonal to the primary loss's normalization choice).
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
        self.siglip_loss_fn = SigLIPLoss() if ssl_loss == "siglip" else None
        self.lejepa_loss_fn = (
            LeJEPALoss(
                lejepa_lambda=lejepa_lambda,
                sigreg_num_projections=sigreg_num_projections,
                sigreg_knots=sigreg_knots,
                sigreg_t_max=sigreg_t_max,
                sigreg_eps=sigreg_eps,
                use_visreg=ijepa_use_visreg,
                visreg_gamma=ijepa_visreg_gamma,
                visreg_shape_coeff=ijepa_visreg_shape_coeff,
                visreg_num_projections=ijepa_visreg_num_projections,
            )
            if uses_lejepa_views
            else None
        )
        self.ijepa_loss_fn = (
            IJEPALoss(
                dim=encoder.adapter.d_adapter,
                lejepa_lambda=lejepa_lambda,
                sigreg_num_projections=sigreg_num_projections,
                sigreg_knots=sigreg_knots,
                sigreg_t_max=sigreg_t_max,
                sigreg_eps=sigreg_eps,
                use_visreg=ijepa_use_visreg,
                visreg_gamma=ijepa_visreg_gamma,
                visreg_shape_coeff=ijepa_visreg_shape_coeff,
                visreg_num_projections=ijepa_visreg_num_projections,
                ijepa_predictor_layers=ijepa_predictor_layers,
                ijepa_predictor_heads=ijepa_predictor_heads,
                ijepa_predictor_ff_mult=ijepa_predictor_ff_mult,
                predict_weight=ijepa_predict_weight,
                glob_weight=ijepa_glob_weight,
                inv_weight=ijepa_inv_weight,
                noise_inv_weight=ijepa_noise_inv_weight,
                gram_weight=ijepa_gram_weight,
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
        # Cross-modal 3D co-encoder + pooled predictor. LeJEPA-style: symmetric
        # alignment (z_2d -> pred_3d -> cos to the *online* z_3d, no EMA/stop-grad),
        # collapse prevented by VISReg on each modality. Mode-agnostic (any ssl_loss).
        self.encoder_3d = encoder_3d
        self.pred_3d: PooledEmbeddingPredictor | None = None
        self.visreg_3d: VISReg | None = None
        self.visreg_2d: VISReg | None = None
        self.view3d_weight = float(view3d_weight)
        self.view3d_loss = str(view3d_loss).lower()
        self.view3d_visreg_1d = bool(view3d_visreg_1d)
        if self.view3d_loss not in ("l2", "cosine"):
            raise ValueError(f"view3d_loss must be 'l2' or 'cosine', got {view3d_loss!r}")
        # With dual attention pooling the contrastive (siglip/ntxent) objective owns
        # the projection pool half; the cross-modal LeJEPA prediction (and its 1D
        # VISReg anchor) then reads *only* the main (regression) pool half so the two
        # 1D objectives stay decoupled. z_m stays the full concat for downstream use.
        self._view3d_main_half = self._contrastive and encoder.adapter.dual_attn_pool
        self._d_pool = int(encoder.adapter.d_pool)
        if encoder_3d is not None and self.view3d_weight > 0.0:
            d_3d = int(getattr(encoder_3d, "output_dim", encoder.adapter.d_adapter))
            if d_3d != encoder.adapter.d_adapter:
                raise ValueError(
                    f"3D encoder output_dim ({d_3d}) must match d_adapter "
                    f"({encoder.adapter.d_adapter}); add a projection otherwise"
                )
            pred_in = self._d_pool if self._view3d_main_half else encoder.adapter.d_adapter
            self.pred_3d = PooledEmbeddingPredictor(
                pred_in, out_dim=encoder.adapter.d_adapter
            )
            visreg_kwargs = dict(
                gamma=ijepa_visreg_gamma,
                shape_coeff=ijepa_visreg_shape_coeff,
                num_projections=ijepa_visreg_num_projections,
            )
            self.visreg_3d = VISReg(**visreg_kwargs)
            # Optional anti-collapse on the 1D modality (unnormalized pooled z_m),
            # for the ntxent + cross-modal-LeJEPA pairing (see __init__ docstring).
            if self.view3d_visreg_1d:
                self.visreg_2d = VISReg(**visreg_kwargs)
        self.dino_head: DINOHead | None = None
        self.dino_head_ema: DINOHead | None = None
        self.dino_pool_predictor: PooledEmbeddingPredictor | None = None
        self.dino_loss_fn: DINOLoss | None = None
        if ssl_loss == "ijepa" and float(ijepa_dino_weight) > 0.0:
            dino_student = str(ijepa_dino_student).lower()
            if dino_student not in ("pool", "predict"):
                raise ValueError(
                    f"ijepa_dino_student must be 'pool' or 'predict', got {ijepa_dino_student!r}"
                )
            dino_head_kwargs = dict(
                dim=encoder.adapter.d_adapter,
                out_dim=int(ijepa_dino_out_dim),
                hidden=int(ijepa_dino_hidden),
                bottleneck=int(ijepa_dino_bottleneck),
            )
            self.dino_head = DINOHead(**dino_head_kwargs)
            self.dino_head_ema = DINOHead(**dino_head_kwargs)
            self.dino_head_ema.load_state_dict(self.dino_head.state_dict())
            self.dino_head_ema.eval()
            for p in self.dino_head_ema.parameters():
                p.requires_grad = False
            if dino_student == "predict":
                self.dino_pool_predictor = PooledEmbeddingPredictor(
                    encoder.adapter.d_adapter,
                )
            self.dino_loss_fn = DINOLoss(
                int(ijepa_dino_out_dim),
                teacher_temp=float(ijepa_dino_teacher_temp),
                student_temp=float(ijepa_dino_student_temp),
                center_momentum=float(ijepa_dino_center_momentum),
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
            "loss": [], "loss_1d": [], "acc": [], "inv": [], "noise_inv": [], "sigreg": [],
            "predict": [], "glob": [], "gram": [], "dino": [],
            "dino_teacher_entropy": [], "dino_active_prototypes": [],
            "view_diversity": [], "fp": [], "ntxent": [],
            "view3d": [], "view3d_reg": [], "view3d_reg_1d": [], "view3d_mix": [],
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
        self._ijepa_noise_pool: tuple[int, ...] = ()
        if ssl_loss == "ijepa":
            b = encoder.bundle
            assert self._mask_token_id is not None
            self._ijepa_noise_pool = ijepa_noise_token_pool(
                vocab_size=b.vocab_size,
                token_id_min=encoder.token_id_min,
                pad_id=b.pad_id,
                bos_id=b.bos_id,
                eos_id=b.eos_id,
                unk_id=self._mask_token_id,
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
            "ijepa_noise_pool=%d hybrid_anneal_steps=%s learnable_time=%s init_time=%.3f frag_sep_id=%d",
            ssl_loss,
            lejepa_n_global_views if uses_lejepa_views else 0,
            lejepa_n_local_views if uses_lejepa_views or ssl_loss == "ijepa" else 0,
            ijepa_mask_mode if ssl_loss == "ijepa" else "n/a",
            ijepa_block_hole_attn if ssl_loss == "ijepa" else "n/a",
            ijepa_predictor_layers if ssl_loss == "ijepa" else "n/a",
            self._mask_token_id if uses_masking else "n/a",
            len(self._ijepa_noise_pool) if ssl_loss == "ijepa" else 0,
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
    def _body_ids(item: str | list[int] | tuple[object, ...], tokenizer) -> list[int]:
        if isinstance(item, tuple) and len(item) == 2:
            item = item[0]
        if isinstance(item, list):
            if not item:
                return []
            if not isinstance(item[0], int):
                text = item[0] if len(item) == 1 else " ".join(str(x) for x in item)
                return tokenizer.encode(text, add_special_tokens=False)
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
            frags = split_fragment_ids(self._body_ids(s, b.tokenizer), sep)
            sa = shuffle_frags(frags, sep, self._rng)
            sb = shuffle_frags(frags, sep, self._rng)
            seqs_a.append([b.bos_id, *sa, b.eos_id])
            seqs_b.append([b.bos_id, *sb, b.eos_id])
        ids_a, mask_a = pad_batch(seqs_a, pad_id=b.pad_id)
        ids_b, mask_b = pad_batch(seqs_b, pad_id=b.pad_id)
        dev = self.device
        return (ids_a.to(dev), mask_a.to(dev)), (ids_b.to(dev), mask_b.to(dev))

    @staticmethod
    def _split_batch(batch):
        """Normalize to ``(views, smiles|None, net3d|None)``.

        Accepts view strings, pretokenized body ids, ``(views, smiles)``, or the
        3D-enabled ``(items, smiles|None, net_input_3d dict)`` triple.
        """
        # 3D-enabled collate: third element is the mol_src_* batch dict.
        if isinstance(batch, (tuple, list)) and len(batch) == 3 and isinstance(batch[2], dict):
            views, smiles, net3d = batch
            return list(views), (list(smiles) if smiles is not None else None), net3d
        # collate_*_with_smiles returns (views, smiles); DataLoader may hand that
        # back as a list, not a tuple — do not treat [views, smiles] as two samples.
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            views, smiles = batch
            if (
                isinstance(views, list)
                and isinstance(smiles, list)
                and (not views or not isinstance(views[0], tuple))
            ):
                return list(views), list(smiles), None
        if batch and isinstance(batch[0], tuple) and len(batch[0]) == 2:
            views, smiles = zip(*batch)
            return list(views), list(smiles), None
        return list(batch), None, None

    def _encode_ntxent_views(self, views):
        """Return ``(z_a_proj, z_b_proj, z_a_pooled)``.

        ``z_*_proj`` are the L2-normalized SimCLR projection outputs (NT-Xent
        loss); ``z_a_pooled`` is the *unnormalized* pooled z_m of view a. Kept raw
        so the cross-modal VISReg-on-1D scale anchor is meaningful; the only other
        consumer (Tanimoto distillation) L2-normalizes internally.
        """
        (ids_a, mask_a), (ids_b, mask_b) = self._two_views(views)
        z_a_pooled, z_a = self.encoder.encode_token_ids(
            ids_a, mask_a, return_projection=True, normalize=False
        )
        _, z_b = self.encoder.encode_token_ids(
            ids_b, mask_b, return_projection=True, normalize=False
        )
        z_a = torch.nn.functional.normalize(z_a, dim=-1)
        z_b = torch.nn.functional.normalize(z_b, dim=-1)
        return z_a, z_b, z_a_pooled

    def _fp_distillation(self, z_pooled, smiles):
        """MSE(cos(z_pooled), Tanimoto(smiles)); ``None`` if disabled.

        Distillation runs on an L2-normalized copy of ``z_pooled`` so the cosine
        geometry is well defined regardless of the primary loss: ntxent latents
        are already unit-norm (no-op), while lejepa latents stay unnormalized for
        SIGReg and are only normalized here for the cosine target.
        """
        if self._effective_fp_weight() <= 0 or self._fp_cache is None:
            return None
        if smiles is None:
            dm = getattr(getattr(self.trainer, "datamodule", None), "return_smiles", None)
            hint = "set data.return_smiles=true"
            if dm is False:
                hint += (
                    " (datamodule has return_smiles=false — pipeline may be using "
                    "a stale frozen config; pass data.return_smiles=true on the CLI)"
                )
            raise ValueError(f"fp_weight > 0 but batch has no SMILES; {hint}")
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
            frags = split_fragment_ids(self._body_ids(s, b.tokenizer), sep)
            first_global: list[int] | None = None
            for _ in range(n_global):
                shuffled = shuffle_frags(frags, sep, self._rng)
                seq = [b.bos_id, *shuffled, b.eos_id]
                if first_global is None:
                    first_global = seq
                elif seq != first_global:
                    n_changed += 1
                seqs.append(seq)
            for _ in range(n_local):
                masked = mask_frags(
                    frags, sep, mask_id, self._rng, frac=self._sample_mask_frac(),
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

    def _pad_bool_batch(
        self,
        sequences: list[list[bool]],
        *,
        max_len: int,
    ) -> torch.Tensor:
        out = torch.zeros(len(sequences), max_len, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            ln = min(len(seq), max_len)
            if ln:
                out[i, :ln] = torch.tensor(seq[:ln], dtype=torch.bool)
        return out

    def _ijepa_skip_online_intact(self) -> bool:
        """Encode only noised locals on the online net (EMA still runs on clean intact).

        Skip when nothing needs online clean pooled ``z_m``: VISReg (``lejepa_lambda``),
        ``inv``, Gram anchoring, or fp distillation. Rank/collapse diagnostics then use
        the EMA teacher global instead.
        """
        if self.ijepa_loss_fn is None:
            return False
        lf = self.ijepa_loss_fn
        if self._view3d_enabled():
            return False  # the 3D predictor needs the online 2D pooled z_m
        if float(self.hparams.lejepa_lambda) > 0.0:
            return False
        if lf.inv_weight > 0.0 or lf.gram_weight > 0.0:
            return False
        if self._effective_fp_weight() > 0.0:
            return False
        return True

    def _encode_ijepa(self, batch):
        """I-JEPA views: one masked encode + intact targets at hole indices.

        Returns ``(tok_masked, hole, valid, target_ema, target_online, z_pooled,
        z_teacher_global, z_teacher_shuffle, z_noised_pooled, gram_inputs)``.
        ``z_noised_pooled`` ``[B_mol*n_local, D]`` is the whole-molecule mean-pool of
        each noise-corrupted view (the noise-invariance term matches it to the clean
        EMA global). ``gram_inputs`` is
        ``(tok_online_intact, tok_ema_intact, intact_valid)`` (all aligned, post-BOS
        frame) when Gram anchoring is on, else ``None``. ``z_teacher_shuffle`` is ``None`` when
        ``ijepa_inv_weight`` is 0; otherwise it is the EMA teacher's pooled embedding
        of a *different* fragment shuffle (stop-grad), the asymmetric invariance
        target for the online ``z_pooled``.

        * ``tok_masked``: online encoder on masked locals, per-token adapter reps ``[B,T,D]``.
          Predictor reads visible rows only. Masked and intact sequences for each row share
          the same fragment order (one ``ordered`` shuffle per molecule); ``hole`` indices
          align with EMA targets at those positions.
        * ``hole`` / ``valid``: boolean masks aligned with ``tok_masked`` (post-BOS frame).
        * ``target_ema``: EMA encoder, intact view at hole columns — stop-grad endpoint.
        * ``target_online``: online encoder, intact view at hole columns — SIGReg.
        * ``z_pooled``: ``[B_mol, D]`` mean-pooled intact online ``z_m``.
        * ``z_teacher_global``: ``[B_mol, D]`` intact-view pooled latents from the EMA teacher.

        ponytail: corrupted positions use uniform noise over body vocab (not ``<UNK>``);
        hole flags are tracked explicitly during corruption.
        """
        b = self.encoder.bundle
        sep = int(self.hparams.frag_sep_id)
        n_local = int(self.hparams.lejepa_n_local_views)
        mask_mode = str(self.hparams.ijepa_mask_mode).lower()
        noise_pool = self._ijepa_noise_pool
        loss_fn = self.ijepa_loss_fn
        need_inv = loss_fn is not None and loss_fn.inv_weight > 0.0
        # Token-level terms (predict/glob/gram) require each local to share the intact
        # fragment order so EMA targets line up position-for-position. When they are
        # all off, the only matching is the alignment-free pooled noise-invariance, so
        # each local can be its own independent fragment shuffle (more augmentation).
        need_aligned = loss_fn is not None and (
            loss_fn.predict_weight > 0.0
            or loss_fn.glob_weight > 0.0
            or loss_fn.gram_weight > 0.0
        )
        intact_seqs: list[list[int]] = []
        intact_b: list[list[int]] = []
        masked_seqs: list[list[int]] = []
        hole_seqs: list[list[bool]] = []
        for s in batch:
            frags = split_fragment_ids(self._body_ids(s, b.tokenizer), sep)
            # One shuffle order shared by the intact view and all n_local masks;
            # masks reuse the pre-split fragments instead of re-splitting `ordered`.
            ordered_frags = list(frags)
            if len(ordered_frags) > 1:
                self._rng.shuffle(ordered_frags)
            ordered = join_fragment_ids(ordered_frags, sep)
            intact_seqs.append([b.bos_id, *ordered, b.eos_id])
            if need_inv:
                intact_b.append(
                    [b.bos_id, *shuffle_frags(frags, sep, self._rng), b.eos_id]
                )
            for _ in range(n_local):
                if need_aligned:
                    local_frags, local_ids = ordered_frags, ordered
                else:
                    # Independent fragment shuffle per local view (built from the
                    # globals), then corrupted with uniform noise.
                    local_frags = list(frags)
                    if len(local_frags) > 1:
                        self._rng.shuffle(local_frags)
                    local_ids = join_fragment_ids(local_frags, sep)
                masked, hole_body = noise_local_frags(
                    local_frags,
                    local_ids,
                    sep,
                    noise_pool,
                    self._rng,
                    frac=self._sample_mask_frac(),
                    mode=mask_mode,
                )
                masked_seqs.append([b.bos_id, *masked, b.eos_id])
                hole_seqs.append([False, *hole_body, False])
        max_len = max(
            max(map(len, intact_seqs), default=0),
            max(map(len, masked_seqs), default=0),
            max(map(len, intact_b), default=0) if need_inv else 0,
        )
        intact_ids, intact_mask = pad_batch(intact_seqs, pad_id=b.pad_id, max_len=max_len)
        masked_ids, masked_mask = pad_batch(masked_seqs, pad_id=b.pad_id, max_len=max_len)
        hole_full = self._pad_bool_batch(hole_seqs, max_len=max_len)
        n_mol = len(intact_seqs)
        n_masked = len(masked_seqs)
        dev = self.device
        nb = dev.type == "cuda"

        def _dev(t: torch.Tensor) -> torch.Tensor:
            return t.to(dev, non_blocking=nb)

        masked_ids = _dev(masked_ids)
        masked_mask = _dev(masked_mask)
        intact_ids = _dev(intact_ids)
        intact_mask = _dev(intact_mask)
        hole = hole_full[:, 1:].to(dev)
        valid = masked_mask[:, 1:].bool()
        hole_attn = hole if bool(self.hparams.ijepa_block_hole_attn) else None
        skip_online_intact = self._ijepa_skip_online_intact()

        ema_ids, ema_mask = intact_ids, intact_mask
        if need_inv:
            ids_b, mask_b = pad_batch(intact_b, pad_id=b.pad_id, max_len=max_len)
            ema_ids = torch.cat([intact_ids, _dev(ids_b)], dim=0)
            ema_mask = torch.cat([intact_mask, _dev(mask_b)], dim=0)

        with torch.no_grad():
            z_teacher_all, tok_ema = self.encoder_ema.encode_token_ids(
                ema_ids, ema_mask,
                normalize=False, return_tokens=True,
            )
        z_teacher_global = z_teacher_all[:n_mol]
        tok_ema_intact = tok_ema[:n_mol]
        z_teacher_shuffle = z_teacher_all[n_mol:] if need_inv else None
        tok_ema_rows = tok_ema_intact.repeat_interleave(n_local, dim=0)

        if skip_online_intact:
            z_online, tok_masked = self.encoder.encode_token_ids(
                masked_ids, masked_mask,
                normalize=False, return_tokens=True,
                hole_mask=hole_attn,
            )
            z_noised_pooled = z_online
            z_pooled = z_teacher_global
            target_ema = self._gather_hole_tokens(tok_ema_rows, hole)
            target_online = target_ema
        else:
            online_ids = torch.cat([masked_ids, intact_ids], dim=0)
            online_mask = torch.cat([masked_mask, intact_mask], dim=0)
            hole_attn_full = None
            if hole_attn is not None:
                hole_attn_full = torch.zeros(
                    online_ids.size(0), hole.size(1), dtype=torch.bool, device=dev,
                )
                hole_attn_full[:n_masked] = hole_attn
            z_online, tok_online_all = self.encoder.encode_token_ids(
                online_ids, online_mask,
                normalize=False, return_tokens=True,
                hole_mask=hole_attn_full,
            )
            tok_masked = tok_online_all[:n_masked]
            z_noised_pooled = z_online[:n_masked]
            z_pooled = z_online[n_masked:n_masked + n_mol]
            tok_online_intact = tok_online_all[n_masked:n_masked + n_mol]
            tok_online_rows = tok_online_intact.repeat_interleave(n_local, dim=0)
            target_ema = self._gather_hole_tokens(tok_ema_rows, hole)
            target_online = self._gather_hole_tokens(tok_online_rows, hole)

        if target_ema.numel() == 0:
            raise RuntimeError("ijepa batch has no hole tokens — check masking")
        gram_inputs = None
        if self.ijepa_loss_fn is not None and self.ijepa_loss_fn.gram_weight > 0.0:
            gram_inputs = (
                tok_online_intact,
                tok_ema_intact,
                intact_mask[:, 1:].bool(),
            )
        return (
            tok_masked, hole, valid, target_ema, target_online,
            z_pooled, z_teacher_global, z_teacher_shuffle, z_noised_pooled, gram_inputs,
        )

    @staticmethod
    def _update_ema(online: torch.nn.Module, ema: torch.nn.Module, decay: float) -> None:
        with torch.no_grad():
            for p_online, p_ema in zip(online.parameters(), ema.parameters()):
                p_ema.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    def _effective_ijepa_ema_decay(self) -> float:
        return cosine_ema_decay(
            self.global_step,
            float(self.hparams.ijepa_ema_decay),
            self._resolve_total_steps(),
        )

    def _update_ijepa_ema(self) -> None:
        assert self.encoder_ema is not None
        decay = self._effective_ijepa_ema_decay()
        self._update_ema(self.encoder, self.encoder_ema, decay)
        if self.dino_head_ema is not None:
            self._update_ema(self.dino_head, self.dino_head_ema, decay)

    @staticmethod
    def _gram_kwargs(gram_inputs) -> dict:
        """Unpack ``_encode_ijepa``'s ``gram_inputs`` into IJEPALoss kwargs (empty
        when Gram anchoring is off)."""
        if gram_inputs is None:
            return {}
        online, ema, valid = gram_inputs
        return {"gram_online": online, "gram_target": ema, "gram_valid": valid}

    def _dino_student_z(self, z_noised_pooled: torch.Tensor) -> torch.Tensor:
        """Student embedding fed to ``dino_head``: pool directly or pool → predictor."""
        mode = str(self.hparams.ijepa_dino_student).lower()
        if mode == "pool":
            return z_noised_pooled
        if mode == "predict":
            assert self.dino_pool_predictor is not None
            return self.dino_pool_predictor(z_noised_pooled)
        raise ValueError(
            f"ijepa_dino_student must be 'pool' or 'predict', got {self.hparams.ijepa_dino_student!r}"
        )

    def _dino_term(
        self,
        z_student: torch.Tensor,
        z_teacher_rows: torch.Tensor,
        *,
        update_center: bool,
    ) -> tuple[torch.Tensor | None, dict[str, float] | None]:
        """DINO CE: student prototypes vs centered, sharpened stop-grad teacher.

        ``z_student`` is the noised whole-molecule pool (``pool``) or the pool
        passed through ``dino_pool_predictor`` (``predict``). Teacher rows are
        EMA clean globals, one per local view.
        """
        if self.dino_loss_fn is None:
            return None, None
        student_logits = self.dino_head(z_student)
        with torch.no_grad():
            teacher_logits = self.dino_head_ema(z_teacher_rows)
            entropy, active = self.dino_loss_fn.utilization(teacher_logits)
        loss = self.dino_loss_fn(student_logits, teacher_logits)
        if update_center:
            self.dino_loss_fn.update_center(teacher_logits)
        util = {
            "diagnostics/dino_teacher_entropy": entropy,
            "diagnostics/dino_active_prototypes": active,
        }
        return loss, util

    def _hybrid_alpha(self) -> float:
        """Linear anneal: 1.0 (pure ntxent) at step 0 -> 0.0 (pure lejepa) by
        ``hybrid_anneal_steps``, held at 0 after."""
        anneal = int(self.hparams.hybrid_anneal_steps)
        if anneal <= 0:
            return 0.0
        return max(0.0, 1.0 - self.global_step / anneal)

    def _effective_fp_weight(self) -> float:
        """``fp_weight`` scaled by ``1 - step/fp_anneal_steps`` when annealing."""
        w = self.fp_weight
        anneal = int(self.hparams.fp_anneal_steps)
        if w <= 0 or anneal <= 0:
            return w
        return w * max(0.0, 1.0 - self.global_step / anneal)

    def _compute_loss(
        self,
        *,
        z_a: torch.Tensor | None = None,
        z_b: torch.Tensor | None = None,
        z_global: torch.Tensor | None = None,
        z_all: torch.Tensor | None = None,
        proj_global: torch.Tensor | None = None,
    ):
        if self._contrastive:
            assert z_a is not None and z_b is not None
            fn = self.ntxent_loss_fn if self.ssl_loss == "ntxent" else self.siglip_loss_fn
            loss = fn(z_a, z_b)
            return loss, {
                "inv": None, "sigreg": None,
                "ntxent": None, "alpha": None,
            }
        assert z_global is not None and z_all is not None
        terms = self.lejepa_loss_fn(z_global, z_all)
        if self.ssl_loss != "hybrid":
            if self.hparams.log_grad_norms and torch.is_grad_enabled():
                self._log_grad_norms(inv=terms.inv, sigreg=terms.sigreg)
            return terms.total, {
                "inv": terms.inv.detach(),
                "sigreg": terms.sigreg.detach(), "ntxent": None, "alpha": None,
                "reg_scale": None if terms.reg_scale is None else terms.reg_scale.detach(),
                "reg_shape": None if terms.reg_shape is None else terms.reg_shape.detach(),
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
            "reg_scale": None if terms.reg_scale is None else terms.reg_scale.detach(),
            "reg_shape": None if terms.reg_shape is None else terms.reg_shape.detach(),
        }

    def _log_grad_norms(self, **terms: torch.Tensor) -> None:
        """L2 norm of each loss term's gradient w.r.t. trainable encoder params."""
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
        # When the 3D view is on, _apply_view3d logs the cross-modal acc@1 instead.
        if self._view3d_enabled():
            pass
        elif z_b is not None:
            logs[f"{prefix}/acc@1"] = top1_paired_accuracy(z_a.detach(), z_b.detach())
        elif z_all is not None and z_a is not None:
            acc = lejepa_retrieval_acc1(z_a.detach(), z_all.detach())
            if acc is not None:
                logs[f"{prefix}/acc@1"] = acc
        if extras["inv"] is not None:
            logs[f"{prefix}/inv"] = extras["inv"]
            logs[f"{prefix}/sigreg"] = extras["sigreg"]
        if extras.get("reg_scale") is not None:
            logs[f"{prefix}/reg_scale"] = extras["reg_scale"]
            logs[f"{prefix}/reg_shape"] = extras["reg_shape"]
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

    def _log_pooled_rep_diag(
        self,
        z_online: torch.Tensor,
        z_teacher: torch.Tensor,
        *,
        spectrum: bool,
        batch_size: int,
        log_rank: bool = False,
    ) -> None:
        """Collapse checks on pooled ``z_m`` (online intact slice + EMA teacher).

        Does not log online-vs-teacher cosine: the training student path is the
        noised local views, not the clean intact online pool in this batch.
        """
        online = z_online.detach().float()
        teacher = z_teacher.detach().float()
        logs: dict[str, float] = {
            "diagnostics/pooled_std": float(online.std(dim=0).mean()),
            "diagnostics/teacher_std": float(teacher.std(dim=0).mean()),
            "diagnostics/n_molecules": float(online.shape[0]),
        }
        if spectrum:
            k = int(self.hparams.rank_subsample)
            online_s = self._subsample_rows(online, k).cpu().numpy()
            _, top = embedding_batch_collapse_diag(online_s, top_k=5)
            for i, v in enumerate(top, start=1):
                logs[f"diagnostics/pooled_eig_{i}"] = v
            if log_rank:
                eff, num = embedding_covariance_rank(online_s)
                logs["train/rank_effective"] = eff
                logs["train/rank_numerical"] = num
        self.log_dict(logs, on_step=True, batch_size=batch_size)

    # -- lifecycle ---------------------------------------------------------- #
    def _view3d_enabled(self) -> bool:
        return self.encoder_3d is not None and self.view3d_weight > 0.0

    def _net3d_to_device(self, net3d: dict) -> dict:
        return {k: v.to(self.device) for k, v in net3d.items()}

    def _view3d_loss(self, z_2d_online, net3d):
        """Cross-modal 3D term: ``(L_3d, L_reg_3d, L_reg_1d, acc)`` or all-``None``.

        ``L_3d`` is a *symmetric* LeJEPA-style alignment (no stop-grad) between
        ``pred_3d(z_2d)`` and the online ``z_3d`` — ``l2`` MSE (LeJEPA default; scale
        made safe by VISReg on each modality) or ``cosine``. Gradient flows into both
        the 2D encoder (via ``z_2d``/predictor) and the 3D co-encoder (via ``z_3d``),
        so both learn to match. ``L_reg_3d = VISReg(z_3d)`` is mixed with ``L_3d`` via
        ``lejepa_lambda`` in :meth:`_apply_view3d`. ``L_reg_1d = VISReg(z_2d)`` is
        added to the reg only when ``view3d_visreg_1d`` is set (else ``None``);
        ``z_2d_online`` is the unnormalized pooled 1D z_m so VISReg's scale anchor
        is meaningful. Under dual-pool contrastive it is sliced to the main
        (regression) half so the projection half is left to the contrastive loss.

        ``acc`` is the cross-modal top-1 retrieval accuracy between the 1D-predicted
        embedding ``pred_3d(z_2d)`` and the online ``z_3d`` (symmetric) — the 3D
        analogue of the 1D-1D ``acc@1``, and the more meaningful sanity metric once
        the 3D view is on.
        """
        if not self._view3d_enabled() or net3d is None:
            return None, None, None, None
        # Dual-pool contrastive: predict/anchor off the main (regression) half only,
        # leaving the projection half to the contrastive loss (see __init__).
        if self._view3d_main_half:
            z_2d_online = z_2d_online[..., : self._d_pool]
        net3d = self._net3d_to_device(net3d)
        z_3d = self.encoder_3d(net3d)                      # online, trainable target
        pred = self.pred_3d(z_2d_online)
        if self.view3d_loss == "l2":
            l_3d = torch.nn.functional.mse_loss(pred, z_3d)
        else:
            l_3d = (
                1.0 - torch.nn.functional.cosine_similarity(pred, z_3d, dim=-1)
            ).mean()
        l_reg_3d = self.visreg_3d(z_3d)
        l_reg_1d = self.visreg_2d(z_2d_online) if self.visreg_2d is not None else None
        acc = top1_paired_accuracy(pred.detach(), z_3d.detach(), symmetric=True)
        return l_3d, l_reg_3d, l_reg_1d, acc

    def _apply_view3d(self, loss, z_2d_online, net3d, *, prefix: str, bs: int):
        """Add the 3D terms to ``loss`` and log them; return the updated loss."""
        l_3d, l_reg_3d, l_reg_1d, acc = self._view3d_loss(z_2d_online, net3d)
        if l_3d is None:
            return loss
        lam = float(self.hparams.lejepa_lambda)
        l_reg = l_reg_3d if l_reg_1d is None else l_reg_3d + l_reg_1d
        view3d = (1.0 - lam) * l_3d + lam * l_reg
        loss = loss + self.view3d_weight * view3d
        if prefix == "train":
            logs = {
                "train/1Dto3Dpred": l_3d.detach(),
                "train/3d_visreg": l_reg_3d.detach(),
                "train/view3d_mix": view3d.detach(),
                # 3D cross-modal retrieval replaces the 1D-1D acc@1 (see _log_step).
                "train/acc@1": acc,
            }
            if l_reg_1d is not None:
                logs["train/1d_visreg"] = l_reg_1d.detach()
            self.log_dict(logs, on_step=True, batch_size=bs)
        else:
            self._val["view3d"].append(float(l_3d))
            self._val["view3d_reg"].append(float(l_reg_3d))
            self._val["view3d_mix"].append(float(view3d))
            self._val["acc"].append(acc)
            if l_reg_1d is not None:
                self._val["view3d_reg_1d"].append(float(l_reg_1d))
        return loss

    def training_step(self, batch, batch_idx):
        views, smiles, net3d = self._split_batch(batch)
        bs = len(views)
        log_rank = self._rank_due()
        log_collapse = self._collapse_diag_due()
        z_rank = None
        if self._contrastive:
            z_a, z_b, z_a_pooled = self._encode_ntxent_views(views)
            loss, extras = self._compute_loss(z_a=z_a, z_b=z_b)
            # The 1D within-modality contrastive loss on its own (logged as
            # train/siglip or train/ntxent), before the fp/3D terms are folded in below.
            self.log(f"train/{self.ssl_loss}", loss.detach(), on_step=True, prog_bar=True, batch_size=bs)
            if self.siglip_loss_fn is not None and log_rank:
                self.log_dict(
                    self.siglip_loss_fn.diagnostics(z_a, z_b),
                    on_step=True, batch_size=bs,
                )
            fp_loss = self._fp_distillation(z_a_pooled, smiles)
            if fp_loss is not None:
                self.log("train/fp", fp_loss.detach(), on_step=True, batch_size=bs)
                loss = loss + self._effective_fp_weight() * fp_loss
            loss = self._apply_view3d(loss, z_a_pooled, net3d, prefix="train", bs=bs)
            self._log_step(
                "train", loss=loss, z_a=z_a, z_b=z_b, extras=extras,
                batch_size=bs, prog_bar=True,
            )
            if log_rank:
                z_rank = self._ntxent_global_z(views)
        elif self.ssl_loss == "ijepa":
            (
                tok_masked, hole, valid, target_ema, target_online, z_pooled,
                z_teacher_global, z_teacher_shuffle, z_noised_pooled, gram_inputs,
            ) = self._encode_ijepa(views)
            n_local = int(self.hparams.lejepa_n_local_views)
            z_teacher_rows = z_teacher_global.repeat_interleave(n_local, dim=0)
            gram_kw = self._gram_kwargs(gram_inputs)
            terms = self.ijepa_loss_fn(
                tok_masked, hole, target_ema, z_pooled, valid=valid,
                z_teacher_rows=z_teacher_rows,
                z_inv_target=z_teacher_shuffle,
                z_noised_pooled=z_noised_pooled,
                **gram_kw,
            )
            loss = terms.total
            # The 1D I-JEPA loss on its own (train/ijepa), before fp/dino/3D fold in.
            self.log(f"train/{self.ssl_loss}", loss.detach(), on_step=True, batch_size=bs)
            fp_loss = self._fp_distillation(z_pooled, smiles)
            if fp_loss is not None:
                self.log("train/fp", fp_loss.detach(), on_step=True, batch_size=bs)
                loss = loss + self._effective_fp_weight() * fp_loss
            if self.dino_loss_fn is not None:
                z_dino_student = self._dino_student_z(z_noised_pooled)
                dino_loss, dino_util = self._dino_term(
                    z_dino_student, z_teacher_rows, update_center=True,
                )
            else:
                dino_loss, dino_util = None, None
            if dino_loss is not None:
                loss = loss + float(self.hparams.ijepa_dino_weight) * dino_loss
            if dino_util is not None and log_collapse:
                self.log_dict(dino_util, on_step=True, batch_size=bs)
            loss = self._apply_view3d(loss, z_pooled, net3d, prefix="train", bs=bs)
            if self.hparams.log_grad_norms and torch.is_grad_enabled():
                self._log_grad_norms(
                    predict=terms.predict, glob=terms.glob,
                    inv=terms.inv, sigreg=terms.sigreg,
                )
            self.log("train/loss", loss.detach(), on_step=True, prog_bar=True, batch_size=bs)
            ijepa_logs = {
                "train/predict": terms.predict.detach(),
                "train/glob": terms.glob.detach(),
                "train/inv": terms.inv.detach(),
                "train/sigreg": terms.sigreg.detach(),
                "train/encode_time": self.encoder.encode_time_value,
            }
            if self.ijepa_loss_fn.noise_inv_weight > 0.0:
                ijepa_logs["train/noise_inv"] = terms.noise_inv.detach()
            if self.ijepa_loss_fn.gram_weight > 0.0:
                ijepa_logs["train/gram"] = terms.gram.detach()
            if terms.reg_scale is not None:
                ijepa_logs["train/reg_scale"] = terms.reg_scale.detach()
                ijepa_logs["train/reg_shape"] = terms.reg_shape.detach()
            if dino_loss is not None:
                ijepa_logs["train/dino"] = dino_loss.detach()
            self.log_dict(ijepa_logs, on_step=True, batch_size=bs)
            if log_collapse:
                self._log_pooled_rep_diag(
                    z_pooled, z_teacher_global,
                    spectrum=True, batch_size=bs, log_rank=log_rank,
                )
            elif log_rank:
                # When online intact is skipped, z_pooled is the EMA teacher global.
                z_rank = z_pooled
        else:
            with_proj = self.ssl_loss == "hybrid"
            out = self._encode_lejepa_views(views, with_projection=with_proj)
            z_global, z_all, diversity = out[0], out[1], out[2]
            proj_global = out[3] if with_proj else None
            loss, extras = self._compute_loss(
                z_global=z_global, z_all=z_all, proj_global=proj_global,
            )
            # The 1D LeJEPA/hybrid loss on its own (train/lejepa or train/hybrid),
            # before the fp/3D terms are folded in below.
            self.log(f"train/{self.ssl_loss}", loss.detach(), on_step=True, prog_bar=True, batch_size=bs)
            fp_loss = self._fp_distillation(z_global[:, 0], smiles)
            if fp_loss is not None:
                self.log("train/fp", fp_loss.detach(), on_step=True, batch_size=bs)
                loss = loss + self._effective_fp_weight() * fp_loss
            loss = self._apply_view3d(loss, z_global[:, 0], net3d, prefix="train", bs=bs)
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
        views, smiles, net3d = self._split_batch(batch)
        z_rank = None
        rank_on = int(self.hparams.train_rank_every_n_steps) > 0
        if self._contrastive:
            z_a, z_b, z_a_pooled = self._encode_ntxent_views(views)
            loss, extras = self._compute_loss(z_a=z_a, z_b=z_b)
            self._val["loss_1d"].append(float(loss))
            if self.siglip_loss_fn is not None:
                # Same siglip/diagnostics_* keys as train; Lightning means over the val epoch.
                self.log_dict(
                    self.siglip_loss_fn.diagnostics(z_a, z_b),
                    on_step=False, on_epoch=True, batch_size=len(views),
                )
            # With 3D on, _apply_view3d appends the cross-modal acc@1 instead.
            if not self._view3d_enabled():
                self._val["acc"].append(top1_paired_accuracy(z_a, z_b))
            fp_loss = self._fp_distillation(z_a_pooled, smiles)
            if fp_loss is not None:
                self._val["fp"].append(float(fp_loss))
                loss = loss + self._effective_fp_weight() * fp_loss
            loss = self._apply_view3d(loss, z_a_pooled, net3d, prefix="val", bs=len(views))
            if rank_on:
                z_rank = self._ntxent_global_z(views)
        elif self.ssl_loss == "ijepa":
            (
                tok_masked, hole, valid, target_ema, target_online, z_pooled,
                z_teacher_global, z_teacher_shuffle, z_noised_pooled, gram_inputs,
            ) = self._encode_ijepa(views)
            n_local = int(self.hparams.lejepa_n_local_views)
            z_teacher_rows = z_teacher_global.repeat_interleave(n_local, dim=0)
            gram_kw = self._gram_kwargs(gram_inputs)
            terms = self.ijepa_loss_fn(
                tok_masked, hole, target_ema, z_pooled, valid=valid,
                z_teacher_rows=z_teacher_rows,
                z_inv_target=z_teacher_shuffle,
                z_noised_pooled=z_noised_pooled,
                **gram_kw,
            )
            loss = terms.total
            self._val["loss_1d"].append(float(loss))
            fp_loss = self._fp_distillation(z_pooled, smiles)
            if fp_loss is not None:
                self._val["fp"].append(float(fp_loss))
                loss = loss + self._effective_fp_weight() * fp_loss
            if self.dino_loss_fn is not None:
                z_dino_student = self._dino_student_z(z_noised_pooled)
                dino_loss, dino_util = self._dino_term(
                    z_dino_student, z_teacher_rows, update_center=False,
                )
            else:
                dino_loss, dino_util = None, None
            if dino_loss is not None:
                loss = loss + float(self.hparams.ijepa_dino_weight) * dino_loss
                self._val["dino"].append(float(dino_loss))
                if dino_util is not None:
                    self._val["dino_teacher_entropy"].append(
                        dino_util["diagnostics/dino_teacher_entropy"]
                    )
                    self._val["dino_active_prototypes"].append(
                        dino_util["diagnostics/dino_active_prototypes"]
                    )
            loss = self._apply_view3d(loss, z_pooled, net3d, prefix="val", bs=len(views))
            extras = {"inv": None, "ntxent": None}
            self._val["predict"].append(float(terms.predict))
            self._val["glob"].append(float(terms.glob))
            self._val["inv"].append(float(terms.inv))
            self._val["sigreg"].append(float(terms.sigreg))
            if self.ijepa_loss_fn.noise_inv_weight > 0.0:
                self._val["noise_inv"].append(float(terms.noise_inv))
            if self.ijepa_loss_fn.gram_weight > 0.0:
                self._val["gram"].append(float(terms.gram))
            # condition_bypass_gap probes the token-level predictor; only meaningful
            # when predict is active (locals share the intact frame).
            if self.ijepa_loss_fn.predict_weight > 0.0:
                gap = self.ijepa_loss_fn.condition_bypass_gap(
                    tok_masked, hole, target_ema, valid=valid,
                )
                self._val["cond_true"].append(gap["predict_true"])
                self._val["cond_shuf"].append(gap["predict_shuf"])
                self._val["cond_zero"].append(gap["predict_zero"])
                self._val["cond_gap_zero"].append(gap["gap_zero"])
                self._val["cond_gap_shuf"].append(gap["gap_shuf"])
            if rank_on:
                z_rank = z_pooled
        else:
            with_proj = self.ssl_loss == "hybrid"
            out = self._encode_lejepa_views(views, with_projection=with_proj)
            z_global, z_all, diversity = out[0], out[1], out[2]
            proj_global = out[3] if with_proj else None
            loss, extras = self._compute_loss(
                z_global=z_global, z_all=z_all, proj_global=proj_global,
            )
            self._val["loss_1d"].append(float(loss))
            # With 3D on, _apply_view3d appends the cross-modal acc@1 instead.
            if not self._view3d_enabled():
                acc = lejepa_retrieval_acc1(z_global, z_all)
                if acc is not None:
                    self._val["acc"].append(acc)
            self._val["view_diversity"].append(diversity)
            fp_loss = self._fp_distillation(z_global[:, 0], smiles)
            if fp_loss is not None:
                self._val["fp"].append(float(fp_loss))
                loss = loss + self._effective_fp_weight() * fp_loss
            loss = self._apply_view3d(loss, z_global[:, 0], net3d, prefix="val", bs=len(views))
            if rank_on:
                z_rank = self._online_global_z(z_global)
        if rank_on and z_rank is not None:
            self._append_val_rank(z_rank)
        self._val["loss"].append(loss.item())
        if extras["inv"] is not None:
            self._val["inv"].append(float(extras["inv"]))
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
        cfg_3d = getattr(self.encoder_3d, "build_config", None)
        if cfg_3d is not None:
            checkpoint["encoder_3d_config"] = dict(cfg_3d)
        # Record the fragment-view variant the adapter was trained on so every
        # downstream stage (decoy/binder precompute, EBM, eval) reads the matching
        # _merge stores without having to be told — mismatch becomes impossible.
        checkpoint["fragment_merge"] = bool(getattr(self, "_fragment_merge", False))

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Backfill EMA twins when loading a checkpoint that predates them."""
        state = checkpoint.get("state_dict", {})
        if self.encoder_ema is not None and not any(
            k.startswith("encoder_ema.") for k in state
        ):
            self.encoder_ema.load_state_dict(self.encoder.state_dict())
        if self.dino_head_ema is not None and not any(
            k.startswith("dino_head_ema.") for k in state
        ):
            self.dino_head_ema.load_state_dict(self.dino_head.state_dict())

    def on_fit_start(self) -> None:
        trainer = getattr(self, "trainer", None)
        dm = getattr(trainer, "datamodule", None) if trainer is not None else None
        if dm is None:
            return
        # Fail fast on the silent no-op: 3D view enabled on the model but the data
        # module isn't producing conformers -> encoder_3d/pred_3d would train on
        # nothing and train/view3d* would never log.
        if self._view3d_enabled() and getattr(dm, "conformer_cache", None) is None:
            raise ValueError(
                "view3d_weight>0 but data.conformer_cache is None: the 3D encoder "
                "would never receive a batch. Set data.conformer_cache to a "
                "precomputed conformers.parquet (lattice_lab.preprocessing."
                "precompute_conformers), or disable the 3D view (view3d_weight=0)."
            )
        self._fragment_merge = str(getattr(dm, "shard_dir", "")).rstrip("/").endswith("_merge")
        self._val_probes.val_ratio = float(dm.val_ratio)
        self._val_probes.test_ratio = float(dm.test_ratio)
        self._val_probes.split_seed = int(dm.split_seed)
        self._val_probes.prepare(dm.shard_dir)
        if self.ssl_loss == "ijepa" and self.ijepa_loss_fn is not None:
            skip = self._ijepa_skip_online_intact()
            logger.info(
                "ijepa online intact encode=%s (lejepa_lambda=%s fp_eff=%.4g inv=%s gram=%s)",
                "skip" if skip else "on",
                self.hparams.lejepa_lambda,
                self._effective_fp_weight(),
                self.ijepa_loss_fn.inv_weight,
                self.ijepa_loss_fn.gram_weight,
            )

    def on_validation_epoch_end(self) -> None:
        out = {
            "val/loss": float(np.mean(self._val["loss"])) if self._val["loss"] else float("nan"),
            "val/acc@1": float(np.mean(self._val["acc"])) if self._val["acc"] else 0.0,
            "val/encode_time": self.encoder.encode_time_value,
        }
        if self._val["loss_1d"]:
            out[f"val/{self.ssl_loss}"] = float(np.mean(self._val["loss_1d"]))
        if self._val["inv"]:
            out["val/inv"] = float(np.mean(self._val["inv"]))
            out["val/sigreg"] = float(np.mean(self._val["sigreg"]))
        if self._val["predict"]:
            out["val/predict"] = float(np.mean(self._val["predict"]))
            out["val/glob"] = float(np.mean(self._val["glob"]))
            out["val/inv"] = float(np.mean(self._val["inv"]))
            out["val/sigreg"] = float(np.mean(self._val["sigreg"]))
        if self._val["noise_inv"]:
            out["val/noise_inv"] = float(np.mean(self._val["noise_inv"]))
        if self._val["gram"]:
            out["val/gram"] = float(np.mean(self._val["gram"]))
        if self._val["dino"]:
            out["val/dino"] = float(np.mean(self._val["dino"]))
        if self._val["dino_teacher_entropy"]:
            out["diagnostics/dino_teacher_entropy"] = float(
                np.mean(self._val["dino_teacher_entropy"])
            )
            out["diagnostics/dino_active_prototypes"] = float(
                np.mean(self._val["dino_active_prototypes"])
            )
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
        if self._val["view3d"]:
            out["val/1Dto3Dpred"] = float(np.mean(self._val["view3d"]))
            out["val/3d_visreg"] = float(np.mean(self._val["view3d_reg"]))
            out["val/view3d_mix"] = float(np.mean(self._val["view3d_mix"]))
        if self._val["view3d_reg_1d"]:
            out["val/1d_visreg"] = float(np.mean(self._val["view3d_reg_1d"]))
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

    def _resolve_total_steps(self) -> int:
        cached = getattr(self, "_total_steps_cache", None)
        if cached is not None:
            return cached
        trainer = getattr(self, "trainer", None)
        est = getattr(trainer, "estimated_stepping_batches", None) if trainer else None
        if est is not None and math.isfinite(est) and est > 0:
            total = int(est)
            logger.info("cosine LR horizon from trainer: %d steps", total)
        else:
            logger.warning("trainer estimate unavailable; cosine LR horizon fallback 30000")
            total = 30_000
        self._total_steps_cache = total
        return total

    def configure_optimizers(self):
        hp = self.hparams
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, p in self.encoder.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if _adamw_no_decay(name, p) else decay).append(p)
        # ijepa's flow head lives on the loss module (not the encoder); it must
        # train too. (lejepa/ntxent loss modules have no trainable params.)
        if self.ijepa_loss_fn is not None:
            for name, p in self.ijepa_loss_fn.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if _adamw_no_decay(name, p) else decay).append(p)
        # SigLIP's learnable temperature + bias live on the loss module (scalars →
        # no weight decay via _adamw_no_decay's p.ndim < 2 rule).
        if self.siglip_loss_fn is not None:
            for name, p in self.siglip_loss_fn.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if _adamw_no_decay(name, p) else decay).append(p)
        # DINO head (online only; EMA twin is not optimized) also lives off the encoder.
        if self.dino_head is not None:
            for name, p in self.dino_head.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if _adamw_no_decay(name, p) else decay).append(p)
        if self.dino_pool_predictor is not None:
            for name, p in self.dino_pool_predictor.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if _adamw_no_decay(name, p) else decay).append(p)
        # 3D co-encoder + cross-modal predictor (EMA twin excluded via requires_grad).
        for mod in (self.encoder_3d, self.pred_3d):
            if mod is None:
                continue
            for name, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if _adamw_no_decay(name, p) else decay).append(p)
        groups: list[dict] = []
        if decay:
            groups.append({"params": decay, "weight_decay": hp.weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        optim = torch.optim.AdamW(groups, lr=hp.learning_rate)
        total_steps = self._resolve_total_steps()
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: cosine_with_warmup(s, hp.warmup_steps, total_steps)
        )
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
