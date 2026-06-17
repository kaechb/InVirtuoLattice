"""Conditional denoising-JEPA SSL for the discrete-flow backbone.

An LLM-JEPA-style alternative to the NT-Xent / LeJEPA collapse prevention in
:class:`~lattice_lab.models.discrete_flow_ssl.DiscreteFlowSSLModule`. A learned,
pooled molecule latent ``z_s`` conditions a denoiser that reconstructs a
corrupted copy of the string; the objective is a **pure generative (token)
loss** (CE to the clean tokens at the noised positions), so there is no
representation matching and **no EMA teacher** (see
:mod:`lattice_lab.training.denoising_jepa`):

    z_s    = encoder(clean,   t=1)               # pooled molecule latent (grad)
    x_t    = corrupt(clean,   t=corrupt_t)        # uniform-source substitution
    logits = denoiser(x_t,    t=corrupt_t | z_s)  # z_s is the conditioning input
    loss   = CE(logits, clean)  over the NOISED positions only

The encoder is trained *through* reconstruction (``z_s`` must carry molecule
info or the denoiser cannot fill the noised positions). The failure mode is an
*inert* ``z_s`` (ignored by the denoiser), guarded by the condition-bypass
diagnostic logged as ``val/condition_gap``; ``effective_rank(z_s)`` is logged as
a secondary tripwire and ``val/acc@1`` is the token reconstruction accuracy at
the noised positions (so the existing ``ssl_basic`` ``ModelCheckpoint`` works
unchanged).

Batches are the same fragment-view strings the rest of Stage-2 consumes
(:class:`~lattice_lab.data.fragment_views.FragmentViewDataModule`); this module
owns the tokenizer through ``student.bundle``.
"""

from __future__ import annotations

import logging
import math
import random

import lightning as L
import numpy as np
import torch

from lattice_lab.backbone.discrete_flow import pad_batch
from lattice_lab.data.fragment_views import shuffle_fragment_ids
from lattice_lab.models.schedules import cosine_with_warmup
from lattice_lab.training.denoising_jepa import (
    JEPAStudent,
    VAEHead,
    condition_bypass_gap,
    denoising_loss,
)
from lattice_lab.training.ssl_loss import (
    SIGReg,
    _FingerprintCache,
    similarity_distillation_loss,
    tanimoto_target_matrix,
)
from lattice_lab.training.ssl_val_probes import JepaValProbes

logger = logging.getLogger(__name__)


class DenoisingJEPAModule(L.LightningModule):
    def __init__(
        self,
        student: JEPAStudent,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 30_000,
        frag_sep_id: int = 4,
        shuffle_fragments: bool = True,
        # View-invariance regularizer: align z_s of two views of the same
        # molecule (1 - cos). Keeps z_s semantic (invariant molecule identity) so
        # the generative loss can't erode property structure. 0 disables.
        align_lambda: float = 0.1,
        # Hardness of the alignment positive: corrupt the second view at this
        # flow time (uniform source) before aligning to the clean anchor. Without
        # it, fragment-shuffle alone is too easy for a permutation-invariant pool
        # (align_cos saturates at 1). A [lo, hi] range → per-sample; scalar pins
        # a level; null → clean second view (shuffle-only, can saturate).
        align_corrupt_t: list[float] | float | None = (0.2, 0.5),
        # Fingerprint (Tanimoto) similarity distillation: pull the cosine geometry
        # of z_s toward the Morgan-FP Tanimoto matrix (loss += fp_weight *
        # MSE(cos(z_s), Tanimoto)). Unlike the alignment term (pure invariance,
        # which saturates), this is a *structural* target that imposes chemical
        # similarity directly — the proven fix for probe_r2 erosion (cf.
        # discrete_flow_ssl, fp_weight=2.0). Needs data.return_smiles=true.
        fp_weight: float = 0.0,
        fp_radius: int = 2,
        fp_bits: int = 2048,
        # SIGReg (Sketched Isotropic Gaussian Regularization): push z_s toward an
        # isotropic Gaussian to prevent dimensional collapse (numerical rank 700→260).
        # Applied as loss += (sigreg_lambda / batch_size) * SIGReg(z_s), following
        # the same batch-size normalisation as LeJEPALoss (the raw Epps-Pulley
        # statistic scales ~linearly with N, so dividing by N keeps the weight
        # batch-size-independent). 0 disables.
        sigreg_lambda: float = 0.0,
        sigreg_num_projections: int = 256,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        # Beta-VAE KL regularizer: add a reparameterization head (mu/log_var) on
        # z_s and penalise KL(N(mu,sigma)||N(0,I)). Forces an information
        # bottleneck — the encoder must compress to the most molecule-relevant
        # features. loss += kl_beta * KL. 0 disables.
        kl_beta: float = 0.0,
        # t-scaled conditioning noise: z_s_cond = z_s + t * scale * ε (training only).
        # At t≈0 (heavy corruption) z_s is clean; at t≈1 (easy task) it is noisy so
        # the denoiser cannot bypass z_s at easy timesteps. 0 disables.
        cond_noise_scale: float = 0.0,
        # Corruption (uniform source). A ``[lo, hi]`` range → uniform per-sample
        # flow time (recommended); a scalar pins a fixed time; ``null`` falls
        # back to discrete-flow timestep sampling. With t in [0.1, 0.6] and
        # path_power=1, ~40–90% of positions are corrupted.
        corrupt_t: list[float] | float | None = (0.1, 0.6),
        corrupt_t_cap: float = 1e-3,
        path_power: float = 1.0,
        # Condition-bypass diagnostic: reconstruct with z_s vs zeros at a fixed
        # strong corruption; gap = recon_zeroed - recon_real. A gap below the
        # margin means z_s is inert (warn, or hard-fail if condition_hard_fail).
        condition_corrupt_t: float = 0.1,
        condition_margin: float = 0.1,
        condition_hard_fail: bool = False,
        # effective_rank(z_s) tripwire cadence (0 disables; else every N steps).
        train_rank_every_n_steps: int = 50,
        # Validation probes: encode a fixed val-split subset to z_s, fit Ridge
        # heads for QED / molWt, log rank diagnostics + PCA→t-SNE plots to W&B.
        val_probe_n_molecules: int = 2000,
        val_probe_every_n_epochs: int = 1,
        val_probe_encode_batch_size: int = 128,
        val_probe_ridge_alpha: float = 1.0,
        val_probe_test_size: float = 0.2,
        val_probe_tsne_perplexity: float | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["student"])
        self.student = student
        if student.bundle is None:
            raise ValueError(
                "DenoisingJEPAModule needs student.bundle (build via "
                "build_denoising_jepa) for tokenization"
            )
        self._val_probes = JepaValProbes(
            n_molecules=val_probe_n_molecules,
            seed=seed,
            every_n_epochs=val_probe_every_n_epochs,
            encode_batch_size=val_probe_encode_batch_size,
            ridge_alpha=val_probe_ridge_alpha,
            probe_test_size=val_probe_test_size,
            tsne_perplexity=val_probe_tsne_perplexity,
        )
        self._rng = random.Random(seed)
        self.fp_weight = float(fp_weight)
        if self.fp_weight < 0:
            raise ValueError(f"fp_weight must be >= 0, got {fp_weight}")
        self._fp_cache = (
            _FingerprintCache(radius=int(fp_radius), n_bits=int(fp_bits))
            if self.fp_weight > 0
            else None
        )
        self._sigreg = (
            SIGReg(
                num_projections=int(sigreg_num_projections),
                knots=int(sigreg_knots),
                t_max=float(sigreg_t_max),
            )
            if float(sigreg_lambda) > 0.0
            else None
        )
        if float(kl_beta) > 0.0:
            student.vae_head = VAEHead(student.encoder.pool.dim)
        self._val: dict[str, list[float]] = {
            "loss": [], "recon": [], "recon_acc": [], "rank_s": [],
            "noised_frac": [], "gap": [], "align_cos": [], "fp": [], "sigreg": [], "kl": [],
        }
        if float(align_lambda) > 0.0 and not shuffle_fragments and align_corrupt_t is None:
            logger.warning(
                "align_lambda=%.3f but shuffle_fragments=False and align_corrupt_t=None: "
                "the two views are identical, so the alignment term is trivial (cos=1). "
                "Enable shuffle_fragments, set align_corrupt_t, or set align_lambda=0.",
                align_lambda,
            )
        logger.info(
            "denoising-JEPA SSL (conditional reconstruction, no EMA): "
            "corrupt_t=%s align_lambda=%.3f align_corrupt_t=%s fp_weight=%.3f "
            "shuffle_fragments=%s freeze_backbone=%s frag_sep_id=%d",
            corrupt_t, align_lambda, align_corrupt_t, self.fp_weight,
            shuffle_fragments,
            (student.build_config or {}).get("freeze_backbone"), frag_sep_id,
        )

    # -- tokenization ------------------------------------------------------- #
    def _tokenize(self, views: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Fragment-view strings → padded ``(ids, mask)`` on the module device.

        A single (optionally fragment-shuffled) view per molecule is the *clean*
        string; the corruption inside ``denoising_loss`` provides the noised copy.
        """
        b = self.student.bundle
        sep = int(self.hparams.frag_sep_id)
        seqs: list[list[int]] = []
        for s in views:
            body = b.tokenizer.encode(s, add_special_tokens=False)
            if self.hparams.shuffle_fragments:
                body = shuffle_fragment_ids(body, sep, self._rng)
            seqs.append([b.bos_id, *body, b.eos_id])
        ids, mask = pad_batch(seqs, pad_id=b.pad_id)
        return ids.to(self.device), mask.to(self.device)

    @staticmethod
    def _split_batch(batch) -> tuple[list[str], list[str] | None]:
        """Accept either ``list[str]`` views or ``(views, smiles)`` tuples.

        SMILES (canonical, order-independent) are kept for the Tanimoto
        fingerprint distillation target; ``None`` when the datamodule does not
        yield them (``data.return_smiles=false``).
        """
        if isinstance(batch, tuple) and len(batch) == 2:
            views, smiles = batch
            return list(views), list(smiles)
        return list(batch), None

    def _sigreg_loss(self, z_s: torch.Tensor) -> torch.Tensor | None:
        """SIGReg on z_s ``[B, D]`` → scalar; ``None`` if disabled.

        Divided by batch size to keep the weight batch-size-independent (the raw
        Epps-Pulley statistic scales ~linearly with N — same normalisation as
        ``LeJEPALoss``).
        """
        if self._sigreg is None:
            return None
        return self._sigreg(z_s.unsqueeze(0)) / z_s.size(0)

    def _fp_distillation(self, z_s: torch.Tensor, smiles: list[str] | None):
        """``MSE(cos(z_s), Tanimoto(smiles))`` (off-diagonal); ``None`` if disabled.

        Distillation runs on an L2-normalized copy of ``z_s`` (which is only
        LayerNorm'd, not unit-norm) so the cosine geometry is well defined. This
        imposes *chemical* structure directly — what the pure generative loss and
        the (saturating) view-alignment term never give ``z_s``.
        """
        if self.fp_weight <= 0 or self._fp_cache is None:
            return None
        if smiles is None:
            raise ValueError(
                "fp_weight > 0 but batch has no SMILES; set data.return_smiles=true"
            )
        bits_np = self._fp_cache.bits(smiles)
        bits = torch.from_numpy(np.ascontiguousarray(bits_np)).to(z_s.device)
        target = tanimoto_target_matrix(bits).to(z_s.dtype)
        z_unit = torch.nn.functional.normalize(z_s, dim=-1)
        return similarity_distillation_loss(z_unit, target)

    def _rank_due(self) -> bool:
        every = int(self.hparams.train_rank_every_n_steps)
        return every > 0 and self.global_step % every == 0

    def _align_on(self) -> bool:
        if float(self.hparams.align_lambda) <= 0.0:
            return False
        # Non-trivial as long as the two views differ: fragment shuffle and/or
        # corrupting the second view both qualify.
        return bool(self.hparams.shuffle_fragments) or self.hparams.align_corrupt_t is not None

    def _second_view(self, views: list[str]):
        """A second, independently fragment-shuffled tokenization of ``views``
        (same molecules → same per-row length → same padded shape as the first).
        """
        return self._tokenize(views)

    # -- lifecycle ---------------------------------------------------------- #
    def on_fit_start(self) -> None:
        trainer = getattr(self, "trainer", None)
        dm = getattr(trainer, "datamodule", None) if trainer is not None else None
        if dm is None:
            return
        # Mirror the val-split selection the datamodule uses so the probe set is
        # drawn from held-out molecules.
        for attr in ("val_ratio", "test_ratio", "split_seed"):
            if hasattr(dm, attr):
                setattr(self._val_probes, attr, type(getattr(self._val_probes, attr))(getattr(dm, attr)))
        if hasattr(dm, "shard_dir"):
            self._val_probes.prepare(dm.shard_dir)

    def training_step(self, batch, batch_idx):
        views, smiles = self._split_batch(batch)
        ids, mask = self._tokenize(views)
        batch_b = self._second_view(views) if self._align_on() else None
        loss, metrics, (z_s, _logits, _noised, _ids, kl) = denoising_loss(
            self.student,
            (ids, mask),
            batch_b=batch_b,
            align_lambda=float(self.hparams.align_lambda),
            align_corrupt_t=self.hparams.align_corrupt_t,
            corrupt_t=self.hparams.corrupt_t,
            t_cap=float(self.hparams.corrupt_t_cap),
            path_power=float(self.hparams.path_power),
            compute_rank=self._rank_due(),
            return_outputs=True,
            cond_noise_scale=float(self.hparams.cond_noise_scale),
        )
        bs = len(views)
        if kl is not None:
            loss = loss + float(self.hparams.kl_beta) * kl
            self.log("train/kl", kl.detach(), on_step=True, batch_size=bs)
        fp_loss = self._fp_distillation(z_s, smiles)
        if fp_loss is not None:
            loss = loss + self.fp_weight * fp_loss
            self.log("train/fp", fp_loss.detach(), on_step=True, prog_bar=True, batch_size=bs)
        sigreg_loss = self._sigreg_loss(z_s)
        if sigreg_loss is not None:
            loss = loss + float(self.hparams.sigreg_lambda) * sigreg_loss
            self.log("train/sigreg", sigreg_loss.detach(), on_step=True, batch_size=bs)
        self.log("train/loss", loss.detach(), on_step=True, prog_bar=True, batch_size=bs)
        self.log("train/recon", metrics["recon"], on_step=True, batch_size=bs)
        self.log("train/recon_acc", metrics["recon_acc"], on_step=True, batch_size=bs)
        self.log("train/noised_frac", metrics["noised_frac"], on_step=True, batch_size=bs)
        if "align_cos" in metrics:
            self.log("train/align_cos", metrics["align_cos"], on_step=True, prog_bar=True, batch_size=bs)
        if self._rank_due():
            self.log("train/rank_s", metrics["rank_s"], on_step=True, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        views, smiles = self._split_batch(batch)
        ids, mask = self._tokenize(views)
        batch_b = self._second_view(views) if self._align_on() else None
        with torch.no_grad():
            _, metrics, (z_s, _logits, _noised, _ids, kl) = denoising_loss(
                self.student,
                (ids, mask),
                batch_b=batch_b,
                align_lambda=float(self.hparams.align_lambda),
                align_corrupt_t=self.hparams.align_corrupt_t,
                corrupt_t=self.hparams.corrupt_t,
                t_cap=float(self.hparams.corrupt_t_cap),
                path_power=float(self.hparams.path_power),
                compute_rank=True,
                return_outputs=True,
                cond_noise_scale=0.0,
            )
            fp_loss = self._fp_distillation(z_s, smiles)
            sigreg_loss = self._sigreg_loss(z_s)
            gap = condition_bypass_gap(
                self.student,
                (ids, mask),
                corrupt_t=float(self.hparams.condition_corrupt_t),
                t_cap=float(self.hparams.corrupt_t_cap),
                path_power=float(self.hparams.path_power),
            )
        self._val["loss"].append(metrics["loss"])
        self._val["recon"].append(metrics["recon"])
        self._val["recon_acc"].append(metrics["recon_acc"])
        self._val["rank_s"].append(metrics["rank_s"])
        self._val["noised_frac"].append(metrics["noised_frac"])
        self._val["gap"].append(gap["gap"])
        if "align_cos" in metrics:
            self._val["align_cos"].append(metrics["align_cos"])
        if fp_loss is not None:
            self._val["fp"].append(float(fp_loss))
        if sigreg_loss is not None:
            self._val["sigreg"].append(float(sigreg_loss.detach()))
        if kl is not None:
            self._val["kl"].append(float(kl.detach()))

    def on_validation_epoch_end(self) -> None:
        def _m(key: str, default: float = float("nan")) -> float:
            xs = self._val[key]
            return float(np.mean(xs)) if xs else default

        gap = _m("gap")
        out = {
            "val/loss": _m("loss"),
            "val/recon": _m("recon"),
            # acc@1 = token reconstruction accuracy at the noised positions
            # (monitored by the ssl_basic ModelCheckpoint, mode="max").
            "val/acc@1": _m("recon_acc", 0.0),
            "val/rank_s": _m("rank_s"),
            "val/noised_frac": _m("noised_frac"),
            "val/condition_gap": gap,
        }
        if self._val["align_cos"]:
            out["val/align_cos"] = _m("align_cos")
        if self._val["fp"]:
            out["val/fp"] = _m("fp")
        if self._val["sigreg"]:
            out["val/sigreg"] = _m("sigreg")
        if self._val["kl"]:
            out["val/kl"] = _m("kl")
        # Probe set (global-zero only): Ridge R² (QED/molWt), rank/{effective,
        # numerical} over a fixed 2k val subset, and PCA→t-SNE plots to W&B.
        out.update(self._val_probes.maybe_run(self))
        self.log_dict(out, prog_bar=True, sync_dist=True)

        # Inert-anchor guard: surface a weak conditioning gap loudly.
        margin = float(self.hparams.condition_margin)
        if np.isfinite(gap) and gap < margin:
            msg = (
                f"condition-bypass: val/condition_gap={gap:.4f} < margin={margin}. "
                f"z_s is (nearly) inert — the denoiser reconstructs without it. "
                f"Increase corruption (lower corrupt_t) so reconstruction must "
                f"rely on z_s."
            )
            if bool(self.hparams.condition_hard_fail):
                raise RuntimeError(msg)
            logger.warning(msg)
        for k in self._val:
            self._val[k].clear()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        cfg = getattr(self.student, "build_config", None)
        if cfg is not None:
            checkpoint["encoder_config"] = dict(cfg)

    def _resolve_total_steps(self) -> int:
        """Cosine-decay horizon. When ``total_steps`` is null/<=0, span the whole
        run via the trainer's estimated step count so the LR does not bottom out
        at the 0.1x floor long before training ends (the schedule otherwise
        completes at a hardcoded 30k while the run continues to ~80k)."""
        configured = int(self.hparams.total_steps or 0)
        if configured > 0:
            return configured
        trainer = getattr(self, "trainer", None)
        est = getattr(trainer, "estimated_stepping_batches", None) if trainer else None
        if est is not None and math.isfinite(est) and est > 0:
            total = int(est)
            logger.info("total_steps auto-derived from trainer: %d", total)
            return total
        logger.warning("total_steps unset and trainer estimate unavailable; using 30000")
        return 30_000

    def configure_optimizers(self):
        hp = self.hparams
        params = [p for p in self.student.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(
            params, lr=hp.learning_rate, weight_decay=hp.weight_decay
        )
        total_steps = self._resolve_total_steps()
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: cosine_with_warmup(s, hp.warmup_steps, total_steps)
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
