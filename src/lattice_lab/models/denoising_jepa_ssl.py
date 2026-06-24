"""Conditional denoising-JEPA SSL (:class:`DenoisingJEPAModule`)."""

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
    VAEHead,
    build_denoising_jepa,
    condition_bypass_gap,
    denoise_logits,
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
        *,
        ckpt_path: str | None,
        tokenizer_path: str,
        pool_heads: int = 8,
        pool_dropout: float = 0.0,
        encode_time: float = 1.0,
        freeze_backbone: bool = True,
        token_id_min: int = 4,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        frag_sep_id: int = 4,
        shuffle_fragments: bool = True,
        align_lambda: float = 0.0,
        fp_weight: float = 0.0,
        fp_radius: int = 2,
        fp_bits: int = 2048,
        sigreg_lambda: float = 0.0,
        sigreg_num_projections: int = 256,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        kl_beta: float = 0.0,
        kl_free_bits: float = 0.0,
        kl_warmup_steps: int = 0,
        cond_noise_scale: float = 0.0,
        latent_consistency_lambda: float = 0.0,
        corrupt_t: list[float] | float | None = (0.1, 0.6),
        corrupt_t_cap: float = 1e-3,
        path_power: float = 1.0,
        condition_corrupt_t: float = 0.1,
        condition_margin: float = 0.1,
        condition_hard_fail: bool = False,
        train_rank_every_n_steps: int = 50,
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
        self.save_hyperparameters()
        build_denoising_jepa(
            ckpt_path=ckpt_path,
            tokenizer_path=tokenizer_path,
            pool_heads=pool_heads,
            pool_dropout=pool_dropout,
            encode_time=encode_time,
            freeze_backbone=freeze_backbone,
            token_id_min=token_id_min,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            parent=self,
        )
        if self.bundle is None:
            raise ValueError("build_denoising_jepa must attach bundle for tokenization")
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
            self.vae_head = VAEHead(
                self.encoder.pool.dim, free_bits=float(kl_free_bits)
            )
        self._val: dict[str, list[float]] = {
            "loss": [], "recon": [], "recon_acc": [], "rank_s": [],
            "noised_frac": [], "gap": [], "align_cos": [], "fp": [], "sigreg": [], "kl": [],
            "latent": [], "latent_cos": [],
            "logvar_mean": [], "logvar_std": [], "logvar_exp_mean": [],
        }
        if float(align_lambda) > 0.0 and not shuffle_fragments:
            logger.warning(
                "align_lambda=%.3f with shuffle_fragments=False: views are identical",
                align_lambda,
            )
        logger.info(
            "denoising-JEPA: corrupt_t=%s align_lambda=%.3f fp_weight=%.3f "
            "latent_consistency_lambda=%.3f shuffle_fragments=%s freeze_backbone=%s frag_sep_id=%d",
            corrupt_t, align_lambda, self.fp_weight, latent_consistency_lambda,
            shuffle_fragments,
            (self.build_config or {}).get("freeze_backbone"), frag_sep_id,
        )

    def denoise_logits(self, x_t, mask, t, conds):
        return denoise_logits(self, x_t, mask, t, conds)

    def _tokenize(self, views: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.bundle
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
        if isinstance(batch, tuple) and len(batch) == 2:
            views, smiles = batch
            return list(views), list(smiles)
        return list(batch), None

    def _sigreg_loss(self, z_s: torch.Tensor) -> torch.Tensor | None:
        if self._sigreg is None:
            return None
        return self._sigreg(z_s.unsqueeze(0)) / z_s.size(0)

    def _fp_distillation(self, z_s: torch.Tensor, smiles: list[str] | None):
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

    def _kl_warmup_factor(self) -> float:
        steps = int(self.hparams.kl_warmup_steps)
        if steps <= 0:
            return 1.0
        return min(1.0, (self.global_step + 1) / steps)

    def _align_on(self) -> bool:
        return (
            float(self.hparams.align_lambda) > 0.0
            and bool(self.hparams.shuffle_fragments)
        )

    def _second_view(self, views: list[str]):
        return self._tokenize(views)

    def _log_grad_norms(self, **terms: torch.Tensor) -> None:
        params = [p for p in self.parameters() if p.requires_grad]
        logs: dict[str, float] = {}
        for name, term in terms.items():
            grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            sq_sum = sum(g.float().pow(2).sum() for g in grads if g is not None)
            logs[f"train/grad_norm_{name}"] = float(sq_sum ** 0.5)
        self.log_dict(logs, on_step=True, batch_size=1)

    def on_fit_start(self) -> None:
        trainer = getattr(self, "trainer", None)
        dm = getattr(trainer, "datamodule", None) if trainer is not None else None
        if dm is None:
            return
        self._fragment_merge = str(getattr(dm, "shard_dir", "")).rstrip("/").endswith("_merge")
        for attr in ("val_ratio", "test_ratio", "split_seed"):
            if hasattr(dm, attr):
                setattr(self._val_probes, attr, type(getattr(self._val_probes, attr))(getattr(dm, attr)))
        if hasattr(dm, "shard_dir"):
            self._val_probes.prepare(dm.shard_dir)

    def training_step(self, batch, batch_idx):
        views, smiles = self._split_batch(batch)
        ids, mask = self._tokenize(views)
        batch_b = self._second_view(views) if self._align_on() else None
        loss, metrics, (z_s, _logits, _noised, _ids, kl), terms = denoising_loss(
            self,
            (ids, mask),
            batch_b=batch_b,
            align_lambda=float(self.hparams.align_lambda),
            corrupt_t=self.hparams.corrupt_t,
            t_cap=float(self.hparams.corrupt_t_cap),
            path_power=float(self.hparams.path_power),
            compute_rank=self._rank_due(),
            return_outputs=True,
            cond_noise_scale=float(self.hparams.cond_noise_scale),
            latent_consistency_lambda=float(self.hparams.latent_consistency_lambda),
        )
        bs = len(views)
        if kl is not None:
            beta = float(self.hparams.kl_beta) * self._kl_warmup_factor()
            loss = loss + beta * kl
            terms["kl"] = beta * kl
            self.log("train/kl", kl.detach(), on_step=True, batch_size=bs)
            self.log("train/kl_beta", beta, on_step=True, batch_size=bs)
        fp_loss = self._fp_distillation(z_s, smiles)
        if fp_loss is not None:
            loss = loss + self.fp_weight * fp_loss
            terms["fp"] = self.fp_weight * fp_loss
            self.log("train/fp", fp_loss.detach(), on_step=True, prog_bar=True, batch_size=bs)
        sigreg_loss = self._sigreg_loss(z_s)
        if sigreg_loss is not None:
            loss = loss + float(self.hparams.sigreg_lambda) * sigreg_loss
            terms["sigreg"] = float(self.hparams.sigreg_lambda) * sigreg_loss
            self.log("train/sigreg", sigreg_loss.detach(), on_step=True, batch_size=bs)
        if self.hparams.log_grad_norms and torch.is_grad_enabled():
            self._log_grad_norms(**terms)
        self.log("train/loss", loss.detach(), on_step=True, prog_bar=True, batch_size=bs)
        self.log("train/recon", metrics["recon"], on_step=True, batch_size=bs)
        self.log("train/recon_acc", metrics["recon_acc"], on_step=True, batch_size=bs)
        self.log("train/noised_frac", metrics["noised_frac"], on_step=True, batch_size=bs)
        if "align_cos" in metrics:
            self.log("train/align_cos", metrics["align_cos"], on_step=True, prog_bar=True, batch_size=bs)
        if "latent_cos" in metrics:
            self.log("train/latent_cos", metrics["latent_cos"], on_step=True, batch_size=bs)
        if "latent" in metrics:
            self.log("train/latent", metrics["latent"], on_step=True, batch_size=bs)
        if self._rank_due():
            self.log("train/rank_s", metrics["rank_s"], on_step=True, prog_bar=True, batch_size=bs)
        for k in ("logvar_mean", "logvar_std", "logvar_exp_mean"):
            if k in metrics:
                self.log(f"diagnostics/{k}", metrics[k], on_step=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        views, smiles = self._split_batch(batch)
        ids, mask = self._tokenize(views)
        batch_b = self._second_view(views) if self._align_on() else None
        with torch.no_grad():
            _, metrics, (z_s, _logits, _noised, _ids, kl), _ = denoising_loss(
                self,
                (ids, mask),
                batch_b=batch_b,
                align_lambda=float(self.hparams.align_lambda),
                corrupt_t=self.hparams.corrupt_t,
                t_cap=float(self.hparams.corrupt_t_cap),
                path_power=float(self.hparams.path_power),
                compute_rank=True,
                return_outputs=True,
                cond_noise_scale=0.0,
                latent_consistency_lambda=float(self.hparams.latent_consistency_lambda),
            )
            fp_loss = self._fp_distillation(z_s, smiles)
            sigreg_loss = self._sigreg_loss(z_s)
            gap = condition_bypass_gap(
                self,
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
        if "latent" in metrics:
            self._val["latent"].append(metrics["latent"])
        if "latent_cos" in metrics:
            self._val["latent_cos"].append(metrics["latent_cos"])
        if fp_loss is not None:
            self._val["fp"].append(float(fp_loss))
        if sigreg_loss is not None:
            self._val["sigreg"].append(float(sigreg_loss.detach()))
        if kl is not None:
            self._val["kl"].append(float(kl.detach()))
        for k in ("logvar_mean", "logvar_std", "logvar_exp_mean"):
            if k in metrics:
                self._val[k].append(metrics[k])

    def on_validation_epoch_end(self) -> None:
        def _m(key: str, default: float = float("nan")) -> float:
            xs = self._val[key]
            return float(np.mean(xs)) if xs else default

        gap = _m("gap")
        out = {
            "val/loss": _m("loss"),
            "val/recon": _m("recon"),
            "val/acc@1": _m("recon_acc", 0.0),
            "val/rank_s": _m("rank_s"),
            "val/noised_frac": _m("noised_frac"),
            "val/condition_gap": gap,
        }
        if self._val["align_cos"]:
            out["val/align_cos"] = _m("align_cos")
        if self._val["latent"]:
            out["val/latent"] = _m("latent")
        if self._val["latent_cos"]:
            out["val/latent_cos"] = _m("latent_cos")
        if self._val["fp"]:
            out["val/fp"] = _m("fp")
        if self._val["sigreg"]:
            out["val/sigreg"] = _m("sigreg")
        if self._val["kl"]:
            out["val/kl"] = _m("kl")
        for k in ("logvar_mean", "logvar_std", "logvar_exp_mean"):
            if self._val[k]:
                out[f"diagnostics/{k}"] = _m(k)
        out.update(self._val_probes.maybe_run(self))
        self.log_dict(out, prog_bar=True, sync_dist=True)

        margin = float(self.hparams.condition_margin)
        if np.isfinite(gap) and gap < margin:
            msg = (
                f"val/condition_gap={gap:.4f} < margin={margin} "
                f"(recon_real vs zeroed-z_s at corrupt_t={self.hparams.condition_corrupt_t})"
            )
            if bool(self.hparams.condition_hard_fail):
                raise RuntimeError(msg)
            logger.warning(msg)
        for k in self._val:
            self._val[k].clear()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        cfg = getattr(self, "build_config", None)
        if cfg is not None:
            checkpoint["encoder_config"] = dict(cfg)
        # See discrete_flow_ssl: record the fragment-view variant so downstream
        # stages auto-match the _merge stores (no env, no mismatch).
        checkpoint["fragment_merge"] = bool(getattr(self, "_fragment_merge", False))

    def _resolve_total_steps(self) -> int:
        trainer = getattr(self, "trainer", None)
        est = getattr(trainer, "estimated_stepping_batches", None) if trainer else None
        if est is not None and math.isfinite(est) and est > 0:
            total = int(est)
            logger.info("cosine LR horizon from trainer: %d steps", total)
            return total
        logger.warning("trainer estimate unavailable; cosine LR horizon fallback 30000")
        return 30_000

    def configure_optimizers(self):
        hp = self.hparams
        params = [p for p in self.parameters() if p.requires_grad]
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
