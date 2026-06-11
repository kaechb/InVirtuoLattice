"""LightningModule for Stage-2 adapter SSL (NT-Xent + optional FP distillation).

Reuses the encoder builder, the NT-Xent loss, the Tanimoto similarity-
distillation objective and the paired top-1 accuracy from the package kernels. The
Stage-2 freeze gate (val-alignment / bioisostere / QM9 sanity checks and the
``adapter_v1.pt`` promotion) lives in
:class:`lattice_lab.callbacks.sanity_gate.SanityGateCallback` so this module
stays focused on the train/val loop.
"""

from __future__ import annotations

import logging

import lightning as L
import numpy as np
import torch

from lattice_lab.backbone.encoder import sync_encoder_device
from lattice_lab.training.ssl_loss import (
    NTXentLoss,
    _FingerprintCache,
    similarity_distillation_loss,
    tanimoto_target_matrix,
    top1_paired_accuracy,
)
from lattice_lab.models.builders import build_adapter_encoder
from lattice_lab.models.schedules import cosine_with_warmup

logger = logging.getLogger(__name__)


class AdapterLitModule(L.LightningModule):
    def __init__(
        self,
        *,
        n_fragmol_layers: int = 4,
        d_adapter: int = 512,
        n_adapter_layers: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 10_000,
        temperature: float = 0.1,
        fp_weight: float = 0.0,
        fp_radius: int = 2,
        fp_bits: int = 2048,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = build_adapter_encoder(
            n_fragmol_layers=n_fragmol_layers,
            d_adapter=d_adapter,
            n_adapter_layers=n_adapter_layers,
        )
        self.encoder.adapter.train()
        self.loss_fn = NTXentLoss(temperature=temperature)
        self.use_fp = fp_weight > 0
        self.fp_cache = _FingerprintCache(fp_radius, fp_bits) if self.use_fp else None
        self._val: dict[str, list[float]] = {"loss": [], "acc": []}
        if self.use_fp:
            logger.info(
                "fingerprint distillation ON: fp_weight=%.3g radius=%d bits=%d",
                fp_weight, fp_radius, fp_bits,
            )

    def on_fit_start(self) -> None:
        sync_encoder_device(self.encoder, self.device)

    def on_validation_start(self) -> None:
        sync_encoder_device(self.encoder, self.device)

    def _encode_pair(self, views_a, views_b):
        z_m_a, z_a = self.encoder.encode_views(views_a, device=self.device, return_projection=True)
        z_m_b, z_b = self.encoder.encode_views(views_b, device=self.device, return_projection=True)
        return z_m_a, z_a, z_m_b, z_b

    def training_step(self, batch, batch_idx):
        hp = self.hparams
        if self.use_fp:
            views_a, views_b, smiles = batch
        else:
            views_a, views_b = batch
        z_m_a, z_a, z_m_b, z_b = self._encode_pair(views_a, views_b)
        loss = self.loss_fn(z_a, z_b)
        log = {"train/acc@1": top1_paired_accuracy(z_a.detach(), z_b.detach())}
        if self.use_fp:
            assert self.fp_cache is not None
            bits = torch.from_numpy(np.asarray(self.fp_cache.bits(smiles))).to(self.device)
            target_sim = tanimoto_target_matrix(bits)
            fp_loss = 0.5 * (
                similarity_distillation_loss(z_m_a, target_sim)
                + similarity_distillation_loss(z_m_b, target_sim)
            )
            loss = loss + hp.fp_weight * fp_loss
            log["train/fp_loss"] = fp_loss.detach()
        log["train/loss"] = loss.detach()
        self.log_dict(log, on_step=True, on_epoch=False, batch_size=len(views_a))
        self.log("train/loss_bar", loss.detach(), prog_bar=True, batch_size=len(views_a))
        return loss

    def validation_step(self, batch, batch_idx):
        views_a, views_b = batch
        _, z_a, _, z_b = self._encode_pair(views_a, views_b)
        self._val["loss"].append(self.loss_fn(z_a, z_b).item())
        self._val["acc"].append(top1_paired_accuracy(z_a, z_b))

    def on_validation_epoch_end(self) -> None:
        out = {
            "val/loss": float(np.mean(self._val["loss"])) if self._val["loss"] else float("nan"),
            "val/acc@1": float(np.mean(self._val["acc"])) if self._val["acc"] else 0.0,
        }
        self.log_dict(out, prog_bar=True, sync_dist=True)
        logger.info("val %s", {k: round(v, 4) for k, v in out.items()})
        self._val["loss"].clear()
        self._val["acc"].clear()

    def configure_optimizers(self):
        hp = self.hparams
        optim = torch.optim.AdamW(
            self.encoder.adapter.parameters(),
            lr=hp.learning_rate, weight_decay=hp.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: cosine_with_warmup(s, hp.warmup_steps, hp.total_steps)
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
