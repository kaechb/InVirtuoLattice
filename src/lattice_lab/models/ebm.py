"""LightningModule for the Stage-5 energy-based head.

All numerics come from the package's own kernels: the encoder/head builders, the
SMILES→z_m encoding, the three loss terms and the EF/BEDROC metrics. This class
only restructures them into the Lightning lifecycle:

- ``training_step`` returns the scalar loss; ``self.log`` streams the train
  metrics (the native ``WandbLogger`` + tqdm bar handle the rest).
- ``validation_step`` computes the training loss on the val batch and accumulates
  per-target ``(binder + N decoys)`` scores; ``on_validation_epoch_end`` reduces
  them to ``val/{loss,infonce,sinkhorn,ef1,ef5,top1,bedroc}`` and logs them so
  ``ModelCheckpoint(monitor="val/bedroc", mode="max")`` can promote the best
  checkpoint. NB: ``val/loss`` (temperature-scaled InfoNCE) is tail-dominated by
  hard false-negatives and *rises* as ranking improves — it is anti-correlated
  with retrieval and must NOT be used as the monitor. ``val/bedroc`` is the
  scale-invariant early-enrichment signal, smoother than few-target ``val/ef1``.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder, sync_encoder_device
from lattice_lab.ebm.losses import (
    InfoNCEEnergyLoss,
    SinkhornEnergyLoss,
    cross_target_margin_loss,
)
from lattice_lab.eval.metrics import bedroc
from lattice_lab.models.builders import build_energy_head
from lattice_lab.models.encode import ef_at, encode_binders
from lattice_lab.models.schedules import cosine_with_warmup, lambda_sink_schedule

logger = logging.getLogger(__name__)


class EBMLitModule(L.LightningModule):
    def __init__(
        self,
        encoder: DiscreteFlowEncoder,
        *,
        resume_from: str | Path | None = None,
        d_adapter: int = 512,
        d_protein: int = 1280,
        n_decoys: int = 600,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        num_steps: int = 20_000,
        warmup_steps: int = 500,
        temperature: float = 0.1,
        head_type: str = "film",
        lambda_sink: float = 1.0,
        lambda_sink_warmup: int = 10_000,
        lambda_neg: float = 1.0,
        cross_target_p: float = 1.0,
        cross_target_margin: float = 2.0,
        hard_mining_mult: int = 1,
        hard_skip_frac: float = 0.05,
        bedroc_alpha: float = 80.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder
        self.head = build_energy_head(
            d_adapter=d_adapter, d_protein=d_protein, head_type=head_type
        )
        logger.info("energy head (%s): params=%d", head_type, self.head.num_trainable_params)

        self.info_loss = InfoNCEEnergyLoss(temperature=temperature)
        self.sink_loss = SinkhornEnergyLoss()
        self._cross_rng = random.Random(seed + 7)
        self._val: dict[str, list[float]] = defaultdict(list)
        # Binder embeddings are a deterministic function of SMILES under the
        # frozen encoder, so encode each unique binder once and reuse it.
        self._zm_cache: dict[str, torch.Tensor] = {}

        if resume_from is not None:
            self._load_resume(Path(resume_from))

    def _load_resume(self, path: Path) -> None:
        """Warm-start the energy head from a full EBM ``.ckpt``."""
        from lattice_lab.models.builders import parse_head_checkpoint, safe_torch_load

        raw = safe_torch_load(path, weights_only=False)
        self.head.load_state_dict(parse_head_checkpoint(raw))
        logger.info("resumed from full checkpoint %s", path)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Embed the encoder skeleton config so the ckpt is self-describing.

        Lets :func:`load_encoder_from_ckpt` rebuild the exact encoder (including
        the DDiT hook layer range) from this one file — no base DDiT, no
        per-caller layer range that could drift from training.
        """
        cfg = getattr(self.encoder, "build_config", None)
        if cfg is not None:
            checkpoint["encoder_config"] = dict(cfg)

    # -- lifecycle -------------------------------------------------------
    def _freeze_eval_modes(self) -> None:
        """Force the frozen encoder into ``eval()``.

        Lightning calls ``self.train()`` at the start of fit/each epoch, which
        recursively re-enables dropout on the DDiT backbone and adapter. Keeping
        them in ``eval()`` makes binder embeddings deterministic — required for
        the z_m cache and precomputed binder store to stay valid.
        """
        self.encoder.backbone.eval()
        self.encoder.adapter.eval()

    def on_fit_start(self) -> None:
        sync_encoder_device(self.encoder, self.device, head=self.head)
        self._freeze_eval_modes()

    def on_train_epoch_start(self) -> None:
        self._freeze_eval_modes()

    def on_validation_start(self) -> None:
        sync_encoder_device(self.encoder, self.device, head=self.head)
        self._freeze_eval_modes()

    def _encode_binders(self, batch: dict) -> torch.Tensor:
        """Resolve binder ``z_m`` → ``[B, d_m]``.

        1. precomputed ``binder_z_m`` from the Stage-4 binder store, else
        2. an in-process SMILES→z_m cache (one frozen encode per unique SMILES).
        """
        if batch.get("binder_z_m") is not None:
            return batch["binder_z_m"].to(self.device, non_blocking=True)

        smiles = batch["binder_smiles"]
        missing = [s for s in dict.fromkeys(smiles) if s not in self._zm_cache]
        if missing:
            z = encode_binders(self.encoder, missing, self.device, grad=False)
            # Single batched D2H copy (one sync) instead of one per row — the
            # per-row transfer is what dragged the warm-up epoch down.
            z_cpu = z.detach().to("cpu")
            for s, row in zip(missing, z_cpu):
                self._zm_cache[s] = row
        rows = [self._zm_cache[s] for s in smiles]
        return torch.stack(rows, dim=0).to(self.device, non_blocking=True)

    # -- training --------------------------------------------------------
    def _mine_hard_negatives(self, z_m_dec: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        hp = self.hparams
        self.head.eval()
        with torch.no_grad():
            p_cand = z_m_dec.shape[1]
            z_p_cand = z_p.unsqueeze(1).expand(-1, p_cand, -1)
            e_cand = self.head(z_m_dec, z_p_cand)
        self.head.train()
        skip = int(round(p_cand * hp.hard_skip_frac))
        order = torch.argsort(e_cand, dim=1)
        sel = order[:, skip : skip + hp.n_decoys]
        return torch.gather(z_m_dec, 1, sel.unsqueeze(-1).expand(-1, -1, z_m_dec.shape[-1]))

    def training_step(self, batch, batch_idx):
        hp = self.hparams
        z_m_pos = self._encode_binders(batch)
        z_p = batch["z_p"].to(self.device)
        z_m_dec = batch["decoy_z_m"].to(self.device)
        if hp.hard_mining_mult > 1:
            z_m_dec = self._mine_hard_negatives(z_m_dec, z_p)

        # ponytail: bf16 InfoNCE+Sinkhorn over ~600 decoys underflows head grads to 0;
        # encoder is frozen so fp32 head is free.
        ct_log: dict[str, float] | None = None
        with torch.autocast(device_type=self.device.type, enabled=False):
            e_pos = self.head(z_m_pos, z_p)
            z_p_dec = z_p.unsqueeze(1).expand(-1, z_m_dec.shape[1], -1)
            e_dec = self.head(z_m_dec, z_p_dec)

            l_info, info_log = self.info_loss(e_pos, e_dec)
            l_sink, sink_log = self.sink_loss(e_pos, e_dec)
            lam = lambda_sink_schedule(self.global_step, hp.lambda_sink_warmup, hp.lambda_sink)
            total = l_info + lam * l_sink

            if self._cross_rng.random() < hp.cross_target_p:
                bs = z_p.shape[0]
                perm = torch.randperm(bs, device=self.device)
                if torch.equal(perm, torch.arange(bs, device=self.device)):
                    perm = torch.roll(perm, shifts=1)
                e_wrong = self.head(z_m_pos, z_p[perm])
                l_ct, ct_log = cross_target_margin_loss(
                    e_pos, e_wrong, margin=hp.cross_target_margin
                )
                total = total + hp.lambda_neg * l_ct

        bs = z_p.shape[0]
        # FiLM gate magnitude: γ/β start at 0 (identity in z_m); if these stay
        # ~0 the head is ignoring the protein. Cheap probe, but LayerNorm after
        # FiLM can absorb part of it — diagnostics/e_protein_gap below is the
        # decisive test of whether the *energy* actually depends on z_p.
        log = {
            "train/loss": total.detach(),
            "train/infonce": info_log["infonce/loss"],
            "train/sinkhorn": sink_log["sinkhorn/loss"],
            "train/lambda_sink": lam,
            "train/top1": info_log["infonce/top1"],
            "train/binder_e_mean": e_pos.detach().mean(),
            "train/decoy_e_mean": e_dec.detach().mean(),
        }
        if hasattr(self.head, "protein_proj"):
            with torch.no_grad():
                gamma, beta = self.head.protein_proj(z_p).chunk(2, dim=-1)
            log["diagnostics/film_gamma_absmean"] = gamma.abs().mean()
            log["diagnostics/film_beta_absmean"] = beta.abs().mean()
        if ct_log is not None:
            log["train/cross_target"] = ct_log["cross_target/loss"]
            log["train/cross_target_viol"] = ct_log["cross_target/violation_rate"]
            # |E(z_m+, z_p) − E(z_m+, z_p_wrong)|: energy-level protein sensitivity.
            log["diagnostics/e_protein_gap"] = (e_wrong - e_pos).detach().abs().mean()

        self.log_dict(log, on_step=True, on_epoch=False, prog_bar=False, batch_size=bs)
        self.log("train/loss_bar", total.detach(), prog_bar=True, batch_size=bs)
        return total

    # -- validation ------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        hp = self.hparams
        z_m_pos = self._encode_binders(batch)
        z_p = batch["z_p"].to(self.device)
        z_m_dec = batch["decoy_z_m"].to(self.device)
        n = z_m_dec.shape[1]

        # fp32 head (matches training_step): bf16 InfoNCE+Sinkhorn over the decoy
        # set underflows, and the frozen-encoder head is cheap in fp32. Compute
        # the same loss as training so val/loss is a smooth checkpoint monitor.
        with torch.autocast(device_type=self.device.type, enabled=False):
            e_pos = self.head(z_m_pos, z_p)
            z_p_dec = z_p.unsqueeze(1).expand(-1, n, -1)
            e_dec = self.head(z_m_dec, z_p_dec)
            l_info, _ = self.info_loss(e_pos, e_dec)
            l_sink, _ = self.sink_loss(e_pos, e_dec)
            lam = lambda_sink_schedule(
                self.global_step, hp.lambda_sink_warmup, hp.lambda_sink
            )
            # Same InfoNCE at T=1: undoes the ×10 logit scaling that makes the
            # T=0.1 training loss tail-dominated and anti-correlated with ranking.
            all_e = torch.cat([e_pos.unsqueeze(1), e_dec], dim=1)
            l_info_t1 = torch.nn.functional.cross_entropy(
                -all_e, torch.zeros(all_e.shape[0], dtype=torch.long, device=all_e.device)
            )
        self._val["loss"].append(float((l_info + lam * l_sink).detach()))
        self._val["infonce"].append(float(l_info.detach()))
        self._val["infonce_t1"].append(float(l_info_t1.detach()))
        self._val["sinkhorn"].append(float(l_sink.detach()))

        scores = -torch.cat([e_pos.unsqueeze(1), e_dec], dim=1).float().cpu().numpy()
        labels = np.zeros_like(scores)
        labels[:, 0] = 1
        for s_row, l_row in zip(scores, labels):
            self._val["ef1"].append(ef_at(1.0, s_row, l_row))
            self._val["ef5"].append(ef_at(5.0, s_row, l_row))
            self._val["top1"].append(float(s_row.argmax() == 0))
            self._val["bedroc"].append(bedroc(l_row, s_row, alpha=hp.bedroc_alpha))

    def on_validation_epoch_end(self) -> None:
        out = {
            "val/loss": float(np.mean(self._val["loss"])) if self._val["loss"] else 0.0,
            "val/infonce": float(np.mean(self._val["infonce"]))
            if self._val["infonce"]
            else 0.0,
            "val/infonce_t1": float(np.mean(self._val["infonce_t1"]))
            if self._val["infonce_t1"]
            else 0.0,
            "val/sinkhorn": float(np.mean(self._val["sinkhorn"]))
            if self._val["sinkhorn"]
            else 0.0,
            "val/ef1": float(np.mean(self._val["ef1"])) if self._val["ef1"] else 0.0,
            "val/ef5": float(np.mean(self._val["ef5"])) if self._val["ef5"] else 0.0,
            "val/top1": float(np.mean(self._val["top1"])) if self._val["top1"] else 0.0,
            "val/bedroc": float(np.nanmean(self._val["bedroc"])) if self._val["bedroc"] else 0.0,
            "val/n_targets": float(len(self._val["ef1"])),
        }
        self.log_dict(out, prog_bar=True, sync_dist=True)
        logger.info("val %s", {k: round(v, 4) for k, v in out.items()})
        self._val.clear()

    # -- optim -----------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams
        optim = torch.optim.AdamW(
            self.head.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay
        )
        # num_steps drives the cosine decay length. Epoch-based runs set
        # trainer.max_steps=-1 (→ num_steps=-1); use Lightning's estimate of the
        # total optimizer steps instead, else the cosine collapses to its 0.1x
        # floor the instant warmup ends (progress = (step-warmup)/max(1,-1-warmup)).
        total = int(hp.num_steps)
        if total <= 0:
            total = int(self.trainer.estimated_stepping_batches)
        logger.info("LR cosine: warmup=%d total=%d", hp.warmup_steps, total)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: cosine_with_warmup(s, hp.warmup_steps, total)
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
