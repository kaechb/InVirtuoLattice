"""LightningModule for the Stage-5 energy-based head.

All numerics come from the package's own kernels: the encoder/head builders, the
SMILES→z_m encoding, the three loss terms and the EF/BEDROC metrics. This class
only restructures them into the Lightning lifecycle:

- ``training_step`` returns the scalar loss; ``self.log`` streams the train
  metrics (the native ``WandbLogger`` + tqdm bar handle the rest).
- ``validation_step`` accumulates per-target ``(binder + N decoys)`` scores;
  ``on_validation_epoch_end`` reduces them to ``val/{ef1,ef5,top1,bedroc}`` and
  logs them so ``ModelCheckpoint(monitor="val/ef1", mode="max")`` can promote
  the best checkpoint — replacing the old bespoke ``ebm_best_*.pt`` bookkeeping.
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
        lambda_sink: float = 1.0,
        lambda_sink_warmup: int = 10_000,
        lambda_neg: float = 1.0,
        cross_target_p: float = 1.0,
        cross_target_margin: float = 2.0,
        hard_mining_mult: int = 1,
        hard_skip_frac: float = 0.05,
        finetune_adapter: bool = False,
        adapter_lr: float = 3e-5,
        bedroc_alpha: float = 80.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder
        self.head = build_energy_head(d_adapter=d_adapter, d_protein=d_protein)
        logger.info("energy head: params=%d", self.head.num_trainable_params)

        self.info_loss = InfoNCEEnergyLoss(temperature=temperature)
        self.sink_loss = SinkhornEnergyLoss()
        self._cross_rng = random.Random(seed + 7)
        self._val: dict[str, list[float]] = defaultdict(list)
        # When the adapter is frozen, binder embeddings are a deterministic
        # function of the SMILES, so we encode each unique binder through the
        # DDiT backbone + adapter exactly once and reuse it. This removes the
        # per-step RDKit fragmentation + full backbone forward that made Stage-5
        # look "frozen". Kept on CPU (a few KB/row) to spare GPU memory.
        self._zm_cache: dict[str, torch.Tensor] = {}

        if finetune_adapter:
            for p in self.encoder.adapter.parameters():
                p.requires_grad = True
            self.encoder.adapter.train()
            logger.info(
                "adapter fine-tuning ON: adapter_lr=%.1e head_lr=%.1e",
                adapter_lr, learning_rate,
            )

        if resume_from is not None:
            self._load_resume(Path(resume_from), finetune_adapter)

    def _load_resume(self, path: Path, finetune_adapter: bool) -> None:
        """Warm-start head (and optionally adapter) from a full EBM ``.ckpt``."""
        from lattice_lab.models.builders import (
            parse_adapter_state,
            parse_head_checkpoint,
            safe_torch_load,
        )

        raw = safe_torch_load(path, weights_only=False)
        self.head.load_state_dict(parse_head_checkpoint(raw))
        if finetune_adapter:
            self.encoder.adapter.load_state_dict(parse_adapter_state(raw))
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
        """Force frozen sub-modules into ``eval()``.

        Lightning calls ``self.train()`` at the start of fit/each epoch, which
        recursively re-enables dropout on the (always-frozen) DDiT backbone and,
        when not fine-tuning, the adapter. Keeping them in ``eval()`` makes
        binder embeddings deterministic — required for the z_m cache to be valid
        and to stop noise leaking into the energy head.
        """
        self.encoder.backbone.eval()
        if not self.hparams.finetune_adapter:
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

        Resolution order (frozen adapter):
        1. precomputed ``binder_z_m`` from the Stage-4b store (no encoding),
        2. an in-process SMILES→z_m cache (fallback when no store is wired),
        3. a live DDiT+adapter forward (when fine-tuning, embeddings change).
        """
        if not self.hparams.finetune_adapter and batch.get("binder_z_m") is not None:
            return batch["binder_z_m"].to(self.device, non_blocking=True)

        smiles = batch["binder_smiles"]
        if self.hparams.finetune_adapter:
            # Adapter weights change every step; embeddings are not cacheable.
            # ``self.training`` keeps the adapter forward in the graph for the
            # train pass and detaches it during validation.
            return encode_binders(
                self.encoder, smiles, self.device, grad=self.training
            )
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

        e_pos = self.head(z_m_pos, z_p)
        z_p_dec = z_p.unsqueeze(1).expand(-1, z_m_dec.shape[1], -1)
        e_dec = self.head(z_m_dec, z_p_dec)

        l_info, info_log = self.info_loss(e_pos, e_dec)
        l_sink, sink_log = self.sink_loss(e_pos, e_dec)
        lam = lambda_sink_schedule(self.global_step, hp.lambda_sink_warmup, hp.lambda_sink)
        total = l_info + lam * l_sink

        bs = z_p.shape[0]
        log = {
            "train/loss": total.detach(),
            "train/infonce": info_log["infonce/loss"],
            "train/sinkhorn": sink_log["sinkhorn/loss"],
            "train/lambda_sink": lam,
            "train/top1": info_log["infonce/top1"],
            "train/binder_e_mean": e_pos.detach().mean(),
            "train/decoy_e_mean": e_dec.detach().mean(),
        }
        if self._cross_rng.random() < hp.cross_target_p:
            perm = torch.randperm(bs, device=self.device)
            if torch.equal(perm, torch.arange(bs, device=self.device)):
                perm = torch.roll(perm, shifts=1)
            e_wrong = self.head(z_m_pos, z_p[perm])
            l_ct, ct_log = cross_target_margin_loss(
                e_pos, e_wrong, margin=hp.cross_target_margin
            )
            total = total + hp.lambda_neg * l_ct
            log["train/cross_target"] = ct_log["cross_target/loss"]
            log["train/cross_target_viol"] = ct_log["cross_target/violation_rate"]

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

        e_pos = self.head(z_m_pos, z_p).unsqueeze(1)
        z_p_dec = z_p.unsqueeze(1).expand(-1, n, -1)
        e_dec = self.head(z_m_dec, z_p_dec)

        scores = -torch.cat([e_pos, e_dec], dim=1).cpu().numpy()
        labels = np.zeros_like(scores)
        labels[:, 0] = 1
        for s_row, l_row in zip(scores, labels):
            self._val["ef1"].append(ef_at(1.0, s_row, l_row))
            self._val["ef5"].append(ef_at(5.0, s_row, l_row))
            self._val["top1"].append(float(s_row.argmax() == 0))
            self._val["bedroc"].append(bedroc(l_row, s_row, alpha=hp.bedroc_alpha))

    def on_validation_epoch_end(self) -> None:
        out = {
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
        if hp.finetune_adapter:
            optim = torch.optim.AdamW(
                [
                    {"params": list(self.head.parameters()), "lr": hp.learning_rate},
                    {"params": list(self.encoder.adapter.parameters()), "lr": hp.adapter_lr},
                ],
                weight_decay=hp.weight_decay,
            )
        else:
            optim = torch.optim.AdamW(
                self.head.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: cosine_with_warmup(s, hp.warmup_steps, hp.num_steps)
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
