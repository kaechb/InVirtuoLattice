"""SSL for the discrete-flow backbone: NT-Xent over fragment-shuffle views.

Each batch is a list of fragmented-SMILES strings. Per molecule we tokenize once
and produce two views by shuffling the fragment order at the token level (split
on the separator id, shuffle, rejoin). Both views go through the *same* encoder,
so the (optionally learnable) encode-time is shared across them by construction.
Only the adapter and — when enabled — the encode-time parameter train; the DDiT
backbone stays frozen.
"""

from __future__ import annotations

import logging
import random

import lightning as L
import numpy as np
import torch

from lattice_lab.backbone.discrete_flow import pad_batch
from lattice_lab.data.fragment_views import shuffle_fragment_ids
from lattice_lab.models.builders import build_discrete_flow_encoder
from lattice_lab.models.schedules import cosine_with_warmup
from lattice_lab.training.ssl_loss import NTXentLoss, top1_paired_accuracy

logger = logging.getLogger(__name__)


class DiscreteFlowSSLModule(L.LightningModule):
    def __init__(
        self,
        *,
        ckpt_path: str,
        tokenizer_path: str,
        d_adapter: int = 512,
        n_adapter_layers: int = 4,
        n_backbone_layers: int = 4,
        learnable_time: bool = True,
        encode_time: float = 0.5,
        frag_sep_id: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 30_000,
        temperature: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = build_discrete_flow_encoder(
            ckpt_path=ckpt_path,
            tokenizer_path=tokenizer_path,
            d_adapter=d_adapter,
            n_adapter_layers=n_adapter_layers,
            n_backbone_layers=n_backbone_layers,
            learnable_time=learnable_time,
            encode_time=encode_time,
        )
        self.loss_fn = NTXentLoss(temperature=temperature)
        self._rng = random.Random(seed)
        self._val: dict[str, list[float]] = {"loss": [], "acc": []}
        logger.info(
            "discrete-flow SSL: learnable_time=%s init_time=%.3f frag_sep_id=%d",
            learnable_time, self.encoder.encode_time_value, frag_sep_id,
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

    # -- lifecycle ---------------------------------------------------------- #
    def training_step(self, batch, batch_idx):
        (ids_a, mask_a), (ids_b, mask_b) = self._two_views(batch)
        _, z_a = self.encoder.encode_token_ids(ids_a, mask_a, return_projection=True)
        _, z_b = self.encoder.encode_token_ids(ids_b, mask_b, return_projection=True)
        loss = self.loss_fn(z_a, z_b)
        bs = len(batch)
        self.log_dict(
            {
                "train/loss": loss.detach(),
                "train/acc@1": top1_paired_accuracy(z_a.detach(), z_b.detach()),
                "train/encode_time": self.encoder.encode_time_value,
            },
            on_step=True, on_epoch=False, batch_size=bs,
        )
        self.log("train/loss_bar", loss.detach(), prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        (ids_a, mask_a), (ids_b, mask_b) = self._two_views(batch)
        _, z_a = self.encoder.encode_token_ids(ids_a, mask_a, return_projection=True)
        _, z_b = self.encoder.encode_token_ids(ids_b, mask_b, return_projection=True)
        self._val["loss"].append(self.loss_fn(z_a, z_b).item())
        self._val["acc"].append(top1_paired_accuracy(z_a, z_b))

    def on_validation_epoch_end(self) -> None:
        out = {
            "val/loss": float(np.mean(self._val["loss"])) if self._val["loss"] else float("nan"),
            "val/acc@1": float(np.mean(self._val["acc"])) if self._val["acc"] else 0.0,
            "val/encode_time": self.encoder.encode_time_value,
        }
        self.log_dict(out, prog_bar=True, sync_dist=True)
        self._val["loss"].clear()
        self._val["acc"].clear()

    def configure_optimizers(self):
        hp = self.hparams
        # Trainable encoder params = adapter (+ encode-time parameter if learnable);
        # the DDiT backbone is frozen.
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=hp.learning_rate, weight_decay=hp.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: cosine_with_warmup(s, hp.warmup_steps, hp.total_steps)
        )
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
