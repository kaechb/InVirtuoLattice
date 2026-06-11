"""DataModule for Stage-2 adapter SSL (paired FragMol views).

Thin wrapper over :class:`lattice.training.ssl_dataset.PairedViewDataset`. When
``fp_weight > 0`` the train batches also carry the SMILES (for the Tanimoto
similarity-distillation target), so the collate function switches accordingly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from lattice_lab.training.ssl_dataset import (
    PairedViewDataset,
    collate_pairs,
    collate_pairs_with_smiles,
)

logger = logging.getLogger(__name__)


class AdapterDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        shard_dir: str | Path,
        batch_size: int = 64,
        val_ratio: float = 0.005,
        test_ratio: float = 0.005,
        split_seed: int = 0,
        seed: int = 0,
        use_fp: bool = False,
        same_view_pairs: bool = False,
        run_validation: bool = True,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.seed = seed
        self.use_fp = use_fp
        self.same_view_pairs = same_view_pairs
        self.run_validation = run_validation
        self.num_workers = num_workers

        self._shards: list[Path] = []
        self._train_ds: PairedViewDataset | None = None
        self._val_ds: PairedViewDataset | None = None

    @property
    def shards(self) -> list[Path]:
        return self._shards

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None:
            return
        self._shards = sorted(self.shard_dir.glob("shard_*.parquet"))
        if not self._shards:
            raise FileNotFoundError(f"no parquet shards in {self.shard_dir}")
        logger.info("loading %d shard(s) from %s", len(self._shards), self.shard_dir)
        self._train_ds = PairedViewDataset(
            self._shards, seed=self.seed, split="train",
            val_ratio=self.val_ratio, test_ratio=self.test_ratio,
            split_seed=self.split_seed, return_smiles=self.use_fp,
            same_view_pairs=self.same_view_pairs,
        )
        logger.info("train molecules: %d", len(self._train_ds))
        if self.run_validation:
            self._val_ds = PairedViewDataset(
                self._shards, seed=self.seed, split="val",
                val_ratio=self.val_ratio, test_ratio=self.test_ratio,
                split_seed=self.split_seed, same_view_pairs=self.same_view_pairs,
            )
            logger.info("val molecules: %d", len(self._val_ds))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            collate_fn=collate_pairs_with_smiles if self.use_fp else collate_pairs,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val_ds is None:
            return None
        return DataLoader(
            self._val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=True, collate_fn=collate_pairs,
        )
