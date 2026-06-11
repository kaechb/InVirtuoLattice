"""DataModule for Stage-5 EBM-head training.

Wraps :class:`lattice_lab.ebm.dataset.BinderDataset` + the existing collators and,
crucially, *pre-stacks the protein embedding* ``z_p`` into every batch. That
keeps the :class:`~lattice_lab.models.ebm.EBMLitModule` pure — it never needs a
handle on the protein store. Batches are plain dicts::

    {"binder_smiles": list[str], "uniprots": list[str],
     "decoy_z_m": Tensor[B, N, d_m], "z_p": Tensor[B, d_p]}

For training with online hard-negative mining the train collator oversamples to
``n_decoys * hard_mining_mult`` candidate decoys; the module mines the hardest
``n_decoys``. Validation always draws plain ``n_decoys`` (optionally from a fixed
mix via ``val_bdb_store``) so ``val/*`` is comparable across runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightning as L
from torch.utils.data import DataLoader, WeightedRandomSampler

from lattice_lab.ebm.dataset import (
    BinderDataset,
    DecoyZmPool,
    EBMCollator,
    HardNegativeCollator,
    stack_z_p,
)
from lattice_lab.protein.store import EmbeddingStore

logger = logging.getLogger(__name__)


def _maybe(path: str | Path | None) -> Path | None:
    return Path(path) if path is not None else None


class EBMDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        train_parquet: str | Path,
        val_parquet: str | Path,
        protein_store: str | Path,
        decoy_store: str | Path,
        bdb_store: str | Path | None = None,
        batch_size: int = 64,
        n_decoys: int = 600,
        hard_mining_mult: int = 1,
        binders_only: bool = True,
        binder_threshold_nm: float | None = None,
        frac_other_binder: float = 0.0,
        frac_non_binder: float = 0.0,
        val_decoy_store: str | Path | None = None,
        val_bdb_store: str | Path | None = None,
        val_frac_other_binder: float = 0.4,
        val_frac_non_binder: float = 0.15,
        cluster_weighted_sampler: bool = False,
        cluster_min_identity: float = 0.5,
        cluster_cache_dir: str | Path = "artifacts/energy/checkpoints",
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.train_parquet = Path(train_parquet)
        self.val_parquet = Path(val_parquet)
        self.protein_store_path = Path(protein_store)
        self.decoy_store_path = Path(decoy_store)
        self.bdb_store_path = _maybe(bdb_store)
        self.batch_size = batch_size
        self.n_decoys = n_decoys
        self.hard_mining_mult = max(1, hard_mining_mult)
        self.binders_only = binders_only
        self.binder_threshold_nm = binder_threshold_nm
        self.frac_other_binder = frac_other_binder
        self.frac_non_binder = frac_non_binder
        self.val_decoy_store_path = _maybe(val_decoy_store)
        self.val_bdb_store_path = _maybe(val_bdb_store)
        self.val_frac_other_binder = val_frac_other_binder
        self.val_frac_non_binder = val_frac_non_binder
        self.cluster_weighted_sampler = cluster_weighted_sampler
        self.cluster_min_identity = cluster_min_identity
        self.cluster_cache_dir = Path(cluster_cache_dir)
        self.num_workers = num_workers
        self.seed = seed

        self._protein_store: EmbeddingStore | None = None
        self._decoy_pool: DecoyZmPool | None = None
        self._train_ds: BinderDataset | None = None
        self._val_ds: BinderDataset | None = None

    # -- properties used by the model / entrypoint -----------------------
    @property
    def decoy_dim(self) -> int:
        assert self._decoy_pool is not None, "call setup() first"
        return self._decoy_pool.dim

    @property
    def use_hard_neg(self) -> bool:
        return self.bdb_store_path is not None and (
            self.frac_other_binder > 0 or self.frac_non_binder > 0
        )

    # -- lifecycle -------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        if self._protein_store is not None:
            return
        self._protein_store = EmbeddingStore.open(self.protein_store_path, mode="r")
        self._decoy_pool = DecoyZmPool.open(self.decoy_store_path)
        logger.info(
            "protein store: %d entries, decoy pool: %d entries (dim=%d)",
            self._protein_store.manifest.count, self._decoy_pool.count, self._decoy_pool.dim,
        )
        self._train_ds = BinderDataset(
            self.train_parquet, self._protein_store,
            binders_only=self.binders_only, binder_threshold_nm=self.binder_threshold_nm,
        )
        self._val_ds = BinderDataset(
            self.val_parquet, self._protein_store,
            binders_only=self.binders_only, binder_threshold_nm=self.binder_threshold_nm,
        )
        logger.info("train binders: %d, val binders: %d", len(self._train_ds), len(self._val_ds))

    # -- collators -------------------------------------------------------
    def _train_collator(self) -> Any:
        n = self.n_decoys * self.hard_mining_mult
        if self.use_hard_neg:
            logger.info(
                "train decoy mix: %d random + %d other-binders + %d non-binders / %d",
                int(round(n * (1 - self.frac_other_binder - self.frac_non_binder))),
                int(round(n * self.frac_other_binder)),
                int(round(n * self.frac_non_binder)), n,
            )
            return HardNegativeCollator.from_paths(
                moses_store_path=self.decoy_store_path,
                bdb_store_path=self.bdb_store_path,
                n_decoys=n,
                frac_other_binder=self.frac_other_binder,
                frac_non_binder=self.frac_non_binder,
                seed=self.seed,
            )
        logger.info("train decoy mix: 100%% MOSES random")
        return EBMCollator(self._decoy_pool, n_decoys=n, seed=self.seed)

    def _val_collator(self) -> Any:
        if self.val_bdb_store_path is not None:
            moses = self.val_decoy_store_path or self.decoy_store_path
            return HardNegativeCollator.from_paths(
                moses_store_path=moses,
                bdb_store_path=self.val_bdb_store_path,
                n_decoys=self.n_decoys,
                frac_other_binder=self.val_frac_other_binder,
                frac_non_binder=self.val_frac_non_binder,
                seed=self.seed + 1,
            )
        if self.use_hard_neg:
            return HardNegativeCollator.from_paths(
                moses_store_path=self.decoy_store_path,
                bdb_store_path=self.bdb_store_path,
                n_decoys=self.n_decoys,
                frac_other_binder=self.frac_other_binder,
                frac_non_binder=self.frac_non_binder,
                seed=self.seed + 1,
            )
        return EBMCollator(self._decoy_pool, n_decoys=self.n_decoys, seed=self.seed + 1)

    def _wrap(self, base_collator: Any):
        """Wrap a collator so the returned batch carries pre-stacked ``z_p``."""
        store = self._protein_store

        def collate(rows):
            batch = base_collator(rows)
            z_p = stack_z_p(batch.uniprots, store, "cpu")
            return {
                "binder_smiles": batch.binder_smiles,
                "uniprots": batch.uniprots,
                "decoy_z_m": batch.decoy_z_m,
                "z_p": z_p,
            }

        return collate

    # -- dataloaders -----------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        sampler = self._build_cluster_sampler() if self.cluster_weighted_sampler else None
        return DataLoader(
            self._train_ds, batch_size=self.batch_size,
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.num_workers, drop_last=True,
            collate_fn=self._wrap(self._train_collator()),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            collate_fn=self._wrap(self._val_collator()),
        )

    def _build_cluster_sampler(self) -> WeightedRandomSampler:
        from lattice_lab.data.cluster_sampler import build_cluster_weighted_sampler

        return build_cluster_weighted_sampler(
            train_parquet=self.train_parquet,
            row_uniprots=self._train_ds._uniprots,
            min_identity=self.cluster_min_identity,
            cache_dir=self.cluster_cache_dir,
            seed=self.seed,
        )
