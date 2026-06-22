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
    load_bdb_index,
    pools_load_to_ram,
    stack_binder_z_m,
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
        binder_store: str | Path | None = None,
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
        num_workers: int = 4,
        load_pools_to_ram: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.train_parquet = Path(train_parquet)
        self.val_parquet = Path(val_parquet)
        self.protein_store_path = Path(protein_store)
        self.decoy_store_path = Path(decoy_store)
        self.bdb_store_path = _maybe(bdb_store)
        self.binder_store_path = _maybe(binder_store)
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
        # Pull the decoy/BDB/protein stores into RAM once instead of hitting the
        # (slow, random-access) Lustre memmap every step. Enabled via the
        # ``load_pools_to_ram`` flag or the ``LATTICE_POOLS_RAM`` env var.
        self.load_to_ram = bool(load_pools_to_ram) or pools_load_to_ram()
        self.seed = seed

        self._protein_store: EmbeddingStore | None = None
        self._binder_store: EmbeddingStore | None = None
        self._decoy_pool: DecoyZmPool | None = None
        self._bdb_pool: DecoyZmPool | None = None
        self._bdb_index: Any = None
        self._val_decoy_pool: DecoyZmPool | None = None
        self._val_bdb_pool: DecoyZmPool | None = None
        self._val_bdb_index: Any = None
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
        ram = self.load_to_ram
        self._protein_store = EmbeddingStore.open(
            self.protein_store_path, mode="r", load_to_ram=ram
        )
        self._decoy_pool = DecoyZmPool.open(self.decoy_store_path, load_to_ram=ram)
        logger.info(
            "protein store: %d entries, decoy pool: %d entries (dim=%d) [ram=%s]",
            self._protein_store.manifest.count, self._decoy_pool.count,
            self._decoy_pool.dim, ram,
        )
        # Precomputed binder z_m (Stage-4b). When present, binders are looked up
        # instead of re-encoded every epoch (valid for a frozen adapter only).
        # A missing/unbuilt store is treated as "not configured" so training
        # falls back to live encoding instead of crashing.
        if self.binder_store_path is not None:
            manifest = self.binder_store_path / EmbeddingStore.MANIFEST
            if manifest.is_file():
                self._binder_store = EmbeddingStore.open(
                    self.binder_store_path, mode="r", load_to_ram=ram
                )
                logger.info(
                    "binder store: %d entries (dim=%d)",
                    self._binder_store.manifest.count,
                    self._binder_store.manifest.embedding_dim,
                )
            else:
                logger.warning(
                    "binder_store=%s not found; encoding binders live (run "
                    "lattice_lab.ebm.precompute_binders to skip the warm-up)",
                    self.binder_store_path,
                )
        # Open the hard-negative (BDB) pools once and share them across the
        # train/val collators so RAM is pinned a single time on LUMI.
        if self.bdb_store_path is not None:
            self._bdb_pool = DecoyZmPool.open(self.bdb_store_path, load_to_ram=ram)
            self._bdb_index = load_bdb_index(self.bdb_store_path)
        if self.val_decoy_store_path is not None:
            self._val_decoy_pool = DecoyZmPool.open(
                self.val_decoy_store_path, load_to_ram=ram
            )
        else:
            self._val_decoy_pool = self._decoy_pool
        if self.val_bdb_store_path is not None:
            self._val_bdb_pool = DecoyZmPool.open(
                self.val_bdb_store_path, load_to_ram=ram
            )
            self._val_bdb_index = load_bdb_index(self.val_bdb_store_path)
        self._train_ds = BinderDataset(
            self.train_parquet, self._protein_store,
            binders_only=self.binders_only, binder_threshold_nm=self.binder_threshold_nm,
        )
        self._val_ds = BinderDataset(
            self.val_parquet, self._protein_store,
            binders_only=self.binders_only, binder_threshold_nm=self.binder_threshold_nm,
        )
        logger.info("train binders: %d, val binders: %d", len(self._train_ds), len(self._val_ds))
        self._check_adapter_consistency()

    def _check_adapter_consistency(self) -> None:
        """Fail loudly if the binder / decoy / BDB ``z_m`` stores were encoded by
        *different* adapters.

        Positives (binder store) and negatives (decoy + BDB pools) MUST live in
        the same latent space. If they don't, the energy head can separate them
        by adapter signature — a target-independent shortcut that inflates
        ``val/*`` while collapsing to random on LIT-PCBA (everything there is
        encoded by a single adapter). This silently happened once when the decoy
        pools were left stale (built with ``adapter_v1.pt``) after the binder
        store was rebuilt from the Stage-2 ``.ckpt``.
        """
        sources: list[tuple[str, str]] = []  # (store label, adapter_ckpt)

        def _record(label: str, store: EmbeddingStore | None) -> None:
            if store is None:
                return
            adapter = store.manifest.extra.get("adapter_ckpt")
            if adapter is None:
                logger.warning(
                    "z_m store '%s' has no adapter_ckpt in its manifest; cannot "
                    "verify adapter consistency", label,
                )
                return
            sources.append((label, adapter))

        _record("binder", self._binder_store)
        _record("decoy", self._decoy_pool.store if self._decoy_pool else None)
        _record("bdb", self._bdb_pool.store if self._bdb_pool else None)
        if self._val_decoy_pool is not None and self._val_decoy_pool is not self._decoy_pool:
            _record("val_decoy", self._val_decoy_pool.store)
        if self._val_bdb_pool is not None and self._val_bdb_pool is not self._bdb_pool:
            _record("val_bdb", self._val_bdb_pool.store)

        distinct = {adapter for _, adapter in sources}
        run_ids = {
            store.manifest.extra.get("adapter_run_id")
            for store in (
                self._binder_store,
                self._decoy_pool.store if self._decoy_pool else None,
                self._bdb_pool.store if self._bdb_pool else None,
            )
            if store is not None
        } - {None}
        if len(run_ids) > 1:
            raise ValueError(
                "EBM z_m stores were built for DIFFERENT adapter run ids — "
                f"rebuild Stage-4 pools for one adapter: {sorted(run_ids)}"
            )
        if len(distinct) > 1:
            detail = "\n".join(f"  - {label:10s} → {adapter}" for label, adapter in sources)
            raise ValueError(
                "EBM z_m stores were encoded by DIFFERENT adapters — positives and "
                "negatives would live in different latent spaces, inflating val/* "
                "and destroying downstream LIT-PCBA performance. Rebuild the stores "
                f"so they all use the SAME adapter checkpoint:\n{detail}"
            )
        if sources:
            logger.info("adapter consistency OK: all z_m stores use %s", next(iter(distinct)))

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
            return HardNegativeCollator.from_pools(
                self._decoy_pool,
                self._bdb_pool,
                self._bdb_index,
                n_decoys=n,
                frac_other_binder=self.frac_other_binder,
                frac_non_binder=self.frac_non_binder,
                seed=self.seed,
            )
        logger.info("train decoy mix: 100%% MOSES random")
        return EBMCollator(self._decoy_pool, n_decoys=n, seed=self.seed)

    def _val_collator(self) -> Any:
        if self.val_bdb_store_path is not None:
            return HardNegativeCollator.from_pools(
                self._val_decoy_pool,
                self._val_bdb_pool,
                self._val_bdb_index,
                n_decoys=self.n_decoys,
                frac_other_binder=self.val_frac_other_binder,
                frac_non_binder=self.val_frac_non_binder,
                seed=self.seed + 1,
            )
        if self.use_hard_neg:
            return HardNegativeCollator.from_pools(
                self._decoy_pool,
                self._bdb_pool,
                self._bdb_index,
                n_decoys=self.n_decoys,
                frac_other_binder=self.frac_other_binder,
                frac_non_binder=self.frac_non_binder,
                seed=self.seed + 1,
            )
        return EBMCollator(self._decoy_pool, n_decoys=self.n_decoys, seed=self.seed + 1)

    def _wrap(self, base_collator: Any):
        """Wrap a collator so the returned batch carries pre-stacked ``z_p``.

        When a precomputed binder store is configured, the batch also carries
        ``binder_z_m`` so the model can skip live encoding.
        """
        store = self._protein_store
        binder_store = self._binder_store

        def collate(rows):
            batch = base_collator(rows)
            z_p = stack_z_p(batch.uniprots, store, "cpu")
            out = {
                "binder_smiles": batch.binder_smiles,
                "uniprots": batch.uniprots,
                "decoy_z_m": batch.decoy_z_m,
                "z_p": z_p,
            }
            if binder_store is not None:
                z_m = stack_binder_z_m(batch.binder_smiles, binder_store)
                if z_m is not None:
                    out["binder_z_m"] = z_m
            return out

        return collate

    # -- dataloaders -----------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        sampler = self._build_cluster_sampler() if self.cluster_weighted_sampler else None
        return DataLoader(
            self._train_ds, batch_size=self.batch_size,
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._wrap(self._train_collator()),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
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
