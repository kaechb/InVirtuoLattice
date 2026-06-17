"""Dataset adapters for Stage-4 EBM training.

The training loop consumes:
- per-batch K binder rows, one per target. Each row has ``smiles`` + ``uniprot``;
  the loop encodes the binder on-the-fly through the frozen
  DDiT+adapter and looks up ``z_p`` in :class:`lattice_lab.protein.store.EmbeddingStore`.
- per-batch K × N decoy ``z_m`` vectors sampled from a precomputed pool.

This module provides:
- :class:`BinderDataset` — wraps a Stage-1 BindingDB parquet, filters by
  presence of ``z_p`` in the store, returns ``(smiles, uniprot)`` pairs.
- :class:`DecoyZmPool` — memory-mapped store of precomputed adapter latents
  for decoy molecules (typically MOSES, used in place of ZINC; see README).
- :func:`build_collator` — pulls N decoys per binder using a shared RNG and
  returns the tensors the training loop forwards into the energy head.

The decoy pool is built by :mod:`lattice_lab.ebm.precompute_decoys`; the format is
purposely the same
``mean.dat`` + ``pids.tsv`` + ``manifest.json`` layout as the protein store so
we can reuse :class:`EmbeddingStore`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lattice_lab.protein.store import EmbeddingStore

logger = logging.getLogger(__name__)


def pools_load_to_ram() -> bool:
    """True when decoy pools should be copied into RAM (``LATTICE_POOLS_RAM``)."""
    return os.environ.get("LATTICE_POOLS_RAM", "").lower() in ("1", "true", "yes")


def load_bdb_index(bdb_store_path: Path | str) -> pd.DataFrame:
    """Read ``index.parquet`` beside a BDB ``z_m`` store."""
    index_path = Path(bdb_store_path) / "index.parquet"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"missing {index_path}; rebuild with lattice_lab.ebm.precompute_bdb_zm"
        )
    return pd.read_parquet(index_path)


def same_decoy_store(pool: DecoyZmPool, path: Path | str) -> bool:
    """True when ``pool`` was opened from ``path``."""
    return Path(pool.store.path).resolve() == Path(path).resolve()


# --------------------------------------------------------------------------
# Decoy pool
# --------------------------------------------------------------------------


class DecoyZmPool:
    """Random-access pool of precomputed adapter latents.

    Backed by the same :class:`EmbeddingStore` layout as the protein store, so
    we can grow it incrementally and memory-map at read time.
    """

    def __init__(self, store: EmbeddingStore) -> None:
        if store.manifest.count == 0:
            raise ValueError(f"decoy pool at {store.path} is empty")
        self.store = store
        self.count = store.manifest.count
        self.dim = store.manifest.embedding_dim

    @classmethod
    def open(cls, path: Path | str, *, load_to_ram: bool = False) -> "DecoyZmPool":
        return cls(
            EmbeddingStore.open(path, mode="r", load_to_ram=load_to_ram)
        )

    def sample(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return ``[n, dim]`` decoy latents as a fresh torch tensor (float32).

        Sampling is done with replacement; for the cardinalities we use (pool
        ≈ 10⁶, draws ≤ 500) collisions are vanishingly rare and the bias is
        negligible. Replacement sampling is simpler and reproducible.
        """
        idx = torch.randint(0, self.count, (n,), generator=generator)
        return self._gather(idx)

    def _gather(self, idx: torch.Tensor) -> torch.Tensor:
        # memmap → np.array copy → torch tensor. The copy is needed because the
        # memmap dtype may be float16; we want float32 on the training device.
        rows = np.asarray(self.store.mean_array[idx.numpy()], dtype=np.float32)
        return torch.from_numpy(rows)


# --------------------------------------------------------------------------
# Binder dataset
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BinderRow:
    """One training example: a binder molecule + the UniProt of its target."""

    smiles: str
    uniprot: str


class BinderDataset(Dataset):
    """Yields binder ``(smiles, uniprot)`` rows from a BindingDB parquet.

    Only rows whose UniProt has an embedding in the protein store are kept;
    anything else can't be trained on without a `z_p`. ``is_binder_10uM=True``
    is the default filter, but the caller can disable it (for ablations that
    want to include weak binders as "hard negatives").

    ``binder_threshold_nm`` overrides the binder label: when set, a row is a
    binder iff ``min(Ki, Kd, IC50) <= binder_threshold_nm`` (EC50 excluded, to
    match :mod:`lattice_lab.preprocessing.bindingdb`), instead of using the
    precomputed ``is_binder_10uM`` column. This drives affinity-threshold
    ablations (e.g. 1 µM actives) without re-running Stage-1 curation. The
    non-binder hard-neg pool keeps its own (10 µM) definition, so the
    ``(threshold, 10 µM]`` grey zone is excluded from both positives and
    negatives — the gap design discussed in the README ablation notes.
    """

    def __init__(
        self,
        parquet_path: Path | str,
        protein_store: EmbeddingStore,
        *,
        binders_only: bool = True,
        binder_threshold_nm: float | None = None,
    ) -> None:
        if binder_threshold_nm is None:
            df = pd.read_parquet(
                parquet_path,
                columns=["smiles", "uniprot", "is_binder_10uM"],
            )
            binder_mask = df["is_binder_10uM"].to_numpy(dtype=bool)
        else:
            if binder_threshold_nm <= 0:
                raise ValueError(
                    f"binder_threshold_nm must be positive, got {binder_threshold_nm}"
                )
            df = pd.read_parquet(
                parquet_path,
                columns=["smiles", "uniprot", "ki_nm", "kd_nm", "ic50_nm"],
            )
            # min over present Ki/Kd/IC50 (NaN where all three missing); EC50 is
            # functional, not binding, so it never makes a row a binder.
            best_nm = df[["ki_nm", "kd_nm", "ic50_nm"]].min(axis=1, skipna=True)
            binder_mask = (best_nm <= binder_threshold_nm).to_numpy(dtype=bool)
            logger.info(
                "binder label recomputed at %.1f nM: %d/%d rows are binders",
                binder_threshold_nm, int(binder_mask.sum()), len(df),
            )
        if binders_only:
            df = df[binder_mask]
        present = df["uniprot"].isin(protein_store.pid_to_row)
        n_dropped = int((~present).sum())
        df = df[present].reset_index(drop=True)
        if n_dropped:
            logger.warning(
                "dropped %d/%d rows missing from protein store (uniprots: %d unique)",
                n_dropped,
                n_dropped + len(df),
                df["uniprot"].nunique(),
            )
        if len(df) == 0:
            raise ValueError(
                f"no usable rows in {parquet_path} (binders_only={binders_only}); "
                "check that the protein store covers the BindingDB uniprots."
            )
        self._smiles: list[str] = df["smiles"].tolist()
        self._uniprots: list[str] = df["uniprot"].tolist()
        self._store = protein_store

    def __len__(self) -> int:
        return len(self._smiles)

    def __getitem__(self, idx: int) -> BinderRow:
        return BinderRow(smiles=self._smiles[idx], uniprot=self._uniprots[idx])

    @property
    def unique_uniprots(self) -> list[str]:
        return sorted(set(self._uniprots))


# --------------------------------------------------------------------------
# Collator
# --------------------------------------------------------------------------


@dataclass
class EBMBatch:
    """One training batch for the EBM head.

    - ``binder_smiles``: ``len == B`` SMILES strings (encoded on the fly).
    - ``uniprots``:      ``len == B`` UniProt strings (key into the protein store).
    - ``decoy_z_m``:     ``[B, N, d_m]`` precomputed decoy latents.

    The training loop is responsible for:
    1. Encoding ``binder_smiles`` → ``z_m+ [B, d_m]``.
    2. Looking up ``z_p`` for each UniProt → ``[B, d_p]``.
    3. Forwarding ``z_m+`` and ``decoy_z_m`` through the energy head.
    """

    binder_smiles: list[str]
    uniprots: list[str]
    decoy_z_m: torch.Tensor


class EBMCollator:
    """Pulls a fresh draw of ``N`` decoy z_m vectors per binder."""

    def __init__(
        self, decoy_pool: DecoyZmPool, *, n_decoys: int, seed: int = 0
    ) -> None:
        if n_decoys <= 0:
            raise ValueError(f"n_decoys must be positive, got {n_decoys}")
        self.pool = decoy_pool
        self.n_decoys = n_decoys
        self._base_seed = seed
        self._gen = torch.Generator().manual_seed(seed)
        self._worker_seeded = False

    def _ensure_worker_seed(self) -> None:
        """Decorrelate RNG across DataLoader workers (fork copies the seed)."""
        if self._worker_seeded:
            return
        info = torch.utils.data.get_worker_info()
        if info is not None:
            self._gen.manual_seed(self._base_seed + info.id + 1)
        self._worker_seeded = True

    def __call__(self, rows: list[BinderRow]) -> EBMBatch:
        self._ensure_worker_seed()
        b = len(rows)
        decoys = self.pool.sample(b * self.n_decoys, generator=self._gen)
        decoys = decoys.view(b, self.n_decoys, self.pool.dim)
        return EBMBatch(
            binder_smiles=[r.smiles for r in rows],
            uniprots=[r.uniprot for r in rows],
            decoy_z_m=decoys,
        )


def stack_z_p(
    uniprots: list[str], store: EmbeddingStore, device: torch.device | str
) -> torch.Tensor:
    """Look up ``z_p`` for each uniprot and return a ``[B, d_p]`` tensor."""
    rows = np.stack(
        [np.asarray(store.get_mean(u), dtype=np.float32) for u in uniprots], axis=0
    )
    return torch.from_numpy(rows).to(device)


def stack_binder_z_m(
    smiles: list[str], store: EmbeddingStore
) -> torch.Tensor | None:
    """Look up precomputed binder ``z_m`` for each SMILES → ``[B, d_m]`` (CPU).

    Returns ``None`` if any SMILES is missing from the store, so the caller can
    fall back to live encoding for that batch instead of crashing.
    """
    try:
        rows = np.stack(
            [np.asarray(store.get_mean(s), dtype=np.float32) for s in smiles], axis=0
        )
    except KeyError:
        return None
    return torch.from_numpy(rows)


# --------------------------------------------------------------------------
# Hard-negative collator (cross-target binders + annotated non-binders + MOSES)
# --------------------------------------------------------------------------


class HardNegativeCollator:
    """Draws ``n_decoys`` decoys per binder from three experimentally grounded pools.

    Composition is fully configurable per call site:

    - ``frac_other_binder``: BDB ligands flagged as binders to *any* target.
      Drug-like by construction; pushing them below the true binder for the
      current target *requires* the head to use ``z_p``.
    - ``frac_non_binder``: BDB ligands annotated as non-binders at the 10 µM
      cutoff. Experimentally tested and shown not to bind.
    - remainder (``1 - frac_other_binder - frac_non_binder``): random draws
      from the existing MOSES pool, kept for chemical-space coverage.

    Notes on collisions:
    - We do **not** exclude "other-target binders that also bind the current
      target". At 1.2 M binders / 7 K targets that's a < 0.1 % collision
      rate, well below the ~1 % false-negative rate of MOSES decoys. Adding
      an explicit per-target filter would cost more in CPU than it saves in
      signal cleanliness.
    - Sampling is with replacement; for the cardinalities we use it's a
      vanishing concern and keeps the sampler O(B × N).
    """

    def __init__(
        self,
        moses_pool: DecoyZmPool,
        bdb_pool: DecoyZmPool,
        bdb_index_df: pd.DataFrame,
        *,
        n_decoys: int,
        frac_other_binder: float,
        frac_non_binder: float,
        seed: int = 0,
    ) -> None:
        if n_decoys <= 0:
            raise ValueError(f"n_decoys must be positive, got {n_decoys}")
        if frac_other_binder < 0 or frac_non_binder < 0:
            raise ValueError("fractions must be non-negative")
        if frac_other_binder + frac_non_binder > 1 + 1e-9:
            raise ValueError(
                f"fractions sum > 1: other_binder={frac_other_binder} "
                f"+ non_binder={frac_non_binder}"
            )
        required = {"row_idx", "is_binder_any_target"}
        missing = required - set(bdb_index_df.columns)
        if missing:
            raise ValueError(f"bdb_index_df missing columns: {missing}")
        if moses_pool.dim != bdb_pool.dim:
            raise ValueError(
                f"pool dim mismatch: moses={moses_pool.dim} bdb={bdb_pool.dim}"
            )

        self.moses_pool = moses_pool
        self.bdb_pool = bdb_pool
        self.dim = moses_pool.dim
        self.n_decoys = n_decoys

        # Split the per-binder budget into three integer counts that sum to
        # exactly n_decoys (round, then absorb any drift into the random
        # bucket so the batch size is exact).
        self.n_other = int(round(n_decoys * frac_other_binder))
        self.n_non = int(round(n_decoys * frac_non_binder))
        self.n_random = n_decoys - self.n_other - self.n_non
        if self.n_random < 0:
            # Rounding can push n_random negative when fractions sum to ~1;
            # rebalance by shrinking the larger of the experimental buckets.
            if self.n_other >= self.n_non:
                self.n_other += self.n_random
            else:
                self.n_non += self.n_random
            self.n_random = 0

        idx = bdb_index_df.reset_index(drop=True)
        self._binder_rows: np.ndarray = (
            idx.loc[idx["is_binder_any_target"], "row_idx"]
            .to_numpy(dtype=np.int64)
        )
        self._nonbinder_rows: np.ndarray = (
            idx.loc[~idx["is_binder_any_target"], "row_idx"]
            .to_numpy(dtype=np.int64)
        )
        if self.n_other > 0 and self._binder_rows.size == 0:
            raise ValueError("frac_other_binder > 0 but no binders in bdb_index_df")
        if self.n_non > 0 and self._nonbinder_rows.size == 0:
            raise ValueError("frac_non_binder > 0 but no non-binders in bdb_index_df")

        self._base_seed = seed
        self._np_rng = np.random.default_rng(seed)
        self._gen = torch.Generator().manual_seed(seed + 1)
        self._worker_seeded = False

    def _ensure_worker_seed(self) -> None:
        """Decorrelate RNG across DataLoader workers (fork copies the seed)."""
        if self._worker_seeded:
            return
        info = torch.utils.data.get_worker_info()
        if info is not None:
            self._np_rng = np.random.default_rng(self._base_seed + info.id)
            self._gen.manual_seed(self._base_seed + info.id + 1)
        self._worker_seeded = True

    @classmethod
    def from_pools(
        cls,
        moses_pool: DecoyZmPool,
        bdb_pool: DecoyZmPool,
        bdb_index_df: pd.DataFrame,
        *,
        n_decoys: int,
        frac_other_binder: float,
        frac_non_binder: float,
        seed: int = 0,
    ) -> "HardNegativeCollator":
        """Build from already-open pools (avoids duplicate RAM pins on LUMI)."""
        return cls(
            moses_pool=moses_pool,
            bdb_pool=bdb_pool,
            bdb_index_df=bdb_index_df,
            n_decoys=n_decoys,
            frac_other_binder=frac_other_binder,
            frac_non_binder=frac_non_binder,
            seed=seed,
        )

    @classmethod
    def from_paths(
        cls,
        moses_store_path: Path | str,
        bdb_store_path: Path | str,
        *,
        n_decoys: int,
        frac_other_binder: float,
        frac_non_binder: float,
        seed: int = 0,
        load_to_ram: bool = False,
    ) -> "HardNegativeCollator":
        moses = DecoyZmPool.open(moses_store_path, load_to_ram=load_to_ram)
        bdb = DecoyZmPool.open(bdb_store_path, load_to_ram=load_to_ram)
        return cls.from_pools(
            moses,
            bdb,
            load_bdb_index(bdb_store_path),
            n_decoys=n_decoys,
            frac_other_binder=frac_other_binder,
            frac_non_binder=frac_non_binder,
            seed=seed,
        )

    def __call__(self, rows: list[BinderRow]) -> EBMBatch:
        self._ensure_worker_seed()
        b = len(rows)
        decoys = torch.empty(b, self.n_decoys, self.dim, dtype=torch.float32)
        offset = 0

        if self.n_random > 0:
            rand_idx = torch.randint(
                0, self.moses_pool.count, (b * self.n_random,), generator=self._gen
            )
            block = self.moses_pool._gather(rand_idx).view(b, self.n_random, self.dim)
            decoys[:, offset : offset + self.n_random] = block
            offset += self.n_random

        if self.n_other > 0:
            picks = self._np_rng.integers(
                0, self._binder_rows.size, size=b * self.n_other
            )
            pool_rows = self._binder_rows[picks]
            arr = np.asarray(self.bdb_pool.store.mean_array[pool_rows], dtype=np.float32)
            block = torch.from_numpy(arr).view(b, self.n_other, self.dim)
            decoys[:, offset : offset + self.n_other] = block
            offset += self.n_other

        if self.n_non > 0:
            picks = self._np_rng.integers(
                0, self._nonbinder_rows.size, size=b * self.n_non
            )
            pool_rows = self._nonbinder_rows[picks]
            arr = np.asarray(self.bdb_pool.store.mean_array[pool_rows], dtype=np.float32)
            block = torch.from_numpy(arr).view(b, self.n_non, self.dim)
            decoys[:, offset : offset + self.n_non] = block
            offset += self.n_non

        return EBMBatch(
            binder_smiles=[r.smiles for r in rows],
            uniprots=[r.uniprot for r in rows],
            decoy_z_m=decoys,
        )
