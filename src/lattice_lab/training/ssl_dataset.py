"""Dataset that yields paired molecule views from parquet shards.

Each parquet shard contains rows of the form (smiles, inchikey, view_idx, fragment_view).
For SSL we want two distinct views of the same molecule per sample. The dataset
groups rows by ``inchikey`` and serves a random pair per molecule.

A deterministic train/val/test split is applied at the **molecule** level using a
SHA-1 hash of the InChIKey, so two molecules never straddle splits regardless of
shard ordering. Re-running with the same ``split_seed`` is reproducible.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


def _bucket_for_inchikey(inchikey: str, split_seed: int) -> float:
    """Map an InChIKey to a stable float in [0, 1) for split assignment.

    Hashes ``inchikey + str(split_seed)`` with SHA-1 (deterministic across Python
    versions, unlike ``hash()``) and uses the first 8 bytes as a uint64.
    """
    h = hashlib.sha1(f"{inchikey}|{split_seed}".encode()).digest()[:8]
    return int.from_bytes(h, "big") / (1 << 64)


def assign_split(inchikey: str, *, val_ratio: float, test_ratio: float,
                 split_seed: int = 0) -> str:
    """Return ``"train"``, ``"val"``, or ``"test"`` for one molecule.

    Determined entirely by the InChIKey + ``split_seed`` so the assignment is
    consistent across shards, processes, and re-runs.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio > 1:
        raise ValueError(f"bad split ratios: val={val_ratio}, test={test_ratio}")
    u = _bucket_for_inchikey(inchikey, split_seed)
    if u < val_ratio:
        return "val"
    if u < val_ratio + test_ratio:
        return "test"
    return "train"


class PairedViewDataset(Dataset):
    """Returns (view_a, view_b) string pairs for the same molecule.

    Only molecules with ≥2 views are usable; molecules with a single view are
    dropped at index time so ``__getitem__`` never has to retry.

    Args:
        shard_paths: parquet shards produced by Stage 1.
        seed: per-instance RNG seed (controls which pair is drawn for a molecule).
        split: ``"train"`` / ``"val"`` / ``"test"`` / ``"all"`` — filter molecules
            by the deterministic ``assign_split`` bucket.
        val_ratio / test_ratio: split fractions, applied via ``assign_split``.
        split_seed: seeds the hash so different seeds produce different splits.

    The default ratios match the README spec (``99/0.5/0.5``).
    """

    def __init__(
        self,
        shard_paths: Iterable[Path | str],
        *,
        seed: int = 0,
        split: str = "all",
        val_ratio: float = 0.005,
        test_ratio: float = 0.005,
        split_seed: int = 0,
        return_smiles: bool = False,
        same_view_pairs: bool = False,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be one of train/val/test/all, got {split!r}")

        self._return_smiles = return_smiles
        self._same_view_pairs = same_view_pairs
        from lattice_lab.preprocessing.molecules import (
            fragment_view_column,
            fragment_view_column_for_parquet,
        )

        view_col = fragment_view_column_for_parquet(shard_paths[0])
        frames = []
        for p in shard_paths:
            cols = ["inchikey", view_col] + (["smiles"] if return_smiles else [])
            frames.append(pd.read_parquet(p, columns=cols))
        if not frames:
            raise ValueError("no shards provided")
        df = pd.concat(frames, ignore_index=True)
        view_col = fragment_view_column(df)

        if split != "all":
            keep = df["inchikey"].map(
                lambda k: assign_split(
                    k, val_ratio=val_ratio, test_ratio=test_ratio, split_seed=split_seed
                )
                == split
            )
            df = df[keep]

        grouped = df.groupby("inchikey")[view_col].apply(list)
        keep_pairs = [(k, v) for k, v in grouped.items() if len(v) >= 2]
        self._inchikeys: list[str] = [k for k, _ in keep_pairs]
        self._views: list[list[str]] = [v for _, v in keep_pairs]
        # One representative SMILES per molecule, aligned with ``_inchikeys``;
        # used by the optional fingerprint-distillation objective.
        self._smiles: list[str] = []
        if return_smiles:
            smi_first = df.groupby("inchikey")["smiles"].first()
            self._smiles = [str(smi_first[k]) for k in self._inchikeys]
        self._rng = random.Random(seed)
        if not self._views:
            raise ValueError(
                f"split={split!r} has no molecules with ≥ 2 views; "
                "check n_views ≥ 2 in preprocessing or relax ratios"
            )

    def __len__(self) -> int:
        return len(self._views)

    def __getitem__(self, idx: int) -> tuple[str, str] | tuple[str, str, str]:
        views = self._views[idx]
        if self._same_view_pairs:
            a = b = self._rng.choice(views)
        else:
            a, b = self._rng.sample(views, 2)
        if self._return_smiles:
            return a, b, self._smiles[idx]
        return a, b

    @property
    def inchikeys(self) -> list[str]:
        """Stable list of InChIKeys, aligned with ``__getitem__`` indices."""
        return list(self._inchikeys)


def collate_pairs(batch: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Trivial collator: keep strings; tokenization happens inside the encoder."""
    a, b = zip(*batch)
    return list(a), list(b)


def collate_pairs_with_smiles(
    batch: list[tuple[str, str, str]],
) -> tuple[list[str], list[str], list[str]]:
    """Collator for ``return_smiles=True``: (views_a, views_b, smiles)."""
    a, b, smi = zip(*batch)
    return list(a), list(b), list(smi)
