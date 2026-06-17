"""SSL data for the discrete-flow backbone: fragment-shuffle contrastive pairs.

Each MOSES molecule has a space-separated fragmented-SMILES string (the
``fragment_view`` column from Stage-1). The contrastive augmentation is a
**fragment shuffle**: tokenize the view, split the token ids on the separator
id (the space token, id 4 in the discrete-flow tokenizer), shuffle the fragment
order, and rejoin. Two independent shuffles give the two views of a molecule.

The shuffle is a pure token-level op (:func:`shuffle_fragment_ids`) so it's unit
tested without the model. The dataset just yields the view strings; tokenization
+ shuffling happen in the LightningModule (which owns the tokenizer via the
encoder), keeping the tokenizer in one place.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def fragment_split_mask(
    inchikeys: pd.Series,
    *,
    split: str,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> pd.Series:
    """Boolean mask selecting molecules for ``train`` / ``val`` / ``test``."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test, got {split!r}")
    buckets = inchikeys.astype(str).map(
        lambda k: (hash((split_seed, k)) % 10_000) / 10_000.0
    )
    if split == "train":
        return buckets >= (val_ratio + test_ratio)
    if split == "val":
        return (buckets >= test_ratio) & (buckets < val_ratio + test_ratio)
    return buckets < test_ratio


def load_fragment_split_df(
    shards: list[Path],
    *,
    split: str,
    val_ratio: float = 0.005,
    test_ratio: float = 0.005,
    split_seed: int = 0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load deduplicated ``view_idx==0`` rows for a MOSES split."""
    from lattice_lab.preprocessing.molecules import (
        fragment_view_column,
        fragment_view_column_for_parquet,
    )

    view_col = fragment_view_column_for_parquet(shards[0])
    use_cols = columns or ["smiles", "inchikey", "view_idx", view_col]
    if view_col not in use_cols:
        use_cols = [*use_cols, view_col]
    frames = [pd.read_parquet(s, columns=use_cols) for s in shards]
    df = pd.concat(frames, ignore_index=True)
    view_col = fragment_view_column(df)
    df = df[df["view_idx"] == 0].drop_duplicates("inchikey").reset_index(drop=True)
    mask = fragment_split_mask(
        df["inchikey"], split=split, val_ratio=val_ratio,
        test_ratio=test_ratio, split_seed=split_seed,
    )
    return df.loc[mask].reset_index(drop=True)


def split_fragment_ids(ids: list[int], sep_id: int) -> list[list[int]]:
    """Split token ids on ``sep_id`` into fragment lists (empty frags allowed)."""
    frags: list[list[int]] = [[]]
    for t in ids:
        if t == sep_id:
            frags.append([])
        else:
            frags[-1].append(t)
    return frags


def join_fragment_ids(frags: list[list[int]], sep_id: int) -> list[int]:
    """Rejoin fragment token lists with ``sep_id`` between fragments."""
    out: list[int] = []
    for i, frag in enumerate(frags):
        if i:
            out.append(sep_id)
        out.extend(frag)
    return out


def shuffle_fragment_ids(ids: list[int], sep_id: int, rng: random.Random) -> list[int]:
    """Split ``ids`` on ``sep_id``, shuffle the fragments, rejoin with ``sep_id``.

    A single-fragment sequence (no ``sep_id``) is returned unchanged. Leading /
    trailing separators yield empty fragments, which are preserved (so the op is
    exactly invertible in fragment count).
    """
    frags = split_fragment_ids(ids, sep_id)
    if len(frags) <= 1:
        return list(ids)
    rng.shuffle(frags)
    return join_fragment_ids(frags, sep_id)


def mask_fragment_ids(
    ids: list[int],
    sep_id: int,
    mask_id: int,
    rng: random.Random,
    *,
    frag_idx: int | None = None,
) -> list[int]:
    """Replace one fragment's tokens with ``mask_id`` (local LeJEPA view).

    Uses ``mask_id`` (not PAD). Single-fragment sequences mask the whole body.
    """
    if not ids:
        return []
    frags = split_fragment_ids(ids, sep_id)
    non_empty = [i for i, frag in enumerate(frags) if frag]
    if not non_empty:
        return [mask_id] * len(ids)
    if frag_idx is None:
        frag_idx = rng.choice(non_empty)
    elif frag_idx not in non_empty:
        frag_idx = rng.choice(non_empty)
    n_mask = max(1, len(frags[frag_idx]))
    frags[frag_idx] = [mask_id] * n_mask
    return join_fragment_ids(frags, sep_id)


class FragmentViewDataset(Dataset):
    """Yields one fragmented-SMILES view string per molecule from MOSES shards."""

    def __init__(
        self,
        shards: list[Path],
        *,
        split: str = "train",
        val_ratio: float = 0.005,
        test_ratio: float = 0.005,
        split_seed: int = 0,
        return_smiles: bool = False,
    ) -> None:
        from lattice_lab.preprocessing.molecules import (
            fragment_view_column,
            fragment_view_column_for_parquet,
        )

        self._return_smiles = return_smiles
        cols = ["inchikey", "view_idx", fragment_view_column_for_parquet(shards[0])]
        if return_smiles:
            cols.append("smiles")
        df = load_fragment_split_df(
            shards, split=split, val_ratio=val_ratio,
            test_ratio=test_ratio, split_seed=split_seed,
            columns=cols,
        )
        view_col = fragment_view_column(df)
        self._views: list[str] = df[view_col].astype(str).tolist()
        # Canonical SMILES per molecule (Morgan-fingerprint source for the
        # optional similarity-distillation objective); aligned with ``_views``.
        self._smiles: list[str] = (
            df["smiles"].astype(str).tolist() if return_smiles else []
        )
        if not self._views:
            raise ValueError(f"no molecules in split={split!r} from {len(shards)} shard(s)")
        logger.info("fragment-view dataset split=%s: %d molecules", split, len(self._views))

    def __len__(self) -> int:
        return len(self._views)

    def __getitem__(self, idx: int) -> str | tuple[str, str]:
        if self._return_smiles:
            return self._views[idx], self._smiles[idx]
        return self._views[idx]


def collate_views(batch: list[str]) -> list[str]:
    return list(batch)


def collate_views_with_smiles(
    batch: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Collator for ``return_smiles=True``: ``(views, smiles)``."""
    views, smiles = zip(*batch)
    return list(views), list(smiles)


class FragmentViewDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        shard_dir: str | Path,
        batch_size: int = 256,
        val_ratio: float = 0.005,
        test_ratio: float = 0.005,
        split_seed: int = 0,
        run_validation: bool = True,
        num_workers: int = 0,
        return_smiles: bool = False,
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.run_validation = run_validation
        self.num_workers = num_workers
        self.return_smiles = return_smiles
        self._shards: list[Path] = []
        self._train: FragmentViewDataset | None = None
        self._val: FragmentViewDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return
        self._shards = sorted(self.shard_dir.glob("shard_*.parquet"))
        if not self._shards:
            raise FileNotFoundError(f"no parquet shards in {self.shard_dir}")
        self._train = FragmentViewDataset(
            self._shards, split="train", val_ratio=self.val_ratio,
            test_ratio=self.test_ratio, split_seed=self.split_seed,
            return_smiles=self.return_smiles,
        )
        if self.run_validation:
            self._val = FragmentViewDataset(
                self._shards, split="val", val_ratio=self.val_ratio,
                test_ratio=self.test_ratio, split_seed=self.split_seed,
                return_smiles=self.return_smiles,
            )

    @property
    def _collate(self):
        return collate_views_with_smiles if self.return_smiles else collate_views

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True, collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val is None:
            return None
        return DataLoader(
            self._val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=True, collate_fn=self._collate,
        )
