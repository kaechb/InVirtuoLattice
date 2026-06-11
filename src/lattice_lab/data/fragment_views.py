"""SSL data for the discrete-flow backbone: fragment-shuffle contrastive pairs.

Each MOSES molecule has a space-separated fragmented-SMILES string (the
``fragmol_view`` column from Stage-1). The contrastive augmentation is a
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


def shuffle_fragment_ids(ids: list[int], sep_id: int, rng: random.Random) -> list[int]:
    """Split ``ids`` on ``sep_id``, shuffle the fragments, rejoin with ``sep_id``.

    A single-fragment sequence (no ``sep_id``) is returned unchanged. Leading /
    trailing separators yield empty fragments, which are preserved (so the op is
    exactly invertible in fragment count).
    """
    frags: list[list[int]] = [[]]
    for t in ids:
        if t == sep_id:
            frags.append([])
        else:
            frags[-1].append(t)
    if len(frags) <= 1:
        return list(ids)
    rng.shuffle(frags)
    out: list[int] = []
    for i, frag in enumerate(frags):
        if i:
            out.append(sep_id)
        out.extend(frag)
    return out


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
    ) -> None:
        frames = [
            pd.read_parquet(s, columns=["inchikey", "view_idx", "fragmol_view"])
            for s in shards
        ]
        df = pd.concat(frames, ignore_index=True)
        # One view per molecule (the canonical view_idx==0 row).
        df = df[df["view_idx"] == 0].drop_duplicates("inchikey").reset_index(drop=True)
        # Deterministic per-molecule split by hashing inchikey (independent of order).
        keys = df["inchikey"].astype(str)
        buckets = keys.map(lambda k: (hash((split_seed, k)) % 10_000) / 10_000.0)
        if split == "train":
            mask = buckets >= (val_ratio + test_ratio)
        elif split == "val":
            mask = (buckets >= test_ratio) & (buckets < val_ratio + test_ratio)
        elif split == "test":
            mask = buckets < test_ratio
        else:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self._views: list[str] = df.loc[mask, "fragmol_view"].astype(str).tolist()
        if not self._views:
            raise ValueError(f"no molecules in split={split!r} from {len(shards)} shard(s)")
        logger.info("fragment-view dataset split=%s: %d molecules", split, len(self._views))

    def __len__(self) -> int:
        return len(self._views)

    def __getitem__(self, idx: int) -> str:
        return self._views[idx]


def collate_views(batch: list[str]) -> list[str]:
    return list(batch)


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
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.run_validation = run_validation
        self.num_workers = num_workers
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
        )
        if self.run_validation:
            self._val = FragmentViewDataset(
                self._shards, split="val", val_ratio=self.val_ratio,
                test_ratio=self.test_ratio, split_seed=self.split_seed,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True, collate_fn=collate_views,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val is None:
            return None
        return DataLoader(
            self._val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=True, collate_fn=collate_views,
        )
