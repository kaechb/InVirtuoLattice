"""SSL data for the discrete-flow backbone: fragment-shuffle contrastive pairs.

Each MOSES molecule has a space-separated fragmented-SMILES string (the
``fragment_view`` column from Stage-1). The contrastive augmentation is a
**fragment shuffle**: tokenize the view, split the token ids on the separator
id (the space token, id 4 in the discrete-flow tokenizer), shuffle the fragment
order, and rejoin. Two independent shuffles give the two views of a molecule.

The shuffle is a pure token-level op (:func:`shuffle_fragment_ids`) so it's unit
tested without the model. Stage 1 may store pretokenized ``body_ids`` in parquet
(``run_preprocessing --tokenizer-path``); otherwise workers or the LightningModule
tokenizes ``fragment_view`` strings at load time. Shuffle/mask always run online.
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
    frac: float = 0.5,
    frag_idx: int | None = None,
) -> list[int]:
    """Replace ~``frac`` of the molecule with ``mask_id`` (local SSL view).

    Masks ``max(1, round(frac * n))`` of the ``n`` non-empty fragments, always
    leaving >= 1 fragment intact so the context keeps real signal (masking one
    fragment of a 3-fragment median molecule is too gentle — the masked and
    intact pooled embeddings stay nearly equal, so the invariance task is
    trivial). A single-fragment molecule has no fragment to spare, so it instead
    masks ~``frac`` of the *tokens* within that fragment (again leaving >= 1),
    keeping the ~19% single-fragment MOSES molecules from degenerating into an
    information-free all-``mask_id`` context. Uses ``mask_id`` (never PAD).

    ``frag_idx`` forces a single specific fragment to be fully masked (test hook;
    ignores ``frac``).
    """
    if not ids:
        return []
    frags = split_fragment_ids(ids, sep_id)
    non_empty = [i for i, frag in enumerate(frags) if frag]
    if not non_empty:
        return [mask_id] * len(ids)
    if frag_idx is not None:
        idx = frag_idx if frag_idx in non_empty else rng.choice(non_empty)
        frags[idx] = [mask_id] * len(frags[idx])
        return join_fragment_ids(frags, sep_id)
    if len(non_empty) >= 2:
        k = min(len(non_empty) - 1, max(1, round(frac * len(non_empty))))
        for idx in rng.sample(non_empty, k):
            frags[idx] = [mask_id] * len(frags[idx])
        return join_fragment_ids(frags, sep_id)
    # Single fragment: mask a fraction of its tokens, keeping >= 1 token.
    only = non_empty[0]
    tok = frags[only]
    k = min(len(tok) - 1, max(1, round(frac * len(tok)))) if len(tok) > 1 else 1
    masked_pos = set(rng.sample(range(len(tok)), k))
    frags[only] = [mask_id if i in masked_pos else t for i, t in enumerate(tok)]
    return join_fragment_ids(frags, sep_id)


def mask_span_ids(
    ids: list[int],
    mask_id: int,
    rng: random.Random,
    *,
    frac: float = 0.5,
) -> list[int]:
    """Replace ~``frac`` of the sequence with one contiguous ``mask_id`` span.

    Always leaves >= 1 real token so the masked view keeps context. Spans may
    cross fragment boundaries (including ``sep_id`` tokens).
    """
    if not ids:
        return []
    n = len(ids)
    if n <= 1:
        return list(ids)
    k = min(n - 1, max(1, round(frac * n)))
    start = rng.randrange(0, n - k + 1)
    return [mask_id if start <= i < start + k else t for i, t in enumerate(ids)]


def mask_local_ids(
    ids: list[int],
    sep_id: int,
    mask_id: int,
    rng: random.Random,
    *,
    frac: float = 0.5,
    mode: str = "fragment",
) -> list[int]:
    """Local-view mask: whole fragments, a contiguous span, or random choice."""
    if mode == "fragment":
        return mask_fragment_ids(ids, sep_id, mask_id, rng, frac=frac)
    if mode == "span":
        return mask_span_ids(ids, mask_id, rng, frac=frac)
    if mode == "mixed":
        pick = "span" if rng.random() < 0.5 else "fragment"
        return mask_local_ids(ids, sep_id, mask_id, rng, frac=frac, mode=pick)
    raise ValueError(f"mask mode must be fragment, span, or mixed, got {mode!r}")


class FragmentViewDataset(Dataset):
    """Yields fragmented-SMILES views or pretokenized ``body_ids`` from MOSES shards."""

    def __init__(
        self,
        shards: list[Path],
        *,
        split: str = "train",
        val_ratio: float = 0.005,
        test_ratio: float = 0.005,
        split_seed: int = 0,
        return_smiles: bool = False,
        tokenizer_path: str | Path | None = None,
    ) -> None:
        from lattice_lab.preprocessing.molecules import (
            fragment_view_column,
            fragment_view_column_for_parquet,
            load_smiles_tokenizer,
            shards_have_body_ids,
        )

        self._return_smiles = return_smiles
        self._use_body_ids = shards_have_body_ids(shards)
        self._tokenizer = None
        if not self._use_body_ids and tokenizer_path is not None:
            self._tokenizer = load_smiles_tokenizer(tokenizer_path)
        cols = ["inchikey", "view_idx"]
        if self._use_body_ids:
            cols.append("body_ids")
        else:
            cols.append(fragment_view_column_for_parquet(shards[0]))
        if return_smiles:
            cols.append("smiles")
        df = load_fragment_split_df(
            shards, split=split, val_ratio=val_ratio,
            test_ratio=test_ratio, split_seed=split_seed,
            columns=cols,
        )
        if self._use_body_ids:
            self._bodies = [list(map(int, row)) for row in df["body_ids"].tolist()]
            self._views: list[str] = []
            logger.info(
                "fragment-view dataset split=%s: %d molecules (precomputed body_ids)",
                split, len(self._bodies),
            )
        else:
            view_col = fragment_view_column(df)
            self._views = df[view_col].astype(str).tolist()
            self._bodies = []
            logger.info("fragment-view dataset split=%s: %d molecules", split, len(self._views))
        self._smiles: list[str] = (
            df["smiles"].astype(str).tolist() if return_smiles else []
        )
        if not self._use_body_ids and not self._views:
            raise ValueError(f"no molecules in split={split!r} from {len(shards)} shard(s)")
        if self._use_body_ids and not self._bodies:
            raise ValueError(f"no molecules in split={split!r} from {len(shards)} shard(s)")

    def __len__(self) -> int:
        return len(self._bodies) if self._use_body_ids else len(self._views)

    def __getitem__(self, idx: int) -> str | list[int] | tuple[str | list[int], str]:
        if self._use_body_ids:
            item: str | list[int] = self._bodies[idx]
        else:
            view = self._views[idx]
            item = view
            if self._tokenizer is not None:
                item = self._tokenizer.encode(view, add_special_tokens=False)
        if self._return_smiles:
            return item, self._smiles[idx]
        return item


def collate_views(batch: list[str]) -> list[str]:
    return list(batch)


def collate_bodies(batch: list[list[int]]) -> list[list[int]]:
    return list(batch)


def collate_views_with_smiles(
    batch: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Collator for ``return_smiles=True``: ``(views, smiles)``."""
    views, smiles = zip(*batch)
    return list(views), list(smiles)


def collate_bodies_with_smiles(
    batch: list[tuple[list[int], str]],
) -> tuple[list[list[int]], list[str]]:
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
        tokenizer_path: str | Path | None = None,
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
        self.tokenizer_path = tokenizer_path
        self._shards: list[Path] = []
        self._train: FragmentViewDataset | None = None
        self._val: FragmentViewDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return
        self._shards = sorted(self.shard_dir.glob("shard_*.parquet"))
        if not self._shards:
            raise FileNotFoundError(f"no parquet shards in {self.shard_dir}")
        ds_kw = dict(
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            split_seed=self.split_seed,
            return_smiles=self.return_smiles,
            tokenizer_path=self.tokenizer_path,
        )
        self._train = FragmentViewDataset(self._shards, split="train", **ds_kw)
        if self.run_validation:
            self._val = FragmentViewDataset(self._shards, split="val", **ds_kw)

    def _pretokenized_batches(self) -> bool:
        from lattice_lab.preprocessing.molecules import shards_have_body_ids

        if self._train is not None:
            return bool(self._train._use_body_ids)
        return shards_have_body_ids(self._shards)

    @property
    def _collate(self):
        pretokenized = self._pretokenized_batches() or self.tokenizer_path is not None
        if pretokenized and self.return_smiles:
            return collate_bodies_with_smiles
        if pretokenized:
            return collate_bodies
        if self.return_smiles:
            return collate_views_with_smiles
        return collate_views

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True, collate_fn=self._collate,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val is None:
            return None
        return DataLoader(
            self._val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=True, collate_fn=self._collate,
            persistent_workers=self.num_workers > 0,
        )
