"""Unit tests for the fragment-shuffle SSL augmentation (model-free)."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from lattice_lab.data.fragment_views import (
    FragmentViewDataset,
    mask_fragment_ids,
    shuffle_fragment_ids,
    split_fragment_ids,
)
from lattice_lab.preprocessing.molecules import fragment_view_column

SEP = 4


def _fragments(ids: list[int], sep: int = SEP) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = [()]
    for t in ids:
        if t == sep:
            out.append(())
        else:
            out[-1] = (*out[-1], t)
    return out


def test_shuffle_preserves_fragment_multiset() -> None:
    ids = [5, 6, SEP, 7, 8, 9, SEP, 10]  # 3 fragments
    out = shuffle_fragment_ids(ids, SEP, random.Random(0))
    assert sorted(_fragments(out)) == sorted(_fragments(ids))
    assert out.count(SEP) == ids.count(SEP)  # same number of joins


def test_single_fragment_is_unchanged() -> None:
    ids = [5, 6, 7, 8]  # no separator
    assert shuffle_fragment_ids(ids, SEP, random.Random(1)) == ids


def test_shuffle_actually_reorders_with_enough_fragments() -> None:
    ids = [10, SEP, 20, SEP, 30, SEP, 40, SEP, 50]
    seen = {tuple(shuffle_fragment_ids(ids, SEP, random.Random(s))) for s in range(20)}
    assert len(seen) > 1  # not always identity
    # every output has the same fragments, just reordered
    for o in seen:
        assert sorted(_fragments(list(o))) == sorted(_fragments(ids))


def test_mask_fragment_replaces_one_fragment() -> None:
    ids = [10, SEP, 20, 21, SEP, 30]
    masked = mask_fragment_ids(ids, SEP, mask_id=99, rng=random.Random(0), frag_idx=1)
    frags = split_fragment_ids(masked, SEP)
    assert frags[0] == [10]
    assert frags[1] == [99, 99]
    assert frags[2] == [30]


def test_mask_fragment_single_fragment_sequence() -> None:
    ids = [5, 6, 7]
    masked = mask_fragment_ids(ids, SEP, mask_id=42, rng=random.Random(1))
    assert masked == [42, 42, 42]


def test_deterministic_given_rng() -> None:
    ids = [5, SEP, 6, SEP, 7]
    a = shuffle_fragment_ids(ids, SEP, random.Random(42))
    b = shuffle_fragment_ids(ids, SEP, random.Random(42))
    assert a == b


@pytest.mark.parametrize("view_col", ["fragment_view", "fragmol_view"])
def test_fragment_view_dataset_reads_view_column(tmp_path: Path, view_col: str) -> None:
    rows = [
        {"smiles": "CCO", "inchikey": "IK1", "view_idx": 0, view_col: "a b"},
        {"smiles": "CCO", "inchikey": "IK1", "view_idx": 1, view_col: "a c"},
        {"smiles": "CCC", "inchikey": "IK2", "view_idx": 0, view_col: "d e"},
    ]
    shard = tmp_path / "shard_0000.parquet"
    pd.DataFrame(rows).to_parquet(shard, index=False)
    assert fragment_view_column(pd.read_parquet(shard).head(0)) == view_col
    ds = FragmentViewDataset([shard], split="train", val_ratio=0.0, test_ratio=0.0)
    assert len(ds) == 2
