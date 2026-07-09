"""Unit tests for the fragment-shuffle SSL augmentation (model-free)."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from lattice_lab.data.fragment_views import (
    FragmentViewDataset,
    join_fragment_ids,
    mask_frags,
    mask_fragment_ids,
    mask_local_frags,
    mask_local_ids,
    mask_span_ids,
    noise_frags,
    noise_local_frags,
    noise_span_ids,
    shuffle_fragment_ids,
    shuffle_frags,
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


def test_mask_fragment_fraction_masks_some_but_leaves_one() -> None:
    ids = [10, SEP, 20, SEP, 30, SEP, 40]  # 4 fragments
    masked = mask_fragment_ids(ids, SEP, mask_id=99, rng=random.Random(0), frac=0.5)
    frags = split_fragment_ids(masked, SEP)
    fully_masked = [f for f in frags if f and all(t == 99 for t in f)]
    intact = [f for f in frags if f and all(t != 99 for t in f)]
    assert len(fully_masked) == 2  # round(0.5 * 4)
    assert len(intact) >= 1  # never masks every fragment


def test_mask_fragment_single_fragment_not_fully_masked() -> None:
    # ~19% of MOSES molecules are single-fragment; masking the whole body gives
    # an information-free context. Mask a token fraction, leaving >= 1 token.
    ids = [5, 6, 7, 8]
    masked = mask_fragment_ids(ids, SEP, mask_id=42, rng=random.Random(1), frac=0.5)
    assert len(masked) == len(ids)
    assert 0 < masked.count(42) < len(ids)  # some masked, some real signal kept


def test_mask_span_is_contiguous_and_leaves_context() -> None:
    ids = [10, 11, 12, SEP, 20, 21, 22]
    masked = mask_span_ids(ids, mask_id=99, rng=random.Random(0), frac=0.5)
    assert len(masked) == len(ids)
    assert 0 < masked.count(99) < len(ids)
    runs = []
    run = 0
    for t in masked:
        if t == 99:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    assert len(runs) == 1  # one contiguous span


def test_mask_local_mixed_picks_fragment_or_span() -> None:
    ids = [10, SEP, 20, SEP, 30, SEP, 40]
    rng = random.Random(0)
    seen = {tuple(mask_local_ids(ids, SEP, 99, rng, frac=0.5, mode="mixed")) for _ in range(40)}
    assert len(seen) > 1


def test_deterministic_given_rng() -> None:
    ids = [5, SEP, 6, SEP, 7]
    a = shuffle_fragment_ids(ids, SEP, random.Random(42))
    b = shuffle_fragment_ids(ids, SEP, random.Random(42))
    assert a == b


# Cases spanning the branches: multi-fragment, single-fragment (token mask),
# empty, leading/trailing separators, two-fragment edge.
_EQUIV_CASES = [
    [10, SEP, 20, 21, SEP, 30, SEP, 40],
    [5, 6, 7, 8],
    [],
    [SEP, 5, 6, SEP],
    [10, SEP, 20],
    [SEP, SEP],
]


@pytest.mark.parametrize("ids", _EQUIV_CASES)
@pytest.mark.parametrize("seed", range(8))
def test_frag_list_path_matches_flat_ids_path(ids: list[int], seed: int) -> None:
    """The pre-split fragment helpers used by the model must produce bit-identical
    output (same RNG draws) to the flat-ids helpers they wrap — the whole point of
    the optimization is to be a pure speedup, not a behavior change."""
    sep, mask_id = SEP, 99
    for frac in (0.1, 0.5):
        for mode in ("fragment", "span", "mixed"):
            frags = split_fragment_ids(ids, sep)
            ref = mask_local_ids(ids, sep, mask_id, random.Random(seed), frac=frac, mode=mode)
            got = mask_local_frags(
                frags, ids, sep, mask_id, random.Random(seed), frac=frac, mode=mode
            )
            assert got == ref, (ids, frac, mode)

    # shuffle_frags reuses one split across two views without mutating it.
    frags = split_fragment_ids(ids, sep)
    ref = shuffle_fragment_ids(ids, sep, random.Random(seed))
    got = shuffle_frags(frags, sep, random.Random(seed))
    assert got == ref
    # second call on the same frags is unaffected by the first (no in-place mutation)
    assert shuffle_frags(frags, sep, random.Random(seed)) == ref

    # mask_frags matches mask_fragment_ids and leaves the input list intact.
    frags = split_fragment_ids(ids, sep)
    snapshot = [list(f) for f in frags]
    ref = mask_fragment_ids(ids, sep, mask_id, random.Random(seed), frac=0.5)
    got = mask_frags(frags, sep, mask_id, random.Random(seed), frac=0.5)
    assert got == ref
    assert frags == snapshot  # not mutated
    assert join_fragment_ids(split_fragment_ids(ids, sep), sep) == ids  # split/join inverse


NOISE_POOL = (11, 12, 13, 14)


def test_noise_frags_marks_holes_and_avoids_unk() -> None:
    ids = [10, SEP, 20, 21, SEP, 30]
    masked, holes = noise_frags(
        split_fragment_ids(ids, SEP), SEP, NOISE_POOL, random.Random(0), frag_idx=1,
    )
    assert len(masked) == len(ids)
    assert len(holes) == len(ids)
    assert holes == [False, False, True, True, False, False]
    assert all(t in NOISE_POOL for i, t in enumerate(masked) if holes[i])


def test_noise_span_is_contiguous_and_leaves_context() -> None:
    ids = [10, 11, 12, SEP, 20, 21, 22]
    masked, holes = noise_span_ids(ids, NOISE_POOL, random.Random(0), frac=0.5)
    assert len(masked) == len(ids) == len(holes)
    assert 0 < sum(holes) < len(ids)
    runs = []
    run = 0
    for h in holes:
        if h:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    assert len(runs) == 1
    assert all(masked[i] in NOISE_POOL for i, h in enumerate(holes) if h)


def test_noise_local_matches_mask_coverage() -> None:
    ids = [10, SEP, 20, SEP, 30, SEP, 40]
    sep, mask_id = SEP, 99
    frags = split_fragment_ids(ids, sep)
    ref = mask_local_ids(ids, sep, mask_id, random.Random(0), frac=0.5, mode="fragment")
    masked, holes = noise_local_frags(
        frags, ids, sep, NOISE_POOL, random.Random(0), frac=0.5, mode="fragment",
    )
    assert holes == [t == mask_id for t in ref]
    assert sum(holes) == ref.count(mask_id)


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


def test_flatten_views_to_rows_body_ids() -> None:
    from lattice_lab.preprocessing.molecules import flatten_views_to_rows

    class _Tok:
        def encode(self, view: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return [len(view), view.count(" ") + 10]

    records = [{"smiles": "CCO", "inchikey": "IK1", "views": ["a b", "c d e"]}]
    rows = flatten_views_to_rows(records, tokenizer=_Tok())
    assert rows[0]["body_ids"] == [3, 11]
    assert rows[1]["body_ids"] == [5, 12]


def test_fragment_view_dataset_reads_body_ids_column(tmp_path: Path) -> None:
    from lattice_lab.preprocessing.molecules import flatten_views_to_rows

    class _Tok:
        def encode(self, view: str, *, add_special_tokens: bool = False) -> list[int]:
            return [ord(view[0])]

    rows = flatten_views_to_rows(
        [
            {"smiles": "CCO", "inchikey": "IK1", "views": ["ab"]},
            {"smiles": "CCC", "inchikey": "IK2", "views": ["cd"]},
        ],
        tokenizer=_Tok(),
    )
    shard = tmp_path / "shard_0000.parquet"
    pd.DataFrame(rows).to_parquet(shard, index=False)
    ds = FragmentViewDataset([shard], split="train", val_ratio=0.0, test_ratio=0.0)
    assert ds._use_body_ids
    assert len(ds) == 2
    assert ds[0] == [ord("a")]
    assert ds[1] == [ord("c")]


def test_fragment_view_dataset_reads_mixed_legacy_body_id_shards(tmp_path: Path) -> None:
    old = tmp_path / "shard_0000.parquet"
    new = tmp_path / "shard_0001.parquet"
    pd.DataFrame(
        [{"smiles": "CCO", "inchikey": "IK1", "view_idx": 0, "fragmol_view": "a b", "body_ids": [1, 2]}]
    ).to_parquet(old, index=False)
    pd.DataFrame(
        [{"smiles": "CCC", "inchikey": "IK2", "view_idx": 0, "fragment_view": "c d", "body_ids": [3, 4]}]
    ).to_parquet(new, index=False)

    ds = FragmentViewDataset([old, new], split="train", val_ratio=0.0, test_ratio=0.0, return_smiles=True)

    assert ds._use_body_ids
    assert ds[0] == ([1, 2], "CCO")
    assert ds[1] == ([3, 4], "CCC")
