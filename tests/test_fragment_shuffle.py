"""Unit tests for the fragment-shuffle SSL augmentation (model-free)."""

from __future__ import annotations

import random

from lattice_lab.data.fragment_views import shuffle_fragment_ids

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


def test_deterministic_given_rng() -> None:
    ids = [5, SEP, 6, SEP, 7]
    a = shuffle_fragment_ids(ids, SEP, random.Random(42))
    b = shuffle_fragment_ids(ids, SEP, random.Random(42))
    assert a == b
