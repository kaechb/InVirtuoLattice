"""Guards for multi-view z_m cache consistency."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from lattice_lab.eval.lit_pcba import (
    N_VIEWS_KEY,
    enforce_cache_n_views,
    require_zm_cache_complete,
)


@dataclass
class _FakeManifest:
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class _FakeStore:
    path: Path
    manifest: _FakeManifest
    pid_to_row: dict[str, int]


def test_enforce_cache_n_views_rejects_mismatch() -> None:
    store = _FakeStore(Path("/tmp/zm"), _FakeManifest(extra={N_VIEWS_KEY: "1"}), {})
    with pytest.raises(ValueError, match="n_views=1"):
        enforce_cache_n_views(store, 4)  # type: ignore[arg-type]


def test_enforce_cache_n_views_requires_metadata_for_mv() -> None:
    store = _FakeStore(Path("/tmp/zm"), _FakeManifest(extra={}), {})
    with pytest.raises(ValueError, match=N_VIEWS_KEY):
        enforce_cache_n_views(store, 4)  # type: ignore[arg-type]


def test_require_zm_cache_complete_raises_on_missing() -> None:
    store = _FakeStore(Path("/tmp/zm"), _FakeManifest(), {"a": 0})
    with pytest.raises(ValueError, match="missing 1"):
        require_zm_cache_complete(store, ["a", "b"])  # type: ignore[arg-type]


def test_require_zm_cache_complete_ok_when_full() -> None:
    store = _FakeStore(Path("/tmp/zm"), _FakeManifest(), {"a": 0, "b": 1})
    require_zm_cache_complete(store, ["a", "b", "a"])  # type: ignore[arg-type]
