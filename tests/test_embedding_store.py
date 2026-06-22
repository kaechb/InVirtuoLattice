"""EmbeddingStore append resilience."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lattice_lab.protein.store import EmbeddingStore


def test_append_mean_recreates_missing_mean_dat(tmp_path: Path) -> None:
    store = EmbeddingStore.create(
        tmp_path, embedding_dim=4, model_name="test", dtype="float32"
    )
    store.append_mean(["a"], np.zeros((1, 4), dtype=np.float32))
    (tmp_path / EmbeddingStore.MEAN).unlink()
    store.append_mean(["b"], np.ones((1, 4), dtype=np.float32))
    assert store.manifest.count == 2
    assert store.get_mean("b")[0] == 1.0
