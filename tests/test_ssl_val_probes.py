"""Tests for SSL validation probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lattice_lab.data.fragment_views import load_fragment_split_df
from lattice_lab.preprocessing.molecules import molecule_qed_molwt
from lattice_lab.training.ssl_val_probes import _pca_tsne_2d, _ridge_r2, _tsne_2d


def test_molecule_qed_molwt_ethanol() -> None:
    row = molecule_qed_molwt("CCO")
    assert row is not None
    qed, mw = row
    assert 0.0 < qed <= 1.0
    assert 40.0 < mw < 50.0


def test_ridge_r2_perfect_linear_relation() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 16)).astype(np.float32)
    y = np.column_stack([z[:, 0] * 0.5 + 0.2, z[:, 1] * 10.0 + 100.0])
    r2, n_tr, n_te = _ridge_r2(z, y, seed=0, test_size=0.2, ridge_alpha=1e-6)
    assert r2["qed"] > 0.95
    assert r2["molwt"] > 0.95
    assert n_tr + n_te == 200


def test_tsne_2d_shape() -> None:
    x = np.random.default_rng(1).normal(size=(40, 8))
    emb = _tsne_2d(x, seed=0, perplexity=10.0)
    assert emb.shape == (40, 2)


def test_pca_tsne_2d_shape() -> None:
    x = np.random.default_rng(2).normal(size=(40, 128))
    emb = _pca_tsne_2d(x, seed=0, perplexity=10.0, pca_components=50)
    assert emb.shape == (40, 2)


def test_load_fragment_split_df_val(tmp_path: Path) -> None:
    rows = [
        {"smiles": "CCO", "inchikey": "IK1", "view_idx": 0, "fragment_view": "a b"},
        {"smiles": "CCC", "inchikey": "IK2", "view_idx": 0, "fragment_view": "c d"},
        {"smiles": "CCCC", "inchikey": "IK3", "view_idx": 0, "fragment_view": "e f"},
    ]
    shard = tmp_path / "shard_0000.parquet"
    pd.DataFrame(rows).to_parquet(shard, index=False)
    # split_seed=0: hash buckets determine split; at least one row in some split
    for split in ("train", "val", "test"):
        df = load_fragment_split_df(
            [shard], split=split, val_ratio=0.34, test_ratio=0.33, split_seed=0,
        )
        assert isinstance(df, pd.DataFrame)
