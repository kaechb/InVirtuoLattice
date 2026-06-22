"""Tests for SSL validation probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lattice_lab.data.fragment_views import load_fragment_split_df
from lattice_lab.preprocessing.molecules import molecule_qed_molwt
from lattice_lab.training.ssl_val_probes import _pca_tsne_2d, _ridge_r2, _tsne_2d
from lattice_lab.training.ssl_val_probes import embedding_batch_collapse_diag


def test_molecule_qed_molwt_ethanol() -> None:
    row = molecule_qed_molwt("CCO")
    assert row is not None
    qed, mw = row
    assert 0.0 < qed <= 1.0
    assert 40.0 < mw < 50.0


def test_molecule_probe_props_ethanol() -> None:
    import math

    from lattice_lab.preprocessing.molecules import (
        PROBE_DESCRIPTOR_NAMES,
        molecule_probe_props,
    )

    row = molecule_probe_props("CCO")
    assert row is not None
    assert len(row) == len(PROBE_DESCRIPTOR_NAMES)
    d = dict(zip(PROBE_DESCRIPTOR_NAMES, row))
    assert 0.0 < d["qed"] <= 1.0
    assert 40.0 < d["molwt"] < 50.0
    assert -1.0 < d["logp"] < 1.0
    assert d["fraction_csp3"] == 1.0  # ethanol: both carbons sp3
    assert d["bertz_ct"] >= 0.0
    assert math.isfinite(d["balaban_j"])


def test_probe_result_as_metrics_keys() -> None:
    from lattice_lab.preprocessing.molecules import PROBE_DESCRIPTOR_NAMES
    from lattice_lab.training.ssl_val_probes import _build_probe_result

    r2 = {name: 0.5 for name in PROBE_DESCRIPTOR_NAMES}
    result = _build_probe_result(r2, n_probe=100, n_train=80, n_test=20, r2_molwt_sum=0.4)
    metrics = result.as_metrics()
    for name in PROBE_DESCRIPTOR_NAMES:
        assert metrics[f"r2/{name}"] == 0.5
    assert "r2/mean" in metrics and "r2/mean_structural" in metrics
    assert metrics["r2/molwt_sum"] == 0.4


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


def test_embedding_batch_collapse_diag_spiky_when_collapsed() -> None:
    rng = np.random.default_rng(0)
    full = rng.standard_normal((128, 32))
    collapsed = rng.standard_normal((128, 1)) @ rng.standard_normal((1, 32))
    std_full, eig_full = embedding_batch_collapse_diag(full, top_k=5)
    std_col, eig_col = embedding_batch_collapse_diag(collapsed, top_k=5)
    assert std_full > std_col
    assert eig_col[0] / max(eig_col[4], 1e-12) > eig_full[0] / max(eig_full[4], 1e-12)


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
