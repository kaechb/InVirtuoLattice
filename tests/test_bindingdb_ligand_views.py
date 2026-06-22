"""Stage-1 BindingDB ligand view enrichment + Stage-4 fallbacks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lattice_lab.ebm.precompute_bdb_zm import _views_for_todo
from lattice_lab.preprocessing import bindingdb
from lattice_lab.preprocessing.molecules import (
    build_smiles_fragment_views,
    fragment_view_for_smiles,
)


def test_fragment_view_for_smiles_simple() -> None:
    v = fragment_view_for_smiles("CCO")
    assert v is not None
    assert isinstance(v, str)


def test_build_smiles_fragment_views_dedupes() -> None:
    views = build_smiles_fragment_views(["CCO", "CCO"], n_jobs=1)
    assert len(views) == 1
    assert "CCO" in views


def test_ligand_view_records_and_row_from_record() -> None:
    row = bindingdb.BindingDbRow(
        monomer_id="m1",
        target_name="t",
        uniprot="P12345",
        sequence="ACDE",
        smiles="CCO",
        inchikey="IK",
        ki_nm=None,
        kd_nm=None,
        ic50_nm=None,
        ec50_nm=None,
        best_nm=None,
        best_assay="",
        is_binder_10uM=True,
    )
    recs = bindingdb.ligand_view_records(
        [row], views={"CCO": "C C O"}, body_ids={"CCO": [1, 2, 3]}
    )
    assert recs[0]["fragment_view"] == "C C O"
    assert recs[0]["body_ids"] == [1, 2, 3]
    roundtrip = bindingdb.row_from_record(recs[0])
    assert roundtrip.smiles == "CCO"


def test_views_for_todo_uses_precomputed_column() -> None:
    fv = pd.Series({"ik1": "precomputed view", "ik2": None})
    views = _views_for_todo(
        ["ik1", "ik2"],
        ["CCO", "CCN"],
        fv,
        n_jobs=1,
    )
    assert views[0] == "precomputed view"
    assert views[1] is not None  # runtime BRICS/canonical fallback


def test_run_bindingdb_enrich_helpers(tmp_path: Path) -> None:
    from lattice_lab.preprocessing import run_bindingdb

    p = tmp_path / "curated.parquet"
    pd.DataFrame({"smiles": ["CCO"], "fragment_view": ["cached"]}).to_parquet(p, index=False)
    assert not run_bindingdb._needs_ligand_view_enrich(p, want_body_ids=False)
    assert run_bindingdb._needs_ligand_view_enrich(p, want_body_ids=True)
