"""BRICS fragmentation resilience."""

from __future__ import annotations

from unittest.mock import patch

from rdkit import Chem

from lattice_lab.preprocessing.molecules import _brics_fragment_smiles, smiles_to_fragment_views


def test_brics_fragment_smiles_returns_empty_on_rdkit_failure() -> None:
    mol = Chem.MolFromSmiles("CCO")
    with patch(
        "rdkit.Chem.BRICS.BRICSDecompose",
        side_effect=AttributeError("'Mol' object has no attribute 'pSmi'"),
    ):
        assert _brics_fragment_smiles(mol) == []


def test_smiles_to_fragment_views_falls_back_to_canon_when_brics_empty() -> None:
    with patch("lattice_lab.preprocessing.molecules._brics_fragment_smiles", return_value=[]):
        views = smiles_to_fragment_views("c1ccccc1", n_views=1, seed=0)
    assert views == ["c1ccccc1"]
