"""BRICS fragmentation resilience."""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

from rdkit import Chem

from lattice_lab.preprocessing.molecules import (
    _brics_fragment_smiles,
    augment_fragment_views,
    fragment_view,
    seeded_views,
)


def _bridge(fragments: list[str]) -> str | None:
    """Reassemble a fragment view by matching same-number ``[i*]`` dummy pairs.

    The inverse of :func:`_brics_partition`'s labelling: each cut stamps isotope
    ``i`` on both stubs, so every label must appear on exactly two distinct atoms;
    bonding their non-dummy neighbours rebuilds the molecule. Returns the
    canonical SMILES, or ``None`` if the pairing invariant is violated (which is
    exactly what stock ``BRICS.BreakBRICSBonds`` env-type labels would do).
    """
    if len(fragments) == 1:
        return Chem.CanonSmiles(fragments[0])
    mol = Chem.MolFromSmiles(".".join(fragments))
    if mol is None:
        return None
    labeled: dict[int, list[int]] = {}
    for atom in mol.GetAtoms():
        iso = atom.GetIsotope()
        if iso > 0 and atom.GetSymbol() == "*":
            labeled.setdefault(iso, []).append(atom.GetIdx())
    ed = Chem.EditableMol(mol)
    conns = []
    for idxs in labeled.values():
        if len(idxs) != 2:
            return None  # not an invertible matched-pair labelling
        nbrs = []
        for d in idxs:
            real = [
                (n.GetIdx(), mol.GetBondBetweenAtoms(d, n.GetIdx()).GetBondType())
                for n in mol.GetAtomWithIdx(d).GetNeighbors()
                if n.GetIsotope() == 0
            ]
            if not real:
                return None
            nbrs.append(real[0])
        if nbrs[0][0] == nbrs[1][0]:
            return None  # self-bond: the [5*]N[5*] failure mode
        conns.append((nbrs[0][0], nbrs[1][0], nbrs[0][1]))
    for a1, a2, bt in conns:
        ed.AddBond(a1, a2, order=bt)
    for idx in sorted((i for v in labeled.values() for i in v), reverse=True):
        ed.RemoveAtom(idx)
    return Chem.MolToSmiles(ed.GetMol())

# Multi-fragment drug-like molecules with several BRICS cut points.
_SMILES = [
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",  # imatinib
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",               # gefitinib
    "CC(=O)Oc1ccccc1C(=O)O",                                        # aspirin
]


def _heavy_atoms(smi: str) -> int:
    """Heavy-atom count, ignoring BRICS dummy atoms (``*``)."""
    m = Chem.MolFromSmiles(smi)
    return sum(a.GetSymbol() != "*" for a in m.GetAtoms())


def test_brics_fragment_smiles_returns_empty_on_rdkit_failure() -> None:
    mol = Chem.MolFromSmiles("CCO")
    with patch(
        "rdkit.Chem.BRICS.BRICSDecompose",
        side_effect=AttributeError("'Mol' object has no attribute 'pSmi'"),
    ):
        assert _brics_fragment_smiles(mol) == []


def test_fragment_view_falls_back_to_canon_when_brics_empty() -> None:
    with patch("lattice_lab.preprocessing.molecules._brics_fragment_smiles", return_value=[]):
        assert fragment_view("c1ccccc1", merge=False) == "c1ccccc1"


def test_faithful_fragment_view_covers_the_whole_molecule() -> None:
    """``fragment_view`` (faithful) must encode the *whole* molecule at both the
    finest (merge=False) and merged (merge=True) granularities — every heavy atom
    preserved, never a lossy subset. Regression guard for the historical
    ``rng.sample(frags, k)`` subset bug."""
    for smi in _SMILES:
        full = _heavy_atoms(smi)
        # merge=False: deterministic finest decomposition.
        v0 = fragment_view(smi, merge=False)
        assert sum(_heavy_atoms(f) for f in v0.split(" ")) == full
        # merge=True: coarser partitions over many seeds, always full coverage.
        for seed in range(40):
            v = fragment_view(smi, merge=True, seed=seed)
            assert sum(_heavy_atoms(f) for f in v.split(" ")) == full, (
                f"merge view dropped atoms: {v!r}"
            )


def test_seeded_views_are_faithful_and_deterministic() -> None:
    for smi in _SMILES:
        full = _heavy_atoms(smi)
        views = seeded_views(smi, k=4)
        assert views and seeded_views(smi, k=4) == views  # reproducible
        for v in views:
            assert sum(_heavy_atoms(f) for f in v.split(" ")) == full


def _attachment_labels(view: str) -> list[int]:
    """Isotope labels of every ``[n*]`` dummy atom in a (space-joined) view."""
    import re

    return [int(n) for n in re.findall(r"\[(\d+)\*\]", view)]


def test_attachment_points_form_unique_same_number_pairs() -> None:
    """Every cut must leave a *matched* attachment pair: the same isotope on both
    stubs, each label used exactly twice. Regression guard against emitting raw
    BRICS environment-type labels (where ``[12*]`` pairs with ``[5*]`` and so has
    no same-number partner). Holds at every granularity / seed."""
    from collections import Counter

    for smi in _SMILES:
        views = [fragment_view(smi, merge=False)]
        views += [fragment_view(smi, merge=True, seed=s) for s in range(40)]
        views += seeded_views(smi, k=4)
        for v in views:
            counts = Counter(_attachment_labels(v))
            assert all(c == 2 for c in counts.values()), (
                f"unmatched attachment label in {v!r}: {dict(counts)}"
            )


def test_fragment_view_dummy_labels_are_matched_pairs() -> None:
    """Every cut must stamp a *unique* number on both its stubs, so each dummy
    label appears exactly twice across the view. This is the invariant that makes
    the view invertible — stock ``BRICS.BreakBRICSBonds`` env-type labels break it
    (e.g. ``[5*]N[5*]``: same label twice on *one* atom; multiple ``[16*]``)."""
    for smi in _SMILES:
        for merge, seeds in ((False, [None]), (True, range(20))):
            for seed in seeds:
                v = fragment_view(smi, merge=merge, seed=seed)
                isos = [
                    a.GetIsotope()
                    for frag in v.split(" ")
                    for a in Chem.MolFromSmiles(frag).GetAtoms()
                    if a.GetSymbol() == "*"
                ]
                counts = Counter(isos)
                assert all(n == 2 for n in counts.values()), (
                    f"non-paired dummy labels {dict(counts)} in {v!r}"
                )


def test_fragment_view_is_invertible() -> None:
    """The matched-pair labelling must let us rebuild the exact original molecule
    from the (order-independent) fragment set, at both granularities."""
    for smi in _SMILES:
        canon = Chem.CanonSmiles(smi)
        for merge, seeds in ((False, [None]), (True, range(20))):
            for seed in seeds:
                v = fragment_view(smi, merge=merge, seed=seed)
                rebuilt = _bridge(v.split(" "))
                assert rebuilt == canon, (
                    f"view not invertible (merge={merge}, seed={seed}): "
                    f"{v!r} -> {rebuilt!r} != {canon!r}"
                )


def test_merge_guards_against_lone_stub_fragment() -> None:
    """If a cut ever leaves a *single* piece still carrying dummy stubs (only
    possible for a ring cut, e.g. if rBRICS is swapped in), both the faithful and
    augment merge paths must fall back to the clean whole molecule — never emit a
    lone unpaired ``[1*]...[1*]`` fragment. Simulated by mocking the partition."""
    smi = "c1ccccc1C2CCCCC2"
    canon = Chem.CanonSmiles(smi)
    lone = ["[1*]C1CCCCC1[1*]"]  # one connected piece, dangling stubs (ring cut)
    with patch(
        "lattice_lab.preprocessing.molecules._brics_partition", return_value=lone
    ):
        v = fragment_view(smi, merge=True, seed=0)
        assert v == canon, f"merge=True leaked a lone-stub fragment: {v!r}"
        for av in augment_fragment_views(smi, n_views=3, merge=True, seed=0):
            assert " " in av or "*" not in av, f"augment leaked lone stub: {av!r}"


def test_augment_views_are_always_full_coverage() -> None:
    """``augment_fragment_views`` must never drop atoms — fragment dropping was
    removed because it orphans attachment stubs (lone ``[1*]c1ccccc1``) and breaks
    the invertible matched-pair labelling. Every view tiles the whole molecule and
    every dummy label still forms a matched pair (so the view is reassemblable)."""
    for smi in _SMILES:
        full = _heavy_atoms(smi)
        for v in augment_fragment_views(smi, n_views=5, merge=True, seed=1):
            assert sum(_heavy_atoms(f) for f in v.split(" ")) == full, (
                f"augment view dropped atoms: {v!r}"
            )
            assert _bridge(v.split(" ")) == Chem.CanonSmiles(smi)
