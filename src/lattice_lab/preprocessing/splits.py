"""Train/val/test split strategies.

Two split families are implemented:

- ``random_split`` — used for adapter SSL (99/0.5/0.5).
- ``scaffold_split`` — Bemis-Murcko scaffold disjoint split for EBM training.
- ``cluster_split`` — generic cluster-id-disjoint split, used for proteins.

Each function returns a dict ``{"train": [...], "val": [...], "test": [...]}`` of
*indices* into the input sequence (so callers can apply the split to parallel arrays).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

SplitDict = dict[str, list[int]]


def _validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    for name, x in (("train", train), ("val", val), ("test", test)):
        if x < 0:
            raise ValueError(f"{name} ratio must be non-negative, got {x}")


def random_split(
    n: int, train: float = 0.99, val: float = 0.005, test: float = 0.005, *, seed: int = 0
) -> SplitDict:
    """Uniformly random index split."""
    _validate_ratios(train, val, test)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    return {
        "train": idx[:n_train].tolist(),
        "val": idx[n_train : n_train + n_val].tolist(),
        "test": idx[n_train + n_val :].tolist(),
    }


def murcko_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES, or ``""`` for unparseable input."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        return ""


def scaffold_split(
    smiles_list: Sequence[str],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    *,
    seed: int = 0,
) -> SplitDict:
    """Bemis-Murcko scaffold-disjoint split.

    Strategy: group by scaffold SMILES, then assign whole scaffold groups to splits
    in descending order of size — packing the largest groups into ``train`` first
    keeps val/test small enough while guaranteeing scaffold disjointness. Within
    same-size groups the order is randomized by ``seed`` for reproducibility.
    """
    _validate_ratios(train, val, test)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(smiles_list):
        groups[murcko_scaffold(s)].append(i)

    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    sizes = np.array([len(groups[k]) for k in keys])
    # Tie-break randomly to avoid alphabetic bias on equal-size scaffolds.
    tiebreak = rng.random(len(keys))
    order = np.lexsort((tiebreak, -sizes))

    n = len(smiles_list)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    out: SplitDict = {"train": [], "val": [], "test": []}
    for i in order:
        members = groups[keys[i]]
        # Greedy: pick the bucket with the most remaining capacity.
        capacities = {
            "train": n_train - len(out["train"]),
            "val": n_val - len(out["val"]),
            "test": (n - n_train - n_val) - len(out["test"]),
        }
        target = max(capacities, key=lambda k: capacities[k])
        out[target].extend(members)
    return out


def cluster_split(
    cluster_ids: Sequence[int],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    *,
    seed: int = 0,
) -> SplitDict:
    """Split entries whose ``cluster_ids[i]`` shares no cluster across train/val/test.

    Same packing strategy as ``scaffold_split`` but groups by an explicit integer
    cluster id (e.g., MMseqs2 cluster index).
    """
    _validate_ratios(train, val, test)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster_ids):
        groups[c].append(i)
    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    sizes = np.array([len(groups[k]) for k in keys])
    tiebreak = rng.random(len(keys))
    order = np.lexsort((tiebreak, -sizes))
    n = len(cluster_ids)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    out: SplitDict = {"train": [], "val": [], "test": []}
    for i in order:
        members = groups[keys[i]]
        capacities = {
            "train": n_train - len(out["train"]),
            "val": n_val - len(out["val"]),
            "test": (n - n_train - n_val) - len(out["test"]),
        }
        target = max(capacities, key=lambda k: capacities[k])
        out[target].extend(members)
    return out


def check_disjoint(split: SplitDict) -> bool:
    """Raise on overlap between any two splits; return True otherwise."""
    tr = set(split["train"])
    va = set(split["val"])
    te = set(split["test"])
    if tr & va or tr & te or va & te:
        raise AssertionError("splits are not disjoint")
    return True
