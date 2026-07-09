"""3D conformer generation, atom dictionary, and point-cloud featurization.

Vendored/adapted from InVirtuoLabs/InVirtuoCLIP (``data/dictionary.py``,
``data/transforms.py``, ``data/clip/pointcloud.py``). Turns a SMILES string into a
single RDKit conformer ``(atoms, coordinates)`` and featurizes heavy-atom
conformers into the ``(tokens, distance, edge_type)`` tensors consumed by
:class:`lattice_lab.backbone.pointcloud.PointCloudEncoder`.

The default atom dictionary ships at ``lattice_lab/assets/dict_mol.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DEFAULT_DICT_PATH = str(Path(__file__).resolve().parents[1] / "assets" / "dict_mol.txt")


class Dictionary:
    """Minimal atom-type dictionary compatible with Uni-Mol ``dict_mol.txt``.

    ``load`` always prepends the ``[PAD]/[CLS]/[SEP]/[UNK]`` specials (so ``[PAD]``
    is index 0, matching ``PointCloudEncoder.padding_idx``) and appends ``[MASK]``.
    """

    def __init__(self, symbols: list[str]):
        self._symbols = symbols
        self._indices = {s: i for i, s in enumerate(symbols)}

    @classmethod
    def load(cls, path: str, add_mask: bool = True) -> "Dictionary":
        symbols = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                tok = parts[0]
                if tok and tok not in symbols:
                    symbols.append(tok)
        if add_mask and "[MASK]" not in symbols:
            symbols.append("[MASK]")
        return cls(symbols)

    def index(self, sym: str) -> int:
        return self._indices.get(sym, self._indices["[UNK]"])

    def bos(self) -> int:
        return self._indices["[CLS]"]

    def eos(self) -> int:
        return self._indices["[SEP]"]

    def pad(self) -> int:
        return self._indices["[PAD]"]

    def __len__(self) -> int:
        return len(self._symbols)


def remove_hydrogens(
    atoms: np.ndarray, coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop all hydrogen atoms (the Uni-Mol mol tower is heavy-atom only)."""
    mask = atoms != "H"
    return atoms[mask], coordinates[mask]


def normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Center coordinates at their mean (Uni-Mol convention)."""
    if len(coordinates) == 0:
        return coordinates.astype(np.float32)
    return (coordinates - coordinates.mean(axis=0)).astype(np.float32)


def generate_conformer(
    smiles: str, seed: int = 42
) -> tuple[np.ndarray, np.ndarray] | None:
    """Generate one 3D conformer (heavy atoms) from SMILES via RDKit ETKDGv3.

    Returns ``(atoms, coordinates)`` (both length-N heavy-atom arrays) or ``None``
    if parsing/embedding fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    coordinates = np.array(conf.GetPositions(), dtype=np.float32)
    return atoms, coordinates


def featurize_conformer(
    atoms: np.ndarray,
    coordinates: np.ndarray,
    dictionary: Dictionary,
    max_seq_len: int = 256,
) -> dict[str, torch.Tensor]:
    """Build ``(tokens, distance, edge_type)`` tensors for one conformer.

    A ``[CLS]`` and ``[SEP]`` bracket the atom tokens; the inner distance matrix
    (scipy pairwise Euclidean) is placed in the ``[1:-1, 1:-1]`` block with zero
    padding for the two specials. ``edge_type = t_i * V + t_j`` indexes the
    Gaussian-basis edge embeddings. Shapes: ``tokens [L]``, ``distance [L, L]``,
    ``edge_type [L, L]`` with ``L = n_atoms + 2``.
    """
    from scipy.spatial import distance_matrix

    n = min(len(atoms), len(coordinates), max_seq_len)
    atoms = atoms[:n]
    coordinates = coordinates[:n]

    tokens = [dictionary.index(a) for a in atoms]
    tokens = [dictionary.bos()] + tokens + [dictionary.eos()]
    tokens_t = torch.tensor(tokens, dtype=torch.long)

    inner = distance_matrix(coordinates, coordinates).astype(np.float32)
    n_full = inner.shape[0] + 2
    dist = np.zeros((n_full, n_full), dtype=np.float32)
    dist[1:-1, 1:-1] = inner
    dist_t = torch.from_numpy(dist)

    edge_type = tokens_t.view(-1, 1) * len(dictionary) + tokens_t.view(1, -1)
    return {"tokens": tokens_t, "distance": dist_t, "edge_type": edge_type}


def _right_pad_1d(tensors: list[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.size(0)] = t
    return out


def _right_pad_2d(tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    max_h = max(t.size(0) for t in tensors)
    max_w = max(t.size(1) for t in tensors)
    out = torch.full((len(tensors), max_h, max_w), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.size(0), : t.size(1)] = t
    return out


def load_conformer_cache(
    path: str, key_col: str = "inchikey"
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load a ``conformers.parquet`` cache into ``{key: (atoms, coords)}``.

    Inverse of :mod:`lattice_lab.preprocessing.precompute_conformers`: ``atoms`` is
    re-split from the space-joined string, ``coords`` reshaped to ``[n_atoms, 3]``.
    ``key_col`` selects the row key (``inchikey`` for MOSES/BindingDB pools keyed
    by InChIKey; ``smiles`` for the Stage-5 binder store keyed by raw SMILES).
    """
    import pandas as pd

    df = pd.read_parquet(path, columns=[key_col, "atoms", "coords"])
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, atoms, coords in zip(df[key_col], df["atoms"], df["coords"]):
        a = np.asarray(str(atoms).split(), dtype=object)
        c = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
        cache[str(key)] = (a, c)
    return cache


def collate_conformers(
    samples: list[dict[str, torch.Tensor]], *, key_prefix: str = "mol"
) -> dict[str, torch.Tensor]:
    """Right-pad a list of :func:`featurize_conformer` outputs into a batch dict.

    Returns ``{key_prefix}_src_tokens [B, L]`` (pad 0), ``..._src_distance
    [B, L, L]`` (pad 0.0) and ``..._src_edge_type [B, L, L]`` (pad 0) — the exact
    keys :meth:`PointCloudEncoder.forward` reads.
    """
    p = key_prefix
    return {
        f"{p}_src_tokens": _right_pad_1d([s["tokens"] for s in samples], 0),
        f"{p}_src_distance": _right_pad_2d([s["distance"] for s in samples], 0.0),
        f"{p}_src_edge_type": _right_pad_2d([s["edge_type"] for s in samples], 0),
    }
