"""Project-relative path helpers.

Anchors the repo root so callers do not hardcode paths. All other modules import
``REPO_ROOT`` / ``FRAGMOL_DIR`` from here rather than computing paths themselves.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# This file is at <repo>/src/lattice_lab/paths.py, so the repo root (which holds
# software/, artifacts/, scripts/) is three parents up.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
# Backbone lives in software/FragMol by default; LATTICE_FRAGMOL_DIR overrides it
# (e.g. point at a shared copy instead of duplicating ~1 GB per checkout).
FRAGMOL_DIR: Path = Path(os.environ.get("LATTICE_FRAGMOL_DIR", REPO_ROOT / "software" / "FragMol"))
FRAGMOL_SAVED_MODEL: Path = FRAGMOL_DIR / "saved_models"
FRAGMOL_TOKENIZER: Path = FRAGMOL_DIR / "tokenizer" / "smiles.json"

# Bundled MMseqs2 binary dir; LATTICE_MMSEQS_DIR overrides (e.g. a module install).
MMSEQS_BIN_DIR: Path = Path(
    os.environ.get("LATTICE_MMSEQS_DIR", REPO_ROOT / "software" / "mmseqs" / "bin")
)


def ensure_mmseqs_on_path() -> None:
    """Prepend the bundled MMseqs2 ``bin/`` to ``PATH`` so ``shutil.which`` /
    ``subprocess`` find ``software/mmseqs/bin/mmseqs`` without a manual export.

    No-op if the bundled binary isn't there (a system MMseqs2 already on PATH
    still works). Idempotent.
    """
    if (MMSEQS_BIN_DIR / "mmseqs").is_file():
        d = str(MMSEQS_BIN_DIR)
        if d not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def ensure_fragmol_on_path() -> None:
    """Insert ``software/FragMol/`` on ``sys.path`` so its ``utils.*`` imports resolve.

    FragMol's modules use bare ``from utils.X import Y`` statements that assume the
    FragMol directory is the import root. We expose a single
    helper so every consumer goes through the same sys.path mutation; calling twice is
    idempotent.
    """
    if not FRAGMOL_DIR.is_dir():
        raise FileNotFoundError(
            f"FragMol backbone not found at {FRAGMOL_DIR}. The backbone is a "
            "required external dependency (like MMseqs2). Place it under "
            "software/FragMol/ (or set LATTICE_FRAGMOL_DIR). See the README "
            "'FragMol backbone' section."
        )
    p = str(FRAGMOL_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    # FragMol writes log files into models/ next to CWD; tolerate missing dir.
    os.environ.setdefault("RDKIT_DISABLE_DEPRECATION_WARNINGS", "1")


def ensure_invirtuo_on_path() -> None:
    """Make the ``in_virtuo_gen`` package (InVirtuoGen — the DDiT architecture for
    the discrete-flow backbone) importable.

    A checkpoint only stores weights; the model *class* must be importable to load
    them. Honors ``LATTICE_INVIRTUO_DIR`` (either the ``in_virtuo_gen`` package dir
    or its parent); otherwise searches ``software/`` for an ``in_virtuo_gen``
    package. No-op if it's already importable.
    """
    import importlib.util

    if importlib.util.find_spec("in_virtuo_gen") is not None:
        return

    def _is_pkg_parent(d: Path) -> bool:
        return (d / "in_virtuo_gen" / "__init__.py").is_file()

    candidates: list[Path] = []
    env = os.environ.get("LATTICE_INVIRTUO_DIR")
    if env:
        candidates += [Path(env), Path(env).parent]
    soft = REPO_ROOT / "software"
    if soft.is_dir():
        if _is_pkg_parent(soft):
            candidates.append(soft)
        for child in soft.iterdir():
            if child.is_dir() and _is_pkg_parent(child):
                candidates.append(child)

    for c in candidates:
        if _is_pkg_parent(c) and str(c) not in sys.path:
            sys.path.insert(0, str(c))
            return
