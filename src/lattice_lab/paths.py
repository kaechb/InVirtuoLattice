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
