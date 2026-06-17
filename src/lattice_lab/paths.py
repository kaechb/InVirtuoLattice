"""Project-relative path helpers.

Anchors the repo root so callers do not hardcode paths. All other modules import
``REPO_ROOT`` from here rather than computing paths themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

# This file is at <repo>/src/lattice_lab/paths.py, so the repo root (which holds
# software/, artifacts/, scripts/) is three parents up.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

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
