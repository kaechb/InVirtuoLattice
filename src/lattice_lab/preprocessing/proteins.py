"""Protein preprocessing primitives — FASTA parsing, length filter, clustering."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from lattice_lab.preprocessing.homology import enforce_mmseqs

logger = logging.getLogger(__name__)


AA_ALPHABET: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class ProteinRecord:
    """One protein entry: identifier + sequence."""

    pid: str
    sequence: str


def parse_fasta(path: str | Path) -> list[ProteinRecord]:
    """Lightweight FASTA parser. Returns one ``ProteinRecord`` per ``>`` block.

    We deliberately avoid biopython as a dependency for one parser.
    """
    records: list[ProteinRecord] = []
    pid: str | None = None
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if pid is not None:
                    records.append(ProteinRecord(pid=pid, sequence="".join(buf)))
                # Use first token after ">" (UniProt-style "sp|P12345|NAME").
                pid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if pid is not None:
        records.append(ProteinRecord(pid=pid, sequence="".join(buf)))
    return records


def filter_length(records: Iterable[ProteinRecord], min_len: int = 50, max_len: int = 1500
                  ) -> list[ProteinRecord]:
    """Keep proteins whose sequence length is within ``[min_len, max_len]``."""
    out: list[ProteinRecord] = []
    for r in records:
        if min_len <= len(r.sequence) <= max_len:
            out.append(r)
    return out


def filter_valid_residues(records: Iterable[ProteinRecord]) -> list[ProteinRecord]:
    """Drop sequences containing non-canonical residues (X, U, etc.)."""
    return [r for r in records if set(r.sequence).issubset(AA_ALPHABET)]


def _mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


def cluster_proteins(
    records: list[ProteinRecord],
    min_identity: float = 0.4,
    workdir: str | Path | None = None,
) -> dict[str, int]:
    """Cluster protein records at ``min_identity`` sequence identity.

    Returns a mapping ``pid -> cluster_id`` (small integer). Uses MMseqs2 when
    available; otherwise falls back to a deterministic per-record cluster (one
    cluster per sequence). That fallback is **opt-in**: a missing MMseqs2 raises
    unless ``LATTICE_ALLOW_KMER_FALLBACK=1`` is set, so real runs fail loudly
    instead of silently losing the identity-based clustering.

    The fallback policy is "every protein is its own cluster". This is the *most
    conservative* choice for split disjointness: it guarantees no leakage but
    yields no clustering benefit. It is a placeholder for use in tests and CI;
    real training runs must install MMseqs2.
    """
    if not records:
        return {}
    if not _mmseqs_available():
        enforce_mmseqs("not found on PATH")
        logger.warning(
            "mmseqs not found on PATH — every protein placed in its own cluster. "
            "Install MMseqs2 to perform identity-based clustering."
        )
        return {r.pid: i for i, r in enumerate(records)}

    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattice_mmseqs_"))
    # Start from a clean workdir: mmseqs refuses to overwrite an existing output
    # database, so a stale `clu`/`db`/`tmp` left by a previous run makes
    # `mmseqs cluster` exit non-zero even though it's installed and on PATH.
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    fasta = workdir / "input.fasta"
    with open(fasta, "w") as fh:
        for r in records:
            fh.write(f">{r.pid}\n{r.sequence}\n")
    db = workdir / "db"
    clu = workdir / "clu"
    tmp = workdir / "tmp"
    tmp.mkdir(exist_ok=True)
    tsv = workdir / "clu.tsv"
    # Capture mmseqs output so a real failure surfaces *its* error (e.g. memory,
    # bad input) instead of the generic "install it" message.
    try:
        subprocess.run(["mmseqs", "createdb", str(fasta), str(db)], check=True,
                       capture_output=True, text=True)
        subprocess.run(
            [
                "mmseqs",
                "cluster",
                str(db),
                str(clu),
                str(tmp),
                "--min-seq-id",
                str(min_identity),
                "-c",
                "0.8",
                "--cov-mode",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(["mmseqs", "createtsv", str(db), str(db), str(clu), str(tsv)],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        err_lines = (exc.stderr or exc.stdout or "").strip().splitlines()
        tail = " | ".join(err_lines[-3:]) if err_lines else f"exit {exc.returncode}"
        enforce_mmseqs(f"cluster run failed ({tail})", cause=exc)
        logger.error("mmseqs failed (%s): %s; falling back to identity clusters.", exc, tail)
        return {r.pid: i for i, r in enumerate(records)}

    rep_to_idx: dict[str, int] = {}
    out: dict[str, int] = {}
    with open(tsv) as fh:
        for line in fh:
            rep, member = line.strip().split("\t")
            if rep not in rep_to_idx:
                rep_to_idx[rep] = len(rep_to_idx)
            out[member] = rep_to_idx[rep]
    return out
