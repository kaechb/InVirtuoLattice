"""Cross-FASTA homology search using MMseqs2.

DrugCLIP's LIT-PCBA evaluation excludes any training target whose sequence
identity to a LIT-PCBA target exceeds a chosen threshold. They report three
levels:

- 30 %  — strict family-level removal (similar to PFAM domain removal).
- 60 %  — sub-family / orthologue removal.
- 90 %  — near-duplicate removal; the default for the single non-ensemble model
          on DUD-E / LIT-PCBA (see DrugCLIP §"In silico validation").

This module wraps ``mmseqs easy-search`` to compute, for each *query* sequence
(BindingDB target), the maximum identity against any *reference* sequence
(LIT-PCBA target). Anything above the threshold is exported to a TSV exclusion
list which the splitter consumes.

A pure-Python fallback (k-mer Jaccard with a conservative scaler) exists for
environments without MMseqs2, but it is **opt-in**: a missing or failing MMseqs2
raises unless ``LATTICE_ALLOW_KMER_FALLBACK=1`` is set. The approximate identities
are only fit for tests / smoke runs and would silently break real splits, so we
refuse to use them by default rather than produce a quietly-wrong leakage split.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# DrugCLIP-style identity cutoffs (fractions).
DRUGCLIP_THRESHOLDS: tuple[float, ...] = (0.30, 0.60, 0.90)
DEFAULT_THRESHOLD: float = 0.90


@dataclass(frozen=True)
class HomologyHit:
    query: str
    target: str
    identity: float    # fraction in [0, 1]
    aln_len: int


def _mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


#: Truthy env var that opts back into the approximate (MMseqs2-free) fallback.
#: Unset by default, so a missing/failing MMseqs2 raises instead of silently
#: degrading the identity split / clustering. Honour the user's expectation that
#: real runs fail loudly; only tests and smoke runs should set this.
ALLOW_KMER_FALLBACK_ENV = "LATTICE_ALLOW_KMER_FALLBACK"


def _fallback_allowed() -> bool:
    return os.environ.get(ALLOW_KMER_FALLBACK_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def enforce_mmseqs(reason: str, *, cause: BaseException | None = None) -> None:
    """Require MMseqs2 unless the approximate fallback is explicitly enabled.

    Call this at every point that would otherwise *silently* drop to the k-mer (or
    singleton-cluster) fallback. Raises ``RuntimeError`` unless
    ``LATTICE_ALLOW_KMER_FALLBACK`` is truthy. ``reason`` describes what went wrong
    (e.g. ``"not found on PATH"``); ``cause`` chains an underlying exception.
    """
    if _fallback_allowed():
        return
    raise RuntimeError(
        f"MMseqs2 {reason}. It is required for accurate identity-based splits and "
        f"protein clustering; without it the result silently degrades to an "
        f"approximate fallback and the anti-leakage guarantee no longer holds. "
        f"Install it with `conda env create -f environment.yml` (or "
        f"`conda install -c bioconda mmseqs2`). For tests / smoke runs only, set "
        f"{ALLOW_KMER_FALLBACK_ENV}=1 to permit the approximate fallback."
    ) from cause


def _write_fasta(seqs: Mapping[str, str], path: Path) -> None:
    with open(path, "w") as fh:
        for pid, seq in seqs.items():
            fh.write(f">{pid}\n{seq}\n")


def mmseqs_easy_search(
    query: Mapping[str, str],
    reference: Mapping[str, str],
    *,
    min_identity: float = 0.30,
    coverage: float = 0.5,
    workdir: str | Path | None = None,
) -> list[HomologyHit]:
    """Run ``mmseqs easy-search`` for all query × reference pairs.

    The lowest ``min_identity`` you ever need should be passed in — MMseqs2's
    sensitivity floor is driven by this number; we then sub-set the hits in
    Python to obtain the 60 % / 90 % cutoffs.

    Returns hits with identity ≥ ``min_identity``. Sequences without any hit
    are simply absent from the result.
    """
    if not query or not reference:
        return []
    if not _mmseqs_available():
        enforce_mmseqs("not found on PATH")
        logger.warning(
            "mmseqs not on PATH; falling back to k-mer Jaccard. "
            "Identity numbers will be approximate — install MMseqs2 for accurate splits."
        )
        return _kmer_jaccard_hits(query, reference, min_identity=min_identity)

    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattice_mmseqs_search_"))
    workdir.mkdir(parents=True, exist_ok=True)
    qf = workdir / "query.fasta"
    rf = workdir / "ref.fasta"
    out = workdir / "hits.tsv"
    tmp = workdir / "tmp"
    tmp.mkdir(exist_ok=True)
    _write_fasta(query, qf)
    _write_fasta(reference, rf)

    cmd = [
        "mmseqs",
        "easy-search",
        str(qf),
        str(rf),
        str(out),
        str(tmp),
        "--min-seq-id",
        f"{min_identity:.3f}",
        "-c",
        f"{coverage:.3f}",
        "--cov-mode",
        "0",
        "--format-output",
        "query,target,pident,alnlen",
        "-s",
        "7.5",
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        enforce_mmseqs("easy-search failed", cause=exc)
        logger.error("mmseqs easy-search failed (%s); falling back to k-mer Jaccard.", exc)
        return _kmer_jaccard_hits(query, reference, min_identity=min_identity)

    hits: list[HomologyHit] = []
    if not out.exists():
        return hits
    with open(out) as fh:
        for line in fh:
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue
            q, t, pident, alnlen = parts[:4]
            try:
                ident = float(pident)
                if ident > 1.5:
                    ident /= 100.0   # MMseqs2 emits 0-100 by default.
                hits.append(HomologyHit(q, t, ident, int(alnlen)))
            except ValueError:
                continue
    return hits


def mmseqs_easy_cluster(
    seqs: Mapping[str, str],
    *,
    min_identity: float = 0.5,
    coverage: float = 0.8,
    workdir: str | Path | None = None,
) -> dict[str, str]:
    """Run ``mmseqs easy-cluster`` and return ``pid → cluster_representative_pid``.

    Used by the Stage-5 cluster-weighted sampler to flatten protein-space
    crowding (so a 200-protein kinase family doesn't dominate gradient mass).

    The cluster file is the standard MMseqs2 output (``<prefix>_cluster.tsv``)
    with two columns: ``representative\tmember``. When ``mmseqs`` is missing
    from PATH this falls back to single-linkage clustering on the k-mer
    Jaccard hits, so smoke tests still work — accuracy is approximate.
    """
    if not seqs:
        return {}
    if not _mmseqs_available():
        enforce_mmseqs("not found on PATH")
        logger.warning(
            "mmseqs not on PATH; falling back to k-mer Jaccard single-link "
            "clustering. Cluster boundaries will be approximate."
        )
        return _kmer_jaccard_cluster(seqs, min_identity=min_identity)

    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattice_mmseqs_cluster_"))
    workdir.mkdir(parents=True, exist_ok=True)
    fasta = workdir / "seqs.fasta"
    prefix = workdir / "cluster"
    tmp = workdir / "tmp"
    tmp.mkdir(exist_ok=True)
    _write_fasta(seqs, fasta)

    cluster_tsv = prefix.with_name(prefix.name + "_cluster.tsv")
    if not cluster_tsv.exists():
        cmd = [
            "mmseqs", "easy-cluster",
            str(fasta), str(prefix), str(tmp),
            "--min-seq-id", f"{min_identity:.3f}",
            "-c", f"{coverage:.3f}",
            "--cov-mode", "0",
            "-s", "7.5",
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            enforce_mmseqs("easy-cluster failed", cause=exc)
            logger.error("mmseqs easy-cluster failed (%s); falling back to k-mer Jaccard.", exc)
            return _kmer_jaccard_cluster(seqs, min_identity=min_identity)

    pid_to_rep: dict[str, str] = {}
    with open(cluster_tsv) as fh:
        for line in fh:
            parts = line.rstrip().split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            pid_to_rep[member] = rep
    # Every pid must be present (mmseqs guarantees this) — singletons cluster to themselves.
    for pid in seqs:
        pid_to_rep.setdefault(pid, pid)
    return pid_to_rep


def _kmer_jaccard_cluster(
    seqs: Mapping[str, str], *, min_identity: float
) -> dict[str, str]:
    """Single-link clustering on k-mer Jaccard ≥ min_identity. O(N²) — only for fallback."""
    pids = list(seqs.keys())
    kmer_sets = {p: _kmers(seqs[p]) for p in pids}
    parent = {p: p for p in pids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, p in enumerate(pids):
        for q in pids[i + 1:]:
            inter = len(kmer_sets[p] & kmer_sets[q])
            if inter == 0:
                continue
            union_size = len(kmer_sets[p] | kmer_sets[q])
            if union_size and inter / union_size >= min_identity:
                union(p, q)
    return {p: find(p) for p in pids}


def max_identity_per_query(hits: Iterable[HomologyHit]) -> dict[str, float]:
    """Reduce hits to ``query -> max(identity)``."""
    out: dict[str, float] = {}
    for h in hits:
        if h.identity > out.get(h.query, 0.0):
            out[h.query] = h.identity
    return out


def excluded_at(max_id: Mapping[str, float], threshold: float) -> set[str]:
    """Return the set of query ids whose max identity is ``>= threshold``.

    ``>=`` matches DrugCLIP's wording ("filtered at X% identity"): a query
    *exactly* at the threshold is considered homologous and removed.
    """
    return {q for q, v in max_id.items() if v >= threshold}


# ---------------------------------------------------------------------------
# Fallback: deterministic k-mer Jaccard. Tests / smoke runs only.
# ---------------------------------------------------------------------------


def _kmers(seq: str, k: int = 5) -> set[str]:
    if len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _kmer_jaccard_hits(
    query: Mapping[str, str],
    reference: Mapping[str, str],
    *,
    min_identity: float,
    k: int = 5,
) -> list[HomologyHit]:
    """Approximate identity via k-mer Jaccard (over-estimates for short seqs).

    This is a *floor* — Jaccard tends to be smaller than true identity at the
    same k. We use a conservative scale factor so a 90 % cutoff still catches
    near-duplicates without false negatives. Use only for tests.
    """
    ref_kmers = {pid: _kmers(seq, k) for pid, seq in reference.items()}
    hits: list[HomologyHit] = []
    for q_pid, q_seq in query.items():
        q_kmers = _kmers(q_seq, k)
        if not q_kmers:
            continue
        for r_pid, r_kmers in ref_kmers.items():
            if not r_kmers:
                continue
            inter = len(q_kmers & r_kmers)
            union = len(q_kmers | r_kmers)
            jacc = inter / union
            # Heuristic conversion to "identity-like" scale.
            ident = min(1.0, jacc * 1.5)
            if ident >= min_identity:
                hits.append(HomologyHit(q_pid, r_pid, ident, min(len(q_seq), 1024)))
    return hits
