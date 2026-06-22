"""MMseqs2 cluster-weighted protein sampler (extracted from the original trainer).

Per-row weight ``1 / (n_rows_for_protein * sqrt(cluster_size))`` so each protein
contributes equal mass and crowded clusters (e.g. kinases) are sqrt-downweighted.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from lattice_lab.preprocessing.homology import mmseqs_easy_cluster

logger = logging.getLogger(__name__)


def build_cluster_weighted_sampler(
    train_parquet: Path,
    row_uniprots: list[str],
    *,
    min_identity: float,
    cache_dir: Path,
    seed: int,
) -> WeightedRandomSampler:
    cache_dir = Path(cache_dir) / f"cluster_{int(round(min_identity * 100)):02d}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    unique_uniprots = sorted(set(row_uniprots))
    df = pd.read_parquet(train_parquet, columns=["uniprot", "sequence"])
    df = df.drop_duplicates("uniprot")
    seqs_all = dict(zip(df["uniprot"].astype(str), df["sequence"].astype(str)))
    seqs = {u: seqs_all[u] for u in unique_uniprots if u in seqs_all}
    n_missing = len(unique_uniprots) - len(seqs)
    if n_missing:
        logger.warning(
            "cluster-weighted sampler: %d/%d train uniprots had no sequence in the parquet",
            n_missing, len(unique_uniprots),
        )

    logger.info(
        "clustering %d train uniprots with MMseqs2 (min_seq_id=%.2f, cache=%s)",
        len(seqs), min_identity, cache_dir,
    )
    pid_to_rep = mmseqs_easy_cluster(seqs, min_identity=min_identity, workdir=cache_dir)
    cluster_size = Counter(pid_to_rep.values())
    sizes = np.array(list(cluster_size.values()), dtype=np.float64)
    logger.info(
        "cluster summary: %d clusters; size min=%d median=%d max=%d mean=%.2f",
        len(cluster_size), int(sizes.min()), int(np.median(sizes)),
        int(sizes.max()), float(sizes.mean()),
    )

    row_count = Counter(row_uniprots)
    weights: list[float] = []
    for u in row_uniprots:
        rep = pid_to_rep.get(u)
        c_size = cluster_size[rep] if rep is not None else 1
        weights.append(1.0 / (row_count[u] * math.sqrt(c_size)))

    by_size = sorted(cluster_size.items(), key=lambda kv: -kv[1])[:5]
    logger.info(
        "top-5 most-crowded cluster reps (rep, size, per-protein weight): %s",
        [(rep, sz, round(1.0 / math.sqrt(sz), 4)) for rep, sz in by_size],
    )

    gen = torch.Generator().manual_seed(seed + 13)
    return WeightedRandomSampler(
        weights=weights, num_samples=len(row_uniprots), replacement=True, generator=gen
    )
