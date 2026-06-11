"""Ranking-metric helpers shared by Stage-6 evaluation.

All functions take ``y_true`` (binary, 1 = active) and ``y_score`` (higher =
more likely active). NaN is returned in pathological cases (e.g. no positives
or no negatives) so the caller can drop those targets when averaging.
"""

from __future__ import annotations

import math

import numpy as np


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC. Returns NaN when only one class is present."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        return float("nan")
    if y_true.sum() in (0, y_true.size):
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def ef_at_k(y_true: np.ndarray, y_score: np.ndarray, percent: float) -> float:
    """Enrichment factor at the top ``percent`` % of the ranked list.

    EF = (fraction of top-k that are active) / (overall active rate).
    Random ≈ 1.0. NaN if no actives or the list is empty.
    """
    if percent <= 0 or percent > 100:
        raise ValueError(f"percent must be in (0, 100], got {percent}")
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = y_true.size
    if n == 0:
        return float("nan")
    n_active = int(y_true.sum())
    if n_active == 0:
        return float("nan")
    k = max(1, int(round(n * percent / 100.0)))
    order = np.argsort(-y_score, kind="stable")
    top_active = int(y_true[order[:k]].sum())
    return (top_active / k) / (n_active / n)


def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 80.5) -> float:
    """BEDROC (Truchon & Bayly, JCIM 2007) — early-recognition metric.

    BEDROC ∈ [0, 1]: 1 = all actives ranked first; 0 = all ranked last;
    ``≈ R_a = N_actives/N`` at random ordering. Higher α emphasises the top
    of the list; α=80.5 is the convention used by DUD-E / DrugCLIP.

    Implementation follows RDKit's ``CalcBEDROC`` exactly: compute the RIE on
    the actives' ranks, then normalise by the analytic min/max RIE so the
    result lives in ``[0, 1]``.

    NaN if ``y_true`` is all-positive or all-negative.
    """
    y_true = (np.asarray(y_true) > 0).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = y_true.size
    if n == 0:
        return float("nan")
    n_active = int(y_true.sum())
    if n_active in (0, n):
        return float("nan")
    R_a = n_active / n

    order = np.argsort(-y_score, kind="stable")
    sorted_labels = y_true[order]
    ranks = np.flatnonzero(sorted_labels) + 1  # 1-indexed positions in the ranked list

    # RIE = mean active-weight divided by expected weight under random ranking.
    denom = (1.0 / n) * ((1.0 - math.exp(-alpha)) / (math.exp(alpha / n) - 1.0))
    sum_exp = float(np.sum(np.exp(-alpha * ranks / n)))
    rie = sum_exp / (n_active * denom)

    # Closed-form bounds.
    rie_max = (1.0 - math.exp(-alpha * R_a)) / (R_a * (1.0 - math.exp(-alpha)))
    rie_min = (1.0 - math.exp(alpha * R_a)) / (R_a * (1.0 - math.exp(alpha)))
    if rie_max == rie_min:  # degenerate (n_active == n)
        return 1.0
    return float((rie - rie_min) / (rie_max - rie_min))
