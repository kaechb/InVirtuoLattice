#!/usr/bin/env python3
"""Ensemble-size scaling on LIT-PCBA (energy averaging).

Scores each seed head once, then for every ensemble size ``k=1..N`` reports
mean±std BEDROC/AUROC/EF over all subsets of size ``k`` (exact combinations),
plus the sequential curve using seeds ``0..k-1`` in submission order.

    python -m lattice_lab.eval.ensemble_scaling \\
        --ckpts artifacts/ablation/energy/checkpoints/<id0>/…/last.ckpt ... \\
        --zm-cache artifacts/ablation/evaluation/<id0>/lit_pcba_zm_mv4 \\
        --protein-store artifacts/protein_store/embeddings/esm2_650M \\
        --out artifacts/ablation/evaluation/w790kdrh/ensemble_scaling.json \\
        --plot artifacts/ablation/evaluation/w790kdrh/ensemble_scaling.png
"""
from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lattice_lab.eval.ensemble_eval import _head_energies, _load_head
from lattice_lab.eval.lit_pcba import (
    N_VIEWS_KEY,
    _inchikey_or_none,
    enforce_cache_adapter,
    enforce_cache_n_views,
)
from lattice_lab.eval.metrics import auroc, bedroc, ef_at_k
from lattice_lab.models.builders import adapter_fingerprint
from lattice_lab.protein.store import EmbeddingStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_METRIC_COLS = ("auroc", "bedroc", "ef@0.5%", "ef@1.0%", "ef@5.0%")


def _target_metrics(y_true: np.ndarray, y_score: np.ndarray, alpha: float) -> dict[str, float]:
    return {
        "auroc": auroc(y_true, y_score),
        "bedroc": bedroc(y_true, y_score, alpha=alpha),
        "ef@0.5%": ef_at_k(y_true, y_score, 0.5),
        "ef@1.0%": ef_at_k(y_true, y_score, 1.0),
        "ef@5.0%": ef_at_k(y_true, y_score, 5.0),
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {m: float(np.mean([r[m] for r in rows])) for m in _METRIC_COLS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--zm-cache", type=Path, required=True)
    ap.add_argument("--protein-store", type=Path, required=True)
    ap.add_argument(
        "--test-parquet",
        type=Path,
        default=Path("artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet"),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--bedroc-alpha", type=float, default=80.5)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d-adapter", type=int, default=None)
    ap.add_argument("--d-protein", type=int, default=None)
    ap.add_argument(
        "--max-combos",
        type=int,
        default=0,
        help="cap random combos per k (0 = all C(n,k); useful if n grows)",
    )
    a = ap.parse_args()

    n = len(a.ckpts)
    if n < 1:
        raise SystemExit("need at least one --ckpts")

    df = pd.read_parquet(a.test_parquet, columns=["target_name", "smiles", "is_active"])
    df["target_name"] = df["target_name"].astype(str)
    df["smiles"] = df["smiles"].astype(str)
    df["is_active"] = df["is_active"].astype(int)

    pstore = EmbeddingStore.open(a.protein_store, mode="r")
    present = [t for t in sorted(df["target_name"].unique()) if t in pstore.pid_to_row]
    work = df[df["target_name"].isin(present)].copy()

    logger.info("computing InChIKeys for %d rows (n_jobs=%d)", len(work), a.n_jobs)
    if a.n_jobs in (0, 1):
        keys = [_inchikey_or_none(s) for s in work["smiles"]]
    else:
        from joblib import Parallel, delayed
        keys = list(Parallel(n_jobs=a.n_jobs, backend="loky")(
            delayed(_inchikey_or_none)(s) for s in work["smiles"]))
    work["inchikey"] = keys
    work = work.dropna(subset=["inchikey"]).reset_index(drop=True)

    zm = EmbeddingStore.open(a.zm_cache, mode="r")
    recorded_views = zm.manifest.extra.get(N_VIEWS_KEY)
    if recorded_views is None or int(recorded_views) < 2:
        raise SystemExit(f"{a.zm_cache} is not a multi-view cache")
    enforce_cache_n_views(zm, int(recorded_views))
    work = work[work["inchikey"].isin(zm.pid_to_row)].reset_index(drop=True)

    fps = [adapter_fingerprint(c) for c in a.ckpts]
    if len(set(fps)) > 1:
        raise ValueError("ensemble ckpts use different adapters")
    enforce_cache_adapter(zm, a.ckpts[0])

    heads = [_load_head(c, a.d_adapter, a.d_protein, a.device) for c in a.ckpts]

    # Per target: energies [n_heads, L]
    target_data: list[tuple[np.ndarray, np.ndarray]] = []
    for t in present:
        sub = work[work["target_name"] == t]
        if sub.empty:
            continue
        idx = np.fromiter((zm.pid_to_row[k] for k in sub["inchikey"]),
                          dtype=np.int64, count=len(sub))
        z_m = np.asarray(zm.mean_array[idx], dtype=np.float32)
        z_p = np.asarray(pstore.get_mean(t), dtype=np.float32)
        y_true = sub["is_active"].to_numpy(dtype=int)
        energies = _head_energies(heads, z_m, z_p, a.device)
        target_data.append((y_true, energies))
        logger.info("%-8s scored n=%d n_a=%d", t, y_true.size, int(y_true.sum()))

    rng = np.random.default_rng(0)
    scaling: list[dict] = []
    sequential: list[dict] = []

    for k in range(1, n + 1):
        combos = list(combinations(range(n), k))
        if a.max_combos > 0 and len(combos) > a.max_combos:
            pick = rng.choice(len(combos), size=a.max_combos, replace=False)
            combos = [combos[i] for i in pick]

        subset_summaries: list[dict[str, float]] = []
        for combo in combos:
            rows = []
            for y_true, energies in target_data:
                y_score = -energies[list(combo)].mean(axis=0)
                rows.append(_target_metrics(y_true, y_score, a.bedroc_alpha))
            subset_summaries.append(_mean_metrics(rows))

        mat = {m: np.asarray([s[m] for s in subset_summaries], dtype=float) for m in _METRIC_COLS}
        entry = {
            "k": k,
            "n_combos": len(combos),
            **{f"mean/{m}": float(mat[m].mean()) for m in _METRIC_COLS},
            **{f"std/{m}": float(mat[m].std(ddof=1) if len(combos) > 1 else 0.0)
               for m in _METRIC_COLS},
            **{f"min/{m}": float(mat[m].min()) for m in _METRIC_COLS},
            **{f"max/{m}": float(mat[m].max()) for m in _METRIC_COLS},
        }
        scaling.append(entry)
        logger.info(
            "k=%d n_combos=%d bedroc=%.4f±%.4f [% .4f, %.4f]",
            k, entry["n_combos"], entry["mean/bedroc"], entry["std/bedroc"],
            entry["min/bedroc"], entry["max/bedroc"],
        )

        # Fixed order: first k seeds as submitted (ebm.0 .. ebm.k-1).
        rows = []
        for y_true, energies in target_data:
            y_score = -energies[:k].mean(axis=0)
            rows.append(_target_metrics(y_true, y_score, a.bedroc_alpha))
        sequential.append({"k": k, **_mean_metrics(rows)})

    per_seed = []
    for hi, ckpt in enumerate(a.ckpts):
        rows = []
        for y_true, energies in target_data:
            rows.append(_target_metrics(y_true, -energies[hi], a.bedroc_alpha))
        per_seed.append({
            "seed": hi,
            "run_id": ckpt.parent.name,
            "ckpt": str(ckpt),
            **{f"mean/{m}": float(np.mean([r[m] for r in rows])) for m in _METRIC_COLS},
        })

    output = {
        "n_seeds": n,
        "n_targets": len(target_data),
        "per_seed": per_seed,
        "scaling": scaling,
        "sequential": sequential,
        "ckpts": [str(c) for c in a.ckpts],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2))
    logger.info("wrote %s", a.out)

    if a.plot is not None:
        _plot(scaling, sequential, a.plot)
        logger.info("wrote %s", a.plot)

    print(json.dumps(output, indent=2))


def _plot(scaling: list[dict], sequential: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in scaling]
    mean_b = [r["mean/bedroc"] for r in scaling]
    std_b = [r["std/bedroc"] for r in scaling]
    seq_b = [r["bedroc"] for r in sequential]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.errorbar(
        ks, mean_b, yerr=std_b, fmt="o-", capsize=3, linewidth=1.5,
        label="all subsets (mean ± std)",
    )
    ax.plot(ks, seq_b, "s--", linewidth=1.5, label="sequential seeds 0..k-1")
    ax.set_xlabel("ensemble size k")
    ax.set_ylabel("mean BEDROC (LIT-PCBA)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Energy-averaging ensemble scaling")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # one check: mean of two identical subsets collapses to that score
    assert _mean_metrics([{"auroc": 1.0, "bedroc": 0.2, "ef@0.5%": 1.0, "ef@1.0%": 1.0, "ef@5.0%": 1.0}] * 2)["bedroc"] == 0.2
    main()
