"""Nesso-1 binding-affinity histogram on THRβ WS1 + Rapposelli.

Ranks by affinity_pred_value = log10(IC50 / µM); lower is stronger.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SOURCES = ("WS1_v1", "WS1_v2", "Rapposelli")


def load() -> pd.DataFrame:
    lat = pd.read_parquet(REPO / "artifacts/predictions/chembl_decoys_thrb_scores_w790kdrh.parquet")
    lat = lat.loc[
        lat.source.isin(SOURCES) & lat.valid & lat.parse_ok,
        ["id", "source", "is_binder"],
    ]
    nesso = pd.read_parquet(REPO / "artifacts/nesso/thrb_ws_rapposelli_nesso_scores.parquet")
    nesso = nesso.loc[nesso.status.eq("ok") & nesso.affinity_pred_value.notna()]
    df = lat.merge(
        nesso[["ivlid", "affinity_pred_value", "affinity_probability_binary"]],
        left_on="id", right_on="ivlid", how="inner", validate="one_to_one",
    )
    assert int(df.is_binder.sum()) >= 1
    return df


def _rank_lower_better(x: np.ndarray, xb: np.ndarray) -> np.ndarray:
    return 1 + np.sum(x[:, None] < xb[None, :], axis=0)


def main() -> None:
    df = load()
    x = df.affinity_pred_value.to_numpy(float)
    y = df.is_binder.astype(bool).to_numpy()
    xb = x[y]
    ranks = _rank_lower_better(x, xb)
    best = int(ranks.min())

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "axes.linewidth": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.hist(x, bins=24, color="#CC79A7", edgecolor="white", linewidth=0.5, alpha=0.85)
    vline = None
    for xi in xb:
        vline = ax.axvline(xi, color=".2", lw=1.2, ls="--", alpha=0.9)
    lo, hi = float(x.min()), float(x.max())
    pad = 0.08 * (hi - lo) if hi > lo else 0.1
    ax.set_xlim(lo - pad, hi + pad)
    ax.set(
        xlabel=r"$\log_{10}(\mathrm{IC}_{50}/\mu\mathrm{M})$  ($\downarrow$ better)",
        ylabel="Count",
        title=f"Nesso-1 affinity  (best rank {best}/{len(x)}; {int(y.sum())} binders)",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        [vline], ["labelled binders"],
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        frameon=False, handlelength=2.2,
    )
    fig.suptitle(f"THR$\\beta$ WS1 + Rapposelli  ($n={len(df)}$)", y=1.10, fontsize=10)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.78)
    stem = REPO / "report/figures/thrb_nesso_affinity_hist"
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    for ext in (".pdf", ".png"):
        shutil.copy2(stem.with_suffix(ext), REPO / "report" / f"thrb_nesso_affinity_hist{ext}")
    print(f"wrote {stem}.pdf")
    print("binder affinity_pred_value + ranks:")
    for src, xi, r in zip(df.loc[y, "source"], xb, ranks, strict=True):
        print(f"  {src:12s}  aff={xi:7.3f}  rank={int(r)}/{len(x)}")
    assert best >= 1


if __name__ == "__main__":
    main()
