"""Histograms on THRβ WS1 + Rapposelli; vlines mark all labelled binders."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCES = ("WS1_v1", "WS1_v2", "Rapposelli")

STYLES = {
    "LATTICE": {
        "column": "energy", "lower_better": True, "color": "#0072B2",
        "xlabel": r"Energy ($\downarrow$ better)",
    },
    "Uni-Dock": {
        "column": "docking_score_kcal_mol", "lower_better": True, "color": "#D55E00",
        "xlabel": r"Energy ($\downarrow$ better)",
    },
    "Boltz-2": {
        "column": "boltz_score", "lower_better": False, "color": "#009E73",
        "xlabel": r"Score ($\uparrow$ better)",
    },
    "Nesso-1": {
        "column": "nesso_score", "lower_better": False, "color": "#CC79A7",
        "xlabel": r"Score ($\uparrow$ better)",
    },
}


def load_ws(repo: Path, unidock_path: Path | None) -> pd.DataFrame:
    lattice = pd.read_parquet(repo / "artifacts/predictions/chembl_decoys_thrb_scores_w790kdrh.parquet")
    lattice = lattice.loc[
        lattice.source.isin(SOURCES) & lattice.valid & lattice.parse_ok & lattice.energy.notna()
    ].copy()
    boltz = pd.read_parquet(repo / "thrb_3imy_boltz_scores_seed_1.parquet")
    boltz = boltz.loc[boltz.status.eq("ok") & boltz.score.notna(), ["ivlid", "score"]]
    df = lattice.merge(
        boltz.rename(columns={"score": "boltz_score"}),
        left_on="id", right_on="ivlid", how="inner", validate="one_to_one",
    )
    df = df.loc[np.isfinite(df.boltz_score)].copy()

    nesso_path = repo / "artifacts/nesso/thrb_ws_rapposelli_nesso_scores.parquet"
    if not nesso_path.is_file():
        nesso_path = repo / "artifacts/nesso/thrb_ws1_nesso_scores.parquet"
    nesso = pd.read_parquet(nesso_path)
    nesso = nesso.loc[nesso.status.eq("ok") & nesso.score.notna(), ["ivlid", "score"]]
    df = df.merge(
        nesso.rename(columns={"score": "nesso_score"}),
        left_on="id", right_on="ivlid", how="inner", validate="one_to_one",
        suffixes=("", "_n"),
    )
    df = df.loc[np.isfinite(df.nesso_score)].copy()

    # ponytail: two files share the name — repo-root has WS1/WS2/Rapposelli;
    # notebooks/ has ChEMBL+decoys only (used for the shared-cohort table).
    candidates = []
    if unidock_path is not None:
        candidates.append(unidock_path)
    candidates += [
        repo / "aggregated_lowest_score_per_ligand.parquet",
        repo / "notebooks/aggregated_lowest_score_per_ligand.parquet",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        unidock = pd.read_parquet(path)
        if "docking_score_kcal_mol" not in unidock.columns or "ivlid" not in unidock.columns:
            continue
        m = df.merge(
            unidock[["ivlid", "docking_score_kcal_mol"]],
            left_on="id", right_on="ivlid", how="left", suffixes=("", "_u"),
        )
        if np.isfinite(m["docking_score_kcal_mol"]).sum() > 0:
            df = m
            break

    if df.empty or int(df.is_binder.sum()) < 1:
        raise ValueError("cohort empty or has no binder")
    return df


def _rank(scores: np.ndarray, binder_scores: np.ndarray, *, lower_better: bool) -> np.ndarray:
    if lower_better:
        return 1 + np.sum(scores[:, None] < binder_scores[None, :], axis=0)
    return 1 + np.sum(scores[:, None] > binder_scores[None, :], axis=0)


def make_figure(ws: pd.DataFrame, output_stem: Path) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5,
        "axes.linewidth": 0.8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    panels = [
        (name, style) for name, style in STYLES.items()
        if style["column"] in ws.columns and np.isfinite(ws[style["column"]]).any()
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(9.2, 2.7), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    binder_mask = ws["is_binder"].astype(bool)
    vline = None
    n_b = int(binder_mask.sum())

    for ax, (name, style) in zip(axes, panels):
        col = style["column"]
        ok = np.isfinite(ws[col].to_numpy(float))
        x = ws.loc[ok, col].to_numpy(float)
        y = binder_mask.to_numpy()[ok]
        xb = x[y]
        ranks = _rank(x, xb, lower_better=style["lower_better"])
        ax.hist(x, bins=20, color=style["color"], edgecolor="white", linewidth=0.5, alpha=0.85)
        for xi in xb:
            vline = ax.axvline(xi, color=".2", lw=1.2, ls="--", alpha=0.9)
        lo, hi = float(np.min(x)), float(np.max(x))
        pad = 0.08 * (hi - lo) if hi > lo else 0.1
        ax.set_xlim(lo - pad, hi + pad)
        best = int(np.min(ranks))
        ax.set_xlabel(style["xlabel"])
        ax.set_title(f"{name}\nbest rank {best}/{len(x)}", fontsize=9, pad=4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)

    axes[0].set_ylabel("Count")
    fig.legend(
        [vline], ["labelled binders"],
        loc="upper center", bbox_to_anchor=(0.5, 1.04),
        frameon=False, ncol=1, handlelength=2.2,
    )
    fig.suptitle(
        f"THR$\\beta$ WS1 + Rapposelli  ($n={len(ws)}$, {n_b} binders)",
        y=1.16, fontsize=10,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.72, wspace=0.32)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unidock", type=Path, default=None,
                    help="optional parquet with ivlid + docking_score_kcal_mol")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    ws = load_ws(repo, a.unidock)
    stem = repo / "report/figures/thrb_ws2_energy_hist"
    make_figure(ws, stem)
    for ext in (".pdf", ".png"):
        shutil.copy2(stem.with_suffix(ext), repo / "report" / f"thrb_ws2_energy_hist{ext}")
    y = ws.is_binder.astype(bool).to_numpy()
    print(f"n={len(ws)} binders={int(y.sum())} by source:")
    print(ws.groupby("source").agg(n=("id", "count"), binders=("is_binder", "sum")).to_string())
    for name, style in STYLES.items():
        col = style["column"]
        if col not in ws.columns or not np.isfinite(ws.loc[y, col]).any():
            print(f"{name:10s}  SKIPPED")
            continue
        x = ws[col].to_numpy(float)
        ok = np.isfinite(x)
        ranks = _rank(x[ok], x[ok & y], lower_better=style["lower_better"])
        print(f"{name:10s}  ranks={sorted(int(r) for r in ranks)}  best={int(ranks.min())}/{ok.sum()}")
    assert int(ws.is_binder.sum()) == 4
    assert len(ws) == 218
