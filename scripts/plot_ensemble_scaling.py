"""Publication figure: LIT-PCBA BEDROC vs energy-ensemble size (from ensemble_scaling.json)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Match scripts/plot_thrb_lattice_unidock.py
C0, C1, C2 = "#0072B2", "#D55E00", "#009E73"


def make_figure(data: dict, output_stem: Path) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "axes.linewidth": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    scaling = data["scaling"]
    sequential = data["sequential"]
    ks = [r["k"] for r in scaling]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45))
    for ax, metric, title, ylabel in (
        (axes[0], "bedroc", "a  Early recognition", "Mean BEDROC"),
        (axes[1], "auroc", "b  Global ranking", "Mean AUROC"),
    ):
        mean = np.asarray([r[f"mean/{metric}"] for r in scaling])
        std = np.asarray([r[f"std/{metric}"] for r in scaling])
        best = np.asarray([r[f"max/{metric}"] for r in scaling])
        seq = np.asarray([r[metric] for r in sequential])
        ax.fill_between(ks, mean - std, mean + std, color=C0, alpha=0.18, linewidth=0)
        ax.plot(ks, mean, "o-", color=C0, lw=1.8, ms=4.5, label="all subsets (mean ± std)")
        ax.plot(ks, seq, "s--", color=C1, lw=1.8, ms=4.5, label="seeds 0..$k{-}1$")
        ax.plot(ks, best, "^:", color=C2, lw=1.6, ms=4.5, label="best subset (post hoc)")
        ax.set(xlabel="Ensemble size $k$", ylabel=ylabel, title=title, xticks=ks)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.04),
               ncol=3, frameon=False, handlelength=2.2, columnspacing=1.6)
    fig.subplots_adjust(left=.09, right=.995, bottom=.22, top=.72, wspace=.34)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    src = repo / "artifacts/ablation/evaluation/w790kdrh/ensemble_scaling.json"
    data = json.loads(src.read_text())
    make_figure(data, repo / "report/figures/ensemble_scaling")
    # one check: k=1 mean BEDROC equals average of per-seed BEDROCs
    k1 = next(r for r in data["scaling"] if r["k"] == 1)
    assert abs(k1["mean/bedroc"] - np.mean([s["mean/bedroc"] for s in data["per_seed"]])) < 1e-9
    print(f"n_seeds={data['n_seeds']}  wrote report/figures/ensemble_scaling.{{pdf,png}}")
