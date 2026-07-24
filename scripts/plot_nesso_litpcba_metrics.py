"""Plot per-target Nesso-1 metrics for the LIT-PCBA subsample."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = REPO / "artifacts/nesso/litpcba_sub100/metrics.csv"
DEFAULT_OUTPUT = REPO / "report/figures/nesso_litpcba_metrics"

PANELS = (
    ("auroc", "AUROC", 0.5),
    ("bedroc", "BEDROC (α = 80.5)", None),
    ("ef@1%", "EF@1%", 1.0),
)
LABELS = {"p_bind": "Binding probability", "affinity": "Predicted affinity"}
COLORS = {"p_bind": "#0072B2", "affinity": "#D55E00"}


def make_figure(metrics: pd.DataFrame, output: Path) -> None:
    required = {"target", "score"} | {column for column, _, _ in PANELS}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    metrics = metrics.loc[metrics.score.isin(LABELS)].copy()
    if metrics.empty:
        raise ValueError("no p_bind or affinity rows")

    order = (
        metrics.loc[metrics.score.eq("p_bind")]
        .sort_values("bedroc", ascending=True)
        .target.tolist()
    )
    if not order:
        order = sorted(metrics.target.unique())
    y = np.arange(len(order))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "axes.linewidth": 0.8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.8), sharey=True)

    for ax, (column, title, baseline) in zip(axes, PANELS, strict=True):
        wide = metrics.pivot(index="target", columns="score", values=column).reindex(order)
        if set(LABELS).issubset(wide.columns):
            for yi, left, right in zip(y, wide.p_bind, wide.affinity, strict=True):
                if np.isfinite(left) and np.isfinite(right):
                    ax.plot([left, right], [yi, yi], color=".82", lw=0.8, zorder=1)
        for score, label in LABELS.items():
            if score in wide:
                ax.scatter(wide[score], y, s=22, color=COLORS[score], label=label, zorder=2)
        if baseline is not None:
            ax.axvline(baseline, color=".45", ls="--", lw=0.8, zorder=0)
        ax.set_xlabel(title)
        ax.grid(axis="x", color=".9", lw=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y, order)
    fig.suptitle(f"Nesso-1 on LIT-PCBA subsample ({len(order)} targets)", y=0.98, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               frameon=False, ncol=2)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.10, top=0.85, wspace=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    table = pd.read_csv(args.metrics)
    make_figure(table, args.output)
    print(f"wrote {args.output}.pdf and {args.output}.png")
