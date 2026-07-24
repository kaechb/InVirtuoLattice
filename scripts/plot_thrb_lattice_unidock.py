"""Publication figure: LATTICE / Uni-Dock / Boltz-2 on the shared THRB set."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from lattice_lab.eval.metrics import auroc, bedroc, ef_at_k

# Colorblind-safe: blue / vermillion / bluish green (Wong)
METHODS = {
    "LATTICE": {"column": "energy", "negate": True, "color": "#0072B2"},
    "Uni-Dock": {"column": "docking_score_kcal_mol", "negate": True, "color": "#D55E00"},
    "Boltz-2": {"column": "boltz_score", "negate": False, "color": "#009E73"},
}


def _rank_score(shared: pd.DataFrame, style: dict) -> np.ndarray:
    x = shared[style["column"]].to_numpy(dtype=float)
    return -x if style["negate"] else x


def load_shared_cohort(
    lattice_path: Path, unidock_path: Path, boltz_path: Path,
) -> pd.DataFrame:
    lattice = pd.read_parquet(lattice_path)
    lattice = lattice.loc[
        (lattice.target == "THRB") & lattice.valid & lattice.parse_ok & lattice.energy.notna()
    ]
    unidock = pd.read_parquet(unidock_path)
    shared = lattice.merge(
        unidock[["ivlid", "docking_score_kcal_mol"]],
        left_on="id", right_on="ivlid", validate="one_to_one",
    )
    shared = shared.loc[np.isfinite(shared.docking_score_kcal_mol)].copy()
    boltz = pd.read_parquet(boltz_path)
    boltz = boltz.loc[boltz.status.eq("ok") & boltz.score.notna(), ["ivlid", "score"]]
    shared = shared.merge(boltz.rename(columns={"score": "boltz_score"}), on="ivlid", validate="one_to_one")
    shared = shared.loc[np.isfinite(shared.boltz_score)].copy()
    if shared.empty or shared.is_binder.nunique() != 2:
        raise ValueError("Shared cohort must contain finite scores and both classes")
    return shared


def comparison_metrics(shared: pd.DataFrame) -> pd.DataFrame:
    y = shared.is_binder.to_numpy(dtype=int)
    rows = []
    for method, style in METHODS.items():
        score = _rank_score(shared, style)
        rows.append({
            "method": method,
            "AUROC": auroc(y, score),
            "BEDROC": bedroc(y, score, alpha=80.5),
            "EF@0.5%": ef_at_k(y, score, 0.5),
            "EF@1%": ef_at_k(y, score, 1.0),
            "EF@5%": ef_at_k(y, score, 5.0),
        })
    return pd.DataFrame(rows).set_index("method")


def _enrichment_curve(y, score):
    ranked = y[np.argsort(-score, kind="stable")]
    screened = np.arange(1, len(ranked) + 1) / len(ranked)
    enrichment = np.cumsum(ranked) / np.arange(1, len(ranked) + 1) / y.mean()
    return screened * 100, enrichment


def make_comparison_figure(shared: pd.DataFrame, output_stem: Path) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "axes.linewidth": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    y = shared.is_binder.to_numpy(dtype=int)
    metrics = comparison_metrics(shared)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), gridspec_kw={"width_ratios": [1, 1.15, 1.05]})

    ax = axes[0]
    for method, style in METHODS.items():
        fpr, tpr, _ = roc_curve(y, _rank_score(shared, style))
        ax.plot(fpr, tpr, lw=1.8, color=style["color"], label=method)
    ax.plot([0, 1], [0, 1], color=".65", lw=0.9, ls="--", label="Random")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="a  Global ranking")
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.98, 0.20, "AUROC", transform=ax.transAxes, ha="right", va="bottom", color=".25", fontsize=7.5)
    for i, method in enumerate(METHODS):
        ax.text(
            0.98, 0.145 - 0.055 * i,
            f"{method}  {metrics.loc[method, 'AUROC']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            color=METHODS[method]["color"], fontsize=7.5,
        )

    ax = axes[1]
    for method, style in METHODS.items():
        screened, enrichment = _enrichment_curve(y, _rank_score(shared, style))
        keep = screened <= 10
        ax.plot(screened[keep], enrichment[keep], lw=1.8, color=style["color"], label=method)
    ax.axhline(1, color=".65", lw=0.9, ls="--")
    ax.set(xlabel="Library screened (%)", ylabel="Enrichment factor", title="b  Early enrichment", xlim=(0, 10))

    ax = axes[2]
    names = ["EF@0.5%", "EF@1%", "EF@5%"]
    x = np.arange(3)
    width = 0.26
    offsets = np.linspace(-(len(METHODS) - 1) / 2, (len(METHODS) - 1) / 2, len(METHODS)) * width
    for offset, (method, style) in zip(offsets, METHODS.items()):
        bars = ax.bar(x + offset, metrics.loc[method, names].to_numpy(float), width, color=style["color"], label=method)
        ax.bar_label(bars, fmt="%.1f", padding=1, fontsize=6.5)
    ax.set_xticks(x, ["0.5%", "1%", "5%"])
    ax.set(xlabel="Top-ranked fraction", ylabel="Enrichment factor", title="c  Fixed cutoffs")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=4, frameon=False, handlelength=2.2, columnspacing=1.6)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.21, top=0.76, wspace=0.48)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    shared = load_shared_cohort(
        repo / "artifacts/predictions/chembl_decoys_thrb_scores_w790kdrh.parquet",
        repo / "notebooks/aggregated_lowest_score_per_ligand.parquet",
        repo / "thrb_3imy_boltz_scores_seed_1.parquet",
    )
    metrics = comparison_metrics(shared)
    print(f"Shared cohort: n={len(shared):,}; binders={shared.is_binder.sum():,}")
    print(metrics.round(3).to_string())
    # one check: Boltz EF@1% matches the previously computed shared-cohort number
    assert abs(metrics.loc["Boltz-2", "EF@1%"] - 3.097) < 0.01
    make_comparison_figure(shared, repo / "report/figures/thrb_lattice_unidock_comparison")
