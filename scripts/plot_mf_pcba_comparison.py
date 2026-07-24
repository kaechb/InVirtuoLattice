"""Plot LATTICE MF-PCBA metrics against Nesso-1 paper Figure 4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    REPO / "artifacts/benchmarks/mf_pcba/lattice_wk8denar_ensemble_seed0.json"
)
DEFAULT_OUTPUT = REPO / "report/figures/mf_pcba_lattice_nesso_comparison"

# Nesso AP is printed in Figure 4. Its EF values are digitized from the plot;
# Boltz-2 values are exact values from Table 13 of the Boltz-2 paper.
PAPER = {
    "Nesso-1": {"ap": 0.031, "ef@0.5%": 17.8, "ef@1.0%": 13.8,
                "ef@2.0%": 11.0, "ef@5.0%": 6.8},
    "Boltz-2": {"ap": 0.0248, "ef@0.5%": 18.3916, "ef@1.0%": 13.9540,
                "ef@2.0%": 10.5706, "ef@5.0%": 7.0448},
}


def make_plot(results: Path, output: Path) -> None:
    lattice = json.loads(results.read_text())
    rows = [{
        "method": "LATTICE",
        "ap": lattice["mean/ap"],
        **{f"ef@{p}%": lattice[f"mean/ef@{p}%"] for p in ("0.5", "1.0", "2.0", "5.0")},
    }]
    rows.extend({"method": method, **metrics} for method, metrics in PAPER.items())
    table = pd.DataFrame(rows).set_index("method")

    colors = {"LATTICE": "#0072B2", "Nesso-1": "#0046CC", "Boltz-2": "#999999"}
    fig, (ap_ax, ef_ax) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    methods = list(table.index)
    ap_ax.bar(methods, table.ap, color=[colors[m] for m in methods], width=0.65)
    ap_ax.set(ylabel="Average precision", title="Mean AP")
    ap_ax.tick_params(axis="x", rotation=20)

    percentiles = ("5.0", "2.0", "1.0", "0.5")
    for method in methods:
        ef_ax.plot(
            [f"{p.rstrip('0').rstrip('.')}%" for p in percentiles],
            [table.loc[method, f"ef@{p}%"] for p in percentiles],
            marker="o", lw=1.8, color=colors[method], label=method,
        )
    ef_ax.set(ylabel="Enrichment factor", xlabel="Top-ranked percentile",
              title="Mean enrichment")
    ef_ax.legend(frameon=False)

    for ax in (ap_ax, ef_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".9", lw=0.6)
    fig.suptitle("MF-PCBA comparison", fontsize=10)
    fig.text(
        0.5, -0.03,
        "LATTICE: 8 public assays; Nesso-1/Boltz-2: 10 public assays with an "
        "unreleased exact 50k sample. Nesso-1 EF values digitized from Figure 4.",
        ha="center", fontsize=7,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    table.to_csv(output.with_suffix(".csv"))
    print(f"wrote {output}.pdf, .png, and .csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    make_plot(args.results, args.output)
