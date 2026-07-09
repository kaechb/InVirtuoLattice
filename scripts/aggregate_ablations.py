#!/usr/bin/env python3
"""Aggregate stage-6 LIT-PCBA results across pipeline runs into one table.

Walks logs/slurm/{ablation,pipeline}/<run>/ dirs, reads each run's config from
pipeline.env + stage2.train.args, resolves its stage-6 metrics (3-seed
ensemble_mv4.json, else single-seed lit_pcba_mv4.csv), and prints:
  * a per-run table sorted by mean BEDROC
  * one-factor-at-a-time group means (method, protein, pool, layers, view3d)
  * seed-variance summary for repeated configs

Run after jobs finish:  python scripts/aggregate_ablations.py --prefix abl_
No results yet is fine — it just prints what exists.

ponytail: config is read from the frozen stage2.train.args (source of truth),
falling back to pipeline.env; only the knobs this ablation round varies are parsed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = ["logs/slurm/ablation", "logs/slurm/pipeline"]

# stage2 encoder defaults (discrete_flow.yaml) — used when an arg isn't overridden.
DEFAULTS = {"adapter_pool": "attn", "adapter_n_layers": "2", "seed": "0"}


def parse_env(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def parse_stage2_args(path: Path) -> dict:
    """Extract the knobs this round varies from the frozen stage-2 Hydra args."""
    out = {}
    if not path.exists():
        return out
    for tok in path.read_text().split():
        k, _, v = tok.partition("=")
        if k == "model.encoder.adapter_pool":
            out["adapter_pool"] = v
        elif k == "model.encoder.adapter_n_layers":
            out["adapter_n_layers"] = v
        elif k == "model.ssl_loss":
            out["ssl_loss"] = v
        elif k == "seed":
            out["seed"] = v
        elif k == "experiment":
            out["experiment"] = v
    return out


def load_metrics(art_root: Path, ebm_ids: list[str]) -> dict | None:
    """3-seed ensemble JSON keyed by seed-0 id, else single-seed CSV mean."""
    if not ebm_ids:
        return None
    ev = art_root / "evaluation" / ebm_ids[0]
    js = ev / "ensemble_mv4.json"
    if js.exists():
        d = json.loads(js.read_text())
        return {
            "bedroc": d.get("mean/bedroc"),
            "bedroc_median": d.get("median/bedroc"),
            "auroc": d.get("mean/auroc"),
            "ef1": d.get("mean/ef@1.0%"),
            "n_targets": d.get("n_targets"),
            "eval": "ensemble",
        }
    csv = ev / "lit_pcba_mv4.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        col = next((c for c in df.columns if c.lower() == "bedroc"), None)
        if col is None:
            return None
        return {
            "bedroc": df[col].mean(),
            "bedroc_median": df[col].median(),
            "auroc": df["auroc"].mean() if "auroc" in df else None,
            "ef1": None,
            "n_targets": len(df),
            "eval": "single",
        }
    return None


def collect(roots: list[str], prefix: str | None) -> pd.DataFrame:
    rows = []
    seen = set()
    for root in roots:
        for env_path in sorted((REPO / root).glob("*/pipeline.env")):
            run_dir = env_path.parent
            env = parse_env(env_path)
            run_id = env.get("ADAPTER_RUN_ID", run_dir.name)
            if run_id in seen:  # _finished dir shadows the live one
                continue
            seen.add(run_id)
            name = env.get("RUN_NAME", run_id)
            if prefix and not name.startswith(prefix):
                continue
            args = parse_stage2_args(run_dir / "stage2.train.args")
            ebm_ids = [
                (run_dir / f"ebm.{s}").read_text().strip()
                for s in range(int(env.get("N_SEEDS", "1")))
                if (run_dir / f"ebm.{s}").exists()
            ]
            art_root = REPO / env.get("ARTIFACTS_ROOT", "artifacts")
            m = load_metrics(art_root, ebm_ids)
            view3d = env.get("VIEW3D", "0") == "1" or args.get("experiment") == "adapter3d"
            rows.append(
                {
                    "run_name": name,
                    "run_id": run_id,
                    "method": args.get("ssl_loss", env.get("METHOD", "?")),
                    "protein": env.get("PROTEIN", "?"),
                    "merge": env.get("MERGE", "0"),
                    "view3d": int(view3d),
                    "pool": args.get("adapter_pool", DEFAULTS["adapter_pool"]),
                    "n_layers": int(args.get("adapter_n_layers", DEFAULTS["adapter_n_layers"])),
                    "seed": int(args.get("seed", DEFAULTS["seed"])),
                    "n_ebm_seeds": len(ebm_ids),
                    "finished": env.get("FINISHED", "0") == "1",
                    **(m or {"bedroc": None}),
                }
            )
    return pd.DataFrame(rows)


def factor_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.dropna(subset=["bedroc"]).groupby(col)["bedroc"]
    return pd.DataFrame({"mean_bedroc": g.mean(), "std": g.std(), "n": g.size()}).round(4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default=None, help="only runs whose RUN_NAME starts with this")
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--csv", default=None, help="also write the per-run table here")
    args = ap.parse_args()

    df = collect(args.roots, args.prefix)
    if df.empty:
        print("no runs found (prefix / roots matched nothing)")
        return
    df = df.sort_values("bedroc", ascending=False, na_position="last").reset_index(drop=True)

    cols = ["run_name", "method", "protein", "view3d", "pool", "n_layers", "seed",
            "bedroc", "bedroc_median", "auroc", "ef1", "n_ebm_seeds", "finished", "eval"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("\n=== per-run (sorted by mean BEDROC) ===")
        print(df[cols].to_string(index=False))

    done = df.dropna(subset=["bedroc"])
    if not done.empty:
        for col in ("method", "protein", "pool", "n_layers", "view3d"):
            if done[col].nunique() > 1:
                print(f"\n=== BEDROC by {col} (OFAT — holds only near the anchor config) ===")
                print(factor_table(done, col).to_string())

        # Seed variance: same config signature, different SSL seed.
        sig = ["method", "protein", "view3d", "pool", "n_layers"]
        rep = done.groupby(sig).filter(lambda g: g["seed"].nunique() > 1)
        if not rep.empty:
            print("\n=== SSL-seed variance (same config, different seed) ===")
            v = rep.groupby(sig)["bedroc"].agg(["mean", "std", "min", "max", "size"]).round(4)
            print(v.to_string())

    n_done = int(done.shape[0])
    print(f"\n{n_done}/{len(df)} runs have stage-6 results.")
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
