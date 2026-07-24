"""Subsample LIT-PCBA (default 100/target), run Nesso-1, report AUROC/BEDROC/EF.

Stratified per target: keep up to ~15 actives (all if fewer), fill with inactives
to --per-target. ~15×100 ≈ 1.5k preds ≈ a few hours on one MI250X.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = REPO / "artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet"


def _record_id(target: str, smiles: str, cid: str | int | None) -> str:
    raw = f"{target}|{cid if cid is not None else ''}|{smiles}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def subsample(df: pd.DataFrame, per_target: int, seed: int, max_actives: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for target, g in df.groupby("target_name", sort=True):
        act = g.loc[g.is_active.astype(bool)]
        ina = g.loc[~g.is_active.astype(bool)]
        n_act = min(len(act), max_actives, per_target)
        n_ina = min(len(ina), per_target - n_act)
        if n_act == 0:
            raise ValueError(f"{target}: no actives")
        pick_act = act.iloc[rng.choice(len(act), size=n_act, replace=False)]
        pick_ina = ina.iloc[rng.choice(len(ina), size=n_ina, replace=False)] if n_ina else ina.iloc[:0]
        rows.append(pd.concat([pick_act, pick_ina], axis=0))
    out = pd.concat(rows, ignore_index=True)
    return out


def write_yamls(sample: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    for r in sample.itertuples(index=False):
        rid = _record_id(r.target_name, r.smiles, getattr(r, "pubchem_cid", None))
        yml = {
            "sequences": [
                {"protein": {"id": "A", "sequence": r.sequence}},
                {"ligand": {"id": "B", "smiles": r.smiles}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        path = out_dir / f"{rid}.yaml"
        path.write_text(yaml.safe_dump(yml, sort_keys=False))
        meta_rows.append({
            "record_id": rid,
            "target_name": r.target_name,
            "smiles": r.smiles,
            "is_active": bool(r.is_active),
            "yaml": str(path),
        })
    meta = pd.DataFrame(meta_rows)
    return meta


def run_nesso(inputs: Path, out_dir: Path, devices: int) -> None:
    cmd = [
        sys.executable, str(REPO / ".deps_nesso/bin/nesso"), "predict", str(inputs),
        "--out_dir", str(out_dir),
        "--no_kernels", "--accelerator", "gpu",
        "--devices", str(devices),
        "--require_affinity", "--num_workers", "1", "--override",
    ]
    env_py = str(REPO / ".deps_nesso")
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = env_py + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("NESSO_CACHE", str(REPO / ".cache/nesso"))
    log = out_dir.with_suffix(".log") if out_dir.suffix else Path(str(out_dir) + ".log")
    out_dir.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"nesso failed ({proc.returncode}); see {log}")


def aggregate(meta: pd.DataFrame, pred_dir: Path) -> pd.DataFrame:
    rows = []
    missing = []
    for r in meta.itertuples(index=False):
        path = pred_dir / r.record_id / "affinity.json"
        if not path.is_file():
            missing.append(r.record_id)
            continue
        aff = json.loads(path.read_text())
        rows.append({
            "record_id": r.record_id,
            "target_name": r.target_name,
            "smiles": r.smiles,
            "is_active": bool(r.is_active),
            "affinity_pred_value": aff.get("affinity_pred_value"),
            "affinity_probability_binary": aff.get("affinity_probability_binary"),
            "status": "ok",
        })
    if missing:
        print(f"missing {len(missing)} predictions", file=sys.stderr)
    return pd.DataFrame(rows)


def metrics_table(scores: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(REPO / "src"))
    from lattice_lab.eval.metrics import auroc, bedroc, ef_at_k

    rows = []
    for target, g in scores.groupby("target_name", sort=True):
        y = g.is_active.astype(int).to_numpy()
        # higher better for both ranking scores used here
        for name, col, negate in (
            ("p_bind", "affinity_probability_binary", False),
            ("affinity", "affinity_pred_value", True),  # lower logIC50 = better
        ):
            s = g[col].to_numpy(float)
            ok = np.isfinite(s)
            if ok.sum() < 2 or y[ok].sum() == 0 or y[ok].sum() == ok.sum():
                continue
            score = (-s if negate else s)[ok]
            yy = y[ok]
            rows.append({
                "target": target,
                "score": name,
                "n": int(ok.sum()),
                "n_active": int(yy.sum()),
                "auroc": auroc(yy, score),
                "bedroc": bedroc(yy, score, alpha=80.5),
                "ef@0.5%": ef_at_k(yy, score, 0.5),
                "ef@1%": ef_at_k(yy, score, 1.0),
                "ef@5%": ef_at_k(yy, score, 5.0),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--per-target", type=int, default=100)
    ap.add_argument("--max-actives", type=int, default=15,
                    help="cap actives kept per target (rest filled with inactives)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", nargs="*", default=None,
                    help="optional subset of target names (default: all)")
    ap.add_argument("--out-root", type=Path,
                    default=REPO / "artifacts/nesso/litpcba_sub100")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--skip-predict", action="store_true",
                    help="only aggregate + metrics from existing predictions")
    a = ap.parse_args()

    cols = ["target_name", "smiles", "is_active", "sequence"]
    df = pd.read_parquet(a.parquet, columns=cols)
    if a.targets:
        df = df.loc[df.target_name.isin(a.targets)]
    sample = subsample(df, a.per_target, a.seed, a.max_actives)
    a.out_root.mkdir(parents=True, exist_ok=True)
    sample_path = a.out_root / "sample.parquet"
    sample.to_parquet(sample_path, index=False)
    print(sample.groupby("target_name").agg(n=("smiles", "count"), n_act=("is_active", "sum")).to_string())

    inputs = a.out_root / "inputs"
    meta = write_yamls(sample, inputs)
    meta.to_parquet(a.out_root / "meta.parquet", index=False)
    print(f"wrote {len(meta)} yamls -> {inputs}")

    pred_dir = a.out_root / "predictions"
    if not a.skip_predict:
        # nesso writes to out_dir/predictions/...
        run_nesso(inputs, a.out_root, a.devices)

    scores = aggregate(meta, pred_dir)
    scores_path = a.out_root / "scores.parquet"
    scores.to_parquet(scores_path, index=False)
    print(f"scored {len(scores)}/{len(meta)} -> {scores_path}")

    tab = metrics_table(scores)
    tab_path = a.out_root / "metrics.csv"
    tab.to_csv(tab_path, index=False)
    print(tab.to_string(index=False))
    if not tab.empty:
        for score, g in tab.groupby("score"):
            print(f"\nmean {score}: AUROC={g.auroc.mean():.3f} BEDROC={g.bedroc.mean():.3f} "
                  f"EF@1%={g['ef@1%'].mean():.2f}  (over {len(g)} targets)")
    assert len(scores) > 0


if __name__ == "__main__":
    main()
