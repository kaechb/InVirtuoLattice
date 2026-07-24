"""Run Nesso-1 on the WS1_v1/WS1_v2 compounds against human THRα."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
SOURCE_META = REPO / "artifacts/nesso/ws1_meta.parquet"
FASTA = REPO / "notebooks/P10827-2_THA_HUMAN_Isoform_Alpha-1.fasta"
OUT = REPO / "artifacts/nesso/thra_ws1_ws2"

AFFINITY_KEYS = (
    "affinity_pred_value",
    "affinity_pred_value1",
    "affinity_pred_value2",
    "affinity_logits_binary",
    "affinity_probability_binary",
    "entropy_crop_pl",
    "entropy_pl",
)


def prepare_inputs() -> pd.DataFrame:
    sequence = "".join(
        line.strip() for line in FASTA.read_text().splitlines()
        if line and not line.startswith(">")
    )
    if len(sequence) != 410:
        raise ValueError(f"expected 410-aa THRα sequence, got {len(sequence)}")

    meta = pd.read_parquet(SOURCE_META)
    expected_sources = {"WS1_v1", "WS1_v2"}
    if set(meta.source) != expected_sources or len(meta) != 214:
        raise ValueError(f"unexpected WS cohort: {len(meta)} rows, sources={set(meta.source)}")

    inputs = OUT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    yaml_paths = []
    for row in meta.itertuples(index=False):
        path = inputs / f"{row.ivlid}.yaml"
        data = {
            "sequences": [
                {"protein": {"id": "A", "sequence": sequence}},
                {"ligand": {"id": "B", "smiles": row.canonical_smiles}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        yaml_paths.append(str(path))

    meta = meta.drop(columns="yaml", errors="ignore").copy()
    meta["yaml"] = yaml_paths
    meta.to_parquet(OUT / "meta.parquet", index=False)
    print(f"prepared {len(meta)} THRα inputs in {inputs}")
    return meta


def run_nesso(devices: int) -> None:
    cmd = [
        sys.executable, str(REPO / ".deps_nesso/bin/nesso"),
        "predict", str(OUT / "inputs"),
        "--out_dir", str(OUT),
        "--no_kernels", "--accelerator", "gpu", "--devices", str(devices),
        "--require_affinity", "--num_workers", "1", "--override",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / ".deps_nesso") + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("NESSO_CACHE", str(REPO / ".cache/nesso"))
    with (OUT.parent / f"{OUT.name}.log").open("w") as log:
        result = subprocess.run(
            cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"Nesso failed ({result.returncode}); see {OUT.parent / (OUT.name + '.log')}")


def aggregate(meta: pd.DataFrame) -> Path:
    rows = []
    missing = []
    for row in meta.itertuples(index=False):
        path = OUT / "predictions" / row.ivlid / "affinity.json"
        if not path.is_file():
            missing.append(row.ivlid)
            continue
        affinity = json.loads(path.read_text())
        rows.append({
            "ivlid": row.ivlid,
            "canonical_smiles": row.canonical_smiles,
            "source": row.source,
            "thrb_binder": bool(row.binder),
            "thrb_p_activity_values": row.p_activity_values,
            "score": affinity.get("affinity_probability_binary"),
            **{key: affinity.get(key) for key in AFFINITY_KEYS},
            "status": "ok",
        })
    if missing:
        raise RuntimeError(f"missing {len(missing)} predictions; first five: {missing[:5]}")
    output = OUT / "scores.parquet"
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"wrote {len(rows)}/{len(meta)} predictions to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    metadata = prepare_inputs()
    if not args.prepare_only:
        if not args.skip_predict:
            run_nesso(args.devices)
        aggregate(metadata)
