"""Aggregate Nesso-1 affinity.json outputs for WS1 compounds into one parquet."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "artifacts/nesso/ws1_out/predictions"
META = REPO / "artifacts/nesso/ws1_meta.parquet"
OUT = REPO / "artifacts/nesso/thrb_ws1_nesso_scores.parquet"

KEYS = [
    "affinity_pred_value",
    "affinity_pred_value1",
    "affinity_pred_value2",
    "affinity_logits_binary",
    "affinity_probability_binary",
    "entropy_crop_pl",
    "entropy_pl",
]


def main() -> None:
    meta = pd.read_parquet(META)
    rows = []
    missing = []
    for _, r in meta.iterrows():
        path = PRED / r.ivlid / "affinity.json"
        if not path.is_file():
            missing.append(r.ivlid)
            continue
        aff = json.loads(path.read_text())
        rows.append({
            "ivlid": r.ivlid,
            "canonical_smiles": r.canonical_smiles,
            "source": r.source,
            "binder": bool(r.binder),
            "p_activity_values": r.p_activity_values,
            "score": aff.get("affinity_probability_binary"),
            **{k: aff.get(k) for k in KEYS},
            "status": "ok",
        })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df)}/{len(meta)} -> {OUT}")
    if missing:
        print(f"missing {len(missing)}: {missing[:5]}...")
    if not df.empty and df.binder.any():
        # lower affinity_pred_value = stronger; higher probability = stronger
        b = df.loc[df.binder]
        print("binder rows:", b[["ivlid", "source", "affinity_pred_value", "affinity_probability_binary"]].to_string(index=False))
    assert len(df) + len(missing) == len(meta)


if __name__ == "__main__":
    main()
