"""Wall-clock timing for a LATTICE THRβ library screen (matches notebook protocol).

Times: model load, ESM-2 target encode, ligand fragmentize+encode (4 views),
and 3-seed energy scoring. Writes JSON for the report.

    PYTHONPATH=src python scripts/time_thrb_screen.py
    PYTHONPATH=src python scripts/time_thrb_screen.py --max-mols 500   # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lattice_lab.inference.predict import (
    build_encoder,
    build_head,
    encode_ligands,
    encode_protein,
    read_target_sequence,
    score as head_energy,
)
from lattice_lab.preprocessing.molecules import standardize_smiles

REPO = Path(__file__).resolve().parents[1]
EBM_ROOT = REPO / "artifacts/ablation/energy/checkpoints"
HEAD_CKPTS = [
    EBM_ROOT / "nr3hsvf0" / "ebm-002-054000.ckpt",
    EBM_ROOT / "q3hzyymw" / "ebm-002-056000.ckpt",
    EBM_ROOT / "dv44weya" / "ebm-002-056000.ckpt",
]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-mols", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "artifacts/predictions/thrb_lattice_timing.json",
    )
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parquet = REPO / "notebooks/chembl_decoys_ws1v1_ws1v2_rapposelli_slim.parquet"
    fasta = REPO / "notebooks/P10828.fasta"
    raw = pd.read_parquet(parquet, columns=["canonical_smiles"])
    if a.max_mols is not None:
        raw = raw.head(a.max_mols)
    smiles = [s for s in (standardize_smiles(x) for x in raw["canonical_smiles"]) if s]
    seq = read_target_sequence(None, fasta)

    args = argparse.Namespace(
        adapter_ckpt=HEAD_CKPTS[0],
        head_ckpt=HEAD_CKPTS[0],
        d_adapter=512,
        d_protein=1280,
        protein_backend="esm2",
        esm_model="facebook/esm2_t33_650M_UR50D",
        device=device,
        batch_size=a.batch_size,
        n_views=a.n_views,
        n_jobs=a.n_jobs,
    )

    times: dict[str, float] = {}
    t0 = time.perf_counter()
    encoder = build_encoder(args)
    args.d_adapter = encoder.adapter.d_adapter
    _sync()
    times["load_encoder_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    heads = []
    for ckpt in HEAD_CKPTS:
        args.head_ckpt = ckpt
        heads.append(build_head(args))
    _sync()
    times["load_heads_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    z_p = encode_protein(args, seq)
    _sync()
    times["encode_protein_s"] = time.perf_counter() - t0

    # Warmup one batch so first-kernel compile isn't in the encode timing.
    warm = smiles[: min(8, len(smiles))]
    encode_ligands(args, encoder, warm, desc="warmup")
    _sync()

    t0 = time.perf_counter()
    z_m, valid = encode_ligands(args, encoder, smiles, desc="library")
    _sync()
    times["encode_ligands_s"] = time.perf_counter() - t0
    n_valid = int(sum(valid))

    # Warmup head
    head_energy(heads[0], z_m[: min(256, len(z_m))], z_p, args)
    _sync()

    per_head = []
    t0 = time.perf_counter()
    for h in heads:
        t1 = time.perf_counter()
        head_energy(h, z_m, z_p, args)
        _sync()
        per_head.append(time.perf_counter() - t1)
    times["score_ensemble_s"] = time.perf_counter() - t0
    times["score_per_head_s"] = per_head

    times["screen_amortized_s"] = (
        times["encode_ligands_s"] + times["score_ensemble_s"]
    )  # exclude one-time load/protein
    times["end_to_end_s"] = (
        times["load_encoder_s"]
        + times["load_heads_s"]
        + times["encode_protein_s"]
        + times["encode_ligands_s"]
        + times["score_ensemble_s"]
    )

    n = len(smiles)
    out = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "n_smiles_input": int(len(raw)),
        "n_smiles_std": n,
        "n_valid": n_valid,
        "n_views": a.n_views,
        "batch_size": a.batch_size,
        "n_heads": len(heads),
        "times_s": times,
        "throughput": {
            "mol_per_s_encode": n_valid / times["encode_ligands_s"],
            "mol_per_s_score_one_head": n_valid / float(np.mean(per_head)),
            "mol_per_s_screen_amortized": n_valid / times["screen_amortized_s"],
            "ms_per_mol_screen_amortized": 1000.0 * times["screen_amortized_s"] / n_valid,
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # one check: scoring should be much cheaper than encoding
    assert times["score_ensemble_s"] < times["encode_ligands_s"]
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
