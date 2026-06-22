"""Stage 7 — ensemble screening.

Score a ligand library against one protein target with the multi-seed LATTICE
ensemble, exactly like :mod:`lattice_lab.inference.predict` but averaging the energy
of N seed heads instead of using one. Supports the same optional reference sets
(known binders / non-binders) and emits the same artifacts:

- a **CSV** ranked best-first with ``energy`` (raw E, lower = stronger binder)
  and ``score`` (``-E``);
- a **violin PNG** of the predicted **energy ``E``** (lower = stronger binder),
  with the screened library bracketed between the known-binder and non-binder
  reference violins when those are provided.

The target protein is given either as a raw sequence (encoded with ESM-2 on the
fly) or as an id already in a precomputed protein store.

Example:

    python -m lattice_lab.inference.predict_ensemble \
        --head-ckpts artifacts/energy/checkpoints/seed0.pt \
                     artifacts/energy/checkpoints/seed1.pt \
                     artifacts/energy/checkpoints/seed2.pt \
        --adapter-ckpt artifacts/adapter/checkpoints_ssl2/adapter_v1.pt \
        --target-fasta thrb.fasta --target-name THRB \
        --smiles-file  my_library.csv \
        --binders      known_binders.smi \
        --nonbinders   known_decoys.smi \
        --output-csv   artifacts/predictions/thrb_predictions.csv \
        --output-png   artifacts/predictions/thrb_affinity_violin.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lattice_lab.inference.predict import (
    build_encoder,
    build_head,
    encode_ligands,
    encode_protein,
    plot_violin,
    read_smiles,
    read_target_sequence,
    score,
)
from lattice_lab.eval.lit_pcba import _inchikey_or_none
from lattice_lab.models.builders import adapter_fingerprint
from lattice_lab.protein.store import EmbeddingStore

logger = logging.getLogger(__name__)


def _score_set(args, encoder, heads, z_p, smiles, *, desc):
    """Encode + score a SMILES list, averaging energy over the seed heads.

    Returns ``(energy, valid)`` with one energy per input SMILES (``NaN`` where
    the SMILES was unparseable) and the per-SMILES parse mask.
    """
    z_m, valid = encode_ligands(args, encoder, smiles, desc=desc)
    e = np.mean([score(h, z_m, z_p, args) for h in heads], axis=0)
    full = np.full(len(smiles), np.nan, dtype=np.float32)
    full[np.array(valid, dtype=bool)] = e
    return full, valid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head-ckpts", type=Path, nargs="+", required=True,
                    help="seed EBM checkpoints (Stage-5 Lightning .ckpt); their energies "
                         "are averaged. The frozen adapter is read from the first one.")
    ap.add_argument("--adapter-ckpt", type=Path, default=None,
                    help="optional Stage-2 adapter ckpt; defaults to the first --head-ckpt "
                         "so the encoder always matches the trained heads.")
    # ligand inputs
    ap.add_argument("--smiles-file", type=Path,
                    help=".csv/.tsv/.parquet (a 'smiles' column) or .smi/.txt library to screen")
    ap.add_argument("--smiles", nargs="+", help="SMILES given inline (instead of --smiles-file)")
    ap.add_argument("--binders", type=Path, default=None,
                    help="optional SMILES of KNOWN BINDERS (reference violin)")
    ap.add_argument("--nonbinders", type=Path, default=None,
                    help="optional SMILES of KNOWN NON-BINDERS (reference violin)")
    # target: a raw sequence, or an id in a precomputed protein store
    ap.add_argument("--target-seq", type=str, help="literal protein sequence (encoded with ESM-2)")
    ap.add_argument("--target-fasta", type=Path, help="FASTA with one protein (encoded with ESM-2)")
    ap.add_argument("--protein-store", type=Path, help="precomputed store (use with --target)")
    ap.add_argument("--target", type=str, help="protein id in --protein-store")
    ap.add_argument("--target-name", type=str, default="target", help="label for CSV / plots")
    # outputs / misc
    ap.add_argument("--output-csv", type=Path, default=Path("artifacts/predictions/predictions.csv"))
    ap.add_argument("--output-png", type=Path, default=Path("artifacts/predictions/affinity_distribution.png"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--n-views", type=int, default=4,
                    help="seeded rBRICS views averaged per molecule (multi-view "
                         "test-time augmentation). 4 matches the reported LIT-PCBA "
                         "encoding; 1 = fast single-view.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.smiles_file and not args.smiles:
        ap.error("provide --smiles-file or --smiles")
    use_store = args.protein_store is not None and args.target is not None
    if not use_store and not (args.target_seq or args.target_fasta):
        ap.error("provide a target: --target-seq / --target-fasta, or --protein-store with --target")

    args.smiles_file = args.smiles_file or Path("-")
    args.binders_file = args.binders
    args.nonbinders_file = args.nonbinders
    args.head_ckpt = args.head_ckpts[0]
    args.adapter_ckpt = args.adapter_ckpt or args.head_ckpts[0]

    # All heads share ONE encoder, so they must share ONE adapter, and the encoder
    # adapter (--adapter-ckpt) must be that same adapter — else z_m is encoded in a
    # latent space the heads weren't trained on.
    head_fps = [adapter_fingerprint(c) for c in args.head_ckpts]
    if len(set(head_fps)) > 1:
        ap.error(
            "ensemble --head-ckpts were trained with DIFFERENT adapters "
            f"(fingerprints {[f[:12] for f in head_fps]}); cannot share one encoder."
        )
    if adapter_fingerprint(args.adapter_ckpt) != head_fps[0]:
        ap.error(
            "--adapter-ckpt does not match the adapter the head(s) were trained with "
            f"(adapter fp={adapter_fingerprint(args.adapter_ckpt)[:12]}, "
            f"head fp={head_fps[0][:12]}); pass the matching adapter or omit it to "
            "use the one baked into the head ckpt."
        )

    args.d_adapter = 512  # provisional; overwritten from the ckpt below
    args.d_protein = 1280
    args.esm_model = "facebook/esm2_t33_650M_UR50D"

    # --- target z_p -----------------------------------------------------
    if use_store:
        z_p = EmbeddingStore.open(args.protein_store, mode="r").get_mean(args.target).astype(np.float32)
        logger.info("target z_p from store %s id=%s", args.protein_store, args.target)
        args.target_seq = ""
    else:
        args.target_seq = read_target_sequence(args.target_seq, args.target_fasta)
        z_p = encode_protein(args, args.target_seq)

    # --- models ---------------------------------------------------------
    encoder = build_encoder(args)
    args.d_adapter = encoder.adapter.d_adapter
    heads = []
    for ckpt in args.head_ckpts:
        args.head_ckpt = ckpt
        heads.append(build_head(args))
    logger.info("ensemble of %d heads", len(heads))

    # --- screened library ----------------------------------------------
    smiles = read_smiles(args.smiles_file) if args.smiles_file != Path("-") else list(args.smiles)
    energy, valid = _score_set(args, encoder, heads, z_p, smiles, desc="screened")
    if not any(valid):
        raise ValueError("no valid SMILES to score")

    df = pd.DataFrame({
        "target": args.target_name,
        "smiles": smiles,
        "inchikey": [_inchikey_or_none(s) for s in smiles],
        "valid": valid,
        "energy": energy,                              # raw E ; lower = stronger
        "score": -energy,                              # -E    ; higher = stronger
    })
    df["rank"] = df["score"].rank(ascending=False, method="min").astype("Int64")
    df = df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info("wrote %d predictions to %s", len(df), args.output_csv)
    screened_scores = df.loc[df["valid"], "score"].to_numpy(dtype=np.float32)

    # --- optional reference sets ---------------------------------------
    def _ref(path, desc):
        if path is None:
            return None
        e, _ = _score_set(args, encoder, heads, z_p, read_smiles(path), desc=desc)
        s = -e[np.isfinite(e)]
        logger.info("[%s] %d valid (median score=%.3f)", desc, s.size,
                    float(np.median(s)) if s.size else float("nan"))
        return s

    binder_scores = _ref(args.binders, "binders")
    nonbinder_scores = _ref(args.nonbinders, "nonbinders")

    # --- violin plot of the ENERGY E (lower = stronger binder) ---------
    def _neg(s):
        return None if s is None else -s

    plot_violin(
        target_name=args.target_name,
        screened=_neg(screened_scores),
        binders=_neg(binder_scores),
        nonbinders=_neg(nonbinder_scores),
        path=args.output_png,
        ylabel="energy E   (lower = stronger binder)",
    )

    top = df.loc[df["valid"]].head(5)
    logger.info("top predicted binders for %s:", args.target_name)
    for _, r in top.iterrows():
        logger.info("  rank=%-3d score=%+.3f  %s",
                    int(r["rank"]), r["score"], r["smiles"])


if __name__ == "__main__":
    main()
