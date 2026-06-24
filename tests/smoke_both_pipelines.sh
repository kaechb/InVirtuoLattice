#!/usr/bin/env bash
# Smoke-test BOTH the old (lattice.training.train_ebm) and new
# (lattice_lab.train) Stage-5 pipelines on a SAMPLE of the real data.
#
# Safety: every output goes to a fresh mktemp dir; the canonical 00_..07_ stage
# dirs are only ever read (protein store / decoy pool / adapter ckpt are opened
# read-only by the code). Nothing under them is written.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/lattice_smoke.XXXXXX")
export WANDB_MODE=offline WANDB_DIR="$TMP" HYDRA_FULL_ERROR=1
echo ">>> temp outputs: $TMP   (canonical 0X_ dirs are read-only)"

PSTORE=artifacts/protein_store/embeddings/esm2_650M
DZM=artifacts/decoys/decoy_zm_ssl2
BDB=artifacts/decoys/bdb_zm_ssl2
ADP="${ADP:-artifacts/adapter/checkpoints/ed7yw5vq/last.ckpt}"
SRC=artifacts/preprocessing/processed/bindingdb/threshold_90

# 1. Sample the real parquet (read-only) into the temp dir.
python - "$TMP" "$SRC" <<'PY'
import sys, pandas as pd
tmp, src = sys.argv[1], sys.argv[2]
for split in ("train", "val"):
    df = pd.read_parquet(f"{src}/{split}.parquet").head(8000)
    df.to_parquet(f"{tmp}/{split}.parquet")
    print(f"  sampled {split}: {len(df)} rows -> {tmp}/{split}.parquet")
PY

COMMON_GPU="trainer.accelerator=gpu trainer.devices=1"

echo "===================== NEW: lattice_lab.train (experiment=ebm_hardneg) ====================="
python -m lattice_lab.train experiment=ebm_hardneg logger=csv \
  data.train_parquet="$TMP/train.parquet" data.val_parquet="$TMP/val.parquet" \
  data.protein_store="$PSTORE" data.decoy_store="$DZM" data.bdb_store="$BDB" \
  model.encoder.ckpt="$ADP" \
  n_decoys=64 data.batch_size=16 trainer.max_steps=20 \
  trainer.val_check_interval=10 trainer.limit_val_batches=20 $COMMON_GPU \
  hydra.run.dir="$TMP/new" callbacks.model_checkpoint.dirpath="$TMP/new/ckpt"

echo "===================== OLD: lattice.training.train_ebm ====================="
python -m lattice.training.train_ebm \
  data.train_parquet="$TMP/train.parquet" data.val_parquet="$TMP/val.parquet" \
  data.protein_store="$PSTORE" data.decoy_store="$DZM" data.bdb_store="$BDB" \
  data.adapter_ckpt="$ADP" data.output_dir="$TMP/old" \
  data.n_decoys=64 data.batch_size=16 optim.num_steps=20 \
  optim.val_every_steps=10 data.val_batches=20 \
  data.hard_mining_mult=3 data.frac_other_binder=0.4 data.frac_non_binder=0.15

echo "===================== artifacts written (temp only) ====================="
find "$TMP" -name '*.ckpt' -o -name 'ebm_*.pt' | sort
echo ">>> SMOKE OK — both pipelines completed. temp=$TMP"
