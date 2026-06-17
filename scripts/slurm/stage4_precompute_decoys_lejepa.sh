#!/usr/bin/env bash
#SBATCH --job-name=lattice-s4-lejepa
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage4/lejepa_%j.out
#SBATCH --error=logs/slurm/stage4/lejepa_%j.err
#
# Stage 4 — precompute decoy pools with the LeJEPA adapter (Stage-2 Lightning ckpt).
#
# Outputs:
#   artifacts/decoys/lejepa/decoy_zm/
#   artifacts/decoys/lejepa/bdb_zm/
#   artifacts/binders/lejepa/binder_zm/
#
# Edit SSL_RUN_ID (W&B run id from stage2_adapter_ssl_lejepa.sh), then:
#   sbatch scripts/slurm/stage4_precompute_decoys_lejepa.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage4_precompute_decoys_lejepa.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

SSL_RUN_ID=4tau7a38
SSL_CKPT="artifacts/adapter/lejepa/checkpoints/${SSL_RUN_ID}/last.ckpt"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

lattice_require_file "${REPO}/${SSL_CKPT}" \
  "edit SSL_RUN_ID in scripts/slurm/stage4_precompute_decoys_lejepa.sh (from Stage 2 W&B run id)"

# --force rebuilds the pools from scratch with ${SSL_CKPT}. This MUST stay in
# sync with the binder store + the EBM encoder adapter (all = ${SSL_CKPT}):
# if positives and negatives are encoded by different adapters the head learns
# an adapter-signature shortcut (huge val/*, random LIT-PCBA). The data module
# now hard-fails on a mismatch, so keep --force here whenever the adapter changes.
srun python -m lattice_lab.ebm.precompute_decoys \
  --shard-dir /flash/project_465003063/grassogi/artifacts/processed/moses \
  --adapter-ckpt "${SSL_CKPT}" \
  --store artifacts/decoys/lejepa/decoy_zm \
  --batch-size 512 \
  --force

srun python -m lattice_lab.ebm.precompute_bdb_zm \
  --bdb-parquet artifacts/processed/bindingdb/threshold_90/train.parquet \
  --adapter-ckpt "${SSL_CKPT}" \
  --store artifacts/decoys/lejepa/bdb_zm \
  --batch-size 512 \
  --n-jobs 6 \
  --force

srun python -m lattice_lab.ebm.precompute_binders \
  --train-parquet artifacts/processed/bindingdb/threshold_90/train.parquet \
  --val-parquet artifacts/processed/bindingdb/threshold_90/val.parquet \
  --adapter-ckpt "${SSL_CKPT}" \
  --store artifacts/binders/lejepa/binder_zm \
  --batch-size 512 \
  --force
