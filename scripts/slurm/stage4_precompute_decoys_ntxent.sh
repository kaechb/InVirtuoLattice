#!/usr/bin/env bash
#SBATCH --job-name=lattice-s4-ntxent
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/stage4/ntxent_%j.out
#SBATCH --error=logs/slurm/stage4/ntxent_%j.err
#
# Stage 4 — precompute decoy pools with the NT-Xent adapter (Stage-2 Lightning ckpt).
#
# Outputs:
#   artifacts/decoys/ntxent/decoy_zm/
#   artifacts/decoys/ntxent/bdb_zm/
#   artifacts/binders/ntxent/binder_zm/
#
# Edit SSL_RUN_ID (W&B run id from stage2_adapter_ssl_ntxent.sh), then:
#   sbatch scripts/slurm/stage4_precompute_decoys_ntxent.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage4_precompute_decoys_ntxent.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

SSL_RUN_ID=29c1bbwj
# Standard NT-Xent adapter location: Stage-2 writes here and the EBM config
# (ebm_hardneg_ntxent.yaml) reads from here. Keep all stages on this one path so
# binders/decoys/bdb + the EBM encoder share the exact same adapter weights.
SSL_CKPT="artifacts/adapter/ntxent/checkpoints/${SSL_RUN_ID}/last.ckpt"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

lattice_require_file "${REPO}/${SSL_CKPT}" \
  "edit SSL_RUN_ID in scripts/slurm/stage4_precompute_decoys_ntxent.sh (from Stage 2 W&B run id)"

# --force keeps all three pools in the SAME adapter space as ${SSL_CKPT} (and the
# EBM encoder adapter). Mismatched adapters between positives (binders) and
# negatives (decoy/bdb) teach the head an adapter-signature shortcut: huge val/*,
# random LIT-PCBA. The EBM data module hard-fails on a mismatch.
WANDB_OFFLINE=1 srun python -m lattice_lab.ebm.precompute_decoys \
  --shard-dir /flash/project_465003063/grassogi/artifacts/processed/moses \
  --adapter-ckpt "${SSL_CKPT}" \
  --store artifacts/decoys/ntxent/decoy_zm \
  --batch-size 512 \
  --force

WANDB_OFFLINE=1 srun python -m lattice_lab.ebm.precompute_bdb_zm \
  --bdb-parquet artifacts/processed/bindingdb/threshold_90/train.parquet \
  --adapter-ckpt "${SSL_CKPT}" \
  --store artifacts/decoys/ntxent/bdb_zm \
  --batch-size 512 \
  --n-jobs 6 \
  --force

WANDB_OFFLINE=1 srun python -m lattice_lab.ebm.precompute_binders \
  --train-parquet artifacts/processed/bindingdb/threshold_90/train.parquet \
  --val-parquet artifacts/processed/bindingdb/threshold_90/val.parquet \
  --adapter-ckpt "${SSL_CKPT}" \
  --store artifacts/binders/ntxent/binder_zm \
  --batch-size 512 \
  --force
