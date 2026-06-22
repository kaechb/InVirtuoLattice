#!/usr/bin/env bash
#SBATCH --job-name=lattice-s4-decoys
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage4/%j.out
#SBATCH --error=logs/slurm/stage4/%j.err
#
# Stage 4 — precompute decoy / BDB / binder z_m pools for any Stage-2 adapter ckpt
# (LeJEPA discrete-flow, denoising-JEPA, etc. — detected from the checkpoint).
#
# Stores land under the adapter W&B run id:
#   artifacts/decoys/<run_id>/{decoy_zm,bdb_zm}
#   artifacts/binders/<run_id>/binder_zm
#
#   RUN_ID=nsw2w2z5 sbatch scripts/slurm/stage4_precompute_decoys.sh
#   sbatch scripts/slurm/stage4_precompute_decoys.sh nsw2w2z5
#   sbatch scripts/slurm/stage4_precompute_decoys.sh  # pipeline: reads ADAPTER_RUN_ID from PIPELINE_ENV
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage4_precompute_decoys.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_pipeline_source_env

RUN_ID="${1:-${RUN_ID:-}}"
if [[ -z "${RUN_ID}" && -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
  lattice_pipeline_source_env
  RUN_ID="${ADAPTER_RUN_ID:?missing ADAPTER_RUN_ID in ${PIPELINE_ENV} — run stage2 first}"
fi
: "${RUN_ID:?set RUN_ID=<stage2_wandb_run_id> (or pass as \$1)}"
SSL_CKPT="artifacts/adapter/checkpoints/${RUN_ID}/last.ckpt"

lattice_load_gpu_modules
lattice_cd_repo
[[ -n "${PIPELINE_LOG_DIR:-}" ]] && trap 'lattice_pipeline_collect_logs_on_exit 4' EXIT
lattice_require_gpu

export WANDB_MODE=disabled

lattice_require_file "${REPO}/${SSL_CKPT}" \
  "missing adapter checkpoint for RUN_ID=${RUN_ID}"

PRECOMPUTE_LIMIT=()
TRAIN_PARQUET="artifacts/preprocessing/processed/bindingdb/threshold_90/train.parquet"
VAL_PARQUET="artifacts/preprocessing/processed/bindingdb/threshold_90/val.parquet"
if lattice_smoke_enabled; then
  PRECOMPUTE_LIMIT=(--limit "$(lattice_smoke_precompute_limit)")
  SMOKE_PARQUET_DIR="${SMOKE_PARQUET_DIR:-${REPO}/logs/slurm/smoke-$$}"
  lattice_ensure_smoke_parquets "${SMOKE_PARQUET_DIR}"
  TRAIN_PARQUET="${SMOKE_PARQUET_DIR}/train.parquet"
  VAL_PARQUET="${SMOKE_PARQUET_DIR}/val.parquet"
  lattice_job_banner "SMOKE: precompute limit=${PRECOMPUTE_LIMIT[*]} parquets=${TRAIN_PARQUET}"
fi

srun python -m lattice_lab.ebm.precompute_decoys \
  --shard-dir "${LATTICE_FLASH_PROCESSED}/moses" \
  --adapter-ckpt "${SSL_CKPT}" \
  --batch-size 512 \
  "${PRECOMPUTE_LIMIT[@]}" \
  --force

srun python -m lattice_lab.ebm.precompute_bdb_zm \
  --bdb-parquet "${TRAIN_PARQUET}" \
  --adapter-ckpt "${SSL_CKPT}" \
  --batch-size 512 \
  --n-jobs 6 \
  "${PRECOMPUTE_LIMIT[@]}" \
  --force

srun python -m lattice_lab.ebm.precompute_binders \
  --train-parquet "${TRAIN_PARQUET}" \
  --val-parquet "${VAL_PARQUET}" \
  --adapter-ckpt "${SSL_CKPT}" \
  --batch-size 512 \
  "${PRECOMPUTE_LIMIT[@]}" \
  --force
