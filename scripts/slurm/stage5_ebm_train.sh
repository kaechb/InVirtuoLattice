#!/usr/bin/env bash
#SBATCH --job-name=lattice-s5-ebm
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/slurm/stage5/%A_%a.out
#SBATCH --error=logs/slurm/stage5/%A_%a.err
#
# Stage 5 — EBM hard-negative training (one seed per array task).
#
# RUN_ID must match the Stage-2 adapter + Stage-4 z_m stores:
#
#   sbatch scripts/slurm/stage5_ebm_train.sh lejepa nsw2w2z5          # seed 0 (default)
#   ./scripts/slurm/stage5_ebm_train.sh --three-seeds lejepa nsw2w2z5 # seeds 0–2
#   sbatch --array=0-2 scripts/slurm/stage5_ebm_train.sh lejepa nsw2w2z5
#
# Checkpoints: artifacts/energy/checkpoints/<wandb_run_id>/last.ckpt
set -euo pipefail

# Login-node convenience: ./stage5_ebm_train.sh [--three-seeds] …  (sbatch uses #SBATCH --array above)
if [[ -z "${SLURM_JOB_ID:-}" && "${BASH_SOURCE[0]}" == "${0}" ]]; then
  array="0-0"
  args=("$@")
  if [[ "${1:-}" == --three-seeds ]]; then
    array="0-2"
    args=("${@:2}")
  fi
  exec sbatch --array="${array}" "${BASH_SOURCE[0]}" "${args[@]}"
fi

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage5_ebm_train.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
  lattice_pipeline_source_env
  RUN_ID="${ADAPTER_RUN_ID:?missing ADAPTER_RUN_ID in ${PIPELINE_ENV} — run stage2 first}"
  METHOD="${PIPELINE_EBM_METHOD:-lejepa}"
  PIPELINE_MARKER="$(lattice_pipeline_marker)"
elif [[ "${1:-}" == lejepa || "${1:-}" == ntxent ]]; then
  METHOD="$1"
  RUN_ID="${2:-${RUN_ID:?set RUN_ID=<stage2_wandb_run_id> as \$2 or env RUN_ID=…}}"
else
  METHOD="${METHOD:-lejepa}"
  RUN_ID="${1:-${RUN_ID:?set RUN_ID=<stage2_wandb_run_id> (or pass as \$1)}}"
fi

case "${METHOD}" in
  lejepa)  EXPERIMENT=ebm_hardneg_lejepa ;;
  ntxent)  EXPERIMENT=ebm_hardneg_ntxent ;;
  *)
    echo "unknown METHOD=${METHOD} (want lejepa or ntxent)" >&2
    exit 1
    ;;
esac

SEED="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"

lattice_job_banner "stage5 ${METHOD} start run_id=${RUN_ID} seed=${SEED}"

lattice_load_gpu_modules
lattice_cd_repo
[[ -n "${PIPELINE_LOG_DIR:-}" ]] && trap 'lattice_pipeline_collect_logs_on_exit 5' EXIT
lattice_job_banner "modules loaded; checking gpu"
lattice_require_gpu

ADAPTER_CKPT="${REPO}/artifacts/adapter/checkpoints/${RUN_ID}/last.ckpt"
DECOY_STORE="${REPO}/artifacts/decoys/${RUN_ID}/decoy_zm/manifest.json"
lattice_require_file "${ADAPTER_CKPT}" \
  "missing adapter ckpt — run stage2 first for RUN_ID=${RUN_ID}"
lattice_require_file "${DECOY_STORE}" \
  "missing decoy store — run stage4 for RUN_ID=${RUN_ID} first"

lattice_job_banner "starting train (pool RAM load can take several minutes before first log line)"

TRAIN_EXTRA=(trainer.max_steps=12000)
if lattice_smoke_enabled; then
  SMOKE_PARQUET_DIR="${SMOKE_PARQUET_DIR:-${REPO}/logs/slurm/pipeline/smoke-s5-${SLURM_JOB_ID:-local}}"
  lattice_ensure_smoke_parquets "${SMOKE_PARQUET_DIR}"
  TRAIN_EXTRA=(
    trainer.max_steps=50
    trainer.limit_val_batches=5
    trainer.val_check_interval=10
    n_decoys=64
    data.batch_size=16
    "data.train_parquet=${SMOKE_PARQUET_DIR}/train.parquet"
    "data.val_parquet=${SMOKE_PARQUET_DIR}/val.parquet"
  )
  lattice_job_banner "SMOKE: max_steps=50 n_decoys=64 parquets=${SMOKE_PARQUET_DIR}"
fi

srun python -m lattice_lab.train "experiment=${EXPERIMENT}" \
  "ssl_run_id=${RUN_ID}" \
  "${TRAIN_EXTRA[@]}" \
  "seed=${SEED}" \
  callbacks.model_checkpoint.dirpath=artifacts/energy/checkpoints \
  trainer.precision=bf16-mixed

if [[ -n "${PIPELINE_ENV:-}" ]]; then
  EBM_RUN_ID="$(lattice_newest_subdir_since "${PIPELINE_MARKER}" "${REPO}/artifacts/energy/checkpoints")"
  lattice_require_file "${REPO}/artifacts/energy/checkpoints/${EBM_RUN_ID}/last.ckpt" \
    "stage5 seed=${SEED} finished but no new EBM checkpoint found (pipeline)"
  echo "${EBM_RUN_ID}" > "${PIPELINE_ENV}.ebm.${SEED}"
  lattice_job_banner "pipeline wrote EBM seed=${SEED} → ${EBM_RUN_ID}"
fi
