#!/usr/bin/env bash
#SBATCH --job-name=lattice-s2-ssl
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage2/%j.out
#SBATCH --error=logs/slurm/stage2/%j.err
#
# Stage 2 — adapter / denoising-JEPA SSL training.
#
#   METHOD=lejepa sbatch scripts/slurm/stage2_ssl.sh
#   sbatch scripts/slurm/stage2_ssl.sh ntxent
#   sbatch scripts/slurm/stage2_ssl.sh ijepa
#   sbatch scripts/slurm/stage2_ssl.sh ijepa my_ablation_name
#   sbatch scripts/slurm/stage2_ssl.sh denoise
#
# Checkpoints: artifacts/adapter/checkpoints/<wandb_run_id>/last.ckpt
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage2_ssl.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

METHOD="${1:-${METHOD:?set METHOD=lejepa|ntxent|ijepa|denoise (or pass as \$1)}}"
RUN_NAME="${2:-${RUN_NAME:-}}"
lattice_pipeline_source_env

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

if [[ -n "${PIPELINE_ENV:-}" ]]; then
  PIPELINE_MARKER="$(lattice_pipeline_marker)"
fi

TRAIN_ARGS=(
  "data.shard_dir=${LATTICE_FLASH_PROCESSED}/moses"
  trainer.max_epochs=10
  trainer.accelerator=gpu
  callbacks.model_checkpoint.dirpath=artifacts/adapter/checkpoints
)

case "${METHOD}" in
  ntxent)
    TRAIN_ARGS=(
      "experiment=adapter_discrete_flow"
      "${TRAIN_ARGS[@]}"
      "model.ssl_loss=ntxent"
      "model.fp_weight=2.0"
      "logger.wandb.group=ssl_ntxent_fpdistill"
      "logger.wandb.name=ntxent_fp2.0_baseline"
    )
    ;;
  lejepa)
    TRAIN_ARGS=(
      "experiment=adapter_discrete_flow"
      "${TRAIN_ARGS[@]}"
    )
    ;;
  ijepa)
    TRAIN_ARGS=(
      "experiment=adapter_discrete_flow"
      "${TRAIN_ARGS[@]}"
      "model.ssl_loss=ijepa"
      "model.ijepa_block_hole_attn=true"
    )
    ;;
  denoise)
    TRAIN_ARGS=(
      "experiment=denoising_jepa"
      "${TRAIN_ARGS[@]}"
    )
    ;;
  *)
    echo "unknown METHOD=${METHOD} (want lejepa, ntxent, ijepa, or denoise)" >&2
    exit 1
    ;;
esac

if [[ -n "${RUN_NAME}" ]]; then
  TRAIN_ARGS+=("logger.wandb.name=${RUN_NAME}")
fi

if lattice_smoke_enabled; then
  TRAIN_ARGS+=(
    trainer.max_epochs=1
    trainer.check_val_every_n_epoch=1
    trainer.val_check_interval=null
    trainer.limit_train_batches=0.01
    trainer.limit_val_batches=2
  )
  [[ -z "${RUN_NAME}" ]] && TRAIN_ARGS+=("logger.wandb.name=smoke_${METHOD}")
  lattice_job_banner "SMOKE: 1 epoch, 1% train batches, 2 val batches"
fi

srun python -m lattice_lab.train "${TRAIN_ARGS[@]}"

if [[ -n "${PIPELINE_ENV:-}" ]]; then
  ADAPTER_RUN_ID="$(lattice_newest_subdir_since "${PIPELINE_MARKER}" "${REPO}/artifacts/adapter/checkpoints")"
  lattice_require_file "${REPO}/artifacts/adapter/checkpoints/${ADAPTER_RUN_ID}/last.ckpt" \
    "stage2 finished but no new adapter checkpoint found (pipeline)"
  echo "ADAPTER_RUN_ID=${ADAPTER_RUN_ID}" >> "${PIPELINE_ENV}"
  lattice_pipeline_init_log_dir "${ADAPTER_RUN_ID}"
  cp "${PIPELINE_ENV}" "${PIPELINE_LOG_DIR}/pipeline.env"
  lattice_pipeline_collect_slurm_logs 2
  lattice_job_banner "pipeline logs → ${PIPELINE_LOG_DIR}"
  lattice_job_banner "pipeline wrote ADAPTER_RUN_ID=${ADAPTER_RUN_ID} → ${PIPELINE_ENV}"
fi
