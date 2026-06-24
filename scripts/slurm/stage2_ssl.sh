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
MERGE_SUFFIX="$(lattice_merge_suffix)"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

TRAIN_ARGS=(
  "data.shard_dir=${LATTICE_FLASH_PROCESSED}/moses${MERGE_SUFFIX}"
  trainer.accelerator=gpu
  callbacks.model_checkpoint.dirpath=artifacts/adapter/checkpoints
)

case "${METHOD}" in
  ntxent)
    TRAIN_ARGS=(
      "experiment=adapter_discrete_flow"
      "${TRAIN_ARGS[@]}"
      "model.ssl_loss=ntxent"
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
      "model.ijepa_gram_weight=${GRAM_WEIGHT:-1.0}"
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

# Morgan fp distillation (model.fp_weight defaults to 2) needs SMILES in each batch.
# Override here so pipeline frozen configs stay correct even if snapshotted before a fix.
if [[ "${METHOD}" != denoise ]]; then
  TRAIN_ARGS+=("data.return_smiles=true")
fi

if [[ -n "${RUN_NAME}" ]] && [[ -z "${PIPELINE_ENV:-}" ]]; then
  TRAIN_ARGS+=("logger.wandb.name=${RUN_NAME}")
fi

if lattice_smoke_enabled; then
  TRAIN_ARGS+=(
    trainer.check_val_every_n_epoch=1
    trainer.val_check_interval=null
    trainer.limit_train_batches=0.01
    trainer.limit_val_batches=2
  )
  [[ -z "${RUN_NAME}" && -z "${PIPELINE_ENV:-}" ]] && TRAIN_ARGS+=("logger.wandb.name=smoke_${METHOD}")
  lattice_job_banner "SMOKE: 1 epoch, 1% train batches, 2 val batches"
fi

PIPELINE_CONFIG=()
if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
  if grep -q '^ADAPTER_RUN_ID=' "${PIPELINE_ENV}"; then
    ADAPTER_RUN_ID="$(grep '^ADAPTER_RUN_ID=' "${PIPELINE_ENV}" | tail -1 | cut -d= -f2-)"
  else
    ADAPTER_RUN_ID="$(python -c 'import wandb; print(wandb.util.generate_id())')"
    echo "ADAPTER_RUN_ID=${ADAPTER_RUN_ID}" >> "${PIPELINE_ENV}"
  fi
  WANDB_NAME="stage2_${ADAPTER_RUN_ID}"
  [[ -n "${RUN_NAME:-}" ]] && WANDB_NAME="${RUN_NAME}_${WANDB_NAME}"
  TRAIN_ARGS+=(
    "logger.wandb.id=${ADAPTER_RUN_ID}"
    "logger.wandb.name=${WANDB_NAME}"
  )
  lattice_pipeline_init_log_dir "${ADAPTER_RUN_ID}"
  lattice_pipeline_snapshot_configs
  PIPELINE_CONFIG=(--config-path="${PIPELINE_LOG_DIR}/configs")
  lattice_pipeline_save_train_args 2 "${PIPELINE_CONFIG[@]}" "${TRAIN_ARGS[@]}"
fi

srun python -m lattice_lab.train "${PIPELINE_CONFIG[@]}" "${TRAIN_ARGS[@]}"

if [[ -n "${PIPELINE_ENV:-}" ]]; then
  lattice_pipeline_source_env
  lattice_require_file "${REPO}/artifacts/adapter/checkpoints/${ADAPTER_RUN_ID}/last.ckpt" \
    "stage2 finished but no adapter checkpoint at ADAPTER_RUN_ID=${ADAPTER_RUN_ID}"
  if lattice_smoke_enabled; then
    SMOKE_PARQUET_DIR="${PIPELINE_LOG_DIR}/smoke_data"
    SMOKE_TEST_PARQUET="${SMOKE_PARQUET_DIR}/test_lit_pcba.parquet"
    mkdir -p "${SMOKE_PARQUET_DIR}"
    echo "SMOKE_PARQUET_DIR=${SMOKE_PARQUET_DIR}" >> "${PIPELINE_ENV}"
    echo "SMOKE_TEST_PARQUET=${SMOKE_TEST_PARQUET}" >> "${PIPELINE_ENV}"
  fi
  lattice_pipeline_install_env
  lattice_pipeline_collect_slurm_logs 2
  lattice_job_banner "pipeline logs → ${PIPELINE_LOG_DIR}"
  lattice_job_banner "pipeline state → ${PIPELINE_LOG_DIR}/pipeline.env"
fi
