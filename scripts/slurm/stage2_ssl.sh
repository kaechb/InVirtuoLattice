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
#   sbatch scripts/slurm/stage2_ssl.sh siglip
#   sbatch scripts/slurm/stage2_ssl.sh ijepa
#   sbatch scripts/slurm/stage2_ssl.sh ijepa my_ablation_name
#   sbatch scripts/slurm/stage2_ssl.sh denoise
#
# Checkpoints: artifacts/adapter/checkpoints/<wandb_run_id>/last.ckpt
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage2_ssl.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

METHOD="${1:-${METHOD:?set METHOD=lejepa|ntxent|siglip|ijepa|denoise (or pass as \$1)}}"
RUN_NAME="${2:-${RUN_NAME:-}}"
lattice_pipeline_source_env
MERGE_SUFFIX="$(lattice_merge_suffix)"

# VIEW3D=1: pretrain the 2D adapter *with* the cross-modal 3D point-cloud view
# (experiment=adapter3d). The 3D encoder is only a pretraining crutch — its weights
# (encoder_3d.*/pred_3d.*) are not under the "encoder." prefix, so every downstream
# stage's load_encoder_from_ckpt drops them automatically. Needs conformers.parquet
# (run stage1b_precompute_conformers first; the pipeline submits it for you).
VIEW3D="${VIEW3D:-0}"
case "${VIEW3D}" in
  0|false|no|"") VIEW3D=0 ;;
  1|true|yes)   VIEW3D=1 ;;
  *) echo "VIEW3D=${VIEW3D} (want 0 or 1)" >&2; exit 1 ;;
esac
EXPERIMENT="adapter_discrete_flow"
if [[ "${VIEW3D}" == 1 ]]; then
  [[ "${METHOD}" == denoise ]] && {
    echo "VIEW3D=1 is not supported with METHOD=denoise (2D adapter only)" >&2
    exit 1
  }
  EXPERIMENT="adapter3d"
fi

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

TRAIN_ARGS=(
  "data.shard_dir=${LATTICE_FLASH_PROCESSED}/moses${MERGE_SUFFIX}"
  trainer.accelerator=gpu
  "callbacks.model_checkpoint.dirpath=$(lattice_artifacts_root)/adapter/checkpoints"
)

case "${METHOD}" in
  ntxent)
    TRAIN_ARGS=(
      "experiment=${EXPERIMENT}"
      "${TRAIN_ARGS[@]}"
      "model.ssl_loss=ntxent"
    )
    ;;
  siglip)
    TRAIN_ARGS=(
      "experiment=${EXPERIMENT}"
      "${TRAIN_ARGS[@]}"
      "model.ssl_loss=siglip"
    )
    ;;
  lejepa)
    TRAIN_ARGS=(
      "experiment=${EXPERIMENT}"
      "${TRAIN_ARGS[@]}"
      "model.ssl_loss=lejepa"
    )
    ;;
  ijepa)
    TRAIN_ARGS=(
      "experiment=${EXPERIMENT}"
      "${TRAIN_ARGS[@]}"
      "model.ssl_loss=ijepa"
    )
    ;;
  denoise)
    TRAIN_ARGS=(
      "experiment=denoising_jepa"
      "${TRAIN_ARGS[@]}"
    )
    ;;
  *)
    echo "unknown METHOD=${METHOD} (want lejepa, ntxent, siglip, ijepa, or denoise)" >&2
    exit 1
    ;;
esac

# Morgan fp distillation (model.fp_weight defaults to 2) needs SMILES in each batch.
# Override here so pipeline frozen configs stay correct even if snapshotted before a fix.
if [[ "${METHOD}" != denoise ]]; then
  TRAIN_ARGS+=("data.return_smiles=true")
fi

# Point the 3D conformer cache at the shard set actually being trained on (repo
# processed dir + merge suffix), overriding the experiment's static default.
if [[ "${VIEW3D}" == 1 ]]; then
  TRAIN_ARGS+=(
    "data.conformer_cache=${REPO}/artifacts/preprocessing/processed/moses${MERGE_SUFFIX}/conformers.parquet"
  )
fi

# Dual attention pooling (two half-width pools, z_m=concat) is ONLY sound when a
# contrastive loss owns the projection half AND view3d trains the regression half.
# Off by default in the yaml; enable it here exactly for that pairing. Enabling it
# elsewhere leaves the regression half untrained (random) and half of every
# downstream z_m becomes noise -> Stage-4/5 EBM collapses.
if [[ "${VIEW3D}" == 1 && ( "${METHOD}" == ntxent || "${METHOD}" == siglip ) ]]; then
  TRAIN_ARGS+=("model.encoder.adapter_dual_pool=true")
fi

# Extra Hydra overrides (space-separated), e.g.
#   EXTRA_TRAIN_ARGS="model.learning_rate=1e-4 model.ijepa_noise_inv_weight=1.0"
# Propagates into pipeline jobs via sbatch --export=ALL and is saved into the
# frozen stageN.train.args record below.
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  TRAIN_ARGS+=(${EXTRA_TRAIN_ARGS})
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
  lattice_pipeline_source_env
  if [[ -z "${ADAPTER_RUN_ID:-}" ]]; then
    ADAPTER_RUN_ID="$(lattice_generate_run_id)"
    echo "ADAPTER_RUN_ID=${ADAPTER_RUN_ID}" >> "${PIPELINE_ENV}"
  fi
  if [[ -z "${PIPELINE_LOG_DIR:-}" ]]; then
    lattice_pipeline_init_log_dir "${ADAPTER_RUN_ID}"
  fi
  WANDB_NAME="stage2_${ADAPTER_RUN_ID}"
  [[ -n "${RUN_NAME:-}" ]] && WANDB_NAME="${RUN_NAME}_${WANDB_NAME}"
  lattice_pipeline_set_env STAGE2_WANDB_NAME "${WANDB_NAME}"
  TRAIN_ARGS+=(
    "logger.wandb.id=${ADAPTER_RUN_ID}"
    "logger.wandb.name=${WANDB_NAME}"
  )
  # ponytail: run_pipeline freezes configs at submit; snapshot here only if missing (legacy submit)
  lattice_pipeline_snapshot_configs
  PIPELINE_CONFIG=(--config-path="${PIPELINE_LOG_DIR}/configs")
  lattice_pipeline_save_train_args 2 "${PIPELINE_CONFIG[@]}" "${TRAIN_ARGS[@]}"
fi

srun python -m lattice_lab.train "${PIPELINE_CONFIG[@]}" "${TRAIN_ARGS[@]}"

if [[ -n "${PIPELINE_ENV:-}" ]]; then
  lattice_pipeline_source_env
  lattice_require_file "${REPO}/$(lattice_artifacts_root)/adapter/checkpoints/${ADAPTER_RUN_ID}/last.ckpt" \
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
