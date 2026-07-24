#!/usr/bin/env bash
#SBATCH --job-name=lattice-s5-ebm
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00
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
  trap 'rm -f "${PIPELINE_MARKER:-}" "${SEED_LOCK:-}"' EXIT
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
lattice_pipeline_track_slurm_logs 5
lattice_job_banner "modules loaded; checking gpu"
lattice_require_gpu

SEED_LOCK=""
if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
  _sidecar="$(lattice_pipeline_ebm_sidecar "${SEED}")"
  if [[ -f "${_sidecar}" ]] && ! lattice_pipeline_ebm_seed_adapter_ok "${SEED}"; then
    lattice_job_banner "seed=${SEED} stale sidecar (wrong adapter) — removing ${_sidecar}"
    rm -f "${_sidecar}"
  fi
  if lattice_pipeline_ebm_seed_done "${SEED}"; then
    lattice_job_banner "seed=${SEED} already finished — skip"
    exit 0
  fi
  SEED_LOCK="$(lattice_pipeline_ebm_seed_lock "${SEED}")"
  if [[ -f "${SEED_LOCK}" ]]; then
    other="$(<"${SEED_LOCK}")"
    if [[ -n "${other}" ]] && squeue -j "${other}" -h &>/dev/null; then
      lattice_job_banner "seed=${SEED} already running as job ${other} — skip"
      exit 0
    fi
    rm -f "${SEED_LOCK}"
  fi
  echo "${SLURM_JOB_ID:-local}" > "${SEED_LOCK}"
fi

ADAPTER_CKPT="${REPO}/$(lattice_artifacts_root)/adapter/checkpoints/${RUN_ID}/last.ckpt"
SSL_CKPT_REL="$(lattice_artifacts_root)/adapter/checkpoints/${RUN_ID}/last.ckpt"
if [[ -n "${PIPELINE_ENV:-}" ]]; then
  SSL_CKPT_REL="$(lattice_pipeline_ssl_ckpt "${RUN_ID}")"
  ADAPTER_CKPT="${REPO}/${SSL_CKPT_REL}"
  lattice_job_banner "pipeline ssl ckpt → ${SSL_CKPT_REL}"
fi
lattice_require_file "${ADAPTER_CKPT}" \
  "missing adapter ckpt — run stage2 first for RUN_ID=${RUN_ID}"

# Merge variant follows the adapter (fragment_merge in its ckpt): read the
# matching _merge z_m stores stage4 wrote. Always override frozen yaml paths
# (ablation uses artifacts/ablation/…; merge uses …_merge pool names).
MERGE_SUFFIX="$(lattice_ckpt_merge_suffix "${ADAPTER_CKPT}")"
# ENCODER_3D=1: read the Uni-Mol 3D-encoded *_zm3d pools stage4 wrote (same keys,
# same latent dim; the training code is identical). Must match how stage4 ran.
ZM_POOL_SUFFIX=""
[[ "${ENCODER_3D:-0}" == 1 ]] && ZM_POOL_SUFFIX="3d"
ZM_STORE_ARGS=(
  "data.decoy_store=$(lattice_zm_store_path decoy_zm${ZM_POOL_SUFFIX} "${RUN_ID}" "${MERGE_SUFFIX}")/"
  "data.bdb_store=$(lattice_zm_store_path bdb_zm${ZM_POOL_SUFFIX} "${RUN_ID}" "${MERGE_SUFFIX}")"
  "data.binder_store=$(lattice_zm_store_path binder_zm${ZM_POOL_SUFFIX} "${RUN_ID}" "${MERGE_SUFFIX}")"
)

DECOY_STORE="${REPO}/$(lattice_zm_store_path decoy_zm${ZM_POOL_SUFFIX} "${RUN_ID}" "${MERGE_SUFFIX}")/manifest.json"
lattice_require_file "${DECOY_STORE}" \
  "missing decoy store — run stage4 for RUN_ID=${RUN_ID} first"

lattice_job_banner "starting train (pool RAM load can take several minutes before first log line)"

# Train 3 full epochs (was a 12k-step cap); best ckpt is picked by val/loss
# (ModelCheckpoint monitor) across the step-based validations, and stage 6 now
# evaluates that best ckpt rather than last.ckpt.
TRAIN_EXTRA=(trainer.max_steps=-1 +trainer.max_epochs=3)
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

LOGGER_EXTRA=()
PIPELINE_CONFIG=()
PROTEIN_HYDRA=()
if [[ -n "${PIPELINE_ENV:-}" ]]; then
  lattice_pipeline_source_env
  if [[ -d "${PIPELINE_LOG_DIR}/configs" ]]; then
    PIPELINE_CONFIG=(--config-path="${PIPELINE_LOG_DIR}/configs")
  fi
  WANDB_NAME="$(lattice_pipeline_stage5_wandb_name "${SEED}")"
  LOGGER_EXTRA=("logger.wandb.name=${WANDB_NAME}")
fi
mapfile -t PROTEIN_HYDRA < <(lattice_protein_hydra_args)

STAGE5_ARGS=(
  "experiment=${EXPERIMENT}"
  "ssl_run_id=${RUN_ID}"
  "${TRAIN_EXTRA[@]}"
  "${LOGGER_EXTRA[@]}"
  "seed=${SEED}"
  "${ZM_STORE_ARGS[@]}"
  "${PROTEIN_HYDRA[@]}"
  data.batch_size=64
  callbacks.model_checkpoint.dirpath=$(lattice_artifacts_root)/energy/checkpoints
  trainer.precision=bf16-mixed
)
if [[ -n "${PIPELINE_ENV:-}" ]]; then
  STAGE5_ARGS+=("model.encoder.ckpt=${SSL_CKPT_REL}")
fi
# ENCODER_3D=1: restrict train/val to binders that have a 3D conformer (the
# filtered parquets stage4 wrote next to the binder_zm3d store) so no batch ever
# falls back to *2D* live encoding for a missing binder — which would silently
# mix latent spaces. The baked 2D encoder in the EBM ckpt then stays unused.
if [[ "${ENCODER_3D:-0}" == 1 ]]; then
  BINDER_STORE3D="$(lattice_zm_store_path binder_zm3d "${RUN_ID}" "${MERGE_SUFFIX}")"
  STAGE5_ARGS+=(
    "data.train_parquet=${BINDER_STORE3D}/train_3d.parquet"
    "data.val_parquet=${BINDER_STORE3D}/val_3d.parquet"
  )
fi
if [[ -n "${EXTRA_EBM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  STAGE5_ARGS+=(${EXTRA_EBM_ARGS})
fi
if [[ -n "${PIPELINE_LOG_DIR:-}" ]]; then
  lattice_pipeline_save_train_args 5 "${PIPELINE_CONFIG[@]}" "${STAGE5_ARGS[@]}"
fi

srun python -m lattice_lab.train "${PIPELINE_CONFIG[@]}" "${STAGE5_ARGS[@]}"

if [[ -n "${PIPELINE_ENV:-}" ]]; then
  EBM_RUN_ID="$(lattice_pipeline_ebm_run_id_from_log "${SEED}")" || {
    echo "could not parse EBM run id from ${PIPELINE_LOG_DIR}/stage5.seed${SEED}.out" >&2
    exit 1
  }
  EBM_CKPT_REL="$(lattice_pipeline_ebm_eval_ckpt "${EBM_RUN_ID}")"
  lattice_require_file "${REPO}/${EBM_CKPT_REL}" \
    "stage5 seed=${SEED} finished but no EBM checkpoint found (pipeline)"
  echo "${EBM_RUN_ID}" > "$(lattice_pipeline_ebm_sidecar "${SEED}")"
  rm -f "${PIPELINE_MARKER}"
  lattice_job_banner "pipeline wrote EBM seed=${SEED} → ${PIPELINE_LOG_DIR}/ebm.${SEED} ckpt=${EBM_CKPT_REL}"
fi
