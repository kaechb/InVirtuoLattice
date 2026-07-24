#!/usr/bin/env bash
#
# Leave-one-out component ablations around a finished winner (default: w790kdrh).
# Each run drops one piece of the full stack; Stage-6 LIT-PCBA BEDROC is the score.
#
#   ./scripts/slurm/run_component_ablations.sh
#   DRY_RUN=1 ./scripts/slurm/run_component_ablations.sh
#   BASE_RUN_ID=w790kdrh N_SEEDS=3 ./scripts/slurm/run_component_ablations.sh
#
# Ablations (vs full w790kdrh):
#   no_ssl          — pretrained DDiT + random adapter (no Stage-2 NT-Xent); rebuilds z_m
#   cosine_match    — FiLM energy head → linear contrastive matching (model.head_type=cosine)
#   no_hardneg      — hard_mining_mult=3 → 1
#   no_bdb_mix      — BindingDB other-binder / non-binder fractions → 0 (MOSES decoys only)
#   no_cross_target — cross_target_p=0
#   no_sinkhorn     — lambda_sink=0
#
# EBM leave-one-outs reuse the base adapter + decoy/binder stores (symlink).
# no_ssl plants an init ckpt and re-runs Stage 4→6.
set -euo pipefail

cd "$(dirname "$0")/../.."
source "scripts/slurm/common.sh"
lattice_cd_repo

BASE_RUN_ID="${BASE_RUN_ID:-w790kdrh}"
N_SEEDS="${N_SEEDS:-3}"
DRY_RUN="${DRY_RUN:-0}"
PREFIX="${PREFIX:-comp}"
PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-logs/slurm/ablation/component}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-artifacts/ablation}"

BASE_ENV="$(lattice_pipeline_env_path "${BASE_RUN_ID}" || true)"
[[ -n "${BASE_ENV}" && -f "${BASE_ENV}" ]] || {
  echo "missing pipeline.env for BASE_RUN_ID=${BASE_RUN_ID}" >&2
  exit 1
}
BASE_CKPT="${REPO}/${ARTIFACTS_ROOT}/adapter/checkpoints/${BASE_RUN_ID}/last.ckpt"
[[ -f "${BASE_CKPT}" ]] || BASE_CKPT="${REPO}/artifacts/ablation/adapter/checkpoints/${BASE_RUN_ID}/last.ckpt"
lattice_require_file "${BASE_CKPT}" "missing base adapter ckpt for ${BASE_RUN_ID}"

mkdir -p "${REPO}/${PIPELINE_LOG_ROOT}" "${REPO}/${ARTIFACTS_ROOT}"/{adapter/checkpoints,decoys,binders,energy/checkpoints,evaluation}

n=0
# fork_ebm <label> <EXTRA_EBM_ARGS...>  — symlink stores, Stage 5→6 only
fork_ebm() {
  local label="$1"; shift
  local ebm_args="$*"
  local rid
  rid="$(lattice_generate_run_id)"
  local run_name="${PREFIX}_${label}"
  local log_dir="${REPO}/${PIPELINE_LOG_ROOT}/${rid}"
  local pe="${log_dir}/pipeline.env"

  n=$((n + 1))
  echo ""
  echo "=== [${n}] ${run_name}  rid=${rid}  EXTRA_EBM_ARGS='${ebm_args}' ==="

  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "  DRY_RUN: would symlink ${BASE_RUN_ID} stores → ${rid}, STAGE_FROM=5"
    return
  fi

  mkdir -p "${log_dir}"
  # Reuse base encoder + z_m pools (same latent space; fingerprint matches).
  ln -sfn "${BASE_RUN_ID}" "${REPO}/${ARTIFACTS_ROOT}/adapter/checkpoints/${rid}"
  ln -sfn "${BASE_RUN_ID}" "${REPO}/${ARTIFACTS_ROOT}/decoys/${rid}"
  ln -sfn "${BASE_RUN_ID}" "${REPO}/${ARTIFACTS_ROOT}/binders/${rid}"

  cat > "${pe}" <<EOF
PIPELINE_ID=$(date +%Y%m%d-%H%M%S)-$$
ADAPTER_RUN_ID=${rid}
PIPELINE_LOG_DIR=${log_dir}
PIPELINE_LOG_ROOT=${PIPELINE_LOG_ROOT}
ARTIFACTS_ROOT=${ARTIFACTS_ROOT}
ABLATION=1
METHOD=ntxent
N_SEEDS=${N_SEEDS}
MULTISEED=0
PROTEIN=esm2
PIPELINE_EBM_METHOD=ntxent
SMOKE=0
MERGE=0
VIEW3D=0
ENCODER_3D=0
RUN_NAME=${run_name}
STAGE2_WANDB_NAME=${run_name}_stage2_${rid}
D_PROTEIN=1280
PROTEIN_STORE=artifacts/protein_store/embeddings/esm2_650M
COMPONENT_ABLATION=${label}
BASE_RUN_ID=${BASE_RUN_ID}
EOF

  PIPELINE_ENV="${pe}" PIPELINE_LOG_DIR="${log_dir}" PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT}" \
    ARTIFACTS_ROOT="${ARTIFACTS_ROOT}" \
    lattice_pipeline_freeze_at_submit "${rid}"
  lattice_pipeline_freeze_git "${log_dir}"

  EXTRA_EBM_ARGS="${ebm_args}" \
  PIPELINE_ENV="${pe}" ADAPTER_RUN_ID="${rid}" \
  N_SEEDS="${N_SEEDS}" ABLATION=1 \
  ARTIFACTS_ROOT="${ARTIFACTS_ROOT}" PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT}" \
  STAGE_FROM=5 \
    ./scripts/slurm/run_pipeline.sh ntxent
}

# no_ssl: plant init adapter, Stage 4→6
submit_no_ssl() {
  local label="no_ssl"
  local rid
  rid="$(lattice_generate_run_id)"
  local run_name="${PREFIX}_${label}"
  local log_dir="${REPO}/${PIPELINE_LOG_ROOT}/${rid}"
  local pe="${log_dir}/pipeline.env"
  local out_ckpt="${REPO}/${ARTIFACTS_ROOT}/adapter/checkpoints/${rid}/last.ckpt"

  n=$((n + 1))
  echo ""
  echo "=== [${n}] ${run_name}  rid=${rid}  (init adapter, STAGE_FROM=4) ==="

  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "  DRY_RUN: would dump no-SSL ckpt → ${out_ckpt}, STAGE_FROM=4"
    return
  fi

  mkdir -p "${log_dir}" "$(dirname "${out_ckpt}")"
  lattice_load_python_container
  export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
  python "${REPO}/scripts/dump_no_ssl_adapter.py" --from-ckpt "${BASE_CKPT}" --out "${out_ckpt}"

  cat > "${pe}" <<EOF
PIPELINE_ID=$(date +%Y%m%d-%H%M%S)-$$
ADAPTER_RUN_ID=${rid}
PIPELINE_LOG_DIR=${log_dir}
PIPELINE_LOG_ROOT=${PIPELINE_LOG_ROOT}
ARTIFACTS_ROOT=${ARTIFACTS_ROOT}
ABLATION=1
METHOD=ntxent
N_SEEDS=${N_SEEDS}
MULTISEED=0
PROTEIN=esm2
PIPELINE_EBM_METHOD=ntxent
SMOKE=0
MERGE=0
VIEW3D=0
ENCODER_3D=0
RUN_NAME=${run_name}
STAGE2_WANDB_NAME=${run_name}_stage2_${rid}
D_PROTEIN=1280
PROTEIN_STORE=artifacts/protein_store/embeddings/esm2_650M
COMPONENT_ABLATION=${label}
BASE_RUN_ID=${BASE_RUN_ID}
EOF

  PIPELINE_ENV="${pe}" PIPELINE_LOG_DIR="${log_dir}" PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT}" \
    ARTIFACTS_ROOT="${ARTIFACTS_ROOT}" \
    lattice_pipeline_freeze_at_submit "${rid}"
  lattice_pipeline_freeze_git "${log_dir}"

  PIPELINE_ENV="${pe}" ADAPTER_RUN_ID="${rid}" \
  N_SEEDS="${N_SEEDS}" ABLATION=1 \
  ARTIFACTS_ROOT="${ARTIFACTS_ROOT}" PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT}" \
  STAGE_FROM=4 \
    ./scripts/slurm/run_pipeline.sh ntxent
}

# --- submit ---
submit_no_ssl
fork_ebm cosine_match "model.head_type=cosine"
fork_ebm no_hardneg "data.hard_mining_mult=1"
fork_ebm no_bdb_mix "data.frac_other_binder=0 data.frac_non_binder=0 data.val_frac_other_binder=0 data.val_frac_non_binder=0"
fork_ebm no_cross_target "model.cross_target_p=0"
fork_ebm no_sinkhorn "model.lambda_sink=0"

echo ""
echo "submitted ${n} component ablations (base=${BASE_RUN_ID}, N_SEEDS=${N_SEEDS}) under ${PIPELINE_LOG_ROOT}/"
echo "baseline numbers stay in logs/slurm/ablation/new_ablation/${BASE_RUN_ID}_finished/"
echo "aggregate: python scripts/aggregate_ablations.py --roots ${PIPELINE_LOG_ROOT}"
