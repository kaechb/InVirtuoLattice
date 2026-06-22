#!/usr/bin/env bash
#SBATCH --job-name=lattice-s6-eval
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage6/%j.out
#SBATCH --error=logs/slurm/stage6/%j.err
#
# Stage 6 — LIT-PCBA eval.
#
#   sbatch scripts/slurm/stage6_eval.sh <ebm_run_id>                    # 4-view cache + CSV
#   sbatch scripts/slurm/stage6_eval.sh <id0> <id1> <id2>             # 4-view cache + ensemble JSON
#   SSL_RUN_ID=<stage2_id> sbatch scripts/slurm/stage6_eval.sh --single-view <ebm_run_id>
#
# Pipeline (no args): reads EBM run ids from PIPELINE_ENV.ebm.* when PIPELINE_ENV is set.
#
# Env fallbacks: RUN_ID (one ckpt), RUN_ID0/1/2 (ensemble), CKPT (single-view override).
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage6_eval.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_pipeline_source_env

SINGLE_VIEW=0
N_VIEWS=4
N_JOBS_CACHE=40
N_JOBS_EVAL=7
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --single-view)
      SINGLE_VIEW=1
      N_VIEWS=1
      shift
      ;;
    --n-views)
      N_VIEWS="$2"
      shift 2
      ;;
    --n-jobs-cache)
      N_JOBS_CACHE="$2"
      shift 2
      ;;
    --n-jobs-eval)
      N_JOBS_EVAL="$2"
      shift 2
      ;;
    -*)
      echo "unknown flag: $1 (want --single-view, --n-views, --n-jobs-cache, --n-jobs-eval)" >&2
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -eq 0 && -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
  lattice_pipeline_source_env
  N_SEEDS="${N_SEEDS:-1}"
  for SEED in $(seq 0 $((N_SEEDS - 1))); do
    F="${PIPELINE_ENV}.ebm.${SEED}"
    lattice_require_file "${F}" "missing ${F} — stage5 seed ${SEED} did not finish"
    POSITIONAL+=("$(cat "${F}")")
  done
fi

PROTEIN_STORE="artifacts/protein_store/embeddings/esm2_650M"
TEST_PARQUET="artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet"
LITPCBA_EXTRA=()
CACHE_EXTRA=()

if lattice_smoke_enabled; then
  N_VIEWS=1
  N_JOBS_CACHE=4
  N_JOBS_EVAL=2
  lattice_job_banner "SMOKE: n_views=1 all litpcba targets test_parquet=${TEST_PARQUET}"
fi

lattice_load_gpu_modules
lattice_cd_repo
[[ -n "${PIPELINE_LOG_DIR:-}" ]] && trap 'lattice_pipeline_collect_logs_on_exit 6' EXIT
lattice_require_gpu

_run_mv4_cache() {
  local ckpt="$1" zm_cache="$2"
  lattice_require_file "${REPO}/${ckpt}" "missing EBM checkpoint: ${ckpt}"
  srun python -m lattice_lab.eval.build_multiview_cache \
    --n-views "${N_VIEWS}" \
    --adapter-ckpt "${ckpt}" \
    --zm-cache "${zm_cache}" \
    --test-parquet "${TEST_PARQUET}" \
    --n-jobs "${N_JOBS_CACHE}" \
    "${CACHE_EXTRA[@]}"
}

if [[ "${SINGLE_VIEW}" -eq 1 ]]; then
  if [[ -n "${CKPT:-}" ]]; then
    :
  elif [[ ${#POSITIONAL[@]} -ge 1 ]]; then
    CKPT="artifacts/energy/checkpoints/${POSITIONAL[0]}/last.ckpt"
  else
    : "${RUN_ID:?set RUN_ID=<ebm_wandb_run_id> (or pass as \$1, or CKPT=<path>)}"
    CKPT="artifacts/energy/checkpoints/${RUN_ID}/last.ckpt"
  fi
  SSL_RUN_ID="${SSL_RUN_ID:?set SSL_RUN_ID=<stage2_wandb_run_id> for --single-view cache path}"
  TAG="$(basename "$(dirname "${CKPT}")")"
  ZM_CACHE="artifacts/evaluation/${SSL_RUN_ID}/lit_pcba_zm_sv"
  OUT_CSV="artifacts/evaluation/${SSL_RUN_ID}/lit_pcba_${TAG}.csv"

  lattice_require_file "${REPO}/${CKPT}" \
    "missing EBM checkpoint (set RUN_ID or CKPT)"

  srun python -m lattice_lab.eval.lit_pcba \
    --head-ckpt "${CKPT}" \
    --adapter-ckpt "${CKPT}" \
    --zm-cache "${ZM_CACHE}" \
    --adapter-run-id "${SSL_RUN_ID}" \
    --protein-store "${PROTEIN_STORE}" \
    --test-parquet "${TEST_PARQUET}" \
    --output-csv "${OUT_CSV}" \
    --n-jobs "${N_JOBS_EVAL}" \
    "${LITPCBA_EXTRA[@]}"

elif [[ ${#POSITIONAL[@]} -eq 3 || ( ${#POSITIONAL[@]} -eq 0 && -n "${RUN_ID0:-}" ) ]]; then
  RUN_ID0="${POSITIONAL[0]:-${RUN_ID0:?set RUN_ID0=<ebm_seed0_wandb_id> (or pass three run ids)}}"
  RUN_ID1="${POSITIONAL[1]:-${RUN_ID1:?set RUN_ID1=<ebm_seed1_wandb_id>}}"
  RUN_ID2="${POSITIONAL[2]:-${RUN_ID2:?set RUN_ID2=<ebm_seed2_wandb_id>}}"
  CKPT0="artifacts/energy/checkpoints/${RUN_ID0}/last.ckpt"
  CKPT1="artifacts/energy/checkpoints/${RUN_ID1}/last.ckpt"
  CKPT2="artifacts/energy/checkpoints/${RUN_ID2}/last.ckpt"
  ZM_CACHE="artifacts/evaluation/${RUN_ID0}/lit_pcba_zm_mv${N_VIEWS}"
  OUT_JSON="artifacts/evaluation/${RUN_ID0}/ensemble_mv${N_VIEWS}.json"
  N_JOBS_EVAL=32

  lattice_require_file "${REPO}/${CKPT0}" "missing checkpoint for RUN_ID0=${RUN_ID0}"
  lattice_require_file "${REPO}/${CKPT1}" "missing checkpoint for RUN_ID1=${RUN_ID1}"
  lattice_require_file "${REPO}/${CKPT2}" "missing checkpoint for RUN_ID2=${RUN_ID2}"

  _run_mv4_cache "${CKPT0}" "${ZM_CACHE}"

  srun python -m lattice_lab.eval.ensemble_eval \
    --ckpts "${CKPT0}" "${CKPT1}" "${CKPT2}" \
    --zm-cache "${ZM_CACHE}" \
    --protein-store "${PROTEIN_STORE}" \
    --test-parquet "${TEST_PARQUET}" \
    --out "${OUT_JSON}" \
    --n-jobs "${N_JOBS_EVAL}"

elif [[ ${#POSITIONAL[@]} -eq 1 || ( ${#POSITIONAL[@]} -eq 0 && -n "${RUN_ID:-}" ) ]]; then
  RUN_ID="${POSITIONAL[0]:-${RUN_ID:?set RUN_ID=<ebm_wandb_run_id> (or pass as \$1)}}"
  CKPT="artifacts/energy/checkpoints/${RUN_ID}/last.ckpt"
  ZM_CACHE="artifacts/evaluation/${RUN_ID}/lit_pcba_zm_mv${N_VIEWS}"
  OUT_CSV="artifacts/evaluation/${RUN_ID}/lit_pcba_mv${N_VIEWS}.csv"

  _run_mv4_cache "${CKPT}" "${ZM_CACHE}"

  srun python -m lattice_lab.eval.lit_pcba \
    --head-ckpt "${CKPT}" \
    --adapter-ckpt "${CKPT}" \
    --zm-cache "${ZM_CACHE}" \
    --protein-store "${PROTEIN_STORE}" \
    --test-parquet "${TEST_PARQUET}" \
    --output-csv "${OUT_CSV}" \
    --n-jobs "${N_JOBS_EVAL}" \
    "${LITPCBA_EXTRA[@]}"

else
  echo "usage: sbatch scripts/slurm/stage6_eval.sh <ebm_run_id>" >&2
  echo "       sbatch scripts/slurm/stage6_eval.sh <id0> <id1> <id2>" >&2
  echo "       SSL_RUN_ID=<stage2_id> sbatch scripts/slurm/stage6_eval.sh --single-view <ebm_run_id>" >&2
  echo "       sbatch scripts/slurm/stage6_eval.sh  # pipeline: reads PIPELINE_ENV.ebm.*" >&2
  exit 1
fi
