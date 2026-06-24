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
# Pipeline (no args): reads EBM run ids from <ADAPTER_RUN_ID>/ebm.* when PIPELINE_ENV is set.
#   N_SEEDS=3 (or MULTISEED=1): build mv4 cache + ensemble_eval over three stage-5 ckpts.
#
# Env fallbacks: RUN_ID (one ckpt), RUN_ID0/1/2 (ensemble), CKPT (single-view override).
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage6_eval.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_pipeline_source_env

SINGLE_VIEW=0
N_VIEWS=4
N_JOBS_CACHE=6
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
    F="$(lattice_pipeline_ebm_sidecar "${SEED}")"
    lattice_require_file "${F}" "missing ${F} — stage5 seed ${SEED} did not finish"
    POSITIONAL+=("$(cat "${F}")")
  done
fi

PROTEIN_STORE="artifacts/protein_store/embeddings/esm2_650M"
LITPCBA_EXTRA=()
CACHE_EXTRA=()

lattice_load_gpu_modules
lattice_cd_repo
lattice_pipeline_track_slurm_logs 6
lattice_require_gpu

_ebm_ckpt() {
  local run_id="$1"
  if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
    lattice_pipeline_ebm_ckpt "${run_id}"
  else
    echo "artifacts/energy/checkpoints/${run_id}/last.ckpt"
  fi
}

# Merge variant follows the EBM checkpoint's adapter (fragment_merge rides in its
# encoder_config). Eval re-encodes ligands faithfully, so this only locates the
# test parquet, which stage1 wrote under bindingdb_merge/ for the merge variant.
_MERGE_CKPT="${CKPT:-}"
if [[ -z "${_MERGE_CKPT}" ]]; then
  _rid="${POSITIONAL[0]:-${RUN_ID0:-${RUN_ID:-}}}"
  [[ -n "${_rid}" ]] && _MERGE_CKPT="$(_ebm_ckpt "${_rid}")"
fi
MERGE_SUFFIX=""
[[ -n "${_MERGE_CKPT}" && -f "${REPO}/${_MERGE_CKPT}" ]] && \
  MERGE_SUFFIX="$(lattice_ckpt_merge_suffix "${REPO}/${_MERGE_CKPT}")"
TEST_PARQUET="artifacts/preprocessing/processed/bindingdb${MERGE_SUFFIX}/test_lit_pcba.parquet"

if lattice_smoke_enabled; then
  N_JOBS_CACHE=4
  N_JOBS_EVAL=2
  [[ -n "${SMOKE_TEST_PARQUET:-}" ]] && TEST_PARQUET="${SMOKE_TEST_PARQUET}"
  lattice_job_banner "SMOKE: n_views=${N_VIEWS} test_parquet=${TEST_PARQUET} (all targets)"
fi

if [[ "${N_SEEDS:-1}" -eq 3 && ${#POSITIONAL[@]} -ne 3 ]]; then
  echo "N_SEEDS=3 requires three ebm.* sidecars (found ${#POSITIONAL[@]} run id(s))" >&2
  exit 1
fi

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
    CKPT="$(_ebm_ckpt "${POSITIONAL[0]}")"
  else
    : "${RUN_ID:?set RUN_ID=<ebm_wandb_run_id> (or pass as \$1, or CKPT=<path>)}"
    CKPT="$(_ebm_ckpt "${RUN_ID}")"
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
    --n-views 1 \
    --n-jobs "${N_JOBS_EVAL}" \
    "${LITPCBA_EXTRA[@]}"

elif [[ ${#POSITIONAL[@]} -eq 3 || ( ${#POSITIONAL[@]} -eq 0 && -n "${RUN_ID0:-}" ) ]]; then
  [[ ${#POSITIONAL[@]} -eq 3 ]] || {
    echo "ensemble eval needs three EBM run ids (set RUN_ID0/1/2 or pass three positional args)" >&2
    exit 1
  }
  RUN_ID0="${POSITIONAL[0]:-${RUN_ID0:?set RUN_ID0=<ebm_seed0_wandb_id> (or pass three run ids)}}"
  RUN_ID1="${POSITIONAL[1]:-${RUN_ID1:?set RUN_ID1=<ebm_seed1_wandb_id>}}"
  RUN_ID2="${POSITIONAL[2]:-${RUN_ID2:?set RUN_ID2=<ebm_seed2_wandb_id>}}"
  CKPT0="$(_ebm_ckpt "${RUN_ID0}")"
  CKPT1="$(_ebm_ckpt "${RUN_ID1}")"
  CKPT2="$(_ebm_ckpt "${RUN_ID2}")"
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
  CKPT="$(_ebm_ckpt "${RUN_ID}")"
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
    --n-views "${N_VIEWS}" \
    --skip-zm-precompute \
    --n-jobs "${N_JOBS_EVAL}" \
    "${LITPCBA_EXTRA[@]}"

else
  echo "usage: sbatch scripts/slurm/stage6_eval.sh <ebm_run_id>" >&2
  echo "       sbatch scripts/slurm/stage6_eval.sh <id0> <id1> <id2>" >&2
  echo "       SSL_RUN_ID=<stage2_id> sbatch scripts/slurm/stage6_eval.sh --single-view <ebm_run_id>" >&2
  echo "       sbatch scripts/slurm/stage6_eval.sh  # pipeline: reads <run_id>/ebm.*" >&2
  exit 1
fi
