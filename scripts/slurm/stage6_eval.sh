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

lattice_protein_resolve
LITPCBA_EXTRA=(--d-protein "${D_PROTEIN}")
ENSEMBLE_EXTRA=(--d-protein "${D_PROTEIN}")
CACHE_EXTRA=()

lattice_load_gpu_modules
lattice_cd_repo
lattice_pipeline_track_slurm_logs 6
lattice_require_gpu

_ebm_ckpt() {
  local run_id="$1"
  if [[ "${EVAL_PREFER_LAST:-0}" == 1 ]]; then
    echo "$(lattice_artifacts_root)/energy/checkpoints/${run_id}/last.ckpt"
    return
  fi
  if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
    lattice_pipeline_ebm_eval_ckpt "${run_id}"
  else
    echo "$(lattice_artifacts_root)/energy/checkpoints/${run_id}/last.ckpt"
  fi
}

# VIEW3D Stage-2 ckpt whose encoder_3d.* encodes LIT-PCBA conformers (ENCODER_3D=1).
_ssl_ckpt() {
  if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
    lattice_pipeline_ssl_ckpt "${ADAPTER_RUN_ID:?missing ADAPTER_RUN_ID for ENCODER_3D eval}"
  else
    echo "$(lattice_artifacts_root)/adapter/checkpoints/${SSL_RUN_ID:?set SSL_RUN_ID=<stage2_id> for ENCODER_3D eval}/last.ckpt"
  fi
}

# Sanity check, logged into stage6.out beside the LIT-PCBA JSON: the encoder
# baked into the EBM ckpt must reproduce the frozen stage-4 binder z_m that
# stage-5 val scored against (catches encoder / fragmentation / merge drift and
# surfaces the protein store dim). Non-fatal — results still compute on FAIL.
_zm_consistency() {
  local ckpt="$1" store
  [[ "${ENCODER_3D:-0}" == 1 ]] && { echo "ZM-CONSISTENCY: SKIP (ENCODER_3D — EBM ckpt bakes the 2D encoder, not the scoring 3D one)"; return 0; }
  [[ -n "${ADAPTER_RUN_ID:-}" ]] || { echo "ZM-CONSISTENCY: SKIP (no ADAPTER_RUN_ID; standalone eval)"; return 0; }
  store="$(lattice_zm_store_path binder_zm "${ADAPTER_RUN_ID}" "${MERGE_SUFFIX}")"
  srun python -m lattice_lab.eval.zm_consistency \
    --ebm-ckpt "${ckpt}" \
    --binder-store "${store}" \
    --protein-store "${PROTEIN_STORE}" \
    --d-protein "${D_PROTEIN}" \
    || echo "ZM-CONSISTENCY: check returned nonzero (see above) — continuing eval"
}

# End-to-end confirmation, logged into stage6.out beside the LIT-PCBA JSON:
# re-run the stage-5 validation (val/ef1|ef5|top1|bedroc) on the deterministic
# val split using the *saved* EBM ckpt (head + baked encoder). Matching the
# stage-5 log proves the promoted checkpoint is the one it was credited with.
# Replays the frozen stage-5 train args verbatim (train.py's validate path
# builds no W&B logger / checkpoints). Non-fatal; set VAL_REPRO=0 to skip.
_reproduce_val() {
  local ckpt="$1" seed="$2" args_file="${PIPELINE_LOG_DIR:-}/stage5.train.args"
  [[ "${VAL_REPRO:-1}" == 1 ]] || return 0
  [[ -f "${args_file}" ]] || { echo "VAL-REPRO: SKIP (no ${args_file}; standalone eval)"; return 0; }
  local _all=() _rest=() a
  mapfile -t _all < "${args_file}"
  # Drop the frozen (possibly pre-_finished, now-stale) --config-path; use the
  # live pipeline dir so re-running on a finished run still composes.
  for a in "${_all[@]}"; do [[ "$a" == --config-path=* ]] || _rest+=("$a"); done
  echo "VAL-REPRO: seed=${seed} validating ${ckpt} on the stage-5 val split ..."
  srun python -m lattice_lab.train \
    --config-path="${PIPELINE_LOG_DIR}/configs" \
    "${_rest[@]}" \
    +validate_only=true "seed=${seed}" "ckpt_path=${ckpt}" \
    || echo "VAL-REPRO: seed=${seed} returned nonzero (see above) — continuing eval"
}

# Merge test parquet: pipeline MERGE env, optionally confirmed from ckpt encoder_config.
MERGE_SUFFIX="$(lattice_merge_suffix)"
_rid="${POSITIONAL[0]:-${RUN_ID0:-${RUN_ID:-}}}"
if [[ -n "${_rid}" ]]; then
  _merge_ckpt="${REPO}/$(_ebm_ckpt "${_rid}")"
  if [[ -f "${_merge_ckpt}" ]]; then
    _from_ckpt="$(lattice_ckpt_merge_suffix "${_merge_ckpt}")"
    [[ -n "${_from_ckpt}" ]] && MERGE_SUFFIX="${_from_ckpt}"
  fi
fi
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
  local run_id="$1" zm_cache="$2" ckpt ssl
  ckpt="$(_ebm_ckpt "${run_id}")"
  lattice_require_file "${REPO}/${ckpt}" "missing EBM checkpoint: ${ckpt}"
  # ENCODER_3D=1: encode LIT-PCBA ligands with encoder_3d (from the Stage-2 ckpt);
  # tag the cache with the EBM ckpt's 2D fingerprint so the scorer accepts it (both
  # reference the same Stage-2 run). Same cache layout → scoring is unchanged.
  if [[ "${ENCODER_3D:-0}" == 1 ]]; then
    ssl="$(_ssl_ckpt)"
    lattice_require_file "${REPO}/${ssl}" "missing Stage-2 VIEW3D ckpt: ${ssl}"
    srun python -m lattice_lab.eval.build_conformer_cache \
      --n-views "${N_VIEWS}" \
      --encoder3d-ckpt "${ssl}" \
      --fp-ckpt "${ckpt}" \
      --zm-cache "${zm_cache}" \
      --protein-store "${PROTEIN_STORE}" \
      --test-parquet "${TEST_PARQUET}" \
      --n-jobs "${N_JOBS_CACHE}"
    return
  fi
  srun python -m lattice_lab.eval.build_multiview_cache \
    --n-views "${N_VIEWS}" \
    --adapter-ckpt "${ckpt}" \
    --zm-cache "${zm_cache}" \
    --protein-store "${PROTEIN_STORE}" \
    --test-parquet "${TEST_PARQUET}" \
    --n-jobs "${N_JOBS_CACHE}" \
    "${CACHE_EXTRA[@]}"
}

if [[ "${SINGLE_VIEW}" -eq 1 && "${ENCODER_3D:-0}" == 1 ]]; then
  echo "ENCODER_3D=1 is not supported with --single-view (that path encodes inline with the 2D encoder). Use the default multi-view mode." >&2
  exit 1
fi

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
  ZM_CACHE="$(lattice_evaluation_path "${SSL_RUN_ID}/lit_pcba_zm_sv")"
  OUT_CSV="$(lattice_evaluation_path "${SSL_RUN_ID}/lit_pcba_${TAG}.csv")"

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
  ZM_CACHE="$(lattice_evaluation_path "${RUN_ID0}/lit_pcba_zm_mv${N_VIEWS}")"
  OUT_JSON="$(lattice_evaluation_path "${RUN_ID0}/ensemble_mv${N_VIEWS}.json")"
  N_JOBS_EVAL=32

  lattice_require_file "${REPO}/${CKPT0}" "missing checkpoint for RUN_ID0=${RUN_ID0}"
  lattice_require_file "${REPO}/${CKPT1}" "missing checkpoint for RUN_ID1=${RUN_ID1}"
  lattice_require_file "${REPO}/${CKPT2}" "missing checkpoint for RUN_ID2=${RUN_ID2}"

  _zm_consistency "${CKPT0}"
  _reproduce_val "${CKPT0}" 0
  _reproduce_val "${CKPT1}" 1
  _reproduce_val "${CKPT2}" 2
  _run_mv4_cache "${RUN_ID0}" "${ZM_CACHE}"

  srun python -m lattice_lab.eval.ensemble_eval \
    --ckpts "${CKPT0}" "${CKPT1}" "${CKPT2}" \
    --zm-cache "${ZM_CACHE}" \
    --protein-store "${PROTEIN_STORE}" \
    --test-parquet "${TEST_PARQUET}" \
    --out "${OUT_JSON}" \
    --n-jobs "${N_JOBS_EVAL}" \
    "${ENSEMBLE_EXTRA[@]}"

elif [[ ${#POSITIONAL[@]} -eq 1 || ( ${#POSITIONAL[@]} -eq 0 && -n "${RUN_ID:-}" ) ]]; then
  RUN_ID="${POSITIONAL[0]:-${RUN_ID:?set RUN_ID=<ebm_wandb_run_id> (or pass as \$1)}}"
  CKPT="$(_ebm_ckpt "${RUN_ID}")"
  ZM_CACHE="$(lattice_evaluation_path "${RUN_ID}/lit_pcba_zm_mv${N_VIEWS}")"
  OUT_CSV="$(lattice_evaluation_path "${RUN_ID}/lit_pcba_mv${N_VIEWS}.csv")"

  _zm_consistency "${CKPT}"
  _reproduce_val "${CKPT}" 0
  _run_mv4_cache "${RUN_ID}" "${ZM_CACHE}"

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

if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
  lattice_pipeline_mark_finished || true
fi
