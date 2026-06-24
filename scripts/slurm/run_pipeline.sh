#!/usr/bin/env bash
#
# Submit stages 2→6 as separate SLURM jobs with dependencies (login node).
#
# Env / positional args:
#   METHOD     lejepa | ntxent | ijepa | denoise   (default: lejepa; positional $1)
#   RUN_NAME   optional label for pipeline stage2 W&B name: {RUN_NAME}_stage2_{run_id}
#              (standalone stage 2: used as logger.wandb.name directly)
#   PROTEIN    esm2 | esmc — stage 3 only, no overwrite (default: esm2)
#   N_SEEDS    EBM seeds in stage 5 (default: 1; N_SEEDS=3 → 3 ckpts / ensemble eval)
#   MULTISEED=1  on an existing pipeline only: stage5 ×3 + stage6 ensemble (stages 2–4 unchanged)
#   SMOKE=1    fast wiring test: 1 val epoch, ~1% SSL data, 5k-row pools, 50 EBM steps
#
# Pipeline W&B names: [{RUN_NAME}_]stage2_{adapter_run_id}, stage5_{adapter_run_id}[_sN]
#
#   ./scripts/slurm/run_pipeline.sh lejepa
#   N_SEEDS=3 ./scripts/slurm/run_pipeline.sh lejepa          # fresh run, 3 EBM seeds
#   MULTISEED=1 ADAPTER_RUN_ID=avy80iqo ./scripts/slurm/run_pipeline.sh   # add 3 seeds to existing run
#   ./scripts/slurm/run_pipeline.sh ijepa ijepa_blockhole_smoke
#   SMOKE=1 ./scripts/slurm/run_pipeline.sh lejepa fpdistill_smoke
#
# Run dir: logs/slurm/pipeline/<adapter_run_id>/ (pipeline.env, configs/, ebm.<seed>, …).
set -euo pipefail

cd "$(dirname "$0")/../.."
source "scripts/slurm/common.sh"
lattice_cd_repo

METHOD="${1:-${METHOD:-lejepa}}"
RUN_NAME="${2:-${RUN_NAME:-}}"
PROTEIN="${PROTEIN:-esm2}"
N_SEEDS="${N_SEEDS:-1}"
MULTISEED="${MULTISEED:-0}"
SMOKE="${SMOKE:-0}"
MERGE="${MERGE:-0}"

case "${MULTISEED}" in
  0|false|no|"") MULTISEED=0 ;;
  1|true|yes)   MULTISEED=1 ;;
  *)
    echo "MULTISEED=${MULTISEED} (want 0 or 1)" >&2
    exit 1
    ;;
esac

case "${MERGE}" in
  0|false|no|"") MERGE=0 ;;
  1|true|yes)   MERGE=1 ;;
  *)
    echo "MERGE=${MERGE} (want 0 or 1)" >&2
    exit 1
    ;;
esac

case "${METHOD}" in
  lejepa|ntxent|ijepa|denoise) ;;
  *)
    echo "unknown METHOD=${METHOD} (want lejepa, ntxent, ijepa, or denoise)" >&2
    exit 1
    ;;
esac

case "${PROTEIN}" in
  esm2|esm|esmc) ;;
  *)
    echo "unknown PROTEIN=${PROTEIN} (want esm2 or esmc)" >&2
    exit 1
    ;;
esac

if ! [[ "${N_SEEDS}" =~ ^[0-9]+$ ]] || [[ "${N_SEEDS}" -lt 1 ]]; then
  echo "N_SEEDS must be a positive integer, got: ${N_SEEDS}" >&2
  exit 1
fi

case "${METHOD}" in
  ntxent) PIPELINE_EBM_METHOD=ntxent ;;
  *)      PIPELINE_EBM_METHOD=lejepa ;;
esac

# MULTISEED=1: re-submit stage 5 (×3) + stage 6 on a pipeline that already finished 2–4.
if [[ "${MULTISEED}" == 1 ]]; then
  _pe=""
  if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
    _pe="${PIPELINE_ENV}"
  elif [[ -n "${ADAPTER_RUN_ID:-}" ]]; then
    _pe="${REPO}/logs/slurm/pipeline/${ADAPTER_RUN_ID}/pipeline.env"
    if [[ ! -f "${_pe}" ]]; then
      # ponytail: O(n) scan — user often passes EBM run id (ebm.* sidecar) not adapter dir name
      _resolved="$(grep -l "^${ADAPTER_RUN_ID}$" "${REPO}/logs/slurm/pipeline"/*/ebm.* 2>/dev/null | head -1 || true)"
      if [[ -n "${_resolved}" ]]; then
        _dir="$(dirname "${_resolved}")"
        echo "note: ${ADAPTER_RUN_ID} is an EBM run id; using pipeline ADAPTER_RUN_ID=$(basename "${_dir}")" >&2
        ADAPTER_RUN_ID="$(basename "${_dir}")"
        _pe="${_dir}/pipeline.env"
      fi
    fi
    [[ -f "${_pe}" ]] || _pe=""
  fi
  if [[ -z "${_pe}" ]]; then
    echo "MULTISEED=1 only adds 3-seed EBM + ensemble eval to an existing pipeline (stages 2–4 are not re-run)." >&2
    echo "Set ADAPTER_RUN_ID=<adapter_run_id> (pipeline folder under logs/slurm/pipeline/, e.g. wk8denar)" >&2
    echo "  or PIPELINE_ENV=logs/slurm/pipeline/<adapter_run_id>/pipeline.env" >&2
    echo "  (EBM W&B run ids in ebm.* sidecars are also accepted as ADAPTER_RUN_ID)" >&2
    echo "For a fresh end-to-end run with 3 seeds: N_SEEDS=3 ./scripts/slurm/run_pipeline.sh ${METHOD}" >&2
    exit 1
  fi
  PIPELINE_ENV="${_pe}"
  lattice_pipeline_source_env
  if [[ -z "${ADAPTER_RUN_ID:-}" ]]; then
    echo "pipeline ${PIPELINE_ENV} has no ADAPTER_RUN_ID (stage 2 not finished?)" >&2
    exit 1
  fi
  N_SEEDS=3
  lattice_pipeline_set_env N_SEEDS 3
  lattice_pipeline_set_env MULTISEED 1
  lattice_pipeline_source_env

  if [[ "${SMOKE:-0}" == 1 ]]; then
    SBATCH_TIME=(--time=01:00:00)
  else
    SBATCH_TIME=()
  fi

  EXPORT="ALL,PIPELINE_ENV=${PIPELINE_ENV},PIPELINE_ID=${PIPELINE_ID},METHOD=${METHOD},N_SEEDS=${N_SEEDS},MULTISEED=1,PROTEIN=${PROTEIN},PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD},SMOKE=${SMOKE:-0},MERGE=${MERGE:-0},OVERWRITE=0"

  ARRAY_MAX=$((N_SEEDS - 1))
  J5="$(sbatch --parsable "${SBATCH_TIME[@]}" \
    --array="0-${ARRAY_MAX}" \
    --export="${EXPORT}" \
    scripts/slurm/stage5_ebm_train.sh)"
  J6="$(sbatch --parsable "${SBATCH_TIME[@]}" \
    --dependency="afterok:${J5}" \
    --export="${EXPORT}" \
    scripts/slurm/stage6_eval.sh)"

  cat <<EOF
multiseed continue on ${ADAPTER_RUN_ID} (stages 2–4 unchanged)
  env:     ${PIPELINE_ENV}
  stage 5: ${J5}  (EBM x${N_SEEDS})
  stage 6: ${J6}  (LIT-PCBA mv4 + 3-seed ensemble, after ${J5})

After stage 5:  cat logs/slurm/pipeline/${ADAPTER_RUN_ID}/ebm.*
EOF
  exit 0
fi

PIPELINE_ID="$(date +%Y%m%d-%H%M%S)-$$"
PIPELINE_ENV="${REPO}/logs/slurm/pipeline/${PIPELINE_ID}.env"
mkdir -p "${REPO}/logs/slurm/pipeline"
cat > "${PIPELINE_ENV}" <<EOF
PIPELINE_ID=${PIPELINE_ID}
METHOD=${METHOD}
N_SEEDS=${N_SEEDS}
MULTISEED=${MULTISEED}
PROTEIN=${PROTEIN}
PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD}
SMOKE=${SMOKE}
MERGE=${MERGE}
EOF
[[ -n "${RUN_NAME}" ]] && echo "RUN_NAME=${RUN_NAME}" >> "${PIPELINE_ENV}"

if [[ "${SMOKE}" == 1 ]]; then
  SBATCH_TIME=(--time=01:00:00)
  echo "SMOKE=1: parquets at stage 4 → logs/slurm/pipeline/<ADAPTER_RUN_ID>/smoke_data" >&2
else
  SBATCH_TIME=()
fi

EXPORT="ALL,PIPELINE_ENV=${PIPELINE_ENV},PIPELINE_ID=${PIPELINE_ID},METHOD=${METHOD},N_SEEDS=${N_SEEDS},MULTISEED=${MULTISEED},PROTEIN=${PROTEIN},PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD},SMOKE=${SMOKE},MERGE=${MERGE},OVERWRITE=0"

S2_ARGS=(scripts/slurm/stage2_ssl.sh "${METHOD}")
[[ -n "${RUN_NAME}" ]] && S2_ARGS+=("${RUN_NAME}")

J2="$(sbatch --parsable "${SBATCH_TIME[@]}" --export="${EXPORT}" "${S2_ARGS[@]}")"
J3="$(sbatch --parsable "${SBATCH_TIME[@]}" --dependency="afterok:${J2}" --export="${EXPORT}" \
  scripts/slurm/stage3_protein_precompute.sh)"
J4="$(sbatch --parsable "${SBATCH_TIME[@]}" --dependency="afterok:${J2}" --export="${EXPORT}" \
  scripts/slurm/stage4_precompute_decoys.sh)"

ARRAY_MAX=$((N_SEEDS - 1))
J5="$(sbatch --parsable "${SBATCH_TIME[@]}" \
  --array="0-${ARRAY_MAX}" \
  --dependency="afterok:${J4}" \
  --export="${EXPORT}" \
  scripts/slurm/stage5_ebm_train.sh)"

J6="$(sbatch --parsable "${SBATCH_TIME[@]}" \
  --dependency="afterok:${J5},afterok:${J3}" \
  --export="${EXPORT}" \
  scripts/slurm/stage6_eval.sh)"

if [[ "${SMOKE}" == 1 ]]; then
  _smoke_suffix=" (SMOKE=1)"
else
  _smoke_suffix=""
fi
cat <<EOF
pipeline ${PIPELINE_ID} submitted${_smoke_suffix}
  env:     ${PIPELINE_ENV}
  stage 2: ${J2}  (adapter SSL)
  stage 3: ${J3}  (protein ${PROTEIN}, after ${J2})
  stage 4: ${J4}  (decoy pools, after ${J2})
  stage 5: ${J5}  (EBM x${N_SEEDS}, after ${J4})
  stage 6: ${J6}  (LIT-PCBA$([ "${N_SEEDS}" -eq 3 ] && echo ' mv4 + 3-seed ensemble' || echo ' mv4 + single ckpt'), after ${J5} + ${J3})

After stage 2:  grep ADAPTER_RUN_ID ${PIPELINE_ENV}
               logs + frozen configs → logs/slurm/pipeline/<ADAPTER_RUN_ID>/
After stage 5:  cat logs/slurm/pipeline/<ADAPTER_RUN_ID>/ebm.*
EOF
