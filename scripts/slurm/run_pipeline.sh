#!/usr/bin/env bash
#
# Submit stages 2→6 as separate SLURM jobs with dependencies (login node).
#
# Env / positional args:
#   METHOD     lejepa | ntxent | ijepa | denoise   (default: lejepa; positional $1)
#   RUN_NAME   optional W&B run name for stage 2   (positional $2)
#   PROTEIN    esm2 | esmc — stage 3 only, no overwrite (default: esm2)
#   N_SEEDS    EBM seeds in stage 5 (default: 1; N_SEEDS=3 → 3 ckpts / ensemble eval)
#   SMOKE=1    fast wiring test: 1 val epoch, ~1% SSL data, 5k-row pools, 50 EBM steps
#
#   ./scripts/slurm/run_pipeline.sh lejepa
#   PROTEIN=esmc ./scripts/slurm/run_pipeline.sh ijepa my_run_name
#   SMOKE=1 ./scripts/slurm/run_pipeline.sh lejepa
#
# Run ids land in logs/slurm/pipeline/<pipeline_id>.env (+ .ebm.<seed> sidecars).
# After stage 2, SLURM logs move to logs/slurm/pipeline/<adapter_wandb_run_id>/.
set -euo pipefail

cd "$(dirname "$0")/../.."
source "scripts/slurm/common.sh"
lattice_cd_repo

METHOD="${1:-${METHOD:-lejepa}}"
RUN_NAME="${2:-${RUN_NAME:-}}"
PROTEIN="${PROTEIN:-esm2}"
N_SEEDS="${N_SEEDS:-1}"
SMOKE="${SMOKE:-0}"

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

PIPELINE_ID="$(date +%Y%m%d-%H%M%S)-$$"
PIPELINE_ENV="${REPO}/logs/slurm/pipeline/${PIPELINE_ID}.env"
SMOKE_DIR="${REPO}/logs/slurm/pipeline/${PIPELINE_ID}/smoke_data"
mkdir -p "${REPO}/logs/slurm/pipeline"
cat > "${PIPELINE_ENV}" <<EOF
PIPELINE_ID=${PIPELINE_ID}
METHOD=${METHOD}
N_SEEDS=${N_SEEDS}
PROTEIN=${PROTEIN}
PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD}
SMOKE=${SMOKE}
EOF

if [[ "${SMOKE}" == 1 ]]; then
  mkdir -p "${SMOKE_DIR}"
  cat >> "${PIPELINE_ENV}" <<EOF
SMOKE_PARQUET_DIR=${SMOKE_DIR}
SMOKE_TEST_PARQUET=${SMOKE_DIR}/test_lit_pcba.parquet
EOF
  SBATCH_TIME=(--time=01:00:00)
  echo "SMOKE=1: parquets will be sampled at stage 4 → ${SMOKE_DIR}" >&2
else
  SBATCH_TIME=()
fi

EXPORT="ALL,PIPELINE_ENV=${PIPELINE_ENV},PIPELINE_ID=${PIPELINE_ID},METHOD=${METHOD},N_SEEDS=${N_SEEDS},PROTEIN=${PROTEIN},PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD},SMOKE=${SMOKE},OVERWRITE=0"
[[ "${SMOKE}" == 1 ]] && EXPORT="${EXPORT},SMOKE_PARQUET_DIR=${SMOKE_DIR},SMOKE_TEST_PARQUET=${SMOKE_DIR}/test_lit_pcba.parquet"

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

cat <<EOF
pipeline ${PIPELINE_ID} submitted${SMOKE:+ (SMOKE=1)}
  env:     ${PIPELINE_ENV}
  stage 2: ${J2}  (adapter SSL)
  stage 3: ${J3}  (protein ${PROTEIN}, after ${J2})
  stage 4: ${J4}  (decoy pools, after ${J2})
  stage 5: ${J5}  (EBM x${N_SEEDS}, after ${J4})
  stage 6: ${J6}  (LIT-PCBA, after ${J5} + ${J3})

After stage 2:  grep ADAPTER_RUN_ID ${PIPELINE_ENV}
               logs → logs/slurm/pipeline/<ADAPTER_RUN_ID>/
After stage 5:  cat ${PIPELINE_ENV}.ebm.*
EOF
