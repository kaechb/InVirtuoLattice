#!/usr/bin/env bash
#
# Train 6 extra EBM seeds (3–8) on a finished winner, then plot ensemble scaling.
#
#   ./scripts/slurm/run_ensemble_scaling.sh
#   BASE_RUN_ID=w790kdrh EXTRA_SEEDS=6 ./scripts/slurm/run_ensemble_scaling.sh
#   DRY_RUN=1 ./scripts/slurm/run_ensemble_scaling.sh
#
# Keeps existing ebm.0..2; submits stage5 array for the new seeds; after they
# finish, runs ensemble_scaling (mean±std BEDROC over all subsets of size k).
set -euo pipefail

cd "$(dirname "$0")/../.."
source "scripts/slurm/common.sh"
lattice_cd_repo

BASE_RUN_ID="${BASE_RUN_ID:-w790kdrh}"
EXTRA_SEEDS="${EXTRA_SEEDS:-6}"
DRY_RUN="${DRY_RUN:-0}"

BASE_ENV="$(lattice_pipeline_env_path "${BASE_RUN_ID}" || true)"
[[ -n "${BASE_ENV}" && -f "${BASE_ENV}" ]] || {
  echo "missing pipeline.env for BASE_RUN_ID=${BASE_RUN_ID}" >&2
  exit 1
}
PIPELINE_ENV="${BASE_ENV}"
lattice_pipeline_source_env
: "${ADAPTER_RUN_ID:?}"
: "${PIPELINE_LOG_DIR:?}"

# Existing sidecars define how many seeds we already have.
HAVE=0
while [[ -f "$(lattice_pipeline_ebm_sidecar "${HAVE}")" ]]; do
  HAVE=$((HAVE + 1))
done
TOTAL=$((HAVE + EXTRA_SEEDS))
START="${HAVE}"
END=$((TOTAL - 1))

if [[ "${EXTRA_SEEDS}" -lt 1 ]]; then
  echo "EXTRA_SEEDS must be >= 1, got ${EXTRA_SEEDS}" >&2
  exit 1
fi

lattice_pipeline_set_env N_SEEDS "${TOTAL}"
lattice_pipeline_source_env

echo "base=${ADAPTER_RUN_ID}  have_seeds=0..$((HAVE - 1))  train=${START}..${END}  total=${TOTAL}"
echo "env=${PIPELINE_ENV}"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN: would sbatch --array=${START}-${END} stage5_ebm_train.sh"
  echo "DRY_RUN: would sbatch --dependency=afterok:<J5> stage6_ensemble_scaling.sh"
  exit 0
fi

EXPORT="ALL,PIPELINE_ENV=${PIPELINE_ENV},PIPELINE_ID=${PIPELINE_ID},METHOD=${METHOD:-ntxent},N_SEEDS=${TOTAL},MULTISEED=0,PROTEIN=${PROTEIN:-esm2},PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD:-ntxent},SMOKE=${SMOKE:-0},MERGE=${MERGE:-0},ABLATION=${ABLATION:-1},ARTIFACTS_ROOT=${ARTIFACTS_ROOT:-artifacts/ablation},PIPELINE_LOG_ROOT=${PIPELINE_LOG_ROOT},OVERWRITE=0"

J5="$(sbatch --parsable --export="${EXPORT}" \
  --array="${START}-${END}" \
  scripts/slurm/stage5_ebm_train.sh)"

J6="$(sbatch --parsable --export="${EXPORT}" \
  --dependency="afterok:${J5}" \
  scripts/slurm/stage6_ensemble_scaling.sh)"

cat <<EOF
ensemble scaling submitted for ${ADAPTER_RUN_ID}
  stage 5: ${J5}  (EBM seeds ${START}..${END})
  stage 6: ${J6}  (scaling curve after ${J5})
  out:     ${ARTIFACTS_ROOT}/evaluation/${ADAPTER_RUN_ID}/ensemble_scaling.{json,png}
           (zm cache reused from ebm.0 if present)
EOF
