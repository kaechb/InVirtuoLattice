#!/usr/bin/env bash
# ponytail: bash -n on pipeline scripts + smoke helper dry-run (no sbatch/GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/slurm/common.sh

for f in scripts/slurm/run_pipeline.sh scripts/slurm/stage{2,3,4,5,6}_*.sh scripts/slurm/common.sh; do
  bash -n "$f"
done

TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"' EXIT
if command -v python >/dev/null 2>&1; then
  lattice_make_smoke_parquets "${TMP}" "${REPO}/artifacts/preprocessing/processed/bindingdb/threshold_90" 100 2
  test -f "${TMP}/train.parquet"
  test -f "${TMP}/val.parquet"
  test -f "${TMP}/test_lit_pcba.parquet"
else
  echo "skip parquet smoke (no python in PATH)"
fi
echo "smoke pipeline scripts OK"
