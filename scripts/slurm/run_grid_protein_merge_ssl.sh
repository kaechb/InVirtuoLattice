#!/usr/bin/env bash
#
# 2×2×2 pipeline grid (8 runs):
#   METHOD:  ijepa | ntxent          (stage2 overrides ssl_loss; yaml unchanged)
#   PROTEIN: esm2 | esmc
#   MERGE:   0 (unmerged) | 1 (merge)
#
# Logs:     logs/slurm/ablation/<run_id>/
# Artifacts: artifacts/ablation/{adapter,energy,decoys,binders,evaluation}/
#
#   ./scripts/slurm/run_grid_protein_merge_ssl.sh
#   DRY_RUN=1 ./scripts/slurm/run_grid_protein_merge_ssl.sh
#
set -euo pipefail

cd "$(dirname "$0")/../.."
source "scripts/slurm/common.sh"
lattice_cd_repo

GRID_PREFIX="${GRID_PREFIX:-grid}"
DRY_RUN="${DRY_RUN:-0}"

case "${DRY_RUN}" in
  0|false|no|"") DRY_RUN=0 ;;
  1|true|yes)   DRY_RUN=1 ;;
  *)
    echo "DRY_RUN=${DRY_RUN} (want 0 or 1)" >&2
    exit 1
    ;;
esac

n=0
for method in ijepa ntxent; do
  for protein in esm2 esmc; do
    for merge in 0 1; do
      merge_tag=$([[ "${merge}" == 1 ]] && echo merge || echo unmerged)
      protein_tag=$([[ "${protein}" == esm2 ]] && echo esm || echo esmc)
      run_name="${GRID_PREFIX}_${method}_${protein_tag}_${merge_tag}"
      n=$((n + 1))
      echo ""
      echo "=== [${n}/8] ${run_name}  METHOD=${method} PROTEIN=${protein} MERGE=${merge} ==="
      if [[ "${DRY_RUN}" == 1 ]]; then
        echo "  DRY_RUN: ABLATION=1 PROTEIN=${protein} MERGE=${merge} ./scripts/slurm/run_pipeline.sh ${method} ${run_name}"
        continue
      fi
      N_SEEDS="${N_SEEDS:-3}" ABLATION=1 PROTEIN="${protein}" MERGE="${merge}" \
        ./scripts/slurm/run_pipeline.sh "${method}" "${run_name}"
    done
  done
done

echo ""
echo "grid done (${n} runs) — logs under logs/slurm/ablation/, artifacts under artifacts/ablation/"
