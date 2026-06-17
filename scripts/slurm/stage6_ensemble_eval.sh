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
# Stage 6b — 3-seed ensemble LIT-PCBA evaluation.
#
# Reads the variant-namespaced Stage-5 checkpoints
#   artifacts/energy/<variant>/<run_id>/last.ckpt
# and the matching mv4 cache built by stage6_build_zm_cache.sh (same VARIANT).
#
# Pick the variant + edit the per-seed run ids, then:
#   VARIANT=lejepa sbatch scripts/slurm/stage6_ensemble_eval.sh
#   VARIANT=ntxent sbatch scripts/slurm/stage6_ensemble_eval.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage6_ensemble_eval.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

VARIANT="${VARIANT:-lejepa}"
case "${VARIANT}" in
  lejepa) RUN_ID0=5ubbm5ah; RUN_ID1=8o965wj8; RUN_ID2=bv41h3wo ;;
  ntxent) RUN_ID0=gi2762bi; RUN_ID1=uqcvdwg9; RUN_ID2=fn93bboy ;;
  *) echo "unknown VARIANT='${VARIANT}' (use lejepa|ntxent)" >&2; exit 1 ;;
esac

CKPT0="artifacts/energy/${VARIANT}/${RUN_ID0}/last.ckpt"
CKPT1="artifacts/energy/${VARIANT}/${RUN_ID1}/last.ckpt"
CKPT2="artifacts/energy/${VARIANT}/${RUN_ID2}/last.ckpt"
ZM_CACHE="artifacts/evaluation/lit_pcba_zm_mv4_${VARIANT}"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

lattice_require_file "${REPO}/${CKPT0}" "edit the ${VARIANT} seed-0 run id in scripts/slurm/stage6_ensemble_eval.sh"
lattice_require_file "${REPO}/${CKPT1}" "edit the ${VARIANT} seed-1 run id in scripts/slurm/stage6_ensemble_eval.sh"
lattice_require_file "${REPO}/${CKPT2}" "edit the ${VARIANT} seed-2 run id in scripts/slurm/stage6_ensemble_eval.sh"
lattice_require_file "${REPO}/${ZM_CACHE}/manifest.json" \
  "build it first: VARIANT=${VARIANT} sbatch scripts/slurm/stage6_build_zm_cache.sh"

srun python -m lattice_lab.eval.ensemble_eval \
  --ckpts "${CKPT0}" "${CKPT1}" "${CKPT2}" \
  --zm-cache "${ZM_CACHE}" \
  --protein-store artifacts/protein_store/embeddings/esm2_650M \
  --test-parquet artifacts/processed/bindingdb/test_lit_pcba.parquet \
  --out "artifacts/evaluation/ensemble_hardneg_mv4_${VARIANT}.json" \
  --n-jobs 32
