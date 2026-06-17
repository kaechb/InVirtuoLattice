#!/usr/bin/env bash
#SBATCH --job-name=lattice-s6-cache
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
# Stage 6a — build a 4-view LIT-PCBA z_m cache for ensemble eval.
#
# Stage-5 now writes variant-namespaced runs:
#   artifacts/energy/<variant>/<run_id>/last.ckpt
# The frozen SSL adapter is IDENTICAL across the 3 seeds of a variant (verified
# byte-for-byte), so one cache per variant is valid for every seed. We build it
# from the seed-0 EBM checkpoint so the cache lives in the exact latent space the
# ensemble heads score in.
#
# Pick the variant + edit the seed-0 run id, then:
#   VARIANT=lejepa sbatch scripts/slurm/stage6_build_zm_cache.sh
#   VARIANT=ntxent sbatch scripts/slurm/stage6_build_zm_cache.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage6_build_zm_cache.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

VARIANT="${VARIANT:-lejepa}"
case "${VARIANT}" in
  lejepa) EBM_RUN_ID=5ubbm5ah ;;   # lejepa seed0 (stage5_ebm_train_lejepa.sh)
  ntxent) EBM_RUN_ID=fn93bboy ;;   # ntxent seed0 (stage5_ebm_train_ntxent.sh)
  *) echo "unknown VARIANT='${VARIANT}' (use lejepa|ntxent)" >&2; exit 1 ;;
esac

EBM_CKPT="artifacts/energy/${VARIANT}/${EBM_RUN_ID}/last.ckpt"
ZM_CACHE="artifacts/evaluation/lit_pcba_zm_mv4_${VARIANT}"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

lattice_require_file "${REPO}/${EBM_CKPT}" \
  "edit the ${VARIANT} seed-0 run id in scripts/slurm/stage6_build_zm_cache.sh"

srun python -m lattice_lab.eval.build_multiview_cache \
  --n-views 4 \
  --zm-cache "${ZM_CACHE}" \
  --adapter-ckpt "${EBM_CKPT}" \
  --n-jobs 40
