#!/usr/bin/env bash
#SBATCH --job-name=lattice-s6-single
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
# Stage 6 (single checkpoint) — LIT-PCBA evaluation of ONE EBM checkpoint.
#
# Self-contained: lit_pcba builds (and then reuses) its own single-view z_m cache
# at lit_pcba_zm_sv_<variant>, encoded with the adapter baked into the EBM ckpt
# (identical across seeds of a variant), so no separate Stage-6a build is needed.
# The adapter-fingerprint guard refuses a cache built by a mismatched adapter.
#
# Point it at a run, either by VARIANT + RUN_ID (resolves the standard path)…
#   VARIANT=ntxent RUN_ID=gi2762bi sbatch scripts/slurm/stage6_single_eval.sh
# …or at an explicit checkpoint (VARIANT still names the per-variant z_m cache):
#   VARIANT=lejepa CKPT=artifacts/energy/lejepa/5ubbm5ah/last.ckpt \
#     sbatch scripts/slurm/stage6_single_eval.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage6_single_eval.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

VARIANT="${VARIANT:-lejepa}"
case "${VARIANT}" in
  lejepa|ntxent) ;;
  *) echo "unknown VARIANT='${VARIANT}' (use lejepa|ntxent)" >&2; exit 1 ;;
esac

# Resolve the checkpoint: explicit CKPT wins, else VARIANT/RUN_ID.
if [[ -z "${CKPT:-}" ]]; then
  : "${RUN_ID:?set RUN_ID=<wandb_run_id> (or CKPT=<path>); e.g. VARIANT=${VARIANT} RUN_ID=<id>}"
  CKPT="artifacts/energy/${VARIANT}/${RUN_ID}/last.ckpt"
fi
TAG="$(basename "$(dirname "${CKPT}")")"   # the run id, for output naming
ZM_CACHE="artifacts/evaluation/lit_pcba_zm_sv_${VARIANT}"
OUT_CSV="artifacts/evaluation/lit_pcba_${VARIANT}_${TAG}.csv"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

lattice_require_file "${REPO}/${CKPT}" \
  "set RUN_ID/CKPT for an existing ${VARIANT} run in scripts/slurm/stage6_single_eval.sh"

# Adapter is baked into the EBM ckpt, so head-ckpt and adapter-ckpt are the same
# file — guaranteeing the z_m cache and the head share one latent space.
srun python -m lattice_lab.eval.lit_pcba \
  --head-ckpt "${CKPT}" \
  --adapter-ckpt "${CKPT}" \
  --zm-cache "${ZM_CACHE}" \
  --protein-store artifacts/protein_store/embeddings/esm2_650M \
  --test-parquet artifacts/processed/bindingdb/test_lit_pcba.parquet \
  --output-csv "${OUT_CSV}" \
  --n-jobs 7
