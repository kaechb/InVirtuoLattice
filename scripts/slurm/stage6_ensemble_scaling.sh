#!/usr/bin/env bash
#SBATCH --job-name=lattice-ens-scale
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
# Stage-6 companion: ensemble-size scaling curve over all ebm.* sidecars.
# Prefer submitting via run_ensemble_scaling.sh (trains missing seeds first).
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root}"
source "scripts/slurm/common.sh"
lattice_pipeline_source_env
: "${ADAPTER_RUN_ID:?set PIPELINE_ENV / ADAPTER_RUN_ID}"
: "${PIPELINE_LOG_DIR:?}"

lattice_load_gpu_modules
lattice_cd_repo
lattice_pipeline_track_slurm_logs 6
lattice_require_gpu
lattice_protein_resolve

N_SEEDS="${N_SEEDS:-9}"
CKPTS=()
RUN_IDS=()
for SEED in $(seq 0 $((N_SEEDS - 1))); do
  F="$(lattice_pipeline_ebm_sidecar "${SEED}")"
  lattice_require_file "${F}" "missing ${F} — train seed ${SEED} first"
  rid="$(<"${F}")"
  ckpt="${REPO}/$(lattice_pipeline_ebm_eval_ckpt "${rid}")"
  lattice_require_file "${ckpt}" "missing EBM ckpt for seed=${SEED} rid=${rid}"
  RUN_IDS+=("${rid}")
  CKPTS+=("${ckpt}")
done

ZM_CACHE="$(lattice_evaluation_path "${RUN_IDS[0]}/lit_pcba_zm_mv4")"
# Reuse the seed-0 multi-view cache if present; else build it.
if [[ ! -d "${REPO}/${ZM_CACHE}" ]]; then
  lattice_job_banner "building mv4 cache → ${ZM_CACHE}"
  SSL_CKPT="$(lattice_pipeline_ssl_ckpt "${ADAPTER_RUN_ID}")"
  srun python -m lattice_lab.eval.build_multiview_cache \
    --adapter-ckpt "${SSL_CKPT}" \
    --fp-ckpt "${CKPTS[0]}" \
    --test-parquet artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet \
    --out "${ZM_CACHE}" \
    --n-views 4 \
    --n-jobs 6
fi

OUT_JSON="$(lattice_evaluation_path "${ADAPTER_RUN_ID}/ensemble_scaling.json")"
OUT_PNG="$(lattice_evaluation_path "${ADAPTER_RUN_ID}/ensemble_scaling.png")"

lattice_job_banner "ensemble scaling n=${#CKPTS[@]} → ${OUT_JSON}"
srun python -m lattice_lab.eval.ensemble_scaling \
  --ckpts "${CKPTS[@]}" \
  --zm-cache "${ZM_CACHE}" \
  --protein-store "${PROTEIN_STORE}" \
  --out "${OUT_JSON}" \
  --plot "${OUT_PNG}" \
  --d-protein "${D_PROTEIN}" \
  --n-jobs 32 \
  --device cuda

lattice_job_banner "done: ${OUT_JSON} ${OUT_PNG}"
