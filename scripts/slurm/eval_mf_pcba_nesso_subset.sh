#!/usr/bin/env bash
#SBATCH --job-name=lattice-mfpcba
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/mf_pcba/%j.out
#SBATCH --error=logs/slurm/mf_pcba/%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root}"
source scripts/slurm/common.sh
lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

DATA=artifacts/benchmarks/mf_pcba/nesso_subset8_seed0.parquet
FASTA=artifacts/benchmarks/mf_pcba/nesso_subset8.fasta
PROTEINS=artifacts/benchmarks/mf_pcba/protein_store_esm2_650M
CACHE=artifacts/benchmarks/mf_pcba/zm_mv4_wk8denar
OUTPUT=artifacts/benchmarks/mf_pcba/lattice_wk8denar_ensemble_seed0.json
CKPT0=artifacts/sota/wk8denar/ebm/seed0/ebm_best.ckpt
CKPT1=artifacts/sota/wk8denar/ebm/seed1/ebm_best.ckpt
CKPT2=artifacts/sota/wk8denar/ebm/seed2/ebm_best.ckpt

for file in "${DATA}" "${FASTA}" "${CKPT0}" "${CKPT1}" "${CKPT2}"; do
  lattice_require_file "${REPO}/${file}" "prepare MF-PCBA data/checkpoints first"
done

srun python -m lattice_lab.protein.precompute \
  --fasta "${FASTA}" \
  --store "${PROTEINS}" \
  --batch-size 8 \
  --device cuda

srun python -m lattice_lab.eval.build_multiview_cache \
  --n-views 4 \
  --zm-cache "${CACHE}" \
  --test-parquet "${DATA}" \
  --adapter-ckpt "${CKPT0}" \
  --protein-store "${PROTEINS}" \
  --device cuda \
  --batch-size 256 \
  --n-jobs "${SLURM_CPUS_PER_TASK}"

srun python -m lattice_lab.eval.ensemble_eval \
  --ckpts "${CKPT0}" "${CKPT1}" "${CKPT2}" \
  --zm-cache "${CACHE}" \
  --protein-store "${PROTEINS}" \
  --test-parquet "${DATA}" \
  --out "${OUTPUT}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --device cuda
