#!/usr/bin/env bash
#SBATCH --job-name=lattice-s1-pre
#SBATCH --account=project_465003063
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/stage1/%j.out
#SBATCH --error=logs/slurm/stage1/%j.err
#
# Stage 1 — BindingDB curation + MOSES fragment-view preprocessing.
#
#   sbatch scripts/slurm/stage1_preprocess.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage1_preprocess.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_load_cpu_modules
lattice_cd_repo

OUT_BINDINGDB="${REPO}/artifacts/preprocessing/processed/bindingdb"
OUT_MOSES="${REPO}/artifacts/preprocessing/processed/moses_id"

srun python -m lattice_lab.preprocessing.run_bindingdb \
  --bindingdb-tsv "${REPO}/artifacts/preprocessing/raw/bindingdb/BindingDB_All.tsv" \
  --lit-pcba-dir "${REPO}/artifacts/preprocessing/raw/lit_pcba" \
  --output-dir "${OUT_BINDINGDB}" \
  --identity 90 \
  --n-jobs 128 \
  --tokenizer-path "${REPO}/artifacts/tokenizer/smiles_new.json"

srun python -m lattice_lab.preprocessing.run_preprocessing \
  --input "${REPO}/artifacts/preprocessing/raw/moses.csv" \
  --output "${OUT_MOSES}" \
  --n-views 3 \
  --n-jobs 128 \
  --tokenizer-path "${REPO}/artifacts/tokenizer/smiles_new.json"
