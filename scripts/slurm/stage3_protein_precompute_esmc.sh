#!/usr/bin/env bash
#SBATCH --job-name=lattice-s3-esmc
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage3/%j.out
#SBATCH --error=logs/slurm/stage3/%j.err
#
# Deprecated wrapper — use PROTEIN=esmc stage3_protein_precompute.sh instead.
set -euo pipefail
export PROTEIN=esmc
exec bash "$(dirname "$0")/stage3_protein_precompute.sh"
