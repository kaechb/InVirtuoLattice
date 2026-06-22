#!/usr/bin/env bash
#SBATCH --job-name=lattice-s6-ens
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
# Deprecated wrapper — use stage6_eval.sh with three run ids instead.
set -euo pipefail
exec bash "$(dirname "$0")/stage6_eval.sh" "$@"
