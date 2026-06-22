#!/usr/bin/env bash
#SBATCH --job-name=djepa-abl
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage2/djepa_abl_%j.out
#SBATCH --error=logs/slurm/stage2/djepa_abl_%j.err
#
# Parameterised denoising-JEPA ablation job. Use submit_djepa_ablation.sh.
#
# Required env vars:
#   DJEPA_RUN_NAME    — W&B run name (e.g. "fp10_cn1")
#   DJEPA_CT_HI       — corrupt_t high bound; lo is fixed at 0 → model.corrupt_t=[0,HI]
#                       (passed as a scalar so the inner comma can't break --export)
#   DJEPA_FP_WEIGHT   — model.fp_weight        (float, e.g. 0.0 / 2.0 / 10.0)
#   DJEPA_SIGREG      — model.sigreg_lambda    (float, e.g. 0.0 / 0.1)
#   DJEPA_KL          — model.kl_beta          (float, e.g. 0.0 / 0.1)
#   DJEPA_COND_NOISE  — model.cond_noise_scale (0=off, 1=on)
#
# Optional env vars (defaults preserve current behaviour):
#   DJEPA_KL_FREE_BITS — model.kl_free_bits        (nats/dim; default 0.0)  [Phase 3]
#   DJEPA_KL_WARMUP    — model.kl_warmup_steps      (int;      default 0)    [Phase 3]
#   DJEPA_FREEZE       — model.freeze_backbone (true/false; default true) [Phase 4]
#   DJEPA_LR           — model.learning_rate        (float;    default 1.0e-4)    [Phase 4]
set -euo pipefail

: "${DJEPA_RUN_NAME:?set DJEPA_RUN_NAME before submitting}"
: "${DJEPA_CT_HI:?set DJEPA_CT_HI (corrupt_t high bound)}"
: "${DJEPA_FP_WEIGHT:?set DJEPA_FP_WEIGHT}"
: "${DJEPA_SIGREG:?set DJEPA_SIGREG}"
: "${DJEPA_KL:?set DJEPA_KL}"
: "${DJEPA_COND_NOISE:?set DJEPA_COND_NOISE}"

cd "${SLURM_SUBMIT_DIR:?submit from repo root}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_load_gpu_modules
lattice_cd_repo
lattice_require_gpu

echo "=== djepa ablation: ${DJEPA_RUN_NAME} ==="
echo "  corrupt_t=[0,${DJEPA_CT_HI}]  fp_weight=${DJEPA_FP_WEIGHT}  sigreg=${DJEPA_SIGREG}  kl=${DJEPA_KL}  free_bits=${DJEPA_KL_FREE_BITS:-0.0}  kl_warmup=${DJEPA_KL_WARMUP:-0}  cond_noise=${DJEPA_COND_NOISE}  freeze_backbone=${DJEPA_FREEZE:-true}  lr=${DJEPA_LR:-1.0e-4}"

srun python -m lattice_lab.train experiment=denoising_jepa \
  data.shard_dir=${LATTICE_FLASH_PROCESSED}/moses \
  trainer.max_epochs=10 \
  trainer.accelerator=gpu \
  callbacks.model_checkpoint.dirpath=artifacts/adapter/checkpoints \
  model.freeze_backbone="${DJEPA_FREEZE:-true}" \
  model.learning_rate="${DJEPA_LR:-1.0e-4}" \
  model.corrupt_t="[0,${DJEPA_CT_HI}]" \
  model.fp_weight="${DJEPA_FP_WEIGHT}" \
  model.sigreg_lambda="${DJEPA_SIGREG}" \
  model.kl_beta="${DJEPA_KL}" \
  model.kl_free_bits="${DJEPA_KL_FREE_BITS:-0.0}" \
  model.kl_warmup_steps="${DJEPA_KL_WARMUP:-0}" \
  model.cond_noise_scale="${DJEPA_COND_NOISE}" \
  logger.wandb.name="${DJEPA_RUN_NAME}" \
  logger.wandb.group=djepa_ablation
