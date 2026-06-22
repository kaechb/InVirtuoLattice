#!/usr/bin/env bash
# Submit denoising-JEPA ablation runs to SLURM.
#
#   bash scripts/slurm/submit_djepa_ablation.sh [phase]
#
#   phase 1 (default) — 8 single-axis sweeps from baseline
#   phase 2           — 4 combos (edit to match phase-1 results)
#   phase 3           — 5 KL free-bits / warmup runs
#   phase 4           — frozen vs unfrozen backbone
#   all               — all phases
#
# W&B group: djepa_ablation
# Checkpoints: artifacts/adapter/checkpoints/<wandb_run_id>/last.ckpt
set -euo pipefail

PHASE="${1:-1}"
JOB="scripts/slurm/ablation_djepa_job.sh"

submit() {
    local name=$1 ct=$2 fp=$3 sr=$4 kl=$5 cn=$6 fb=${7:-0.0} wu=${8:-0} fr=${9:-true} lr=${10:-1.0e-4}
    echo "  submitting ${name} (ct=[0,${ct}] fp=${fp} sr=${sr} kl=${kl} cn=${cn} fb=${fb} wu=${wu} freeze=${fr} lr=${lr})"
    sbatch \
        --job-name="djepa-${name}" \
        --export="ALL,DJEPA_RUN_NAME=${name},DJEPA_CT_HI=${ct},DJEPA_FP_WEIGHT=${fp},DJEPA_SIGREG=${sr},DJEPA_KL=${kl},DJEPA_COND_NOISE=${cn},DJEPA_KL_FREE_BITS=${fb},DJEPA_KL_WARMUP=${wu},DJEPA_FREEZE=${fr},DJEPA_LR=${lr}" \
        "${JOB}"
}

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [[ "${PHASE}" == "1" || "${PHASE}" == "all" ]]; then
    echo "=== Phase 1 ==="
    submit base   1.0   0.0   0.0   0.0   0
    submit fp2    1.0   2.0   0.0   0.0   0
    submit fp10   1.0   10.0  0.0   0.0   0
    submit cn1    1.0   0.0   0.0   0.0   1
    submit ct04   0.4   0.0   0.0   0.0   0
    submit ct07   0.7   0.0   0.0   0.0   0
    submit sr     1.0   0.0   0.1   0.0   0
    submit kl     1.0   0.0   0.0   0.1   0
fi

if [[ "${PHASE}" == "2" || "${PHASE}" == "all" ]]; then
    echo "=== Phase 2 ==="
    submit fp10_cn1        1.0   10.0  0.0   0.0   1
    submit fp10_cn1_ct04   0.4   10.0  0.0   0.0   1
    submit fp10_cn1_sr     1.0   10.0  0.1   0.0   1
    submit full            0.4   10.0  0.1   0.1   1
fi

if [[ "${PHASE}" == "3" || "${PHASE}" == "all" ]]; then
    echo "=== Phase 3 ==="
    submit kl_fb02        1.0   0.0   0.0   0.5   0    0.2   0
    submit kl_fb05        1.0   0.0   0.0   0.5   0    0.5   0
    submit kl_wu          1.0   0.0   0.0   0.5   0    0.0   1000
    submit kl_fb05_wu     1.0   0.0   0.0   0.5   0    0.5   1000
    submit fp10_kl_fb_wu  1.0   10.0  0.0   0.5   0    0.5   1000
fi

if [[ "${PHASE}" == "4" || "${PHASE}" == "all" ]]; then
    echo "=== Phase 4 ==="
    submit base_frozen    1.0   0.0  0.0  0.0  0   0.0  0   true    1.0e-4
    submit base_unfrozen  1.0   0.0  0.0  0.0  0   0.0  0   false   1.0e-4
fi

echo "Done. W&B group: djepa_ablation"
