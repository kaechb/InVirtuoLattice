#!/usr/bin/env bash
#
# Pretraining-strategy ablation round (full stage 2->6 pipelines, 3 EBM seeds each,
# judged on stage-6 LIT-PCBA BEDROC). Submit from the login node:
#
#   ./scripts/slurm/run_ablation_round.sh            # submit all
#   DRY_RUN=1 ./scripts/slurm/run_ablation_round.sh  # print the plan, submit nothing
#
# Design (one-factor-at-a-time around an anchor):
#   * method x protein grid: {lejepa,ntxent,siglip} x {esm2,esmc}  -> which SSL + which backbone
#   * variance: anchor retrained from scratch with 2 extra SSL seeds  -> BEDROC noise floor
#   * pooling: anchor with adapter_pool=mean (anchor uses attn)
#   * depth:   anchor with adapter_n_layers in {0,1,4} (anchor uses 2)
#   * 3D:      anchor with VIEW3D=1 (anchor is 2D-only)
# Anchor = ntxent + esm2, attn pool, 2 adapter layers, no 3D, seed 0 (in the grid).
# ntxent+esm2 is the strongest config in this repo's run history, so the OFAT
# ablations hang off it. ijepa is intentionally excluded.
#
# All runs: N_SEEDS=3 (ensemble BEDROC), ABLATION=1, MERGE=0.
set -euo pipefail

cd "$(dirname "$0")/../.."

PREFIX="${PREFIX:-abl}"
DRY_RUN="${DRY_RUN:-0}"
export N_SEEDS="${N_SEEDS:-3}"
export ABLATION=1
export MERGE=0

n=0
# submit <run_name> <method> <protein> [EXTRA_TRAIN_ARGS] [VIEW3D]
submit() {
  local name="$1" method="$2" protein="$3" extra="${4:-}" view3d="${5:-0}"
  n=$((n + 1))
  echo ""
  echo "=== [${n}] ${PREFIX}_${name}  METHOD=${method} PROTEIN=${protein} VIEW3D=${view3d} EXTRA='${extra}' ==="
  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "  DRY_RUN: EXTRA_TRAIN_ARGS='${extra}' VIEW3D=${view3d} PROTEIN=${protein} \\"
    echo "           ./scripts/slurm/run_pipeline.sh ${method} ${PREFIX}_${name}"
    return
  fi
  EXTRA_TRAIN_ARGS="${extra}" VIEW3D="${view3d}" PROTEIN="${protein}" \
    ./scripts/slurm/run_pipeline.sh "${method}" "${PREFIX}_${name}"
}

# 1) SSL method x protein backbone grid (6)
submit lejepa_esm    lejepa esm2
submit lejepa_esmc   lejepa esmc
submit ntxent_esm    ntxent esm2
submit ntxent_esmc   ntxent esmc
submit siglip_esm    siglip esm2
submit siglip_esmc   siglip esmc

# 2) Variance of the anchor (ntxent+esm2) — retrain from scratch, new SSL seed (2)
submit ntxent_esm_seed1 ntxent esm2 "seed=1"
submit ntxent_esm_seed2 ntxent esm2 "seed=2"

# 3) Adapter pooling on the anchor (1) — anchor already covers attn
submit ntxent_esm_poolmean ntxent esm2 "model.encoder.adapter_pool=mean"

# 4) Adapter depth on the anchor (3) — anchor already covers n_layers=2
submit ntxent_esm_layers0 ntxent esm2 "model.encoder.adapter_n_layers=0"
submit ntxent_esm_layers1 ntxent esm2 "model.encoder.adapter_n_layers=1"
submit ntxent_esm_layers4 ntxent esm2 "model.encoder.adapter_n_layers=4"

# 5) 3D view on the anchor (1)
submit ntxent_esm_view3d ntxent esm2 "" 1

echo ""
echo "submitted ${n} full pipelines (prefix=${PREFIX}, N_SEEDS=${N_SEEDS}) under logs/slurm/ablation/"
echo "aggregate when done:  python scripts/aggregate_ablations.py --prefix ${PREFIX}_"
