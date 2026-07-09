#!/usr/bin/env bash
#SBATCH --job-name=lattice-s4-decoys
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage4/%j.out
#SBATCH --error=logs/slurm/stage4/%j.err
#
# Stage 4 — precompute decoy / BDB / binder z_m pools for any Stage-2 adapter ckpt
# (LeJEPA discrete-flow, denoising-JEPA, etc. — detected from the checkpoint).
#
# Stores land under the adapter W&B run id:
#   artifacts/decoys/<run_id>/{decoy_zm,bdb_zm}
#   artifacts/binders/<run_id>/binder_zm
#
#   RUN_ID=nsw2w2z5 sbatch scripts/slurm/stage4_precompute_decoys.sh
#   sbatch scripts/slurm/stage4_precompute_decoys.sh nsw2w2z5
#   sbatch scripts/slurm/stage4_precompute_decoys.sh  # pipeline: reads ADAPTER_RUN_ID from PIPELINE_ENV
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repo root: sbatch scripts/slurm/stage4_precompute_decoys.sh}"
# shellcheck source=scripts/slurm/common.sh
source "scripts/slurm/common.sh"

lattice_pipeline_source_env

RUN_ID="${1:-${RUN_ID:-}}"
if [[ -z "${RUN_ID}" && -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
  lattice_pipeline_source_env
  RUN_ID="${ADAPTER_RUN_ID:?missing ADAPTER_RUN_ID in ${PIPELINE_ENV} — run stage2 first}"
fi
: "${RUN_ID:?set RUN_ID=<stage2_wandb_run_id> (or pass as \$1)}"
if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
  SSL_CKPT="$(lattice_pipeline_ssl_ckpt "${RUN_ID}")"
else
  SSL_CKPT="$(lattice_artifacts_root)/adapter/checkpoints/${RUN_ID}/last.ckpt"
fi

lattice_load_gpu_modules
lattice_cd_repo
lattice_pipeline_track_slurm_logs 4
lattice_require_gpu

if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
  SSL_CKPT="$(lattice_pipeline_ssl_ckpt "${RUN_ID}")"
  lattice_job_banner "pipeline ssl ckpt → ${SSL_CKPT}"
fi

export WANDB_MODE=disabled

lattice_require_file "${REPO}/${SSL_CKPT}" \
  "missing adapter checkpoint for RUN_ID=${RUN_ID}"

# Merge variant follows the adapter itself (fragment_merge flag in its ckpt), so
# the moses/bindingdb sources and the decoy_zm/bdb_zm/binder_zm stores always
# match how the adapter was trained — the precompute CLIs derive the store suffix
# from the same ckpt, so they need no flag.
MERGE_SUFFIX="$(lattice_ckpt_merge_suffix "${REPO}/${SSL_CKPT}")"
[[ -n "${MERGE_SUFFIX}" ]] && lattice_job_banner "adapter trained on merge views → ${MERGE_SUFFIX} stores"
lattice_sync_moses_shards_to_flash "${MERGE_SUFFIX}"

PRECOMPUTE_LIMIT=()
TRAIN_PARQUET="artifacts/preprocessing/processed/bindingdb${MERGE_SUFFIX}/threshold_90/train.parquet"
VAL_PARQUET="artifacts/preprocessing/processed/bindingdb${MERGE_SUFFIX}/threshold_90/val.parquet"
if lattice_smoke_enabled; then
  PRECOMPUTE_LIMIT=(--limit "$(lattice_smoke_precompute_limit)")
  SMOKE_PARQUET_DIR="${SMOKE_PARQUET_DIR:-${REPO}/logs/slurm/smoke-$$}"
  lattice_ensure_smoke_parquets "${SMOKE_PARQUET_DIR}"
  TRAIN_PARQUET="${SMOKE_PARQUET_DIR}/train.parquet"
  VAL_PARQUET="${SMOKE_PARQUET_DIR}/val.parquet"
  lattice_job_banner "SMOKE: precompute limit=${PRECOMPUTE_LIMIT[*]} parquets=${TRAIN_PARQUET}"
fi

# ENCODER_3D=1: encode ligands with the Uni-Mol 3D encoder (encoder_3d.*) baked
# into a VIEW3D Stage-2 ckpt instead of the 2D DDiT+adapter. Writes parallel
# *_zm3d stores (same keys) that stage5 consumes via ENCODER_3D=1. Requires the
# MOSES conformers.parquet (stage1b) for decoys; builds BDB/binder conformer
# caches here.
if [[ "${ENCODER_3D:-0}" == 1 ]]; then
  lattice_job_banner "ENCODER_3D=1: precomputing z_m with the Uni-Mol 3D encoder"
  MOSES_CONF="${REPO}/artifacts/preprocessing/processed/moses${MERGE_SUFFIX}/conformers.parquet"
  lattice_require_file "${MOSES_CONF}" \
    "missing ${MOSES_CONF} — run stage1b (VIEW3D conformers) first"

  DECOY_STORE3D="$(lattice_zm_store_path decoy_zm3d "${RUN_ID}" "${MERGE_SUFFIX}")"
  BDB_STORE3D="$(lattice_zm_store_path bdb_zm3d "${RUN_ID}" "${MERGE_SUFFIX}")"
  BINDER_STORE3D="$(lattice_zm_store_path binder_zm3d "${RUN_ID}" "${MERGE_SUFFIX}")"
  BDB_CONF="$(dirname "${BDB_STORE3D}")/bdb_conformers${MERGE_SUFFIX}.parquet"
  BINDER_CONF="$(dirname "${BINDER_STORE3D}")/binder_conformers${MERGE_SUFFIX}.parquet"

  srun python -m lattice_lab.preprocessing.precompute_conformers \
    --parquet "${TRAIN_PARQUET}" --key-col inchikey \
    --output "${BDB_CONF}" --n-jobs 6 --overwrite "${PRECOMPUTE_LIMIT[@]}"
  srun python -m lattice_lab.preprocessing.precompute_conformers \
    --parquet "${TRAIN_PARQUET}" --parquet "${VAL_PARQUET}" --key-col smiles \
    --output "${BINDER_CONF}" --n-jobs 6 --overwrite "${PRECOMPUTE_LIMIT[@]}"

  srun python -m lattice_lab.ebm.precompute_zm3d --pool decoy \
    --adapter-ckpt "${SSL_CKPT}" --conformer-cache "${MOSES_CONF}" \
    --store "${DECOY_STORE3D}" --batch-size 256 "${PRECOMPUTE_LIMIT[@]}" --force
  srun python -m lattice_lab.ebm.precompute_zm3d --pool bdb \
    --adapter-ckpt "${SSL_CKPT}" --conformer-cache "${BDB_CONF}" \
    --bdb-parquet "${TRAIN_PARQUET}" \
    --store "${BDB_STORE3D}" --batch-size 256 "${PRECOMPUTE_LIMIT[@]}" --force
  srun python -m lattice_lab.ebm.precompute_zm3d --pool binder \
    --adapter-ckpt "${SSL_CKPT}" --conformer-cache "${BINDER_CONF}" \
    --train-parquet "${TRAIN_PARQUET}" --val-parquet "${VAL_PARQUET}" \
    --store "${BINDER_STORE3D}" --batch-size 256 "${PRECOMPUTE_LIMIT[@]}" --force
  lattice_job_banner "ENCODER_3D=1: done — stage5/stage6 must also run with ENCODER_3D=1"
  exit 0
fi

DECOY_MANIFEST="${REPO}/$(lattice_zm_store_path decoy_zm "${RUN_ID}" "${MERGE_SUFFIX}")/manifest.json"
if [[ "${STAGE4_RESUME:-0}" == 1 && -f "${DECOY_MANIFEST}" ]]; then
  lattice_job_banner "STAGE4_RESUME: decoy_zm present, skipping precompute_decoys"
else
  srun python -m lattice_lab.ebm.precompute_decoys \
    --shard-dir "${LATTICE_FLASH_PROCESSED}/moses${MERGE_SUFFIX}" \
    --adapter-ckpt "${SSL_CKPT}" \
    --store "$(lattice_zm_store_path decoy_zm "${RUN_ID}" "${MERGE_SUFFIX}")" \
    --batch-size 512 \
    "${PRECOMPUTE_LIMIT[@]}" \
    --force
fi

srun python -m lattice_lab.ebm.precompute_bdb_zm \
  --bdb-parquet "${TRAIN_PARQUET}" \
  --adapter-ckpt "${SSL_CKPT}" \
  --store "$(lattice_zm_store_path bdb_zm "${RUN_ID}" "${MERGE_SUFFIX}")" \
  --batch-size 512 \
  --n-jobs 6 \
  "${PRECOMPUTE_LIMIT[@]}" \
  --force

srun python -m lattice_lab.ebm.precompute_binders \
  --train-parquet "${TRAIN_PARQUET}" \
  --val-parquet "${VAL_PARQUET}" \
  --adapter-ckpt "${SSL_CKPT}" \
  --store "$(lattice_zm_store_path binder_zm "${RUN_ID}" "${MERGE_SUFFIX}")" \
  --batch-size 512 \
  "${PRECOMPUTE_LIMIT[@]}" \
  --force
