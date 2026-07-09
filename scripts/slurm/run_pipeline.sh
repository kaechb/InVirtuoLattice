#!/usr/bin/env bash
#
# Submit stages 2→6 as separate SLURM jobs with dependencies (login node).
#
# Env / positional args:
#   METHOD     lejepa | ntxent | siglip | ijepa | denoise   (default: lejepa; positional $1)
#   RUN_NAME   optional label for pipeline stage2 W&B name: {RUN_NAME}_stage2_{run_id}
#              (standalone stage 2: used as logger.wandb.name directly)
#   PROTEIN    esm2 | esmc — stages 3/5/6 protein store + d_protein (default: esm2)
#   N_SEEDS    EBM seeds in stage 5 (default: 1; N_SEEDS=3 → 3 ckpts / ensemble eval)
#   STAGE_FROM=5  resume stage5 + stage6 on an existing pipeline (stages 2–4 unchanged)
#   MULTISEED=1  on an existing pipeline: train only missing EBM seeds (0–2) + stage6 ensemble
#   SMOKE=1    fast wiring test: 1 val epoch, ~1% SSL data, 5k-row pools, 50 EBM steps
#   VIEW3D=1   stage 2 pretrains the adapter with the cross-modal 3D point-cloud
#              view (experiment=adapter3d); submits conformer precompute (1b)
#              only when conformers.parquet is missing, then stage 2+.
#
# Pipeline W&B names: {RUN_NAME}_stage2_{id}, {RUN_NAME}_stage5_{id}[_seedN]
#
#   ./scripts/slurm/run_pipeline.sh lejepa
#   N_SEEDS=3 ./scripts/slurm/run_pipeline.sh lejepa          # fresh run, 3 EBM seeds
#   STAGE_FROM=5 ADAPTER_RUN_ID=rq6fpmxs ./scripts/slurm/run_pipeline.sh   # resume after stage5 failure
#   MULTISEED=1 ADAPTER_RUN_ID=avy80iqo ./scripts/slurm/run_pipeline.sh   # add 3 seeds to existing run
#   ./scripts/slurm/run_pipeline.sh ijepa ijepa_blockhole_smoke
#   SMOKE=1 ./scripts/slurm/run_pipeline.sh lejepa fpdistill_smoke
#
# Run dir: logs/slurm/pipeline/<adapter_run_id>/ (pipeline.env, configs/, ebm.<seed>, …).
set -euo pipefail

cd "$(dirname "$0")/../.."
source "scripts/slurm/common.sh"
lattice_cd_repo

METHOD="${1:-${METHOD:-lejepa}}"
RUN_NAME="${2:-${RUN_NAME:-}}"
PROTEIN="${PROTEIN:-esm2}"
N_SEEDS="${N_SEEDS:-1}"
MULTISEED="${MULTISEED:-0}"
SMOKE="${SMOKE:-0}"
MERGE="${MERGE:-0}"
VIEW3D="${VIEW3D:-0}"

case "${MULTISEED}" in
  0|false|no|"") MULTISEED=0 ;;
  1|true|yes)   MULTISEED=1 ;;
  *)
    echo "MULTISEED=${MULTISEED} (want 0 or 1)" >&2
    exit 1
    ;;
esac

case "${MERGE}" in
  0|false|no|"") MERGE=0 ;;
  1|true|yes)   MERGE=1 ;;
  *)
    echo "MERGE=${MERGE} (want 0 or 1)" >&2
    exit 1
    ;;
esac

case "${VIEW3D}" in
  0|false|no|"") VIEW3D=0 ;;
  1|true|yes)   VIEW3D=1 ;;
  *)
    echo "VIEW3D=${VIEW3D} (want 0 or 1)" >&2
    exit 1
    ;;
esac
if [[ "${VIEW3D}" == 1 && "${METHOD}" == denoise ]]; then
  echo "VIEW3D=1 is not supported with METHOD=denoise" >&2
  exit 1
fi

# ENCODER_3D=1: stages 4-6 encode ligands with the Uni-Mol 3D encoder instead of
# the 2D adapter. Needs a 3D-pretrained adapter, so it implies VIEW3D=1.
ENCODER_3D="${ENCODER_3D:-0}"
case "${ENCODER_3D}" in
  0|false|no|"") ENCODER_3D=0 ;;
  1|true|yes)    ENCODER_3D=1 ;;
  *) echo "ENCODER_3D=${ENCODER_3D} (want 0 or 1)" >&2; exit 1 ;;
esac
if [[ "${ENCODER_3D}" == 1 && "${VIEW3D}" != 1 ]]; then
  echo "ENCODER_3D=1 requires VIEW3D=1 (needs a 3D-pretrained adapter)" >&2
  exit 1
fi

case "${METHOD}" in
  lejepa|ntxent|siglip|ijepa|denoise) ;;
  *)
    echo "unknown METHOD=${METHOD} (want lejepa, ntxent, siglip, ijepa, or denoise)" >&2
    exit 1
    ;;
esac

case "${PROTEIN}" in
  esm2|esm|esmc) ;;
  *)
    echo "unknown PROTEIN=${PROTEIN} (want esm2 or esmc)" >&2
    exit 1
    ;;
esac

if ! [[ "${N_SEEDS}" =~ ^[0-9]+$ ]] || [[ "${N_SEEDS}" -lt 1 ]]; then
  echo "N_SEEDS must be a positive integer, got: ${N_SEEDS}" >&2
  exit 1
fi

# siglip is a within-1D contrastive method like ntxent (normalized-projection
# InfoNCE vs sigmoid), so it shares ntxent's downstream EBM treatment.
case "${METHOD}" in
  ntxent|siglip) PIPELINE_EBM_METHOD=ntxent ;;
  *)             PIPELINE_EBM_METHOD=lejepa ;;
esac

# STAGE_FROM=4: re-submit stages 4→6 on a pipeline that already finished 2–3.
if [[ "${STAGE_FROM:-}" == 4 ]]; then
  _pe=""
  if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
    _pe="${PIPELINE_ENV}"
  elif [[ -n "${ADAPTER_RUN_ID:-}" ]]; then
    _pe="$(lattice_pipeline_env_path "${ADAPTER_RUN_ID}")"
  fi
  [[ -f "${_pe}" ]] || {
    echo "STAGE_FROM=4 needs ADAPTER_RUN_ID or PIPELINE_ENV (pipeline folder under logs/slurm/pipeline/)" >&2
    exit 1
  }
  PIPELINE_ENV="${_pe}"
  lattice_pipeline_source_env
  : "${ADAPTER_RUN_ID:?pipeline ${PIPELINE_ENV} has no ADAPTER_RUN_ID}"

  if [[ "${SMOKE:-0}" == 1 ]]; then
    SBATCH_TIME=(--time=01:00:00)
  else
    SBATCH_TIME=()
  fi

  EXPORT="ALL,PIPELINE_ENV=${PIPELINE_ENV},PIPELINE_ID=${PIPELINE_ID},METHOD=${METHOD},N_SEEDS=${N_SEEDS},MULTISEED=${MULTISEED},PROTEIN=${PROTEIN},PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD},SMOKE=${SMOKE:-0},MERGE=${MERGE:-0},OVERWRITE=0,STAGE4_RESUME=1"

  J4="$(sbatch --parsable "${SBATCH_TIME[@]}" --export="${EXPORT}" \
    scripts/slurm/stage4_precompute_decoys.sh)"
  ARRAY_MAX=$((N_SEEDS - 1))
  J5="$(sbatch --parsable "${SBATCH_TIME[@]}" \
    --array="0-${ARRAY_MAX}" \
    --dependency="afterok:${J4}" \
    --export="${EXPORT}" \
    scripts/slurm/stage5_ebm_train.sh)"
  J6="$(sbatch --parsable "${SBATCH_TIME[@]}" \
    --dependency="afterok:${J5}" \
    --export="${EXPORT}" \
    scripts/slurm/stage6_eval.sh)"

  cat <<EOF
resume from stage 4 on ${ADAPTER_RUN_ID} (stages 2–3 unchanged)
  env:     ${PIPELINE_ENV}
  stage 4: ${J4}  (bdb + binders; decoy_zm skipped if present)
  stage 5: ${J5}  (EBM x${N_SEEDS}, after ${J4})
  stage 6: ${J6}  (LIT-PCBA$([ "${N_SEEDS}" -eq 3 ] && echo ' mv4 + 3-seed ensemble' || echo ''), after ${J5})
EOF
  exit 0
fi

# STAGE_FROM=5: re-submit stage 5 + stage 6 on a pipeline that already finished 2–4.
if [[ "${STAGE_FROM:-}" == 5 ]]; then
  _pe=""
  if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
    _pe="${PIPELINE_ENV}"
  elif [[ -n "${ADAPTER_RUN_ID:-}" ]]; then
    _pe="$(lattice_pipeline_env_path "${ADAPTER_RUN_ID}")"
  fi
  [[ -f "${_pe}" ]] || {
    echo "STAGE_FROM=5 needs ADAPTER_RUN_ID or PIPELINE_ENV (pipeline folder under logs/slurm/{pipeline,ablation}/)" >&2
    exit 1
  }
  PIPELINE_ENV="${_pe}"
  lattice_pipeline_source_env
  : "${ADAPTER_RUN_ID:?pipeline ${PIPELINE_ENV} has no ADAPTER_RUN_ID}"
  METHOD="${METHOD:-${PIPELINE_EBM_METHOD:-lejepa}}"
  lattice_pipeline_backfill_env

  if [[ "${SMOKE:-0}" == 1 ]]; then
    SBATCH_TIME=(--time=01:00:00)
  else
    SBATCH_TIME=()
  fi

  EXPORT="ALL,PIPELINE_ENV=${PIPELINE_ENV},PIPELINE_ID=${PIPELINE_ID},METHOD=${METHOD},N_SEEDS=${N_SEEDS},MULTISEED=${MULTISEED},PROTEIN=${PROTEIN},PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD},SMOKE=${SMOKE:-0},MERGE=${MERGE:-0},ABLATION=${ABLATION:-0},ARTIFACTS_ROOT=${ARTIFACTS_ROOT:-artifacts},PIPELINE_LOG_ROOT=${PIPELINE_LOG_ROOT:-logs/slurm/pipeline},OVERWRITE=0"

  ARRAY_MAX=$((N_SEEDS - 1))
  J5="$(sbatch --parsable "${SBATCH_TIME[@]}" \
    --array="0-${ARRAY_MAX}" \
    --export="${EXPORT}" \
    scripts/slurm/stage5_ebm_train.sh)"
  J6="$(sbatch --parsable "${SBATCH_TIME[@]}" \
    --dependency="afterok:${J5}" \
    --export="${EXPORT}" \
    scripts/slurm/stage6_eval.sh)"

  cat <<EOF
resume from stage 5 on ${ADAPTER_RUN_ID} (stages 2–4 unchanged)
  env:     ${PIPELINE_ENV}
  stage 5: ${J5}  (EBM x${N_SEEDS})
  stage 6: ${J6}  (LIT-PCBA$([ "${N_SEEDS}" -eq 3 ] && echo ' mv4 + 3-seed ensemble' || echo ''), after ${J5})
EOF
  exit 0
fi

# MULTISEED=1: re-submit stage 5 (×3) + stage 6 on a pipeline that already finished 2–4.
if [[ "${MULTISEED}" == 1 ]]; then
  _pe=""
  if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
    _pe="${PIPELINE_ENV}"
  elif [[ -n "${ADAPTER_RUN_ID:-}" ]]; then
    _pe="$(lattice_pipeline_env_path "${ADAPTER_RUN_ID}")"
    if [[ ! -f "${_pe}" ]]; then
      # ponytail: O(n) scan — user often passes EBM run id (ebm.* sidecar) not adapter dir name
      _resolved="$(grep -l "^${ADAPTER_RUN_ID}$" \
        "${REPO}/logs/slurm/pipeline"/*/ebm.* \
        "${REPO}/logs/slurm/ablation"/*/ebm.* 2>/dev/null | head -1 || true)"
      if [[ -n "${_resolved}" ]]; then
        _dir="$(dirname "${_resolved}")"
        echo "note: ${ADAPTER_RUN_ID} is an EBM run id; using pipeline ADAPTER_RUN_ID=$(basename "${_dir}")" >&2
        ADAPTER_RUN_ID="$(basename "${_dir}")"
        _pe="${_dir}/pipeline.env"
      fi
    fi
    [[ -f "${_pe}" ]] || _pe=""
  fi
  if [[ -z "${_pe}" ]]; then
    echo "MULTISEED=1 only adds 3-seed EBM + ensemble eval to an existing pipeline (stages 2–4 are not re-run)." >&2
    echo "Set ADAPTER_RUN_ID=<adapter_run_id> (folder under logs/slurm/{pipeline,ablation}/, e.g. wk8denar)" >&2
    echo "  or PIPELINE_ENV=logs/slurm/<pipeline|ablation>/<adapter_run_id>/pipeline.env" >&2
    echo "  (EBM W&B run ids in ebm.* sidecars are also accepted as ADAPTER_RUN_ID)" >&2
    echo "For a fresh end-to-end run with 3 seeds: N_SEEDS=3 ./scripts/slurm/run_pipeline.sh ${METHOD}" >&2
    exit 1
  fi
  PIPELINE_ENV="${_pe}"
  lattice_pipeline_source_env
  if [[ -z "${ADAPTER_RUN_ID:-}" ]]; then
    echo "pipeline ${PIPELINE_ENV} has no ADAPTER_RUN_ID (stage 2 not finished?)" >&2
    exit 1
  fi
  N_SEEDS=3
  lattice_pipeline_set_env N_SEEDS 3
  lattice_pipeline_set_env MULTISEED 1
  lattice_pipeline_backfill_env
  lattice_pipeline_source_env

  if [[ "${SMOKE:-0}" == 1 ]]; then
    SBATCH_TIME=(--time=01:00:00)
  else
    SBATCH_TIME=()
  fi

  EXPORT="ALL,PIPELINE_ENV=${PIPELINE_ENV},PIPELINE_ID=${PIPELINE_ID},METHOD=${METHOD},N_SEEDS=${N_SEEDS},MULTISEED=1,PROTEIN=${PROTEIN},PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD},SMOKE=${SMOKE:-0},MERGE=${MERGE:-0},ABLATION=${ABLATION:-0},ARTIFACTS_ROOT=${ARTIFACTS_ROOT:-artifacts},PIPELINE_LOG_ROOT=${PIPELINE_LOG_ROOT:-logs/slurm/pipeline},OVERWRITE=0"

  MISSING="$(lattice_pipeline_missing_ebm_seeds "${N_SEEDS}")"
  ACTIVE_DEPS="$(lattice_pipeline_ebm_active_deps "${N_SEEDS}")"
  SUBMIT_S6=1
  for _s in $(seq 0 $((N_SEEDS - 1))); do
    if lattice_pipeline_ebm_seed_in_progress "${_s}" && \
       [[ -z "$(lattice_pipeline_ebm_active_job "${_s}")" ]]; then
      SUBMIT_S6=0
      echo "note: seed ${_s} in-flight (log, no lock) — stage 6 deferred until all seeds finish" >&2
    fi
  done
  J5=""
  if [[ -n "${MISSING}" ]]; then
    J5="$(sbatch --parsable "${SBATCH_TIME[@]}" \
      --array="${MISSING}" \
      --export="${EXPORT}" \
      scripts/slurm/stage5_ebm_train.sh)"
  fi

  S6_DEPS=()
  [[ -n "${ACTIVE_DEPS}" ]] && S6_DEPS+=("${ACTIVE_DEPS}")
  [[ -n "${J5}" ]] && S6_DEPS+=("afterok:${J5}")
  J6=""
  S6_DEP_STR=""
  if [[ "${SUBMIT_S6}" == 1 ]]; then
    if [[ ${#S6_DEPS[@]} -eq 0 ]]; then
      J6="$(sbatch --parsable "${SBATCH_TIME[@]}" \
        --export="${EXPORT}" \
        scripts/slurm/stage6_eval.sh)"
    else
      S6_DEP_STR="$(IFS=,; echo "${S6_DEPS[*]}")"
      J6="$(sbatch --parsable "${SBATCH_TIME[@]}" \
        --dependency="${S6_DEP_STR}" \
        --export="${EXPORT}" \
        scripts/slurm/stage6_eval.sh)"
    fi
  fi

  _log_root="${PIPELINE_LOG_ROOT:-logs/slurm/pipeline}"
  cat <<EOF
multiseed continue on ${ADAPTER_RUN_ID} (stages 2–4 unchanged)
  env:     ${PIPELINE_ENV}
  stage 5: ${J5:-skipped — all seeds done or in-flight}  (submit seeds: ${MISSING:-none})
  stage 6: ${J6:-deferred — rerun MULTISEED=1 when all seeds finish}  (LIT-PCBA mv4 + 3-seed ensemble${S6_DEP_STR:+, after ${S6_DEP_STR}})

After stage 5:  cat ${_log_root}/${ADAPTER_RUN_ID}/ebm.*
EOF
  exit 0
fi

PIPELINE_ID="$(date +%Y%m%d-%H%M%S)-$$"
ADAPTER_RUN_ID="$(lattice_generate_run_id)"
if [[ "${ABLATION:-0}" == 1 ]]; then
  PIPELINE_LOG_ROOT=logs/slurm/ablation
  ARTIFACTS_ROOT=artifacts/ablation
  mkdir -p "${REPO}/${ARTIFACTS_ROOT}"
else
  PIPELINE_LOG_ROOT=logs/slurm/pipeline
  ARTIFACTS_ROOT=artifacts
fi
PIPELINE_LOG_DIR="${REPO}/${PIPELINE_LOG_ROOT}/${ADAPTER_RUN_ID}"
PIPELINE_ENV="${PIPELINE_LOG_DIR}/pipeline.env"
mkdir -p "${PIPELINE_LOG_DIR}"
cat > "${PIPELINE_ENV}" <<EOF
PIPELINE_ID=${PIPELINE_ID}
ADAPTER_RUN_ID=${ADAPTER_RUN_ID}
PIPELINE_LOG_DIR=${PIPELINE_LOG_DIR}
PIPELINE_LOG_ROOT=${PIPELINE_LOG_ROOT}
ARTIFACTS_ROOT=${ARTIFACTS_ROOT}
ABLATION=${ABLATION:-0}
METHOD=${METHOD}
N_SEEDS=${N_SEEDS}
MULTISEED=${MULTISEED}
PROTEIN=${PROTEIN}
PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD}
SMOKE=${SMOKE}
MERGE=${MERGE}
VIEW3D=${VIEW3D}
ENCODER_3D=${ENCODER_3D}
EOF
[[ -n "${RUN_NAME}" ]] && echo "RUN_NAME=${RUN_NAME}" >> "${PIPELINE_ENV}"
lattice_pipeline_source_env
lattice_protein_resolve
echo "D_PROTEIN=${D_PROTEIN}" >> "${PIPELINE_ENV}"
echo "PROTEIN_STORE=${PROTEIN_STORE}" >> "${PIPELINE_ENV}"
export PIPELINE_LOG_ROOT ARTIFACTS_ROOT
lattice_pipeline_freeze_at_submit "${ADAPTER_RUN_ID}"

if [[ "${SMOKE}" == 1 ]]; then
  SBATCH_TIME=(--time=01:00:00)
  echo "SMOKE=1: parquets at stage 4 → logs/slurm/pipeline/<ADAPTER_RUN_ID>/smoke_data" >&2
else
  SBATCH_TIME=()
fi

EXPORT="ALL,PIPELINE_ENV=${PIPELINE_ENV},PIPELINE_ID=${PIPELINE_ID},METHOD=${METHOD},N_SEEDS=${N_SEEDS},MULTISEED=${MULTISEED},PROTEIN=${PROTEIN},PIPELINE_EBM_METHOD=${PIPELINE_EBM_METHOD},SMOKE=${SMOKE},MERGE=${MERGE},VIEW3D=${VIEW3D},ENCODER_3D=${ENCODER_3D},ABLATION=${ABLATION:-0},ARTIFACTS_ROOT=${ARTIFACTS_ROOT},PIPELINE_LOG_ROOT=${PIPELINE_LOG_ROOT},OVERWRITE=0"

S2_ARGS=(scripts/slurm/stage2_ssl.sh "${METHOD}")
[[ -n "${RUN_NAME}" ]] && S2_ARGS+=("${RUN_NAME}")

# VIEW3D=1: precompute the conformer cache when missing (CPU, idempotent).
# Skip stage 1b if conformers.parquet already exists — stage 2+ only.
S2_DEP=()
J1B=""
if [[ "${VIEW3D}" == 1 ]]; then
  MERGE_SUFFIX="$(lattice_merge_suffix)"
  CONFORMERS="${REPO}/artifacts/preprocessing/processed/moses${MERGE_SUFFIX}/conformers.parquet"
  if [[ -f "${CONFORMERS}" ]]; then
    echo "conformers cache exists (${CONFORMERS}); skip stage 1b → stage 2+" >&2
  else
    J1B="$(sbatch --parsable --export="${EXPORT}" scripts/slurm/stage1b_precompute_conformers.sh)"
    S2_DEP=(--dependency="afterok:${J1B}")
  fi
fi

J2="$(sbatch --parsable "${SBATCH_TIME[@]}" "${S2_DEP[@]}" --export="${EXPORT}" "${S2_ARGS[@]}")"
J3="$(sbatch --parsable "${SBATCH_TIME[@]}" --dependency="afterok:${J2}" --export="${EXPORT}" \
  scripts/slurm/stage3_protein_precompute.sh)"
J4="$(sbatch --parsable "${SBATCH_TIME[@]}" --dependency="afterok:${J2}" --export="${EXPORT}" \
  scripts/slurm/stage4_precompute_decoys.sh)"

ARRAY_MAX=$((N_SEEDS - 1))
J5="$(sbatch --parsable "${SBATCH_TIME[@]}" \
  --array="0-${ARRAY_MAX}" \
  --dependency="afterok:${J4}" \
  --export="${EXPORT}" \
  scripts/slurm/stage5_ebm_train.sh)"

J6="$(sbatch --parsable "${SBATCH_TIME[@]}" \
  --dependency="afterok:${J5},afterok:${J3}" \
  --export="${EXPORT}" \
  scripts/slurm/stage6_eval.sh)"

if [[ "${SMOKE}" == 1 ]]; then
  _smoke_suffix=" (SMOKE=1)"
else
  _smoke_suffix=""
fi
cat <<EOF
pipeline ${PIPELINE_ID} submitted${_smoke_suffix}
  run id:  ${ADAPTER_RUN_ID}  (configs frozen at submit)
  env:     ${PIPELINE_ENV}
$([ "${VIEW3D}" == 1 ] && echo "  stage 1b:${J1B}  (conformer precompute; 3D pretraining)")
  stage 2: ${J2}  (adapter SSL$([ "${VIEW3D}" == 1 ] && echo ' + 3D view'))
  stage 3: ${J3}  (protein ${PROTEIN}, after ${J2})
  stage 4: ${J4}  (decoy pools, after ${J2})
  stage 5: ${J5}  (EBM x${N_SEEDS}, after ${J4})
  stage 6: ${J6}  (LIT-PCBA$([ "${N_SEEDS}" -eq 3 ] && echo ' mv4 + 3-seed ensemble' || echo ' mv4 + single ckpt'), after ${J5} + ${J3})

After stage 2:  configs already at logs/slurm/pipeline/${ADAPTER_RUN_ID}/configs/
After stage 5:  cat logs/slurm/pipeline/<ADAPTER_RUN_ID>/ebm.*
EOF
