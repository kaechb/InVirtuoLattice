#!/usr/bin/env bash
# Shared LUMI environment for lattice_lab SLURM stage scripts.

export PROJ=project_465003063
export REPO_USER=grassogi
export PYTORCH_MODULE=PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
export MMSEQS_BIN=/projappl/project_465003063/grassogi/software/mmseqs/bin

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common.sh" ]]; then
  export REPO="${SLURM_SUBMIT_DIR}"
else
  _SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export REPO="$(cd "${_SLURM_DIR}/../.." && pwd)"
fi
export SCRATCH="/scratch/${PROJ}/${REPO_USER}"
export FLASH="/flash/${PROJ}/${REPO_USER}"
# Stage-1 MOSES/BDB shards on flash keep the legacy layout (not artifacts/preprocessing/).
export LATTICE_FLASH_PROCESSED="${FLASH}/artifacts/processed"

# Batch jobs log to .err files, not a TTY — tqdm bars flood SLURM output.
export TQDM_DISABLE=1

lattice_cd_repo() {
  cd "${REPO}"
  # Per-stage log dirs. SLURM opens --output/--error at job start and fails if
  # the parent dir is missing, so these are also committed (with .gitkeep) to
  # exist before the first submission; this just self-heals after a cleanup.
  mkdir -p logs/slurm/stage{1,2,3,4,5,6,7} logs/slurm/pipeline logs/slurm/ablation
}

lattice_prepend_mmseqs() {
  if [[ -d "${MMSEQS_BIN}" ]]; then
    export PATH="${MMSEQS_BIN}:${PATH}"
  fi
}

lattice_load_python_container() {
  export PYTHONUNBUFFERED=1
  module load "${PYTORCH_MODULE}"
  module unload lumi-CrayPath 2>/dev/null || true
  module load lumi-CrayPath
}

lattice_load_cpu_modules() {
  module load LUMI/25.09 2>/dev/null || module load LUMI
  module load partition/G
  lattice_load_python_container
  lattice_prepend_mmseqs
}

lattice_load_gpu_modules() {
  module load LUMI/25.09 partition/G
  lattice_load_python_container
  lattice_prepend_mmseqs
}

lattice_require_gpu() {
  python -c "import torch; ok = torch.cuda.is_available(); print('gpu', ok, torch.__version__); import sys; sys.exit(0 if ok else 1)"
}

lattice_require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "${path}" ]]; then
    echo "missing file: ${path}" >&2
    [[ -n "${hint}" ]] && echo "${hint}" >&2
    exit 1
  fi
}

lattice_job_banner() {
  # ponytail: flush early so SLURM logs aren't empty while modules/container warm up
  echo "[$(date -Is)] job=${SLURM_JOB_ID:-?} array=${SLURM_ARRAY_TASK_ID:-} node=$(hostname) $*" >&2
}

# W&B run id from this job's stage5 log (not global newest — parallel grid runs race).
lattice_pipeline_ebm_run_id_from_log() {
  local seed="$1"
  lattice_pipeline_source_env
  local log="${PIPELINE_LOG_DIR}/stage5.seed${seed}.out"
  local rid
  rid="$(grep -oE 'energy/checkpoints/[a-z0-9]+' "${log}" 2>/dev/null | tail -1 | sed 's|.*/||')"
  [[ -n "${rid}" ]] || return 1
  echo "${rid}"
}

# Newest immediate subdir of $dir whose mtime is after marker file $marker.
lattice_newest_subdir_since() {
  local marker="$1" dir="$2"
  find "${dir}" -mindepth 1 -maxdepth 1 -type d -newer "${marker}" -printf '%T@ %f\n' 2>/dev/null \
    | sort -n | tail -1 | awk '{print $2}'
}

lattice_pipeline_marker() {
  lattice_pipeline_source_env
  local dir="${PIPELINE_LOG_DIR:-${REPO}/logs/slurm/pipeline}"
  mkdir -p "${dir}"
  mktemp "${dir}/marker.XXXXXX"
}

lattice_pipeline_ebm_sidecar() {
  local seed="$1"
  lattice_pipeline_source_env
  if [[ -n "${PIPELINE_LOG_DIR:-}" ]]; then
    echo "${PIPELINE_LOG_DIR}/ebm.${seed}"
  else
    echo "${PIPELINE_ENV:?missing PIPELINE_ENV}.ebm.${seed}"
  fi
}

# stage2: {RUN_NAME}_stage2_{id}  →  stage5: {RUN_NAME}_stage5_{id}[_seedN]
lattice_pipeline_stage5_wandb_name() {
  local seed="$1"
  lattice_pipeline_source_env
  local name
  if [[ -n "${STAGE2_WANDB_NAME:-}" ]]; then
    name="${STAGE2_WANDB_NAME/stage2_/stage5_}"
  elif [[ -n "${RUN_NAME:-}" ]]; then
    name="${RUN_NAME}_stage5_${ADAPTER_RUN_ID}"
  else
    name="stage5_${ADAPTER_RUN_ID:-${RUN_ID:-unknown}}"
  fi
  [[ "${N_SEEDS:-1}" -gt 1 ]] && name="${name}_seed${seed}"
  echo "${name}"
}

lattice_pipeline_ebm_seed_lock() {
  echo "$(lattice_pipeline_ebm_sidecar "${1:?seed}")".lock
}

# EBM ckpt was trained with this pipeline's adapter (guards cross-run sidecar mixups).
lattice_pipeline_ebm_seed_adapter_ok() {
  local seed="$1" run_id ssl_ckpt ebm_ckpt
  lattice_pipeline_source_env
  [[ -n "${ADAPTER_RUN_ID:-}" ]] || return 0
  run_id="$(<"$(lattice_pipeline_ebm_sidecar "${seed}")")"
  ssl_ckpt="${REPO}/$(lattice_pipeline_ssl_ckpt "${ADAPTER_RUN_ID}")"
  ebm_ckpt="${REPO}/$(lattice_pipeline_ebm_eval_ckpt "${run_id}")"
  [[ -f "${ssl_ckpt}" && -f "${ebm_ckpt}" ]] || return 1
  python - "${ssl_ckpt}" "${ebm_ckpt}" <<'PY'
import sys
from lattice_lab.models.builders import adapter_fingerprint
sys.exit(0 if adapter_fingerprint(sys.argv[1]) == adapter_fingerprint(sys.argv[2]) else 1)
PY
}

# Sidecar + last.ckpt present + adapter fingerprint matches pipeline ssl ckpt.
lattice_pipeline_ebm_seed_done() {
  local seed="$1" sidecar run_id ckpt
  lattice_pipeline_source_env
  sidecar="$(lattice_pipeline_ebm_sidecar "${seed}")"
  [[ -f "${sidecar}" ]] || return 1
  run_id="$(<"${sidecar}")"
  [[ -n "${run_id}" ]] || return 1
  ckpt="${REPO}/$(lattice_artifacts_root)/energy/checkpoints/${run_id}/last.ckpt"
  [[ -f "${ckpt}" ]] || return 1
  lattice_pipeline_ebm_seed_adapter_ok "${seed}"
}

# Slurm job id if this seed is in-flight; clears stale lock files.
lattice_pipeline_ebm_active_job() {
  local seed="$1" lock jid
  lattice_pipeline_ebm_seed_done "${seed}" && return 0
  lock="$(lattice_pipeline_ebm_seed_lock "${seed}")"
  [[ -f "${lock}" ]] || return 0
  jid="$(<"${lock}")"
  [[ -n "${jid}" ]] || { rm -f "${lock}"; return 0; }
  if squeue -j "${jid}" -h &>/dev/null; then
    echo "${jid}"
    return 0
  fi
  rm -f "${lock}"
}

# Started training but not finished (covers pre-lock in-flight jobs). No job id.
lattice_pipeline_ebm_seed_in_progress() {
  local seed="$1" err
  lattice_pipeline_source_env
  lattice_pipeline_ebm_seed_done "${seed}" && return 1
  [[ -n "$(lattice_pipeline_ebm_active_job "${seed}")" ]] && return 0
  err="${PIPELINE_LOG_DIR}/stage5.seed${seed}.err"
  [[ -f "${err}" ]] && grep -q "starting train" "${err}" && \
    ! grep -q "pipeline wrote EBM seed=${seed}" "${err}" && \
    find "${err}" -mmin -480 -print -quit | grep -q .
}

# Comma-separated seed indices still needed (skips done + in-flight).
lattice_pipeline_missing_ebm_seeds() {
  local n="${1:-3}" seed missing=() aj
  lattice_pipeline_source_env
  for seed in $(seq 0 $((n - 1))); do
    lattice_pipeline_ebm_seed_done "${seed}" && continue
    aj="$(lattice_pipeline_ebm_active_job "${seed}")"
    [[ -n "${aj}" ]] && continue
    lattice_pipeline_ebm_seed_in_progress "${seed}" && continue
    missing+=("${seed}")
  done
  (IFS=,; echo "${missing[*]}")
}

# afterok:… deps for in-flight seeds (stage 6 waits on these too).
lattice_pipeline_ebm_active_deps() {
  local n="${1:-3}" seed deps=() aj
  for seed in $(seq 0 $((n - 1))); do
    aj="$(lattice_pipeline_ebm_active_job "${seed}")"
    [[ -n "${aj}" ]] && deps+=("afterok:${aj}")
  done
  (IFS=,; echo "${deps[*]}")
}

# Repo-relative checkpoint paths (needs python env — call after lattice_load_*_modules).
lattice_rel_ckpt() {
  python - "${REPO}" "$@" <<'PY'
import sys
from pathlib import Path
from lattice_lab.models.builders import resolve_adapter_ckpt, resolve_ebm_ckpt

repo = Path(sys.argv[1])
kind, arg = sys.argv[2], sys.argv[3]
if kind == "ebm_eval":
    p = Path(resolve_ebm_ckpt(arg, prefer_last=True))
elif kind == "ebm":
    p = Path(resolve_ebm_ckpt(arg))
else:
    p = Path(resolve_adapter_ckpt(arg))
try:
    print(p.relative_to(repo))
except ValueError:
    print(p)
PY
}

lattice_pipeline_ssl_ckpt() {
  local r
  r="$(lattice_artifacts_root)"
  lattice_rel_ckpt ssl "${r}/adapter/checkpoints/${1:?run_id}"
}

lattice_pipeline_ebm_ckpt() {
  local r
  r="$(lattice_artifacts_root)"
  lattice_rel_ckpt ebm "${r}/energy/checkpoints/${1:?run_id}"
}

# Stage 6: the monitored best ckpt (ebm-*.ckpt, min val/loss), falling back to
# last.ckpt. Stage 6 runs afterok:stage5, so the best ckpt is final and stable —
# no rotation race, so we no longer force last.ckpt here.
lattice_pipeline_ebm_eval_ckpt() {
  local r
  r="$(lattice_artifacts_root)"
  lattice_rel_ckpt ebm "${r}/energy/checkpoints/${1:?run_id}"
}

# artifacts | artifacts/ablation (from ABLATION=1 / grid ablation runs)
lattice_artifacts_root() {
  lattice_pipeline_source_env
  echo "${ARTIFACTS_ROOT:-artifacts}"
}

lattice_pipeline_log_root() {
  lattice_pipeline_source_env
  echo "${PIPELINE_LOG_ROOT:-logs/slurm/pipeline}"
}

lattice_zm_store_path() {
  local pool="$1" run_id="$2" merge_suffix="${3:-}"
  local art
  art="$(lattice_artifacts_root)"
  # binder_zm / binder_zm3d live under binders/; decoy_zm(3d) / bdb_zm(3d) under decoys/.
  if [[ "${pool}" == binder_zm* ]]; then
    echo "${art}/binders/${run_id}/${pool}${merge_suffix}"
  else
    echo "${art}/decoys/${run_id}/${pool}${merge_suffix}"
  fi
}

lattice_evaluation_path() {
  echo "$(lattice_artifacts_root)/evaluation/${1:?subpath}"
}

# Move pipeline.env into the run-id log dir; leave PIPELINE_ENV as a symlink so
# already-submitted jobs keep the submit-time path.
lattice_pipeline_install_env() {
  [[ -n "${PIPELINE_ENV:-}" && -n "${PIPELINE_LOG_DIR:-}" ]] || return 0
  local canonical="${PIPELINE_LOG_DIR}/pipeline.env"
  mkdir -p "${PIPELINE_LOG_DIR}"
  if [[ -L "${PIPELINE_ENV}" ]]; then
    return 0
  fi
  if [[ -f "${PIPELINE_ENV}" ]] && ! [[ "${PIPELINE_ENV}" -ef "${canonical}" ]]; then
    mv -f "${PIPELINE_ENV}" "${canonical}"
  elif [[ ! -f "${canonical}" ]]; then
    echo "pipeline.env missing at install (${PIPELINE_ENV})" >&2
    return 1
  fi
  if ! [[ "${PIPELINE_ENV}" -ef "${canonical}" ]]; then
    ln -sf "${canonical}" "${PIPELINE_ENV}"
  fi
}

# PROTEIN=esm2|esmc → store path, embedding dim, and stage-3 CLI extras.
lattice_protein_resolve() {
  lattice_pipeline_source_env
  PROTEIN="${PROTEIN:-esm2}"
  case "${PROTEIN}" in
    esm2|esm)
      PROTEIN_STORE=artifacts/protein_store/embeddings/esm2_650M
      D_PROTEIN=1280
      PROTEIN_EXTRA=(--no-canonical-filter)
      ;;
    esmc)
      PROTEIN_STORE=artifacts/protein_store/embeddings/esmc_600m
      D_PROTEIN=1152
      PROTEIN_EXTRA=(--backend esmc --no-canonical-filter)
      ;;
    *)
      echo "unknown PROTEIN=${PROTEIN} (want esm2 or esmc)" >&2
      return 1
      ;;
  esac
  export PROTEIN PROTEIN_STORE D_PROTEIN PROTEIN_EXTRA
}

# Hydra overrides so stage 5 model head dim matches the protein store.
lattice_protein_hydra_args() {
  lattice_protein_resolve
  echo "data.protein_store=${PROTEIN_STORE}/"
  echo "d_protein=${D_PROTEIN}"
}

# Freeze Hydra yaml tree + CLI overrides under the run dir so later pipeline
# stages (e.g. stage 5) are not affected by repo config edits after stage 2.
lattice_pipeline_snapshot_configs() {
  [[ -n "${PIPELINE_LOG_DIR:-}" ]] || return 0
  local dst="${PIPELINE_LOG_DIR}/configs"
  [[ -d "${dst}" ]] && return 0
  local src="${REPO}/src/lattice_lab/configs"
  if [[ ! -d "${src}" ]]; then
    echo "missing Hydra configs: ${src}" >&2
    return 1
  fi
  cp -a "${src}" "${dst}"
  if lattice_protein_resolve && [[ "${PROTEIN}" == esmc ]]; then
    sed -i \
      "s|protein_store: artifacts/protein_store/embeddings/esm2_650M/|protein_store: ${PROTEIN_STORE}/|" \
      "${dst}/data/ebm.yaml"
    sed -i "s/^d_protein: 1280/d_protein: ${D_PROTEIN}/" "${dst}/train.yaml" "${dst}/eval.yaml"
  fi
  lattice_job_banner "snapshotted configs → ${dst} (PROTEIN=${PROTEIN:-esm2})"
}

# Freeze the exact code behind a run alongside its config snapshot: HEAD sha +
# dirty patch + porcelain status. Exact repro is then
# `git checkout $(cat git.sha) && git apply git.diff`, replay stageN.train.args.
# ponytail: git.diff captures tracked edits only; untracked file *contents* are
# not saved (they are listed in git.status). Upgrade path: tar the untracked set
# if a run ever depends on an untracked module.
lattice_pipeline_freeze_git() {
  local dst="${1:-${PIPELINE_LOG_DIR:-}}"
  [[ -n "${dst}" ]] || return 0
  git -C "${REPO}" rev-parse HEAD > "${dst}/git.sha" 2>/dev/null || return 0
  git -C "${REPO}" diff HEAD > "${dst}/git.diff" 2>/dev/null || true
  git -C "${REPO}" status --porcelain > "${dst}/git.status" 2>/dev/null || true
  if [[ -s "${dst}/git.status" ]]; then
    lattice_job_banner "dirty tree at submit: recorded HEAD+diff → ${dst}/git.{sha,diff}; $(wc -l < "${dst}/git.status") uncommitted path(s) in git.status (untracked file contents NOT captured)"
  fi
}

lattice_pipeline_save_train_args() {
  local stage="$1"
  shift
  [[ -n "${PIPELINE_LOG_DIR:-}" ]] || return 0
  printf '%s\n' "$@" > "${PIPELINE_LOG_DIR}/stage${stage}.train.args"
}

lattice_pipeline_source_env() {
  if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${PIPELINE_ENV}"
  fi
}

# Update or append KEY=VAL in an existing pipeline.env (login-node only).
lattice_pipeline_set_env() {
  local key="$1" val="$2" f="${PIPELINE_ENV:?missing PIPELINE_ENV}"
  local tmp="${f}.tmp.$$"
  grep -v "^${key}=" "${f}" > "${tmp}" 2>/dev/null || : > "${tmp}"
  echo "${key}=${val}" >> "${tmp}"
  mv "${tmp}" "${f}"
}

# Idempotent: add derived keys to older pipeline.env files (pre-STAGE2_WANDB_NAME, etc.).
lattice_pipeline_backfill_env() {
  [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]] || return 0
  lattice_pipeline_source_env
  if [[ -n "${RUN_NAME:-}" && -z "${STAGE2_WANDB_NAME:-}" && -n "${ADAPTER_RUN_ID:-}" ]]; then
    lattice_pipeline_set_env STAGE2_WANDB_NAME "${RUN_NAME}_stage2_${ADAPTER_RUN_ID}"
  fi
  lattice_protein_resolve
  if ! grep -q "^D_PROTEIN=" "${PIPELINE_ENV}" 2>/dev/null; then
    lattice_pipeline_set_env D_PROTEIN "${D_PROTEIN}"
    lattice_pipeline_set_env PROTEIN_STORE "${PROTEIN_STORE}"
  fi
  lattice_pipeline_source_env
}

# Resolve pipeline.env for an adapter run id (plain or _finished dir).
lattice_pipeline_env_path() {
  local id="$1" root pe
  for root in logs/slurm/ablation logs/slurm/pipeline; do
    pe="${REPO}/${root}/${id}/pipeline.env"
    if [[ -f "${pe}" ]]; then
      echo "${pe}"
      return 0
    fi
    pe="${REPO}/${root}/${id}_finished/pipeline.env"
    if [[ -f "${pe}" ]]; then
      echo "${pe}"
      return 0
    fi
  done
}

# Rename pipeline log dir to <run_id>_finished after stage 6 succeeds.
lattice_pipeline_mark_finished() {
  [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]] || return 0
  lattice_pipeline_source_env
  [[ -n "${PIPELINE_LOG_DIR:-}" && -d "${PIPELINE_LOG_DIR}" ]] || return 0
  local base old new
  old="${PIPELINE_LOG_DIR}"
  base="$(basename "${old}")"
  [[ "${base}" == *_finished ]] && return 0
  new="${old}_finished"
  mv "${old}" "${new}"
  PIPELINE_LOG_DIR="${new}"
  PIPELINE_ENV="${new}/pipeline.env"
  export PIPELINE_LOG_DIR PIPELINE_ENV
  lattice_pipeline_set_env PIPELINE_LOG_DIR "${new}"
  lattice_pipeline_set_env FINISHED 1
  local f tmp
  for f in "${REPO}/logs/slurm/pipeline"/*.env; do
    [[ -f "${f}" ]] || continue
    grep -q "^PIPELINE_LOG_DIR=${old}$" "${f}" || continue
    tmp="${f}.tmp.$$"
    grep -v "^PIPELINE_LOG_DIR=\|^FINISHED=" "${f}" > "${tmp}" 2>/dev/null || : > "${tmp}"
    echo "PIPELINE_LOG_DIR=${new}" >> "${tmp}"
    echo "FINISHED=1" >> "${tmp}"
    mv "${tmp}" "${f}"
  done
  lattice_job_banner "pipeline finished → ${new}"
}

# Suffix for the merge (multi-granularity) dataset/store variant. Driven by the
# MERGE env var (0/1), threaded through the pipeline so every stage reads the
# matching moses_merge / bindingdb_merge shards and decoy_zm_merge / bdb_zm_merge
# / binder_zm_merge stores. Empty for the default finest-partition variant.
lattice_merge_suffix() {
  case "${MERGE:-0}" in
    1|true|yes) echo "_merge" ;;
    0|false|no|"") echo "" ;;
    *)
      echo "MERGE=${MERGE} (want 0 or 1)" >&2
      return 1
      ;;
  esac
}

# Stage 1 writes MOSES shards under the repo; stage 2/4 read from LUSTRE flash.
# ponytail: ~300MB–2GB rsync once beats debugging path mismatches every run.
lattice_sync_moses_shards_to_flash() {
  local suffix="${1:-}"
  local src="${REPO}/artifacts/preprocessing/processed/moses${suffix}"
  local dst="${LATTICE_FLASH_PROCESSED}/moses${suffix}"
  if compgen -G "${src}/shard_"*.parquet > /dev/null; then
    mkdir -p "${dst}"
    lattice_job_banner "sync moses${suffix} → ${dst}"
    rsync -r --omit-dir-times --no-perms --no-owner --no-group "${src}/" "${dst}/"
    return 0
  fi
  if compgen -G "${dst}/shard_"*.parquet > /dev/null; then
    lattice_job_banner "moses${suffix} already on flash (${dst})"
    return 0
  fi
  echo "no moses shards in ${src} or ${dst} — run stage1 first" >&2
  return 1
}

# Authoritative merge suffix for a *trained adapter*: reads the fragment_merge
# flag the SSL module embedded in its checkpoint, so Stage 4/5 pick the matching
# stores no matter how (or whether) they were launched — no env, no mismatch.
# Needs the python env loaded (call after lattice_load_gpu_modules).
lattice_ckpt_merge_suffix() {
  python - "$1" <<'PY'
import sys
from lattice_lab.models.builders import merge_from_ckpt
print("_merge" if merge_from_ckpt(sys.argv[1]) else "", end="")
PY
}

# 8-char id (wandb-compatible). Login-node safe — no GPU module required.
lattice_generate_run_id() {
  python3 -c 'import wandb; print(wandb.util.generate_id())' 2>/dev/null \
    || tr -dc 'a-z0-9' </dev/urandom | head -c 8
}

# Freeze Hydra tree at submit time so queued ablations don't all pick up the last edit.
lattice_pipeline_freeze_at_submit() {
  local run_id="$1"
  local log_root
  log_root="$(lattice_pipeline_log_root)"
  PIPELINE_LOG_DIR="${REPO}/${log_root}/${run_id}"
  mkdir -p "${PIPELINE_LOG_DIR}"
  export PIPELINE_LOG_DIR
  lattice_pipeline_snapshot_configs
  lattice_pipeline_freeze_git
  echo "froze configs + code (git.sha/diff) at submit → ${PIPELINE_LOG_DIR}" >&2
}

# Stage 2 calls this once ADAPTER_RUN_ID is known; later stages read PIPELINE_LOG_DIR from env.
lattice_pipeline_init_log_dir() {
  local run_id="$1"
  local log_root
  log_root="$(lattice_pipeline_log_root)"
  PIPELINE_LOG_DIR="${REPO}/${log_root}/${run_id}"
  mkdir -p "${PIPELINE_LOG_DIR}"
  if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]] \
    && ! grep -q '^PIPELINE_LOG_DIR=' "${PIPELINE_ENV}"; then
    echo "PIPELINE_LOG_DIR=${PIPELINE_LOG_DIR}" >> "${PIPELINE_ENV}"
  fi
  export PIPELINE_LOG_DIR
}

# SLURM --output/--error paths are fixed at submit time; move into PIPELINE_LOG_DIR at job end.
lattice_pipeline_collect_slurm_logs() {
  local stage="$1"
  if [[ -z "${PIPELINE_LOG_DIR:-}" ]]; then
    lattice_pipeline_source_env
  fi
  [[ -n "${PIPELINE_LOG_DIR:-}" ]] || return 0
  mkdir -p "${PIPELINE_LOG_DIR}"

  local base="${REPO}/logs/slurm/stage${stage}" out err stem
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    stem="stage${stage}.seed${SLURM_ARRAY_TASK_ID}"
    out="${base}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    err="${base}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
  else
    stem="stage${stage}"
    out="${base}/${SLURM_JOB_ID}.out"
    err="${base}/${SLURM_JOB_ID}.err"
  fi

  [[ -f "${out}" ]] && mv -f "${out}" "${PIPELINE_LOG_DIR}/${stem}.out"
  [[ -f "${err}" ]] && mv -f "${err}" "${PIPELINE_LOG_DIR}/${stem}.err"
}

# Pipeline jobs: symlink SLURM logs into PIPELINE_LOG_DIR immediately, mv on EXIT.
lattice_pipeline_track_slurm_logs() {
  local stage="$1"
  [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]] || return 0
  if [[ -z "${PIPELINE_LOG_DIR:-}" ]]; then
    lattice_pipeline_source_env
  fi
  [[ -n "${PIPELINE_LOG_DIR:-}" ]] || return 0
  mkdir -p "${PIPELINE_LOG_DIR}"

  local base="${REPO}/logs/slurm/stage${stage}" out err stem
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    stem="stage${stage}.seed${SLURM_ARRAY_TASK_ID}"
    out="${base}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    err="${base}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
  else
    stem="stage${stage}"
    out="${base}/${SLURM_JOB_ID}.out"
    err="${base}/${SLURM_JOB_ID}.err"
  fi

  ln -sf "${out}" "${PIPELINE_LOG_DIR}/${stem}.out"
  ln -sf "${err}" "${PIPELINE_LOG_DIR}/${stem}.err"
  trap "lattice_pipeline_collect_logs_on_exit ${stage}" EXIT
  lattice_job_banner "pipeline logs → ${PIPELINE_LOG_DIR}/${stem}.{out,err}"
}

lattice_pipeline_collect_logs_on_exit() {
  local stage="$1"
  lattice_pipeline_collect_slurm_logs "${stage}" || true
}

lattice_smoke_enabled() {
  [[ "${SMOKE:-0}" == 1 ]]
}

lattice_smoke_precompute_limit() {
  echo "${SMOKE_PRECOMPUTE_LIMIT:-5000}"
}

# Sample BDB train/val + LIT-PCBA test rows into $1 (needs python — call after module load).
lattice_make_smoke_parquets() {
  local out_dir="$1"
  local bdb_dir="${2:-${REPO}/artifacts/preprocessing/processed/bindingdb/threshold_90}"
  local n_rows="${3:-${SMOKE_PARQUET_ROWS:-5000}}"
  local n_lit_targets="${4:-${SMOKE_LITPCBA_TARGETS:-0}}"
  local py=""
  mkdir -p "${out_dir}"
  if command -v python >/dev/null 2>&1; then
    py=python
  elif command -v python3 >/dev/null 2>&1; then
    py=python3
  else
    lattice_load_cpu_modules
    py=python
  fi
  "${py}" - "${out_dir}" "${bdb_dir}" "${n_rows}" "${n_lit_targets}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

out = Path(sys.argv[1])
bdb = Path(sys.argv[2])
n_rows, n_lit = int(sys.argv[3]), int(sys.argv[4])

for split in ("train", "val"):
    df = pd.read_parquet(bdb / f"{split}.parquet").head(n_rows)
    p = out / f"{split}.parquet"
    df.to_parquet(p)
    print(f"smoke parquet {split}: {len(df)} rows -> {p}")

test_src = bdb.parent / "test_lit_pcba.parquet"
test = pd.read_parquet(test_src)
targets = sorted(test["target_name"].astype(str).unique())
if n_lit > 0:
    targets = targets[:n_lit]
sub = test[test["target_name"].astype(str).isin(targets)]
p = out / "test_lit_pcba.parquet"
sub.to_parquet(p)
print(f"smoke test_lit_pcba: {len(targets)} targets, {len(sub)} rows -> {p}")
PY
}

# Idempotent; safe to call from stage 4/5 after lattice_load_*_modules.
lattice_ensure_smoke_parquets() {
  local out_dir="$1"
  if [[ -f "${out_dir}/train.parquet" && -f "${out_dir}/val.parquet" && -f "${out_dir}/test_lit_pcba.parquet" ]]; then
    return 0
  fi
  lattice_make_smoke_parquets "$@"
}
