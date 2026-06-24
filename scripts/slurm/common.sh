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
  mkdir -p logs/slurm/stage{1,2,3,4,5,6,7} logs/slurm/pipeline
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

# Repo-relative checkpoint paths (needs python env — call after lattice_load_*_modules).
lattice_rel_ckpt() {
  python - "${REPO}" "$@" <<'PY'
import sys
from pathlib import Path
from lattice_lab.models.builders import resolve_ebm_ckpt, resolve_ssl_best_ckpt

repo = Path(sys.argv[1])
kind, arg = sys.argv[2], sys.argv[3]
resolver = resolve_ebm_ckpt if kind == "ebm" else resolve_ssl_best_ckpt
p = Path(resolver(arg))
try:
    print(p.relative_to(repo))
except ValueError:
    print(p)
PY
}

lattice_pipeline_ssl_ckpt() {
  lattice_rel_ckpt ssl "artifacts/adapter/checkpoints/${1:?run_id}"
}

lattice_pipeline_ebm_ckpt() {
  lattice_rel_ckpt ebm "artifacts/energy/checkpoints/${1:?run_id}"
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
  if [[ -f "${PIPELINE_ENV}" ]]; then
    mv -f "${PIPELINE_ENV}" "${canonical}"
  elif [[ ! -f "${canonical}" ]]; then
    echo "pipeline.env missing at install (${PIPELINE_ENV})" >&2
    return 1
  fi
  ln -sf "${canonical}" "${PIPELINE_ENV}"
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
  lattice_job_banner "snapshotted configs → ${dst}"
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

# Stage 2 calls this once ADAPTER_RUN_ID is known; later stages read PIPELINE_LOG_DIR from env.
lattice_pipeline_init_log_dir() {
  local run_id="$1"
  PIPELINE_LOG_DIR="${REPO}/logs/slurm/pipeline/${run_id}"
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
