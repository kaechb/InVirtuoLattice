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
  mkdir -p "${REPO}/logs/slurm/pipeline"
  mktemp "${REPO}/logs/slurm/pipeline/marker.XXXXXX"
}

lattice_pipeline_source_env() {
  if [[ -n "${PIPELINE_ENV:-}" && -f "${PIPELINE_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${PIPELINE_ENV}"
  fi
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

# Sample BDB train/val + a few LIT-PCBA targets into $1 (needs python — call after module load).
lattice_make_smoke_parquets() {
  local out_dir="$1"
  local bdb_dir="${2:-${REPO}/artifacts/preprocessing/processed/bindingdb/threshold_90}"
  local n_rows="${3:-${SMOKE_PARQUET_ROWS:-5000}}"
  local n_lit_targets="${4:-${SMOKE_LITPCBA_TARGETS:-3}}"
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
targets = sorted(test["target_name"].astype(str).unique())[:n_lit]
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
