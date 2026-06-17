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

lattice_cd_repo() {
  cd "${REPO}"
  # Per-stage log dirs. SLURM opens --output/--error at job start and fails if
  # the parent dir is missing, so these are also committed (with .gitkeep) to
  # exist before the first submission; this just self-heals after a cleanup.
  mkdir -p logs/slurm/stage{1,2,3,4,5,6,7}
}

lattice_prepend_mmseqs() {
  if [[ -d "${MMSEQS_BIN}" ]]; then
    export PATH="${MMSEQS_BIN}:${PATH}"
  fi
}

lattice_load_cpu_modules() {
  module load LUMI/25.09 2>/dev/null || module load LUMI
  lattice_prepend_mmseqs
}

lattice_load_gpu_modules() {
  module load LUMI/25.09 partition/G
  module load "${PYTORCH_MODULE}"
  lattice_prepend_mmseqs
  # lumi-CrayPath fixes LD_LIBRARY_PATH for the Cray PE and MUST be (re)loaded
  # LAST, after every other module change, or the container fails to link its
  # libraries on compute nodes. Force a reload so it re-applies even if already
  # loaded by the default environment.
  module unload lumi-CrayPath 2>/dev/null || true
  module load lumi-CrayPath
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
