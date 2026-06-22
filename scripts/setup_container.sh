#!/usr/bin/env bash
#
# Reproducible LUMI container setup for lattice_lab.
#
# Installs the PyTorch container into your /projappl EasyBuild tree, installs
# lattice_lab (editable) + the `esmc` extra (the EvolutionaryScale `esm` SDK)
# into the container's overlay venv, freezes that venv into a SquashFS image,
# and verifies the result. Run this once per container (re)build; afterwards
# `lattice_lab` and `esm` are importable in every job/session with no per-run
# `pip install`.
#
#   RUN ON A LOGIN NODE (EasyBuild must not run on compute nodes):
#     bash scripts/setup_container.sh
#
# Re-running is safe: if the overlay is already squashed it is unpacked, updated,
# and repacked. Code edits to src/ are picked up live (editable install) and do
# NOT require re-running this — only dependency/packaging changes do.
#
# Override any of these via the environment if your project layout differs:
#   PROJECT, LUMI_STACK, PYTORCH_MODULE, EBU_USER_PREFIX, HF_HOME, EASYCONFIG
set -uo pipefail

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT="${PROJECT:-project_465003063}"
LUMI_STACK="${LUMI_STACK:-25.09}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827}"
EBU_USER_PREFIX="${EBU_USER_PREFIX:-/projappl/${PROJECT}/${USER}/EasyBuild}"
HF_HOME="${HF_HOME:-/scratch/${PROJECT}/${USER}/hf-home}"

# The container easyconfig lives in LUMI's container repo, which is NOT on the
# default robot path, so we pass the full path. Derive the .eb name from the
# module name (PyTorch/<ver> -> PyTorch-<ver>.eb).
_EC_NAME="${PYTORCH_MODULE/\//-}.eb"
EASYCONFIG="${EASYCONFIG:-/appl/local/containers/LUMI-EasyBuild-containers/easybuild/easyconfigs/p/PyTorch/${_EC_NAME}}"

# Repo root = parent of this script's directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Whether to delete the expanded user-software/ dir after a verified squash
# (reclaims inodes; the squashfs becomes authoritative). Set to 0 to keep it.
REMOVE_USER_SOFTWARE="${REMOVE_USER_SOFTWARE:-1}"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

ensure_module_cmd() {
  if ! command -v module >/dev/null 2>&1 && [[ -z "${LMOD_CMD:-}" ]]; then
    for init in /usr/share/lmod/lmod/init/bash /opt/cray/pe/lmod/lmod/init/bash; do
      [[ -f "$init" ]] && source "$init" && break
    done
  fi
  command -v module >/dev/null 2>&1 || type module >/dev/null 2>&1 \
    || die "the 'module' command is unavailable; run this on a LUMI login node."
}

persist_env() {
  # Add `export NAME=VALUE` to ~/.bashrc unless NAME is already configured there.
  local name="$1" value="$2"
  if ! grep -q "export ${name}=" "${HOME}/.bashrc" 2>/dev/null; then
    echo "export ${name}=${value}" >> "${HOME}/.bashrc"
    log "persisted ${name} in ~/.bashrc"
  fi
}

# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #
ensure_module_cmd

case "$(hostname)" in
  uan*) : ;;  # login node, good
  nid*) warn "you appear to be on a COMPUTE node ($(hostname)). EasyBuild should run on a login node (uan*). Continuing, but 'eb' may misbehave." ;;
  *)    warn "unrecognised host $(hostname); expected a LUMI login node (uan*)." ;;
esac

[[ -f "${EASYCONFIG}" ]] || die "easyconfig not found: ${EASYCONFIG}
  Locate it with:  eb -S ${PYTORCH_MODULE%%/*}-${PYTORCH_MODULE#*/}
  then set EASYCONFIG=/full/path/to.eb and re-run."

[[ -f "${REPO_ROOT}/pyproject.toml" ]] || die "no pyproject.toml at repo root ${REPO_ROOT}"

log "Configuration"
cat <<EOF
  PROJECT          = ${PROJECT}
  LUMI_STACK       = ${LUMI_STACK}
  PYTORCH_MODULE   = ${PYTORCH_MODULE}
  EBU_USER_PREFIX  = ${EBU_USER_PREFIX}
  HF_HOME          = ${HF_HOME}
  EASYCONFIG       = ${EASYCONFIG}
  REPO_ROOT        = ${REPO_ROOT}
EOF

# --------------------------------------------------------------------------- #
# 1. Persistent environment (EasyBuild prefix + HuggingFace cache off $HOME)
# --------------------------------------------------------------------------- #
log "Setting up persistent environment"
export EBU_USER_PREFIX HF_HOME
persist_env EBU_USER_PREFIX "${EBU_USER_PREFIX}"
persist_env HF_HOME "${HF_HOME}"
mkdir -p "${HF_HOME}"

# --------------------------------------------------------------------------- #
# 2. Install the container into the /projappl EasyBuild tree
# --------------------------------------------------------------------------- #
log "Loading LUMI/${LUMI_STACK} + EasyBuild-user"
module purge
module load "LUMI/${LUMI_STACK}" partition/G || die "failed to load LUMI/${LUMI_STACK} partition/G"
module load EasyBuild-user || die "failed to load EasyBuild-user"

log "Installing container (copies the prebuilt .sif; minutes, no rebuild)"
eb "${EASYCONFIG}" -r || die "eb install failed; see the EasyBuild log above."

# --------------------------------------------------------------------------- #
# 3. Load the freshly installed container module
# --------------------------------------------------------------------------- #
log "Loading ${PYTORCH_MODULE}"
module purge
module load "LUMI/${LUMI_STACK}" partition/G
module load "${PYTORCH_MODULE}" || module --ignore_cache load "${PYTORCH_MODULE}" \
  || die "could not load ${PYTORCH_MODULE} after install."
module load lumi-CrayPath  # MUST be last; fixes LD_LIBRARY_PATH for the Cray PE

[[ -n "${CONTAINERROOT:-}" && -d "${CONTAINERROOT}" ]] \
  || die "CONTAINERROOT is unset/invalid after module load: '${CONTAINERROOT:-}'"
command -v python >/dev/null 2>&1 || die "container 'python' wrapper not on PATH."
log "Container ready: CONTAINERROOT=${CONTAINERROOT}"
python --version

# --------------------------------------------------------------------------- #
# 4. Make the overlay venv writable (unpack squashfs if this is a re-run)
# --------------------------------------------------------------------------- #
if [[ -f "${CONTAINERROOT}/user-software.squashfs" && ! -d "${CONTAINERROOT}/user-software" ]]; then
  log "Existing squashfs found -> unpacking to update it"
  ( cd "${CONTAINERROOT}" && unmake-squashfs ) || die "unmake-squashfs failed."
  rm -f "${CONTAINERROOT}/user-software.squashfs"
  # Reload so /user-software mounts the now-writable directory, not the squashfs.
  module purge
  module load "LUMI/${LUMI_STACK}" partition/G
  module load "${PYTORCH_MODULE}"
  module load lumi-CrayPath
fi

# --------------------------------------------------------------------------- #
# 5. Install lattice_lab (editable) + esmc extra into the overlay venv
# --------------------------------------------------------------------------- #
log "Installing lattice_lab[esmc] (editable) into the overlay venv"
( cd "${REPO_ROOT}" && pip install -e '.[esmc]' ) \
  || die "pip install failed."

log "Verifying imports in the live (unsquashed) venv"
python -c "import lattice_lab.protein.encoder, esm, transformers; \
print('imports OK | esm', esm.__version__, '| transformers', transformers.__version__)" \
  || die "import check failed before squashing."

# --------------------------------------------------------------------------- #
# 6. Freeze the venv into a SquashFS image
# --------------------------------------------------------------------------- #
log "Packing overlay venv into SquashFS"
rm -f "${CONTAINERROOT}/user-software.squashfs"
# NOTE: the make-squashfs wrapper prints a 'Created ...' message even when
# mksquashfs fails (its 'mksquashfs | grep' pipeline hides the exit code), so we
# do NOT trust it. We verify by reloading and importing from the squashfs below.
( cd "${CONTAINERROOT}" && make-squashfs )
[[ -s "${CONTAINERROOT}/user-software.squashfs" ]] \
  || die "user-software.squashfs was not created (likely disk quota). Check 'lumi-quota'."

# --------------------------------------------------------------------------- #
# 7. Reload from the squashfs and verify BEFORE removing the writable dir
# --------------------------------------------------------------------------- #
log "Reloading module to mount the squashfs, then verifying"
module purge
module load "LUMI/${LUMI_STACK}" partition/G
module load "${PYTORCH_MODULE}"
module load lumi-CrayPath

if python -c "import lattice_lab.protein.encoder, esm, transformers; print('squashfs import OK')"; then
  if [[ "${REMOVE_USER_SOFTWARE}" == "1" ]]; then
    log "Verified. Removing expanded user-software/ to reclaim inodes."
    rm -r "${CONTAINERROOT}/user-software"
  else
    log "Verified. Keeping user-software/ (REMOVE_USER_SOFTWARE=0)."
  fi
else
  warn "import from squashfs FAILED — keeping user-software/ so nothing is lost."
  warn "Inspect the squashfs / re-run; do NOT delete ${CONTAINERROOT}/user-software."
  exit 1
fi

# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #
log "Container setup complete."
cat <<EOF

  lattice_lab + esm are now baked into the container overlay at:
    ${CONTAINERROOT}/user-software.squashfs

  In any new session / SLURM job (scripts/slurm/common.sh already does this):
    export EBU_USER_PREFIX=${EBU_USER_PREFIX}
    module load LUMI/${LUMI_STACK} partition/G
    module load ${PYTORCH_MODULE}
    module load lumi-CrayPath

  Run the ESM C precompute from the repo root:
    cd ${REPO_ROOT}
    python -m lattice_lab.protein.precompute --backend esmc \\
      --fasta artifacts/preprocessing/processed/bindingdb/bindingdb_targets.fasta \\
      --store artifacts/protein_store/embeddings/esmc_600m \\
      --device cuda --batch-size 8

  Code edits under src/ are picked up live (editable install). Re-run this
  script only when dependencies or packaging change.
EOF
