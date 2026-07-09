#!/usr/bin/env bash
# ponytail: bash -n on pipeline scripts + smoke helper dry-run (no sbatch/GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/slurm/common.sh

for f in scripts/slurm/run_pipeline.sh scripts/slurm/stage{2,3,4,5,6}_*.sh scripts/slurm/common.sh; do
  bash -n "$f"
done

TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"' EXIT
if command -v python >/dev/null 2>&1; then
  lattice_make_smoke_parquets "${TMP}" "${REPO}/artifacts/preprocessing/processed/bindingdb/threshold_90" 100 2
  test -f "${TMP}/train.parquet"
  test -f "${TMP}/val.parquet"
  test -f "${TMP}/test_lit_pcba.parquet"
else
  echo "skip parquet smoke (no python in PATH)"
fi

# freeze_git: records HEAD sha + dirty patch + porcelain status and warns on a
# dirty tree (records tracked edits; leaves untracked file contents uncaptured).
if command -v git >/dev/null 2>&1; then
  (
    G=$(mktemp -d "${TMP}/gitXXXX"); cd "${G}"
    git init -q && git config user.email t@t && git config user.name t
    echo base > f.py && git add f.py && git commit -qm init
    echo edit >> f.py       # tracked dirty change
    echo new > untracked.py # untracked file (contents not captured)
    export REPO="${G}"
    DST="${G}/run"; mkdir -p "${DST}"
    _WARN=""; lattice_job_banner() { _WARN="$*"; }
    lattice_pipeline_freeze_git "${DST}"
    test "$(cat "${DST}/git.sha")" = "$(git rev-parse HEAD)"
    grep -q 'f.py' "${DST}/git.diff"
    grep -q '?? untracked.py' "${DST}/git.status"
    test -n "${_WARN}"
  )
  echo "freeze_git OK"
else
  echo "skip freeze_git (no git in PATH)"
fi

# zm_consistency.row_cosine: identical rows -> 1, negated rows -> -1.
if command -v python >/dev/null 2>&1; then
  python - <<'PY'
import numpy as np
from lattice_lab.eval.zm_consistency import row_cosine
a = np.random.RandomState(0).randn(5, 8)
assert abs(row_cosine(a, a).min() - 1.0) < 1e-5
assert abs(row_cosine(a, -a).max() + 1.0) < 1e-5
print("row_cosine OK")
PY
else
  echo "skip row_cosine (no python in PATH)"
fi
echo "smoke pipeline scripts OK"
