#!/usr/bin/env bash
# Stage 0 — download + stage the LIT-PCBA benchmark into artifacts/raw/lit_pcba/.
#
# Primary path uses the huggingface_hub client (CDN + automatic retry/resume),
# which is far more robust than raw curl against HF rate-limiting (HTTP 429) —
# common from shared cluster egress IPs (e.g. LUMI login nodes). Export HF_TOKEN
# to raise the anonymous rate limit. Falls back to curl --retry if the client
# isn't available.
#
# Idempotent: skips the download if the zip is already present.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
DL="${REPO}/artifacts/downloads"
ZIP="${DL}/LIT-PCBA.zip"
REPO_ID="THU-ATOM/DrugCLIP_data"
URL="https://huggingface.co/datasets/${REPO_ID}/resolve/main/LIT-PCBA.zip"
mkdir -p "$DL"

if [[ -s "$ZIP" ]]; then
    echo "[skip] $ZIP already present ($(du -h "$ZIP" | awk '{print $1}'))"
elif python -c "import huggingface_hub" 2>/dev/null; then
    echo "[download] ${REPO_ID}:LIT-PCBA.zip -> $ZIP (huggingface_hub, with retry/resume)"
    python - "$REPO_ID" "$DL" <<'PY'
import sys
from huggingface_hub import hf_hub_download
path = hf_hub_download(sys.argv[1], "LIT-PCBA.zip", repo_type="dataset", local_dir=sys.argv[2])
print("[done]", path)
PY
else
    echo "[download] $URL -> $ZIP (curl fallback, --retry on 429)"
    curl -fL -C - --retry 8 --retry-delay 15 --retry-all-errors -o "$ZIP" "$URL"
fi

echo "[unzip] $ZIP -> $DL"
unzip -q -o "$ZIP" -d "$DL"

# Stage into artifacts/raw/lit_pcba/ via the existing copy step.
LIT_PCBA_SRC="${DL}/lit_pcba" bash "${SCRIPT_DIR}/copy_lit_pcba.sh"
