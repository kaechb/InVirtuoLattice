#!/usr/bin/env bash
# Fetch QM9 (~134K molecules with computed quantum properties).
# Used by Stage 2's linear-probe sanity check on HOMO/LUMO.
# Idempotent: re-running with an existing file skips the download.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${REPO}/artifacts/preprocessing/raw"
DEST="${RAW_DIR}/qm9.csv"
URL="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

mkdir -p "$RAW_DIR"

if [[ -s "$DEST" ]]; then
    echo "[skip] $DEST already exists ($(wc -l <"$DEST") lines)"
    exit 0
fi

echo "[download] $URL -> $DEST"
curl -sL --max-time 600 -o "$DEST" "$URL"
echo "[done] $(wc -l <"$DEST") lines, $(du -h "$DEST" | awk '{print $1}')"
