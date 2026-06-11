#!/usr/bin/env bash
# Fetch the MOSES v1 benchmark (1.9M drug-like SMILES, ZINC subset).
# Idempotent: re-running with an existing file skips the download.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${REPO}/artifacts/raw"
DEST="${RAW_DIR}/moses.csv"
URL="https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"

mkdir -p "$RAW_DIR"

if [[ -s "$DEST" ]]; then
    echo "[skip] $DEST already exists ($(wc -l <"$DEST") lines)"
    exit 0
fi

echo "[download] $URL -> $DEST"
curl -sL --max-time 600 -o "$DEST" "$URL"
echo "[done] $(wc -l <"$DEST") lines, $(du -h "$DEST" | awk '{print $1}')"
