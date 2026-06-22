#!/usr/bin/env bash
# Stage 0 — copy the LIT-PCBA test set into artifacts/preprocessing/raw/lit_pcba/.
#
# Source layout (per-target subfolder produced by the 23AIBox-CSCo-DTA repo):
#     <SRC>/<TARGET>/actives.smi          SMILES + PubChem CID, whitespace-separated
#     <SRC>/<TARGET>/inactives.smi        same format
#     <SRC>/<TARGET>/<pdb>_protein.mol2   one representative complex per target
#     <SRC>/<TARGET>/<pdb>_ligand.mol2
#
# We skip _feat_cache/ (3D-coord feature cache from another pipeline, not used here).
# Idempotent: re-running with existing destination is a no-op.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${REPO}/artifacts/preprocessing/raw/lit_pcba"
SRC="${LIT_PCBA_SRC:-/home/euser/Gianvito/23AIBox-CSCo-DTA-main/data/source/lit_pcba}"

if [[ ! -d "$SRC" ]]; then
    echo "[error] LIT-PCBA source not found: $SRC" >&2
    echo "        Override with LIT_PCBA_SRC=<path> $0" >&2
    exit 1
fi

mkdir -p "$DEST"

shopt -s nullglob
n_targets=0
for tdir in "$SRC"/*/; do
    name="$(basename "$tdir")"
    [[ "$name" == "_feat_cache" ]] && continue
    out="$DEST/$name"
    if [[ -d "$out" && -s "$out/actives.smi" ]]; then
        echo "[skip] $name (already present)"
    else
        mkdir -p "$out"
        cp -f "$tdir"/*.smi "$out/" 2>/dev/null || true
        cp -f "$tdir"/*_protein.mol2 "$out/" 2>/dev/null || true
        echo "[copy] $name"
    fi
    n_targets=$((n_targets + 1))
done

echo "[done] $n_targets targets in $DEST"
