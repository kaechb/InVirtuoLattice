#!/usr/bin/env bash
# Stage 0 — fetch the BindingDB-All TSV (curated bioactivity database, ~2M rows).
#
# Source: Gilson et al., BindingDB-Curated. Nucleic Acids Res. 53, D1633 (2025).
#         https://academic.oup.com/nar/article/53/D1/D1633/7906836
#
# We download the canonical "BindingDB_All" monthly TSV release. The columns of
# interest are:
#   Ligand SMILES, BindingDB MonomerID, Target Name,
#   Ki / IC50 / Kd / EC50 (nM),  Number of Protein Chains in Target,
#   BindingDB Target Chain Sequence,
#   UniProt (SwissProt) Primary ID of Target Chain.
#
# The curated subset described in the 2025 NAR paper is the BindingDB-All TSV
# *with* the in-database curation flags already applied — the same file the
# online BindingDB UI uses. Per-record QC happens in
# `lattice/preprocessing/bindingdb.py`.
#
# Override via env vars:
#     BINDINGDB_URL=<url>   point at a specific monthly release
#     BINDINGDB_DATE=YYYYMM (e.g. 202605) — picks that monthly release. The
#                           BindingDB filename pattern is YYYYMM (no 'm' separator).
#
# Find current releases at:
#     https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp
#
# Idempotent: re-running with an existing TSV is a no-op.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${REPO}/artifacts/preprocessing/raw/bindingdb"
DEFAULT_DATE="202606"
DATE="${BINDINGDB_DATE:-$DEFAULT_DATE}"
DEFAULT_URL="https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_${DATE}_tsv.zip"
URL="${BINDINGDB_URL:-$DEFAULT_URL}"

ZIP_DEST="${RAW_DIR}/BindingDB_All_${DATE}.tsv.zip"
TSV_DEST="${RAW_DIR}/BindingDB_All.tsv"

mkdir -p "$RAW_DIR"

if [[ -s "$TSV_DEST" ]]; then
    n=$(wc -l <"$TSV_DEST")
    echo "[skip] $TSV_DEST already exists ($n lines)"
    exit 0
fi

if [[ ! -s "$ZIP_DEST" ]]; then
    echo "[download] $URL"
    if ! curl -fL --max-time 1800 -o "$ZIP_DEST" "$URL"; then
        echo "[error] download failed. Pick a valid release date and re-run:" >&2
        echo "        BINDINGDB_DATE=202604 bash $0   # YYYYMM format (no 'm')" >&2
        echo "        See https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp" >&2
        rm -f "$ZIP_DEST"
        exit 1
    fi
fi

echo "[unzip] $ZIP_DEST -> $RAW_DIR"
unzip -o "$ZIP_DEST" -d "$RAW_DIR" >/dev/null

# Recent releases unpack as BindingDB_All.tsv; older monthly zips used a dated
# name (e.g. BindingDB_All_2026m04.tsv). Normalise to TSV_DEST in either case.
if [[ -s "$TSV_DEST" ]]; then
    : # already at the stable path
else
    extracted=$(find "$RAW_DIR" -maxdepth 1 -name "BindingDB_All*.tsv" ! -name "BindingDB_All.tsv" | head -n 1)
    if [[ -z "$extracted" ]]; then
        echo "[error] no BindingDB_All*.tsv found after unzip" >&2
        exit 1
    fi
    mv -f "$extracted" "$TSV_DEST"
fi

n=$(wc -l <"$TSV_DEST")
if (( n < 3_000_000 )); then
    echo "[warn] $TSV_DEST has only $n lines; expected ~3.1–3.2M for the full dump." >&2
    echo "[warn] Delete the zip/TSV and re-download from Download.jsp." >&2
fi

echo "[done] $n lines, $(du -h "$TSV_DEST" | awk '{print $1}')"
