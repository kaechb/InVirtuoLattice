#!/usr/bin/env bash
# Stage 0 — fetch the DUD-E benchmark into artifacts/preprocessing/raw/dude/.
#
# DUD-E (Mysinger et al., J. Med. Chem. 2012) is a 102-target virtual-screening
# benchmark: experimentally-confirmed actives plus property-matched, topologically
# dissimilar decoys (~50 decoys per active) drawn from ZINC. DrugCLIP and most
# structure-free screeners report EF@1% / BEDROC on it zero-shot.
#
# Per-target tarballs are served at:
#     http://dude.docking.org/targets/<TARGET>/<TARGET>.tar.gz
# each expanding to a <TARGET>/ folder containing (we keep the three we use):
#     actives_final.ism     SMILES  <space>  internal_id  [<space> ChEMBL_id]
#     decoys_final.ism      SMILES  <space>  ZINC-style decoy_id
#     receptor.pdb          one representative receptor structure / target
#
# We download only those three files per target (the .ism ligand lists + the
# receptor PDB we parse for the sequence in lattice/preprocessing/dude.py); the
# docked-pose mol2s and feature caches in the full tarball are not used here.
#
# Override via env vars:
#     DUDE_BASE_URL=<url>   point at a mirror (default http://dude.docking.org/targets)
#     DUDE_TARGETS="a b c"  download a subset instead of all 102
#
# Idempotent: a target whose actives_final.ism is already present is skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${REPO}/artifacts/preprocessing/raw/dude"
BASE_URL="${DUDE_BASE_URL:-http://dude.docking.org/targets}"

# Canonical 102-target DUD-E set (the same names DrugCLIP reports).
DEFAULT_TARGETS="aa2ar abl1 ace aces ada ada17 adrb1 adrb2 akt1 akt2 aldr ampc \
andr aofb bace1 braf cah2 casp3 cdk2 comt cp2c9 cp3a4 csf1r cxcr4 def dhi1 dpp4 \
drd3 dyr egfr esr1 esr2 fa10 fa7 fabp4 fak1 fgfr1 fkb1a fnta fpps gcr glcm gria2 \
grik1 hdac2 hdac8 hivint hivpr hivrt hmdh hs90a hxk4 igf1r inha ital jak2 kif11 \
kit kith kpcb lck lkha4 mapk2 mcr met mk01 mk10 mk14 mmp13 mp2k1 nos1 nram pa2ga \
parp1 pde5a pgh1 pgh2 plk1 pnph ppara ppard pparg prgr ptn1 pur2 pygm pyrd reni \
rock1 rxra sahh src tgfr1 thb thrb try1 tryb1 tysy urok vgfr2 wee1 xiap"
TARGETS="${DUDE_TARGETS:-$DEFAULT_TARGETS}"

mkdir -p "$DEST"

n_done=0
n_skip=0
for name in $TARGETS; do
    out="$DEST/$name"
    if [[ -s "$out/actives_final.ism" ]]; then
        echo "[skip] $name (already present)"
        n_skip=$((n_skip + 1))
        continue
    fi
    tgz="$DEST/$name.tar.gz"
    url="$BASE_URL/$name/$name.tar.gz"
    echo "[download] $url"
    if ! curl -fL --max-time 600 -o "$tgz" "$url"; then
        echo "[error] download failed for $name ($url)" >&2
        echo "        DUD-E can be slow/offline; retry later or set DUDE_BASE_URL=<mirror>." >&2
        rm -f "$tgz"
        exit 1
    fi
    # Extract only the three files we consume; tolerate a leading ./ in the archive.
    mkdir -p "$out"
    tar -xzf "$tgz" -C "$DEST" \
        --wildcards "*/actives_final.ism" "*/decoys_final.ism" "*/receptor.pdb"
    rm -f "$tgz"
    if [[ ! -s "$out/actives_final.ism" ]]; then
        echo "[error] $name: expected files missing after extraction" >&2
        exit 1
    fi
    echo "[done] $name"
    n_done=$((n_done + 1))
done

echo "[summary] $n_done downloaded, $n_skip already present -> $DEST"
