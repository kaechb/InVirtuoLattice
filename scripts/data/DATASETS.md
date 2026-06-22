# Datasets

Raw datasets and download scripts. Files in `artifacts/preprocessing/raw/` are never modified
— Stage 1 reads from there and writes processed shards into
`artifacts/preprocessing/processed/`.

## Currently available

| Path                         | Source                                                      | Rows  | Size  |
|------------------------------|-------------------------------------------------------------|-------|-------|
| `raw/moses.csv`              | MOSES benchmark v1 (ZINC drug-like subset, scaffold split)  | 1.9M  | ~81MB |
| `raw/qm9.csv`                | QM9 (Stage-2 sanity check)                                  | 134K  | ~30MB |
| `raw/bindingdb/BindingDB_All.tsv` | BindingDB-All full monthly TSV (Gilson et al., NAR 2025) | ~3.2M measurements | TSV ~3 GB; zip ~550–570 MB |
| `raw/lit_pcba/<TARGET>/`     | LIT-PCBA actives/inactives + reference protein mol2 (15 targets) | ~2.8M | ~2GB |

The MOSES file has columns `SMILES, SPLIT` (train/test/test_scaffolds). Stage 1
reads only the first column.

## Reproduce

```bash
# Stage 0a — MOSES + QM9 (Stage 1/2)
bash scripts/download_moses.sh
bash scripts/download_qm9.sh

# Stage 0b — BindingDB-All TSV (Stage 1 → 4/5 training).
# Default in download_bindingdb.sh = latest known YYYYMM release. Override with
# any monthly zip listed at
# https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp
#
# Idempotent: skips download if BindingDB_All.tsv already exists. To refresh,
# remove the old files first, then re-download:
#   rm -f artifacts/preprocessing/raw/bindingdb/BindingDB_All.tsv artifacts/preprocessing/raw/bindingdb/BindingDB_All_*.tsv.zip
BINDINGDB_DATE=202606 bash scripts/download_bindingdb.sh

# Sanity-check (must pass before Stage 1):
wc -l artifacts/preprocessing/raw/bindingdb/BindingDB_All.tsv          # ~3.1–3.2M lines incl. header
ls -lh artifacts/preprocessing/raw/bindingdb/BindingDB_All_*.tsv.zip # zip ~550–570 MB, not ~130 MB

# Stage 0c — LIT-PCBA (Stage 6 held-out benchmark).
# Downloads via huggingface_hub (CDN + retry/resume; robust to HTTP 429),
# unzips, and stages into artifacts/preprocessing/raw/lit_pcba/. Set HF_TOKEN if throttled.
bash scripts/download_lit_pcba.sh
# The _feat_cache subfolder is intentionally skipped — 3D-coord cache from another pipeline.
```

## TODO (not yet integrated)

| Dataset       | Purpose                          | Source                              |
|---------------|----------------------------------|-------------------------------------|
| ChEMBL 34     | Curated binders for EBM training | chembl.eu (FTP)                     |
| PDBbind v2020 | High-quality binder anchors      | pdbbind.org.cn                      |
| UniProt FASTA | Protein sequences for ESM-2      | uniprot.org                         |
