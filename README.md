# lattice_lab

A **self-contained** Hydra + Lightning implementation of LATTICE (Latent
Affinity Training with Target-Informed Contrastive Estimation). This package
carries its own copy of every stage — preprocessing, FragMol backbone + adapter,
ESM-2 protein store, the energy head + losses, evaluation and inference — plus a
clean Hydra/Lightning orchestration layer.

## What changed vs. the original `lattice` training stack

The science kernels were re-homed unchanged (same numerics, byte-for-byte). The
*orchestration* was rewritten from the "Lightning-in-name-only" monoliths:

| Concern             | Old (`lattice.training`)                      | New (`lattice_lab`)                                            |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------- |
| Config wiring       | `train_cli` flatten/dataclass/`path_fields`   | Hydra structured configs + `hydra.utils.instantiate`          |
| Data                | procedural inside `train()`                   | `LightningDataModule` (`data/ebm.py`, `data/adapter.py`)      |
| Train/val loop      | `_EBMModule` + a callback that hand-validated | `training_step` / `validation_step` / `on_validation_epoch_end` |
| Checkpoints         | bespoke `ebm_best_{ef1,top1,bedroc}.pt`       | `ModelCheckpoint(monitor="val/ef1", mode="max")` + `save_last` |
| LR logging          | `current_lr()` + manual log                   | `LearningRateMonitor`                                          |
| W&B                 | `RunLogger` wrapper                           | `WandbLogger`                                                  |
| Grad clip           | manual `clip_grad_norm_` in a callback        | `trainer.gradient_clip_val`                                    |
| Stage-2 freeze gate | inline block in `train_adapter.train`         | `callbacks/sanity_gate.py` (`on_fit_end`)                     |

The three old monoliths (`train_cli.py`, `train_ebm.py`, `train_adapter.py`)
were **not** carried over; their reusable helpers were extracted into small,
single-purpose modules (`models/builders.py`, `models/schedules.py`,
`models/encode.py`, `data/cluster_sampler.py`).

## Layout

```
lattice_lab/                      # repo root
├── pyproject.toml                # install:  pip install -e .
├── README.md
├── scripts/                      # data-download CLIs + DATASETS.md
├── tests/                        # import + config-resolution + parity tests
├── artifacts/                    # all pipeline data (git-ignored; tree below)
└── src/lattice_lab/              # the importable package
    ├── train.py / evaluate.py    # Hydra @main entrypoints
    ├── export_adapter.py         # Lightning .ckpt → legacy adapter_v1.pt
    ├── configs/                  # Hydra config tree (see below)
    ├── data/ models/ callbacks/ utils/    # Lightning/Hydra orchestration
    └── preprocessing/ backbone/ protein/ ebm/ eval/ inference/ training/
                                   #    re-homed science kernels (no `import lattice`)
```

All pipeline inputs/outputs live under one git-ignored `artifacts/` tree
(meaningful names, no numeric stage prefixes):

```
artifacts/
├── raw/            # downloads: moses.csv, qm9.csv, bindingdb/, lit_pcba/, dude/
├── processed/      # Stage 1: bindingdb/threshold_90/*.parquet, moses/shard_*.parquet, *.fasta
├── adapter/        # Stage 2: adapter checkpoints (adapter_v1.pt)
├── protein_store/  # Stage 3: frozen ESM-2 embedding store
├── decoys/         # Stage 4: decoy + BindingDB z_m pools
├── energy/         # Stage 5: EBM head checkpoints
├── evaluation/     # Stage 6: LIT-PCBA caches, results, violins
└── predictions/    # Stage 7: inference outputs
```

```
configs/
├── train.yaml  eval.yaml           # entrypoint defaults + shared dims
├── experiment/                     # one file = one runnable setup (override layer)
│   ├── ebm_baseline.yaml  ebm_hardneg.yaml
│   └── adapter_ssl.yaml   adapter_fp.yaml
├── data/ {ebm,adapter}.yaml    model/ {ebm,adapter}.yaml
├── trainer/ {ebm,adapter,smoke}.yaml
├── callbacks/ {ebm,adapter}.yaml  logger/ {wandb,csv,none}.yaml
└── paths/ default.yaml
```

`n_decoys`, `d_adapter`, `d_protein` live at the top of `train.yaml` and are
interpolated into **both** `data` and `model` so a collator and its consumer
can't silently disagree. `model.num_steps = ${trainer.max_steps}` and
`model.hard_mining_mult = ${data.hard_mining_mult}` for the same reason.

## Install

The package lives under `src/`, so it must be installed before
`python -m lattice_lab.*` (or the `lattice-train` / `lattice-eval` scripts) work.
From the repo root:

```bash
pip install -e .          # editable: code changes take effect without reinstall
# (add --no-build-isolation on an offline cluster where setuptools is preinstalled)
```

Verify:

```bash
python -c "import lattice_lab; print(lattice_lab.__file__)"
python -m lattice_lab.train --help
```

> Not installing it (or relying on the old "run from inside the folder") gives
> `ModuleNotFoundError: No module named 'lattice_lab'` — `src/` is intentionally
> **not** on `sys.path` until you `pip install -e .`. If you can't install,
> `export PYTHONPATH=$PWD/src:$PYTHONPATH` from the repo root is an equivalent
> stop-gap.

## Usage

Run the CLIs from the repo root so the relative `artifacts/` paths in
`configs/data/*.yaml` resolve (Hydra keeps the CWD via `hydra.job.chdir=false`).
All pipeline inputs/outputs live under `artifacts/`.

```bash
# Stage 5 — EBM head
python -m lattice_lab.train experiment=ebm_baseline
python -m lattice_lab.train experiment=ebm_hardneg

# Stage 2 — adapter SSL (+ optional fingerprint distillation)
python -m lattice_lab.train experiment=adapter_ssl
python -m lattice_lab.train experiment=adapter_fp

# Override any field on the CLI
python -m lattice_lab.train experiment=ebm_baseline model.learning_rate=1e-4 trainer.max_steps=5000

# End-to-end smoke (tiny, CPU)
python -m lattice_lab.train experiment=ebm_baseline trainer=smoke logger=none

# Evaluate a checkpoint (held-out EF/BEDROC via Trainer.validate)
python -m lattice_lab.evaluate ckpt_path=logs/train/<run>/checkpoints/last.ckpt
```

## End-to-end pipeline

Every stage lives **inside this package** (no `import lattice`). The data-prep,
encoding and eval stages are argparse CLIs (one-shot data jobs); training is
Hydra/Lightning. **Run `pip install -e .` first** (see [Install](#install)), then
run from the repo root; all stage data lives under `artifacts/`. Paths below match
the released `ssl2` artifact set.

### Stage 0 — Data acquisition

Download scripts live in `scripts/` and write into `artifacts/raw/`. See
`scripts/DATASETS.md` for dataset details, sizes, and licences.

```bash
# MOSES — molecules for adapter self-supervision        → artifacts/raw/moses.csv
bash scripts/download_moses.sh

# BindingDB-All — full monthly dump (~3.2M measurements) → artifacts/raw/bindingdb/
# Pick the latest YYYYMM release from
# https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp
# download_bindingdb.sh is idempotent: if BindingDB_All.tsv already exists it
# will NOT re-fetch. Delete stale files before refreshing:
#   rm -f artifacts/raw/bindingdb/BindingDB_All.tsv artifacts/raw/bindingdb/BindingDB_All_*.tsv.zip
BINDINGDB_DATE=202606 bash scripts/download_bindingdb.sh

# Sanity-check before Stage 1 (a truncated TSV silently poisons every later stage):
wc -l artifacts/raw/bindingdb/BindingDB_All.tsv          # ~3.1–3.2M lines incl. header
ls -lh artifacts/raw/bindingdb/BindingDB_All_*.tsv.zip   # zip ~550–570 MB; TSV ~3 GB

# LIT-PCBA held-out benchmark                            → artifacts/raw/lit_pcba/
# Downloads via huggingface_hub (CDN + retry/resume), unzips, and stages it.
# Robust to HF rate-limiting (HTTP 429), common from shared cluster IPs — export
# HF_TOKEN to raise the anonymous limit if it still throttles.
bash scripts/download_lit_pcba.sh

# DUD-E benchmark (optional secondary eval)              → artifacts/raw/dude/
# 102 targets from dude.docking.org; override the mirror with DUDE_BASE_URL=<url>.
bash scripts/download_dude.sh
```

### MMseqs2 (required for Stage 1)

Stage 1 needs MMseqs2 for the identity split + protein clustering. The repo
**auto-discovers a bundled binary** at `software/mmseqs/bin/mmseqs` (it's
prepended to `PATH` automatically — no manual export needed); `LATTICE_MMSEQS_DIR`
overrides the location. Otherwise it falls back to whatever `mmseqs` is on
`PATH`. On a normal box: `conda install -c bioconda mmseqs2`.

**On LUMI** there is no MMseqs2 module and it isn't pip-installable — drop in the
static AVX2 binary (the EPYC nodes are AVX2, **not** AVX512):
```bash
mkdir -p software && cd software
curl -LO https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz   # wget on a login node
tar xzf mmseqs-linux-avx2.tar.gz
./mmseqs/bin/mmseqs version                                   # smoke-check
# Put on PATH in ~/.zshrc AND in every sbatch script (after `module load`):
export PATH=/pfs/lustrep4/scratch/project_465003063/benno/software:$PATH
```
It runs as a plain binary outside the PyTorch container, so it's independent of
the squashfs env. (Fallback URL: the `mmseqs-linux-avx2.tar.gz` asset on
github.com/soedinglab/MMseqs2/releases.) See `setup_lumi.md` for the full LUMI
environment setup.

### FragMol backbone (required for Stage 1+)

The frozen FragMol backbone is a large (~1 GB) external dependency and is **not**
shipped in this repo (`software/` is git-ignored). Stage 1 fragmolization needs
its `utils.fragments` module; Stage 2+ needs its weights. Place it at
`software/FragMol/` (so `software/FragMol/utils/fragments.py` exists), or point
elsewhere without copying:

```bash
export LATTICE_FRAGMOL_DIR=/shared/path/to/FragMol
```

If it's missing you'll get a clear `FileNotFoundError: FragMol backbone not
found at …` rather than a cryptic `No module named 'utils'`.

### Discrete-flow backbone (optional, alternative to FragMol)

`backbone/discrete_flow.py` is an alternative encoder (a DDiT discrete-flow
SMILES model). The DDiT architecture is **vendored** in
`backbone/ddit/` (no `in_virtuo_gen` dependency), so it needs only:

1. a **checkpoint** (weights) — e.g. `artifacts/checkpoints/invirtuo_gen.ckpt`;
2. a **tokenizer** JSON — e.g. `artifacts/tokenizer/smiles_new.json` (tracked in git).

`backbone/ddit/model_ddit.py` does `from . import rotary`, so `backbone/ddit/rotary.py`
(providing `Rotary` and `apply_rotary_emb_torch`) must be present — it's part of
the vendored architecture.

The encode-time fed to the backbone can be a **learnable parameter**
(`DiscreteFlowConfig.learnable_time=True`): it's sigmoid-bounded to (0, 1) and the
SSL gradient reaches it through the frozen backbone.

### Stage 1 — Preprocessing
If you are on LUMI - use this:
```bash
  # In your sbatch script / before running preprocessing:
  export PROJ=project_465003063
  export FLASH=/flash/$PROJ/$USER
  mkdir -p $FLASH/tmp $FLASH/artifacts
  export TMPDIR=$FLASH/tmp
  python -m lattice_lab.preprocessing.run_bindingdb \
    --bindingdb-tsv artifacts/raw/bindingdb/BindingDB_All.tsv \
    --lit-pcba-dir  artifacts/raw/lit_pcba \
    --output-dir     $FLASH/artifacts/processed/bindingdb \
    --identity 90 --n-jobs 16

# Canonicalise MOSES (adapter SSL set) into FragMol-view parquet shards.
python -m lattice_lab.preprocessing.run_preprocessing \
    --input  artifacts/raw/moses.csv \
    --output  $FLASH/artifacts/processed/moses \
    --n-views 3 --n-jobs 16

cp -r $FLASH/artifacts/processed/* /scratch/$PROJ/benno/lattice_lab/artifacts/processed/
```

```bash
# Curate BindingDB + build the 90% MMseqs2 identity split held out vs LIT-PCBA.
# Needs MMseqs2 (hard dep). LATTICE_ALLOW_KMER_FALLBACK=1 only for smoke/tests.
python -m lattice_lab.preprocessing.run_bindingdb \
    --bindingdb-tsv artifacts/raw/bindingdb/BindingDB_All.tsv \
    --lit-pcba-dir  artifacts/raw/lit_pcba \
    --output-dir    artifacts/processed/bindingdb \
    --identity 90 --n-jobs 16

# Canonicalise MOSES (adapter SSL set) into FragMol-view parquet shards.
python -m lattice_lab.preprocessing.run_preprocessing \
    --input  artifacts/raw/moses.csv \
    --output artifacts/processed/moses \
    --n-views 3 --n-jobs 16
```

### Stage 2 — Molecule encoder (FragMol + adapter SSL)
```bash
python -m lattice_lab.train experiment=adapter_fp \
    data.shard_dir=artifacts/processed/moses \
    data.use_fp=true model.fp_weight=2.0 \
    trainer.max_epochs=10 data.batch_size=512 \
    callbacks.model_checkpoint.dirpath=artifacts/adapter/checkpoints_ssl2

# Convert the best Lightning .ckpt → legacy adapter_v1.pt that the next stages load:
python -m lattice_lab.export_adapter \
    --ckpt   artifacts/adapter/checkpoints_ssl2/last.ckpt \
    --output artifacts/adapter/checkpoints_ssl2/adapter_v1.pt
```

### Stage 3 — Protein encoder (frozen ESM-2 650M)
```bash
python -m lattice_lab.protein.precompute \
    --fasta artifacts/processed/bindingdb/bindingdb_targets.fasta \
    --store artifacts/protein_store/embeddings/esm2_650M --model-name esm2_t33_650M_UR50D
python -m lattice_lab.protein.precompute \
    --fasta artifacts/processed/bindingdb/lit_pcba_targets.fasta \
    --store artifacts/protein_store/embeddings/esm2_650M --model-name esm2_t33_650M_UR50D
```

### Stage 4 — Decoy `z_m` pools (run BOTH before Stage 5)
```bash
python -m lattice_lab.ebm.precompute_decoys \
    --shard-dir artifacts/processed/moses \
    --adapter-ckpt artifacts/adapter/checkpoints_ssl2/adapter_v1.pt \
    --store artifacts/decoys/decoy_zm_ssl2 --batch-size 512
python -m lattice_lab.ebm.precompute_bdb_zm \
    --bdb-parquet artifacts/processed/bindingdb/threshold_90/train.parquet \
    --adapter-ckpt artifacts/adapter/checkpoints_ssl2/adapter_v1.pt \
    --store artifacts/decoys/bdb_zm_ssl2 --batch-size 512
```

### Stage 5 — Energy-head training (3 seeds)
```bash
for S in 0 1 2; do
  python -m lattice_lab.train experiment=ebm_hardneg \
    model.head_arch=film trainer.max_steps=12000 seed=$S \
    callbacks.model_checkpoint.dirpath=artifacts/energy/exp_hardneg_seed$S/checkpoints
done
```
`ebm_hardneg` sets `data.hard_mining_mult=3` + the 0.4/0.15 hard-neg mix.
`ModelCheckpoint(monitor="val/ef1", mode="max")` keeps the best head per seed.

### Stage 6 — Evaluation (LIT-PCBA)
```bash
# 1. Build the 4-view LIT-PCBA z_m cache.
python -m lattice_lab.eval.build_multiview_cache \
    --n-views 4 --zm-cache artifacts/evaluation/lit_pcba_zm_mv4 \
    --adapter-ckpt artifacts/adapter/checkpoints_ssl2/adapter_v1.pt --n-jobs 4

# 2. Score the 3-seed ensemble (the reported result).
python -m lattice_lab.eval.ensemble_eval \
    --ckpts artifacts/energy/exp_hardneg_seed{0,1,2}/checkpoints/last.ckpt \
    --zm-cache artifacts/evaluation/lit_pcba_zm_mv4 \
    --protein-store artifacts/protein_store/embeddings/esm2_650M \
    --test-parquet artifacts/processed/bindingdb/test_lit_pcba.parquet \
    --out artifacts/evaluation/ensemble_hardneg_mv4.json --n-jobs 32

# Held-out (in-distribution) ranking metrics for a single checkpoint:
python -m lattice_lab.evaluate ckpt_path=artifacts/energy/exp_hardneg_seed0/checkpoints/last.ckpt
```

### Stage 7 — Inference / virtual screening
```bash
python -m lattice_lab.inference.predict_ensemble \
    --head-ckpts artifacts/energy/exp_hardneg_seed{0,1,2}/checkpoints/last.ckpt \
    --adapter-ckpt artifacts/adapter/checkpoints_ssl2/adapter_v1.pt \
    --target-fasta thrb.fasta --target-name THRB \
    --smiles-file my_library.csv --n-views 4 \
    --output-csv artifacts/predictions/thrb_predictions.csv
```

> **Note on checkpoint formats.** The precompute / inference CLIs and
> `ensemble_eval` expect the energy-head state under the keys produced by the
> original trainers. Lightning `.ckpt` files store params under `state_dict`
> with `head.*` / `encoder.adapter.*` prefixes; `export_adapter.py` handles the
> adapter case. If you load a `lattice_lab`-trained EBM `.ckpt` into the legacy
> `ensemble_eval`/`predict_ensemble`, strip the `head.` prefix the same way (a
> 3-line `torch.load` + dict-comprehension), or open an issue and I'll add an
> `export_head.py` mirror.

## Not re-homed (intentionally)

The `artifacts/energy/` ablation tooling (`run_ablations.py`, `collect_ablation_results.py`,
the reproduce/sweep shell scripts) is monorepo experiment-management, not part of
the model pipeline — it stays in the monorepo.

## Spinning out into its own repo

Move this directory to be the new repo root. `pyproject.toml` maps the package
`lattice_lab` onto `.` (`package-dir = {"lattice_lab" = "."}`), so
`pip install -e .` works unchanged and `lattice-train` / `lattice-eval` console
scripts become available.

## Tests

```bash
python -m pytest lattice_lab/tests -q
```

`test_imports.py` imports every module (orchestration **and** re-homed kernels);
`test_configs.py` composes **and fully resolves** every config (catching
interpolation typos and bad `_target_` paths) and asserts the shared dims stay
in sync across `data`/`model`.
