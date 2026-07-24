# lattice_lab

A **self-contained** Hydra + Lightning implementation of LATTICE (Latent
Affinity Training with Target-Informed Contrastive Estimation). This package
carries its own copy of every stage — preprocessing, DDiT discrete-flow backbone + adapter,
ESM-2 protein store, the energy head + losses, evaluation and inference — plus a
clean Hydra/Lightning orchestration layer.

## What changed vs. the original `lattice` training stack

The science kernels were re-homed unchanged (same numerics, byte-for-byte). The
*orchestration* was rewritten from the "Lightning-in-name-only" monoliths:

| Concern             | Old (`lattice.training`)                      | New (`lattice_lab`)                                            |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------- |
| Config wiring       | `train_cli` flatten/dataclass/`path_fields`   | Hydra structured configs + `hydra.utils.instantiate`          |
| Data                | procedural inside `train()`                   | `LightningDataModule` (`data/ebm.py`, `data/fragment_views.py`) |
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

## Molecule fragment views — faithful representation & the `merge` variant

Every molecule (binder, decoy, MOSES SSL example) is encoded by the DDiT+adapter
from a **fragment view**: a space-separated string of BRICS fragments. Two
functions in `preprocessing/molecules.py` produce them, and the split is
load-bearing:

- **`fragment_view(smiles, *, merge=False)` — faithful, full coverage.** Used
  everywhere `z_m` must *be* the molecule: decoy/binder precompute (Stage 4), the
  EBM (Stage 5), eval (Stage 6). It cuts the molecule at BRICS bonds into a true
  partition via `BRICS.BreakBRICSBonds`, so every heavy atom is preserved.
  `merge=False` cuts *all* BRICS bonds (finest); `merge=True` cuts a random subset
  (coarser, still full coverage).
- **`augment_fragment_views(...)` — SSL augmentation only.** May *drop* fragments
  (I-JEPA-style masking) and vary granularity, then shuffle. Used only to build
  the MOSES SSL corpus, where the model should learn to be robust to perturbed
  views.

> **Correctness fix (why this split exists).** The earlier single
> `smiles_to_fragment_views` sampled a *random subset* of BRICS fragments
> (`rng.sample`) as the one stored view, so ~45% of molecules — binders **and**
> decoys — were encoded from an *incomplete* molecule. With online hard-negative
> mining this capped Stage-5 InfoNCE / EF well below the original `lattice` stack
> regardless of `d_adapter`. Two traps to avoid if you touch this: never use
> `BRICS.BRICSDecompose` for a faithful view (it returns a *deduplicated set of
> building blocks*, dropping repeated fragments — not a partition), and never bake
> `drop` into the stored view (the SSL datamodule applies mask/shuffle **online**
> per contrastive view, so nothing lossy belongs in the stores).

### The `merge` variant and how it stays consistent

`merge=True` is an alternative *coarser* granularity for the whole pipeline,
toggled by a single `MERGE` env var (default `MERGE=0` = finest faithful):

- **Stages 1–2 choose it** (`MERGE=1`): Stage 1 writes a parallel dataset under
  `*_merge` dirs (`moses_merge`, `bindingdb_merge`) and Stage 2 trains the adapter
  on it. *Stage 1 and Stage 2 must use the same `MERGE`* — Stage 2 reads the
  `moses${suffix}` shards Stage 1 produced.
- **Stage 2 records the choice in the adapter checkpoint** — `fragment_merge` is
  embedded by the SSL module's `on_save_checkpoint`.
- **Stages 4/5/6 read it back from the checkpoint** (`builders.merge_from_ckpt`,
  via `lattice_ckpt_merge_suffix`) and auto-select the matching
  `decoy_zm_merge` / `bdb_zm_merge` / `binder_zm_merge` stores — and for eval the
  flag rides into the EBM ckpt's `encoder_config`. A merge-trained adapter can
  never be paired with finest-partition stores; set `MERGE` wrong on a downstream
  stage and it's ignored, the checkpoint wins.

## Layout

```
lattice_lab/                      # repo root
├── pyproject.toml                # install:  pip install -e .
├── README.md
├── scripts/                      # data download CLIs + SLURM submit scripts
│   ├── data/                     # Stage 0 download helpers + DATASETS.md
│   └── slurm/                    # sbatch wrappers for Stages 1–7 (LUMI)
├── tests/                        # import + config-resolution tests
├── artifacts/                    # all pipeline data (git-ignored; tree below)
└── src/lattice_lab/              # the importable package
    ├── train.py / evaluate.py    # Hydra @main entrypoints
    ├── configs/                  # Hydra config tree (see below)
    ├── data/ models/ callbacks/ utils/    # Lightning/Hydra orchestration
    └── preprocessing/ backbone/ protein/ ebm/ eval/ inference/ training/
                                   #    re-homed science kernels (no `import lattice`)
```

All pipeline inputs/outputs live under one git-ignored `artifacts/` tree
(meaningful names, no numeric stage prefixes):

```
artifacts/
├── preprocessing/  # Stages 0–1
│   ├── raw/        # downloads: moses.csv, qm9.csv, bindingdb/, lit_pcba/, dude/
│   └── processed/  # bindingdb/threshold_90/*.parquet, moses/shard_*.parquet, *.fasta
├── adapter/        # Stage 2: checkpoints/<run_id>/last.ckpt (flat — no variant subdirs)
├── protein_store/  # Stage 3: frozen ESM-2 embedding store
├── decoys/         # Stage 4: <adapter_run_id>/{decoy_zm,bdb_zm} (binders → binders/<run_id>/)
├── energy/         # Stage 5: checkpoints/<run_id>/last.ckpt EBM heads (flat)
├── evaluation/     # Stage 6: <adapter_run_id>/lit_pcba_zm_* caches + results
└── predictions/    # Stage 7: inference outputs
```

```
configs/
├── train.yaml  eval.yaml           # entrypoint defaults + shared dims
├── experiment/                     # one file = one runnable setup (override layer)
│   ├── ebm_baseline.yaml  ebm_hardneg.yaml
│   └── adapter_discrete_flow.yaml
├── data/ {ebm,fragment_views}.yaml    model/ {ebm,discrete_flow}.yaml
├── trainer/ {ebm,adapter,smoke}.yaml
├── callbacks/ {ebm,ssl_basic}.yaml  logger/ {wandb,csv,none}.yaml
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

## Setting up on LUMI

LUMI runs PyTorch inside a read-only Singularity container, so the environment
setup (container install, overlay venv + SquashFS, quota-safe paths, the
`lumi-CrayPath` module ordering, and MMseqs2) is different from a normal machine.
It's scripted as a one-time step:

```bash
# On a LOGIN node (uan…), from the repo root.
cd ~/benno/lattice_lab
bash scripts/setup_container.sh
```

See **[setup_lumi.md](setup_lumi.md)** for the full walkthrough and the manual
steps. After that, `lattice_lab` and `esm` are importable in every job/session
with no per-run `pip install` (code edits under `src/` are still picked up live).

## Usage

Run the CLIs from the repo root so the relative `artifacts/` paths in
`configs/data/*.yaml` resolve (Hydra keeps the CWD via `hydra.job.chdir=false`).
All pipeline inputs/outputs live under `artifacts/`.

When W&B logging is enabled, every ``ModelCheckpoint`` writes under
``{dirpath}/{wandb_run_id}/`` (e.g. ``artifacts/adapter/checkpoints/73miv4j1/last.ckpt``).
The run id is printed in the W&B URL and in the training log line
``ModelCheckpoint dirpath → …``.

```bash
# Stage 5 — EBM head
python -m lattice_lab.train experiment=ebm_baseline
python -m lattice_lab.train experiment=ebm_hardneg

# Stage 2 — discrete-flow adapter SSL
python -m lattice_lab.train experiment=adapter_discrete_flow

# Override any field on the CLI
python -m lattice_lab.train experiment=ebm_baseline model.learning_rate=1e-4 trainer.max_steps=5000

# End-to-end smoke (tiny, CPU)
python -m lattice_lab.train experiment=ebm_baseline trainer=smoke logger=none

# Evaluate a checkpoint (held-out EF/BEDROC via Trainer.validate)
python -m lattice_lab.evaluate ckpt_path=logs/train/<run>/checkpoints/<run_id>/last.ckpt
```

## End-to-end pipeline

Every stage lives **inside this package** (no `import lattice`). The data-prep,
encoding and eval stages are argparse CLIs (one-shot data jobs); training is
Hydra/Lightning. **Run `pip install -e .` first** (see [Install](#install)), then
run from the repo root; all stage data lives under `artifacts/`. Paths below match
the released `ssl2` artifact set.

### Running on SLURM (LUMI)

`scripts/slurm/` has one sbatch wrapper per stage. They all `source common.sh`
(module loads, `$REPO`, GPU check, and self-healing `logs/slurm/stage<N>/` log
dirs). Submit from the repo root. Stages 2–6 take **run ids as arguments or env
vars** — you don't edit the script files.

| Stage | Script | Args / env | Output |
|---|---|---|---|
| 1 | `stage1_preprocess.sh` | `MERGE` (0/1) | BindingDB curation + 90% MMseqs2 split |
| 2 | `stage2_ssl.sh` | `METHOD`, `MERGE` (0/1), optional `RUN_NAME` | `artifacts/adapter/checkpoints/<run_id>/last.ckpt` |
| 3 | `stage3_protein_precompute.sh` | `PROTEIN` (`esm2`\|`esmc`), `OVERWRITE` | frozen ESM-2 / ESM C store |
| 4 | `stage4_precompute_decoys.sh` | `RUN_ID` (Stage-2 W&B id) | `artifacts/decoys/<run_id>/…`, `artifacts/binders/<run_id>/…` |
| 5 | `stage5_ebm_train.sh` | `METHOD`, `RUN_ID` (Stage-2); optional `--three-seeds` | `artifacts/energy/checkpoints/<run_id>/last.ckpt` |
| 6 | `stage6_eval.sh` | `RUN_ID` or three EBM ids; optional `--single-view` + `SSL_RUN_ID` | mv4 cache + CSV, ensemble JSON, or 1-view CSV |
| 7 | `stage7_predict_ensemble.sh` | edit paths + seed ids in script | virtual screening CSV |
| — | `run_pipeline.sh` | `METHOD`, `PROTEIN`, `N_SEEDS`, `MERGE`, optional `RUN_NAME` | submits stages 2→6 (same stage scripts) with SLURM dependencies |

**`MERGE` (default `0`).** Selects the fragment-view granularity (see *Molecule
fragment views* above). Set it only on **Stage 1 and Stage 2** (and `run_pipeline.sh`,
which threads it through); Stages 4–6 derive it from the adapter checkpoint, so you
never pass it there. `MERGE=1` reads/writes the parallel `*_merge` datasets and
`*_zm_merge` stores. The two variants coexist on disk, so you can build both.

**Run-id convention.** Stage 2 prints a W&B run id; that same id keys Stage-4
decoy pools and is passed to Stage 5 as `RUN_ID`. Stage 5 prints one W&B run id
per array task (one per seed); those EBM ids go to Stage 6.

```bash
# --- Stage 2: adapter SSL (pick one METHOD) ---
sbatch scripts/slurm/stage2_ssl.sh lejepa
sbatch scripts/slurm/stage2_ssl.sh ntxent
sbatch scripts/slurm/stage2_ssl.sh ijepa
sbatch scripts/slurm/stage2_ssl.sh ijepa my_ablation_name   # optional W&B run name
sbatch scripts/slurm/stage2_ssl.sh denoise
MERGE=1 sbatch scripts/slurm/stage1_preprocess.sh           # coarser variant: build it first…
MERGE=1 sbatch scripts/slurm/stage2_ssl.sh lejepa           # …then train on it (Stage 4–6 follow the ckpt)
# → note the W&B run id, e.g. nsw2w2z5

# --- Stage 4: z_m pools (any Stage-2 ckpt; encoder type auto-detected) ---
sbatch scripts/slurm/stage4_precompute_decoys.sh nsw2w2z5
# RUN_ID=nsw2w2z5 sbatch scripts/slurm/stage4_precompute_decoys.sh

# --- Stage 5: EBM training (seed 0 by default) ---
sbatch scripts/slurm/stage5_ebm_train.sh lejepa nsw2w2z5
sbatch scripts/slurm/stage5_ebm_train.sh ntxent 1kfmwoar
./scripts/slurm/stage5_ebm_train.sh --three-seeds lejepa nsw2w2z5   # seeds 0–2
sbatch --array=0-2 scripts/slurm/stage5_ebm_train.sh lejepa nsw2w2z5
# → note EBM run id(s), e.g. abc123 (one seed) or abc123, def456, ghi789 (three)

# --- Stage 6: LIT-PCBA (rebuilds mv4 cache every time, then scores) ---
sbatch scripts/slurm/stage6_eval.sh abc123                              # one seed
sbatch scripts/slurm/stage6_eval.sh abc123 def456 ghi789                # 3-seed ensemble
SSL_RUN_ID=nsw2w2z5 sbatch scripts/slurm/stage6_eval.sh --single-view abc123  # 1-view debug

# chain jobs:
sbatch --dependency=afterok:<stage4_jobid> scripts/slurm/stage5_ebm_train.sh lejepa nsw2w2z5
```

**Full pipeline (stages 2→6, dependency chain):** run from the login node — **not**
`sbatch`. `run_pipeline.sh` submits the same stage scripts with `PIPELINE_ENV` set;
stage 2 snapshots Hydra configs into that directory; stage 5 trains against the
snapshot so later repo edits cannot change EBM settings mid-pipeline.

**The bare defaults reproduce the ablation winner `w790kdrh`** — `METHOD=ntxent`,
a 4-layer adapter (`model.encoder.adapter_n_layers=4`, now the config default),
esm2 proteins, the hard-neg EBM (`ebm_hardneg_ntxent`) and a 3-seed ensemble.
So `./scripts/slurm/run_pipeline.sh` with no arguments trains and evaluates that
recipe end-to-end; pass `METHOD`/`N_SEEDS`/etc. to explore anything else.

`run_pipeline.sh` starts at Stage 2, so build the matching Stage-1 dataset first
(e.g. `MERGE=1 sbatch scripts/slurm/stage1_preprocess.sh`) and pass the same
`MERGE` to the pipeline; it seeds `MERGE` into `PIPELINE_ENV` and every stage
follows the adapter checkpoint from there.

```bash
./scripts/slurm/run_pipeline.sh                          # reproduce winner: ntxent, 4-layer adapter, 3-seed ensemble
N_SEEDS=1 ./scripts/slurm/run_pipeline.sh lejepa         # single-seed lejepa run
PROTEIN=esmc RUN_NAME=ijepa_ablation ./scripts/slurm/run_pipeline.sh ijepa
MULTISEED=1 ADAPTER_RUN_ID=avy80iqo ./scripts/slurm/run_pipeline.sh   # existing run: only stage 5 ×3 + stage 6
MERGE=1 ./scripts/slurm/run_pipeline.sh lejepa            # coarser fragment-view variant end-to-end
SMOKE=1 ./scripts/slurm/run_pipeline.sh                   # fast wiring test (~1h total)
```

**Smoke test (`SMOKE=1`):** exercises the full dependency chain with tiny data —
1 SSL epoch on 1% of train batches, 5k-row BDB/decoy pools, 50 EBM steps, 3
LIT-PCBA targets with 1-view cache. Sampled parquets are created at **stage 4**
(when python is available) under `logs/slurm/pipeline/<id>/smoke_data/`. Jobs get
`--time=01:00:00`. Tunables: `SMOKE_PRECOMPUTE_LIMIT`, `SMOKE_PARQUET_ROWS`,
`SMOKE_LITPCBA_TARGETS`.

```bash
SMOKE=1 ./scripts/slurm/run_pipeline.sh lejepa
SMOKE=1 SMOKE_LITPCBA_TARGETS=2 ./scripts/slurm/run_pipeline.sh ijepa smoke_ijepa
```

Dependency graph (stage 3 and 4 run in parallel after stage 2; stage 6 waits for
both stage 5 and stage 3):

```
stage2 ─┬─→ stage3 ─┐
        └─→ stage4 ─→ stage5 ─┴─→ stage6
```

| Env | Default | Meaning |
|---|---|---|
| `METHOD` | `ntxent` | Stage-2 SSL objective (positional `$1`) |
| `RUN_NAME` | — | Optional W&B run name for stage 2 (positional `$2`) |
| `PROTEIN` | `esm2` | Stage 3: `esm2` or `esmc` (pipeline sets `OVERWRITE=0` for incremental embed) |
| `N_SEEDS` | `3` | Stage-5 seeds; `3` runs ensemble eval, else single-checkpoint eval |
| `MULTISEED` | `0` | `1` + `ADAPTER_RUN_ID` (or `PIPELINE_ENV`) = re-submit stage 5 ×3 + stage 6 only |
| `VIEW3D` | `0` | `1` = pretrain the adapter with the 3D point-cloud view (`experiment=adapter3d`; auto-submits stage 1b) |
| `ENCODER_3D` | `0` | `1` = also encode Stage 4–6 ligands with the 3D encoder (implies `VIEW3D=1`) |
| `STAGE_FROM` | — | `4` or `5` + `ADAPTER_RUN_ID` = resume an existing pipeline from that stage |
| `SMOKE` | `0` | `1` = smoke mode (see above) |

Stage 3 skips pids already in the protein store; stages 4–6 rebuild decoy pools
and LIT-PCBA caches from scratch (`--force` / cache clear).

Stage-2 `METHOD` values: `lejepa`, `ntxent`, `siglip`, `ijepa`, `denoise` (maps
to Hydra `experiment=`; `siglip` is a sigmoid-loss contrastive variant of
`ntxent`). Stage-5 `METHOD` is `lejepa` or `ntxent` (W&B grouping only; `siglip`
folds into `ntxent`; pools always come from the Stage-2 `RUN_ID`).

**3D cross-modal pretraining (`VIEW3D` / `ENCODER_3D`).** `VIEW3D=1` pretrains
the 2D adapter *with* an auxiliary 3D point-cloud view (`experiment=adapter3d`);
`run_pipeline.sh` first submits `stage1b_precompute_conformers.sh` when the
conformer cache is missing. The 3D encoder is a pretraining crutch by default and
its weights are discarded downstream — unless you also set `ENCODER_3D=1`, which
keeps the 3D encoder and uses it (instead of the 2D adapter) to encode ligands in
Stages 4–6. `ENCODER_3D=1` implies `VIEW3D=1`; neither is compatible with
`METHOD=denoise`.

Stage 6 eval artifacts for EBM run id `<ebm_id>`:

```
artifacts/evaluation/<ebm_id>/lit_pcba_zm_mv4      # 4-view z_m cache
artifacts/evaluation/<ebm_id>/lit_pcba_mv4.csv    # single-checkpoint metrics
artifacts/evaluation/<ebm_id>/ensemble_mv4.json   # 3-seed ensemble (RUN_ID0 key)
```

The `adapter_fingerprint` guard refuses to score a `z_m` cache built with a
different adapter than the checkpoint.

Stage 7 still has per-target paths and seed run ids in the script header — update
those after Stage 5. The manual `python -m …` commands behind each wrapper are
documented per stage below.

### Stage 0 — Data acquisition

Download scripts live in `scripts/` and write into `artifacts/preprocessing/raw/`. See
`scripts/DATASETS.md` for dataset details, sizes, and licences.

```bash
# MOSES — molecules for adapter self-supervision        → artifacts/preprocessing/raw/moses.csv
bash scripts/download_moses.sh

# BindingDB-All — full monthly dump (~3.2M measurements) → artifacts/preprocessing/raw/bindingdb/
# Pick the latest YYYYMM release from
# https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp
# download_bindingdb.sh is idempotent: if BindingDB_All.tsv already exists it
# will NOT re-fetch. Delete stale files before refreshing:
#   rm -f artifacts/preprocessing/raw/bindingdb/BindingDB_All.tsv artifacts/preprocessing/raw/bindingdb/BindingDB_All_*.tsv.zip
BINDINGDB_DATE=202606 bash scripts/download_bindingdb.sh

# Sanity-check before Stage 1 (a truncated TSV silently poisons every later stage):
wc -l artifacts/preprocessing/raw/bindingdb/BindingDB_All.tsv          # ~3.1–3.2M lines incl. header
ls -lh artifacts/preprocessing/raw/bindingdb/BindingDB_All_*.tsv.zip   # zip ~550–570 MB; TSV ~3 GB

# LIT-PCBA held-out benchmark                            → artifacts/preprocessing/raw/lit_pcba/
# Downloads via huggingface_hub (CDN + retry/resume), unzips, and stages it.
# Robust to HF rate-limiting (HTTP 429), common from shared cluster IPs — export
# HF_TOKEN to raise the anonymous limit if it still throttles.
bash scripts/download_lit_pcba.sh

# DUD-E benchmark (optional secondary eval)              → artifacts/preprocessing/raw/dude/
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
github.com/soedinglab/MMseqs2/releases.) See [setup_lumi.md](setup_lumi.md) for
the full LUMI environment setup.

### Discrete-flow backbone (DDiT)

`backbone/discrete_flow.py` wraps a frozen DDiT discrete-flow SMILES encoder.
The DDiT architecture is **vendored** in `backbone/ddit/` (no `in_virtuo_gen`
dependency). You need:

1. a **checkpoint** (weights) — e.g. `artifacts/checkpoints/invirtuo_gen.ckpt`;
2. a **tokenizer** JSON — e.g. `artifacts/tokenizer/smiles_new.json` (tracked in git).

Stage 1 fragmentation uses RDKit BRICS (`preprocessing/molecules.py`); no
external FragMol install is required.

The encode-time fed to the backbone can be a **learnable parameter**
(`learnable_time: true` in `configs/model/discrete_flow.yaml`): it's
sigmoid-bounded to (0, 1) and the SSL gradient reaches it through the frozen
backbone.

### Stage 1 — Preprocessing
If you are on LUMI - use this:
```bash
  # In your sbatch script / before running preprocessing:
  export PROJ=project_465003063
  export FLASH=/flash/$PROJ/$USER
  mkdir -p $FLASH/tmp $FLASH/artifacts
  export TMPDIR=$FLASH/tmp
  python -m lattice_lab.preprocessing.run_bindingdb \
    --bindingdb-tsv artifacts/preprocessing/raw/bindingdb/BindingDB_All.tsv \
    --lit-pcba-dir  artifacts/preprocessing/raw/lit_pcba \
    --output-dir     $FLASH/artifacts/processed/bindingdb \
    --identity 90 --n-jobs 16

# Canonicalise MOSES (adapter SSL set) into fragment-view parquet shards.
python -m lattice_lab.preprocessing.run_preprocessing \
    --input  artifacts/preprocessing/raw/moses.csv \
    --output  $FLASH/artifacts/processed/moses \
    --n-views 3 --n-jobs 16

cp -r $FLASH/artifacts/processed/* /scratch/$PROJ/benno/lattice_lab/artifacts/preprocessing/processed/
```

```bash
# Curate BindingDB + build the 90% MMseqs2 identity split held out vs LIT-PCBA.
# Needs MMseqs2 (hard dep). LATTICE_ALLOW_KMER_FALLBACK=1 only for smoke/tests.
python -m lattice_lab.preprocessing.run_bindingdb \
    --bindingdb-tsv artifacts/preprocessing/raw/bindingdb/BindingDB_All.tsv \
    --lit-pcba-dir  artifacts/preprocessing/raw/lit_pcba \
    --output-dir    artifacts/preprocessing/processed/bindingdb \
    --identity 90 --n-jobs 16

# Canonicalise MOSES (adapter SSL set) into fragment-view parquet shards.
python -m lattice_lab.preprocessing.run_preprocessing \
    --input  artifacts/preprocessing/raw/moses.csv \
    --output artifacts/preprocessing/processed/moses \
    --n-views 3 --n-jobs 16
```

Both commands write **faithful, full-coverage** fragment views by default (no
atoms dropped); SSL masking/shuffle happens online in the datamodule. Add
`--merge` to either command to build the coarser multi-granularity variant
instead — it appends `_merge` to the output dir (`moses_merge`,
`bindingdb_merge`), so the two datasets coexist. On SLURM this is the `MERGE=1`
env var (see the stage table above); keep it consistent across Stage 1 and 2.

MMseqs2 (the Stage-1 clustering dep) runs in a fresh per-run workdir
(`_mmseqs_cluster/`), wiped each run, so a previous run's leftovers can't make
`mmseqs cluster` abort; on a genuine failure its own stderr is surfaced.

### Stage 2 — Molecule encoder (DDiT + adapter SSL)

Pick an SSL objective via `METHOD` on SLURM (`lejepa`, `ntxent`, `siglip`,
`ijepa`, `denoise`). Each lands at `artifacts/adapter/checkpoints/<wandb_run_id>/last.ckpt`,
which Stage 4+ load **directly** — no export step.

```bash
sbatch scripts/slurm/stage2_ssl.sh lejepa
sbatch scripts/slurm/stage2_ssl.sh ijepa my_ablation_name   # custom W&B run name
VIEW3D=1 sbatch scripts/slurm/stage2_ssl.sh lejepa          # + auxiliary 3D point-cloud view
```

Manual equivalents:

```bash
# NT-Xent / InfoNCE (LATTICE baseline — fp distillation):
python -m lattice_lab.train experiment=adapter_discrete_flow model.ssl_loss=ntxent model.fp_weight=2.0 \
    data.shard_dir=artifacts/preprocessing/processed/moses \
    trainer.max_epochs=10 trainer.accelerator=gpu \
    callbacks.model_checkpoint.dirpath=artifacts/adapter/checkpoints

# LeJEPA (invariance + SIGReg):
python -m lattice_lab.train experiment=adapter_discrete_flow model.ssl_loss=lejepa model.fp_weight=0.0 \
    data.shard_dir=artifacts/preprocessing/processed/moses data.batch_size=128 \
    trainer.max_epochs=10 trainer.accelerator=gpu \
    callbacks.model_checkpoint.dirpath=artifacts/adapter/checkpoints

# I-JEPA (masked-fragment prediction + SIGReg):
python -m lattice_lab.train experiment=adapter_discrete_flow model.ssl_loss=ijepa \
    data.shard_dir=artifacts/preprocessing/processed/moses data.batch_size=256 \
    trainer.max_epochs=10 trainer.accelerator=gpu \
    callbacks.model_checkpoint.dirpath=artifacts/adapter/checkpoints

# Denoising-JEPA:
python -m lattice_lab.train experiment=denoising_jepa \
    data.shard_dir=artifacts/preprocessing/processed/moses \
    trainer.max_epochs=10 trainer.accelerator=gpu \
    callbacks.model_checkpoint.dirpath=artifacts/adapter/checkpoints
```

### Stage 3 — Protein encoder (frozen ESM-2 650M)

Loads `facebook/esm2_t33_650M_UR50D` from Hugging Face (the CLI default — do **not**
pass the bare `esm2_t33_650M_UR50D` name; HF needs the `facebook/` prefix). A
harmless transformers warning about uninitialized `pooler` weights is expected —
we mean-pool residue hidden states and never use the pooler head.

**GPU required for the BindingDB FASTA** (~7.5k targets). On LUMI you need **both**
a GPU allocation **and** the ROCm PyTorch module — a plain conda env (e.g.
`/opt/miniconda3/envs/pytorch`) is not ROCm-aware and will raise
`RuntimeError: No HIP GPUs are available` even on a GPU node.

```bash
# From a login node (uan…): srun lands you on a compute node (nid…).
# salloc alone reserves nodes but leaves you on the login node.
srun --account=project_465003063 --partition=small-g --nodes=1 \
  --gpus-per-node=1 --cpus-per-task=7 --mem=60G --time=02:00:00 --pty bash

module load LUMI PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
cd /pfs/lustrep4/scratch/project_465003063/benno/lattice_lab

# Sanity-check BEFORE loading the 2.5 GB checkpoint:
python -c "import torch; print('gpu', torch.cuda.is_available(), torch.__version__)"
# expect: gpu True 2.7.1

python -m lattice_lab.protein.precompute \
    --fasta artifacts/preprocessing/processed/bindingdb/bindingdb_targets.fasta \
    --store artifacts/protein_store/embeddings/esm2_650M \
    --device cuda --batch-size 8
python -m lattice_lab.protein.precompute \
    --fasta artifacts/preprocessing/processed/bindingdb/lit_pcba_targets.fasta \
    --store artifacts/protein_store/embeddings/esm2_650M \
    --device cuda --batch-size 8
```

Omit `--device cuda` to fall back to CPU (fine for the tiny LIT-PCBA target set;
very slow for BindingDB). Full LUMI env setup (container install, MMseqs2, sbatch
templates): see [setup_lumi.md](setup_lumi.md).

**Alternative backend — ESM C 600M.** Pass `--backend esmc` to embed with
EvolutionaryScale's ESM C (Cambrian) instead of ESM-2. It needs the `esm` SDK
(`pip install 'lattice_lab[esmc]'`; on LUMI this is baked into the container by
`scripts/setup_container.sh` — see [Setting up on LUMI](#setting-up-on-lumi)).
ESM C produces **1152-d** embeddings, so write it to a separate store and tell
the later stages about the new dimension:

```bash
python -m lattice_lab.protein.precompute --backend esmc \
    --fasta artifacts/preprocessing/processed/bindingdb/bindingdb_targets.fasta \
    --store artifacts/protein_store/embeddings/esmc_600m \
    --device cuda --batch-size 8
```

Because the embedding dim changes, an esmc store is **not** interchangeable with
an esm2 store or a head trained on one: rebuild the decoy/BDB `z_m` pools is not
needed (those are molecule latents), but you must retrain Stage 5 against the
esmc store and set `d_protein=1152`. Point training and eval at it with two
overrides (no new config files needed):

```bash
python -m lattice_lab.train experiment=ebm_hardneg_lejepa \
    d_protein=1152 \
    data.protein_store=artifacts/protein_store/embeddings/esmc_600m \
    trainer.max_steps=12000 seed=0
```

### Stage 4 — `z_m` pools (run all three before Stage 5)

Encode the decoy, BindingDB-negative, and binder pools **with the same Stage-2
adapter** (replace `<run_id>` with that adapter's W&B run id). Positives and
negatives **must** share one adapter — otherwise the head learns an
adapter-signature shortcut (inflated `val/*`, random LIT-PCBA). The EBM data
module hard-fails on a mismatch. Encoder type (LeJEPA, I-JEPA, denoising-JEPA,
…) is auto-detected from the checkpoint.

```bash
sbatch scripts/slurm/stage4_precompute_decoys.sh <run_id>
```

Manual equivalent:

```bash
ADAPTER=artifacts/adapter/checkpoints/<run_id>/last.ckpt

python -m lattice_lab.ebm.precompute_decoys \
    --shard-dir artifacts/preprocessing/processed/moses \
    --adapter-ckpt "$ADAPTER" --batch-size 512 --force
python -m lattice_lab.ebm.precompute_bdb_zm \
    --bdb-parquet artifacts/preprocessing/processed/bindingdb/threshold_90/train.parquet \
    --adapter-ckpt "$ADAPTER" --batch-size 512 --force
python -m lattice_lab.ebm.precompute_binders \
    --train-parquet artifacts/preprocessing/processed/bindingdb/threshold_90/train.parquet \
    --val-parquet   artifacts/preprocessing/processed/bindingdb/threshold_90/val.parquet \
    --adapter-ckpt "$ADAPTER" --batch-size 512 --force
```

Stores: `artifacts/decoys/<run_id>/{decoy_zm,bdb_zm}`,
`artifacts/binders/<run_id>/binder_zm`.

### Stage 5 — Energy-head training

```bash
# One seed (default --array=0-0 in the sbatch script):
sbatch scripts/slurm/stage5_ebm_train.sh lejepa <stage2_run_id>
sbatch scripts/slurm/stage5_ebm_train.sh ntxent <stage2_run_id>

# Three seeds (reported protocol):
./scripts/slurm/stage5_ebm_train.sh --three-seeds lejepa <stage2_run_id>
sbatch --array=0-2 scripts/slurm/stage5_ebm_train.sh lejepa <stage2_run_id>
```

Manual equivalent (three seeds):

```bash
for S in 0 1 2; do
  python -m lattice_lab.train experiment=ebm_hardneg_lejepa \
    ssl_run_id=<stage2_run_id> \
    trainer.max_steps=12000 seed=$S \
    callbacks.model_checkpoint.dirpath=artifacts/energy/checkpoints
done
# Each run lands at artifacts/energy/checkpoints/<wandb_run_id>/last.ckpt
```
`ebm_hardneg` sets `data.hard_mining_mult=3` + the 0.4/0.15 hard-neg mix.
`ModelCheckpoint(monitor="val/ef1", mode="max")` keeps the best head per seed.

**Head type (`model.head_type`, default `film`).** The energy head is a FiLM
MLP; `model.head_type=cosine` swaps in `CosineMatchHead`, a linear
contrastive-matching baseline (`E = -cos(proj_m(z_m), proj_p(z_p))`) with the same
`(z_m, z_p) → scalar` contract, so Stage 5/6 wiring is unchanged. The type is
embedded in the checkpoint and read back at eval, so callers never re-specify it.

### Stage 6 — Evaluation (LIT-PCBA)

```bash
# Single EBM checkpoint (4-view cache + CSV):
sbatch scripts/slurm/stage6_eval.sh <ebm_run_id>

# 3-seed ensemble (reported protocol):
sbatch scripts/slurm/stage6_eval.sh <id0> <id1> <id2>

# Single-view debug (1-view cache built inline; faster to iterate):
SSL_RUN_ID=<stage2_run_id> sbatch scripts/slurm/stage6_eval.sh --single-view <ebm_run_id>
```

`stage6_ensemble_eval.sh` and `stage6_single_eval.sh` are thin wrappers around
`stage6_eval.sh` for backward compatibility.

Artifacts under `artifacts/evaluation/<ebm_run_id>/` (`lit_pcba_zm_mv4`,
`lit_pcba_mv4.csv`, or `ensemble_mv4.json`). The cache is rebuilt from scratch
on every run.

Manual equivalent:

```bash
# 1. Build 4-view z_m cache + score one checkpoint (stage6_eval.sh does both):
python -m lattice_lab.eval.build_multiview_cache \
    --n-views 4 --zm-cache artifacts/evaluation/<ebm_id>/lit_pcba_zm_mv4 \
    --adapter-ckpt artifacts/energy/checkpoints/<ebm_id>/last.ckpt --n-jobs 40
python -m lattice_lab.eval.lit_pcba \
    --head-ckpt artifacts/energy/checkpoints/<ebm_id>/last.ckpt \
    --adapter-ckpt artifacts/energy/checkpoints/<ebm_id>/last.ckpt \
    --zm-cache artifacts/evaluation/<ebm_id>/lit_pcba_zm_mv4 \
    --output-csv artifacts/evaluation/<ebm_id>/lit_pcba_mv4.csv

# 2. 3-seed ensemble:
python -m lattice_lab.eval.ensemble_eval \
    --ckpts artifacts/energy/checkpoints/<id0>/last.ckpt \
            artifacts/energy/checkpoints/<id1>/last.ckpt \
            artifacts/energy/checkpoints/<id2>/last.ckpt \
    --zm-cache artifacts/evaluation/<id0>/lit_pcba_zm_mv4 \
    --protein-store artifacts/protein_store/embeddings/esm2_650M \
    --test-parquet artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet \
    --out artifacts/evaluation/<id0>/ensemble_mv4.json --n-jobs 32

# Held-out (in-distribution) ranking metrics for a single checkpoint:
python -m lattice_lab.evaluate ckpt_path=artifacts/energy/checkpoints/<ebm_id>/last.ckpt
```

### Stage 7 — Inference / virtual screening
```bash
python -m lattice_lab.inference.predict_ensemble \
    --head-ckpts artifacts/energy/checkpoints/<run_id0>/last.ckpt \
                 artifacts/energy/checkpoints/<run_id1>/last.ckpt \
                 artifacts/energy/checkpoints/<run_id2>/last.ckpt \
    --target-fasta thrb.fasta --target-name THRB \
    --smiles-file my_library.csv --n-views 4 \
    --output-csv artifacts/predictions/thrb_predictions.csv
```

> **Note on checkpoint formats.** There is exactly one format: a full Lightning
> `.ckpt` (the whole module) written by `ModelCheckpoint`, and it is
> **self-describing** — alongside the weights it embeds an `encoder_config`
> (DDiT hook layer range, dims, `d_adapter`) so the encoder can be rebuilt from
> the single file. `load_encoder_from_ckpt` builds a fresh skeleton from that
> config and loads the entire `encoder.*` state (backbone + adapter + learnable
> time); the energy head is pulled out by the `head.*` prefix. **No base DDiT is
> needed at eval, and the layer range is never re-specified by callers** — the
> adapter is always served exactly the DDiT layers it was trained on. The same
> `.ckpt` works as both the encoder source and the head source (the EBM ckpt
> froze the encoder), so Stage 4/6/7 all point at one file. No partial bundles,
> no export step. (Old pre-`encoder_config` ckpts fall back to documented
> defaults; re-saving them embeds the config.)

## Ablations & ensemble scaling

Experiment-management helpers around a finished winner (they reuse the standard
stage scripts, so nothing about the core pipeline changes):

```bash
# Leave-one-out component ablations vs a base run (default BASE_RUN_ID=w790kdrh).
# Each run drops one piece (no_ssl, cosine_match, no_hardneg, no_bdb_mix,
# no_cross_target, no_sinkhorn); Stage-6 LIT-PCBA BEDROC is the score.
./scripts/slurm/run_component_ablations.sh
DRY_RUN=1 ./scripts/slurm/run_component_ablations.sh          # print the plan

# Ensemble-size scaling: train extra EBM seeds, then mean±std BEDROC over all
# subsets of size k.
BASE_RUN_ID=w790kdrh EXTRA_SEEDS=6 ./scripts/slurm/run_ensemble_scaling.sh

# Aggregate stage-6 results across pipeline/ablation runs into one ranked table.
python scripts/aggregate_ablations.py --prefix comp_
```

EBM leave-one-outs symlink the base adapter + decoy/binder stores; `no_ssl`
plants a random-adapter init and re-runs Stage 4→6. Extra ablation env knobs feed
through the stage scripts: `EXTRA_EBM_ARGS` (append Hydra overrides to Stage 5)
and `EVAL_PREFER_LAST=1` (Stage 6 scores `last.ckpt` instead of the monitored
best).

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
