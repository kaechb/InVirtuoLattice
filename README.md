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
├── raw/            # downloads: moses.csv, qm9.csv, bindingdb/, lit_pcba/, dude/
├── processed/      # Stage 1: bindingdb/threshold_90/*.parquet, moses/shard_*.parquet, *.fasta
├── adapter/        # Stage 2: <variant>/checkpoints/<run_id>/last.ckpt
├── protein_store/  # Stage 3: frozen ESM-2 embedding store
├── decoys/         # Stage 4: <variant>/{decoy_zm,bdb_zm} pools (binders → binders/<variant>/)
├── energy/         # Stage 5: <variant>/<run_id>/last.ckpt EBM heads
├── evaluation/     # Stage 6: LIT-PCBA caches, results, violins
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
``{dirpath}/{wandb_run_id}/`` (e.g. ``artifacts/adapter/lejepa/checkpoints/73miv4j1/last.ckpt``).
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
dirs), so just submit from the repo root. Most are driven by **environment
variables — you don't edit the files**:

| Stage | Script | Knobs | Output |
|---|---|---|---|
| 1 | `stage1_preprocess.sh` | — | BindingDB curation + 90% MMseqs2 split |
| 2 | `stage2_adapter_ssl_{lejepa,ntxent}.sh` | — | adapter SSL ckpt (one per variant) |
| 3 | `stage3_protein_precompute.sh` (or `…_esmc.sh`) | — | frozen ESM-2 / ESM C store |
| 4 | `stage4_precompute_decoys_{lejepa,ntxent}.sh` | edit `SSL_RUN_ID` | decoy + BDB + binder `z_m` pools |
| 5 | `stage5_ebm_train_{lejepa,ntxent}.sh` | `--array=0-2` (3 seeds) | `artifacts/energy/<variant>/<run_id>/last.ckpt` |
| 6a | `stage6_build_zm_cache.sh` | `VARIANT` | 4-view cache `lit_pcba_zm_mv4_<variant>` |
| 6b | `stage6_ensemble_eval.sh` | `VARIANT` | 3-seed LIT-PCBA ensemble (needs 6a) |
| 6 | `stage6_single_eval.sh` | `VARIANT`, `RUN_ID` (or `CKPT`) | single-checkpoint LIT-PCBA (self-contained cache) |
| 7 | `stage7_predict_ensemble.sh` | `VARIANT` | virtual screening |

```bash
# variant-aware stages take VARIANT=lejepa|ntxent:
VARIANT=ntxent sbatch scripts/slurm/stage6_build_zm_cache.sh
VARIANT=ntxent sbatch scripts/slurm/stage6_ensemble_eval.sh         # after 6a finishes
VARIANT=ntxent RUN_ID=gi2762bi sbatch scripts/slurm/stage6_single_eval.sh

# chain a job to start only once another succeeds:
sbatch --dependency=afterok:<jobid> scripts/slurm/stage6_ensemble_eval.sh
```

The three per-seed run ids for the **ensemble** (6b) and **screening** (7) are
wired in a `case "$VARIANT"` block near the top of those two scripts — update
them after a fresh Stage-5 run. Stage-4 takes its Stage-2 adapter id via
`SSL_RUN_ID` (also at the top). Everything else needs no per-run edits. The
`adapter_fingerprint` guard refuses to score a `z_m` cache that was built with a
different adapter than the checkpoint, so a stale cache fails loudly instead of
silently returning wrong numbers.

The manual `python -m …` commands behind each wrapper are documented per stage
below.

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
    --bindingdb-tsv artifacts/raw/bindingdb/BindingDB_All.tsv \
    --lit-pcba-dir  artifacts/raw/lit_pcba \
    --output-dir     $FLASH/artifacts/processed/bindingdb \
    --identity 90 --n-jobs 16

# Canonicalise MOSES (adapter SSL set) into fragment-view parquet shards.
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

# Canonicalise MOSES (adapter SSL set) into fragment-view parquet shards.
python -m lattice_lab.preprocessing.run_preprocessing \
    --input  artifacts/raw/moses.csv \
    --output artifacts/processed/moses \
    --n-views 3 --n-jobs 16
```

### Stage 2 — Molecule encoder (DDiT + adapter SSL)

Two interchangeable adapter objectives; pick one (or train both to compare).
Each lands at `artifacts/adapter/<variant>/checkpoints/<wandb_run_id>/last.ckpt`,
which Stage 4+ load **directly** — no export step.

```bash
# NT-Xent / InfoNCE adapter:
python -m lattice_lab.train experiment=adapter_discrete_flow_baseline \
    data.shard_dir=artifacts/processed/moses \
    trainer.max_epochs=10 data.batch_size=64 trainer.accelerator=gpu \
    callbacks.model_checkpoint.dirpath=artifacts/adapter/ntxent/checkpoints

# LeJEPA (invariance + SIGReg) adapter:
python -m lattice_lab.train experiment=adapter_discrete_flow_lejepa \
    data.shard_dir=artifacts/processed/moses \
    trainer.max_epochs=10 data.batch_size=64 trainer.accelerator=gpu \
    callbacks.model_checkpoint.dirpath=artifacts/adapter/lejepa/checkpoints \
    model.lejepa_lambda=0.5
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
    --fasta artifacts/processed/bindingdb/bindingdb_targets.fasta \
    --store artifacts/protein_store/embeddings/esm2_650M \
    --device cuda --batch-size 8
python -m lattice_lab.protein.precompute \
    --fasta artifacts/processed/bindingdb/lit_pcba_targets.fasta \
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
    --fasta artifacts/processed/bindingdb/bindingdb_targets.fasta \
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
adapter** (here `<variant>` = `lejepa` or `ntxent`; replace `<run_id>` with that
variant's Stage-2 W&B run id). Positives and negatives **must** share one
adapter — otherwise the head learns an adapter-signature shortcut (inflated
`val/*`, random LIT-PCBA). The EBM data module hard-fails on a mismatch, and
`--force` rebuilds a pool cleanly when the adapter changes.

```bash
ADAPTER=artifacts/adapter/<variant>/checkpoints/<run_id>/last.ckpt

python -m lattice_lab.ebm.precompute_decoys \
    --shard-dir artifacts/processed/moses \
    --adapter-ckpt "$ADAPTER" \
    --store artifacts/decoys/<variant>/decoy_zm --batch-size 512 --force
python -m lattice_lab.ebm.precompute_bdb_zm \
    --bdb-parquet artifacts/processed/bindingdb/threshold_90/train.parquet \
    --adapter-ckpt "$ADAPTER" \
    --store artifacts/decoys/<variant>/bdb_zm --batch-size 512 --force
python -m lattice_lab.ebm.precompute_binders \
    --train-parquet artifacts/processed/bindingdb/threshold_90/train.parquet \
    --val-parquet   artifacts/processed/bindingdb/threshold_90/val.parquet \
    --adapter-ckpt "$ADAPTER" \
    --store artifacts/binders/<variant>/binder_zm --batch-size 512 --force
```

### Stage 5 — Energy-head training (3 seeds)
```bash
for S in 0 1 2; do
  python -m lattice_lab.train experiment=ebm_hardneg_lejepa \
    trainer.max_steps=12000 seed=$S \
    callbacks.model_checkpoint.dirpath=artifacts/energy/lejepa
done
# Each run lands at artifacts/energy/<variant>/<wandb_run_id>/last.ckpt
# (swap experiment=ebm_hardneg_ntxent + dirpath=artifacts/energy/ntxent for NT-Xent).
```
`ebm_hardneg` sets `data.hard_mining_mult=3` + the 0.4/0.15 hard-neg mix.
`ModelCheckpoint(monitor="val/ef1", mode="max")` keeps the best head per seed.

### Stage 6 — Evaluation (LIT-PCBA)
```bash
# 1. Build the 4-view LIT-PCBA z_m cache (adapter read straight from an EBM ckpt).
python -m lattice_lab.eval.build_multiview_cache \
    --n-views 4 --zm-cache artifacts/evaluation/lit_pcba_zm_mv4_lejepa \
    --adapter-ckpt artifacts/energy/lejepa/<run_id0>/last.ckpt --n-jobs 4

# 2. Score the 3-seed ensemble (the reported result).
# Replace each <run_id> with the W&B run id for that seed's training job.
python -m lattice_lab.eval.ensemble_eval \
    --ckpts artifacts/energy/lejepa/<run_id0>/last.ckpt \
            artifacts/energy/lejepa/<run_id1>/last.ckpt \
            artifacts/energy/lejepa/<run_id2>/last.ckpt \
    --zm-cache artifacts/evaluation/lit_pcba_zm_mv4_lejepa \
    --protein-store artifacts/protein_store/embeddings/esm2_650M \
    --test-parquet artifacts/processed/bindingdb/test_lit_pcba.parquet \
    --out artifacts/evaluation/ensemble_hardneg_mv4_lejepa.json --n-jobs 32

# On SLURM, the above is wrapped by (VARIANT=lejepa|ntxent):
#   VARIANT=lejepa sbatch scripts/slurm/stage6_build_zm_cache.sh
#   VARIANT=lejepa sbatch scripts/slurm/stage6_ensemble_eval.sh
#   VARIANT=lejepa RUN_ID=<run_id> sbatch scripts/slurm/stage6_single_eval.sh   # single ckpt

# Held-out (in-distribution) ranking metrics for a single checkpoint:
python -m lattice_lab.evaluate ckpt_path=artifacts/energy/lejepa/<run_id0>/last.ckpt
```

### Stage 7 — Inference / virtual screening
```bash
python -m lattice_lab.inference.predict_ensemble \
    --head-ckpts artifacts/energy/lejepa/<run_id0>/last.ckpt \
                 artifacts/energy/lejepa/<run_id1>/last.ckpt \
                 artifacts/energy/lejepa/<run_id2>/last.ckpt \
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
