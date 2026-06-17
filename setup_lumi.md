# Setting up `lattice_lab` on LUMI

This guide covers the LUMI-specific environment: the PyTorch container, where to
put large files (quota), the per-session module chain, and MMseqs2 for Stage 1.
For the science pipeline itself, see the [README](README.md).

## TL;DR — one-time setup

```bash
# On a LOGIN node (uan…), from the repo root. EasyBuild must NOT run on a compute node.
cd ~/benno/lattice_lab
bash scripts/setup_container.sh
```

That script does the whole container setup end to end (idempotent — safe to
re-run). The rest of this document explains what it does and why, plus the manual
steps if you ever need them.

## The container model

LUMI runs PyTorch inside a **read-only Singularity container** (an EasyBuild
"container module"), not a normal conda/venv environment. Extra Python packages —
including `lattice_lab` itself and the optional `esm` SDK — live in an **overlay
venv** that the module mounts at `/user-software`. For performance on Lustre you
then freeze that venv into a single **SquashFS** image (one file instead of tens
of thousands of small ones).

So the lifecycle is: install the container module → `pip install` into the
overlay venv → `make-squashfs` to freeze it → reload the module so it mounts the
squashfs read-only. `scripts/setup_container.sh` automates all of this and
**verifies imports from the squashfs before deleting the writable copy**, so a
failed pack (e.g. quota) can never lose your venv.

## Quota: keep large files off `$HOME`

LUMI home is tiny (**~20 GB**) while the container `.sif` alone is **~18 GB**.
Everything must go into project space, not home. The two variables that matter:

```bash
# Persisted in ~/.bashrc by the setup script. $HOME is too small for either.
export EBU_USER_PREFIX=/projappl/project_465003063/$USER/EasyBuild  # container + modules → 54 G /projappl
export HF_HOME=/scratch/project_465003063/$USER/hf-home             # HF model cache     → 55 T /scratch
```

- If **`EBU_USER_PREFIX`** is unset, EasyBuild-user defaults to `$HOME/EasyBuild`
  and the 18 GB `.sif` fills your home quota (then `make-squashfs` fails with
  `Disk quota exceeded`, and the wrapper misleadingly still prints `Created …`).
- If **`HF_HOME`** is unset, HuggingFace downloads (e.g. the 2.3 GB ESM C weights)
  go to `~/.cache/huggingface` and fail with `Disk quota exceeded (os error 122)`.

Check usage any time with `lumi-quota`. Move an existing HF cache off home with:

```bash
mkdir -p "$HF_HOME" && mv ~/.cache/huggingface/* "$HF_HOME"/ 2>/dev/null || true
```

## Per-session module chain

In any new shell or SLURM job, load modules in this order. **`lumi-CrayPath` must
come last** (and be reloaded after any later module change), or the container
can't link its libraries on compute nodes (`LD_LIBRARY_PATH` isn't adapted):

```bash
export EBU_USER_PREFIX=/projappl/project_465003063/$USER/EasyBuild  # so the user module is visible
module load LUMI/25.09 partition/G
module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
module load lumi-CrayPath
which python   # → /projappl/.../bin/python   (NOT /usr/bin/python or "command not found")
```

`scripts/slurm/common.sh` (`lattice_load_gpu_modules`) already does this for batch
jobs, so your sbatch scripts just `source scripts/slurm/common.sh`.

> If `module load PyTorch/...` reports the module is **unknown** right after an
> install, your Lmod cache is stale: `module --ignore_cache load PyTorch/...`.

## Manual container install (what the script automates)

Only needed if you're not using `scripts/setup_container.sh`:

```bash
# 1. Persist the EasyBuild prefix + HF cache (one time)
echo 'export EBU_USER_PREFIX=/projappl/project_465003063/$USER/EasyBuild' >> ~/.bashrc
echo 'export HF_HOME=/scratch/project_465003063/$USER/hf-home'           >> ~/.bashrc
export EBU_USER_PREFIX=/projappl/project_465003063/$USER/EasyBuild
export HF_HOME=/scratch/project_465003063/$USER/hf-home

# 2. Install the container (copies a prebuilt .sif into /projappl; minutes, no rebuild).
#    The easyconfig lives in LUMI's container repo, which isn't on the default
#    robot path, so pass the full path.
module load LUMI/25.09 partition/G
module load EasyBuild-user
eb /appl/local/containers/LUMI-EasyBuild-containers/easybuild/easyconfigs/p/PyTorch/PyTorch-2.7.1-rocm-6.2.4-python-3.12-singularity-20250827.eb -r

# 3. Load it (lumi-CrayPath last)
module purge
module load LUMI/25.09 partition/G
module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
module load lumi-CrayPath

# 4. Install lattice_lab + esm into the overlay venv, from the repo root
cd ~/benno/lattice_lab
pip install -e '.[esmc]'
python -c "import lattice_lab.protein.encoder, esm, transformers; print('ok')"

# 5. Freeze and verify (the wrapper's success message is unreliable — verify the import)
cd "$CONTAINERROOT" && make-squashfs
module purge && module load LUMI/25.09 partition/G && module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827 && module load lumi-CrayPath
python -c "import lattice_lab.protein.encoder, esm; print('squashfs ok')"   # only AFTER this succeeds:
rm -r "$CONTAINERROOT/user-software"
```

### Editable install + SquashFS

`pip install -e .` writes only a **path pointer** into the venv (not a copy of
the code), so after `make-squashfs`:

- **Code edits under `src/` are picked up live** — no reinstall, no re-squash.
- The repo must stay at the same path (the pointer records the absolute path).
- Re-run `scripts/setup_container.sh` only when **dependencies or packaging**
  change (new dep, `pyproject.toml` edit, new top-level package, entry-point
  change). It unpacks the squashfs, reinstalls, and repacks.

## MMseqs2 (required for Stage 1)

There's no MMseqs2 module on LUMI and it isn't pip-installable. Drop in the static
**AVX2** binary (the EPYC nodes are AVX2, **not** AVX512). It runs as a plain
binary outside the container, so it's independent of the squashfs env:

```bash
mkdir -p software && cd software
curl -LO https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz   # on a login node
tar xzf mmseqs-linux-avx2.tar.gz
./mmseqs/bin/mmseqs version                                    # smoke-check
```

The repo auto-discovers `software/mmseqs/bin/mmseqs` and prepends it to `PATH`
(`LATTICE_MMSEQS_DIR` overrides the location); `scripts/slurm/common.sh` also
prepends `MMSEQS_BIN`. Fallback URL: the `mmseqs-linux-avx2.tar.gz` asset on
github.com/soedinglab/MMseqs2/releases.

## Stage 1 scratch/tmp on LUMI

Stage 1 preprocessing is I/O heavy; point `TMPDIR` and intermediate outputs at
fast project storage (`/flash`), then copy results back to `/scratch`:

```bash
export PROJ=project_465003063
export FLASH=/flash/$PROJ/$USER
mkdir -p $FLASH/tmp $FLASH/artifacts
export TMPDIR=$FLASH/tmp
# ... run preprocessing with --output-dir $FLASH/artifacts/... then copy back to /scratch ...
```
