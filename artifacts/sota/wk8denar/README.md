# SOTA: I-JEPA adapter + 3-seed EBM ensemble (pipeline `wk8denar`)

Canonical checkpoint bundle for the best LIT-PCBA run from the `lattice_lab`
discrete-flow pipeline (June 2026).

## Training recipe

| Stage | Setting |
|-------|---------|
| SSL (stage 2) | I-JEPA, `ijepa_block_hole_attn=true`, `ijepa_gram_weight=1.0`, `adapter_n_layers=2`, frozen DDiT backbone, MOSES fragment views (unmerged) |
| Protein (stage 3) | ESM-2 650M, incremental store |
| EBM (stage 5) | Hard-negative LeJEPA recipe, 3 seeds, 12k steps (seed 1 best @ 10k), val-selected `ebm-*.ckpt` |
| Eval (stage 6) | 4-view rBRICS multiview cache + 3-head energy ensemble on LIT-PCBA (15 targets) |

W&B: [stage 2 `wk8denar`](https://wandb.ai/luisbenno/lattice/runs/wk8denar),
EBM seeds [`qit0ptvp`](https://wandb.ai/luisbenno/lattice/runs/qit0ptvp),
[`026fdpv4`](https://wandb.ai/luisbenno/lattice/runs/026fdpv4),
[`bbxnh19w`](https://wandb.ai/luisbenno/lattice/runs/bbxnh19w).

Frozen Hydra configs: `configs/` (snapshot from `logs/slurm/pipeline/wk8denar/`).

## Layout

```
ssl/last.ckpt                 Stage-2 full encoder (backbone + adapter + time)
ebm/seed{0,1,2}/ebm_best.ckpt Stage-5 energy heads (val EF@1% best)
zm_cache/lit_pcba_zm_mv4/     4-view averaged z_m cache (383265 ligands)
results/ensemble_mv4.{json,csv}  LIT-PCBA per-target + summary metrics
runs.json                     run ids, paths, repro notes
pipeline.env                  pipeline metadata
```

Symlinks remain at the original `artifacts/adapter|energy|evaluation/...` paths.

## LIT-PCBA results (stage 6, job 19486268)

**Summary** (`results/ensemble_mv4.json`):

| metric | mean | median |
|--------|-----:|-------:|
| AUROC | 0.603 | 0.564 |
| BEDROC (α=80.5) | 0.075 | 0.051 |
| EF@0.5% | 9.15 | 5.12 |
| EF@1.0% | 6.93 | 5.12 |
| EF@5.0% | 2.80 | 2.05 |

Per-target CSV: `results/ensemble_mv4.csv` (15 targets).

> **Note:** Stage-6 pipeline ensemble accidentally loaded seed-0's head twice
> (sidecar race on `ebm.1`). The three distinct heads in `ebm/seed*/` are the
> correct multiseed bundle; re-run `ensemble_eval` on those files for a true
> 3-seed average.

## Reproduce ensemble eval

From repo root (after `module load` / container as usual):

```bash
SOTA=artifacts/sota/wk8denar
python -m lattice_lab.eval.build_multiview_cache \
  --adapter-ckpt "${SOTA}/ssl/last.ckpt" \
  --output-dir "${SOTA}/zm_cache/lit_pcba_zm_mv4" \
  --n-views 4 --n-jobs 32 --force

python -m lattice_lab.eval.ensemble_eval \
  --head-ckpts \
    "${SOTA}/ebm/seed0/ebm_best.ckpt" \
    "${SOTA}/ebm/seed1/ebm_best.ckpt" \
    "${SOTA}/ebm/seed2/ebm_best.ckpt" \
  --adapter-ckpt "${SOTA}/ssl/last.ckpt" \
  --zm-cache "${SOTA}/zm_cache/lit_pcba_zm_mv4" \
  --protein-store artifacts/protein_store/embeddings/esm2_650M \
  --test-parquet artifacts/preprocessing/processed/bindingdb/test_lit_pcba.parquet \
  --output-json "${SOTA}/results/ensemble_mv4_rerun.json" \
  --output-csv "${SOTA}/results/ensemble_mv4_rerun.csv"
```

Each checkpoint is a full Lightning `.ckpt` with embedded `encoder_config`
(backbone hook layers, `d_adapter`, `adapter_n_layers`, etc.).
