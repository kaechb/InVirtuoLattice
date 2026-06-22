#!/usr/bin/env python3
"""Stage-5 EBM diagnostics: FiLM gate deadlock + loss ablations.

    cd lattice_lab && source scripts/slurm/common.sh && lattice_load_gpu_modules
    python scripts/diag_ebm_stage5.py [--ssl-run-id nsw2w2z5] [--steps 200]
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from lattice_lab.ebm.head import EnergyHead
from lattice_lab.ebm.losses import InfoNCEEnergyLoss, SinkhornEnergyLoss, cross_target_margin_loss
from lattice_lab.models.builders import load_encoder_from_ckpt
from lattice_lab.models.ebm import EBMLitModule
from lattice_lab.models.schedules import lambda_sink_schedule


@dataclass
class StepStats:
    step: int
    loss: float
    infonce: float
    sinkhorn: float
    cross_target: float
    viol: float
    top1: float
    binder_e: float
    decoy_e: float
    gap: float
    gamma_norm: float
    beta_norm: float
    grad_protein: float
    grad_mlp: float


def _grad_norm(params) -> float:
    total = 0.0
    n = 0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
            n += 1
    return (total**0.5) if n else 0.0


def _real_batch(ssl_run_id: str, device: torch.device, *, b: int, n_decoys: int) -> dict:
    from pathlib import Path

    import numpy as np

    from lattice_lab.ebm.dataset import DecoyZmPool, stack_z_p
    from lattice_lab.protein.store import EmbeddingStore

    repo = Path(__file__).resolve().parents[1]
    decoy = DecoyZmPool.open(repo / f"artifacts/decoys/{ssl_run_id}/decoy_zm", load_to_ram=True)
    binder = EmbeddingStore.open(
        repo / f"artifacts/binders/{ssl_run_id}/binder_zm", mode="r", load_to_ram=True
    )
    protein = EmbeddingStore.open(
        repo / "artifacts/protein_store/embeddings/esm2_650M", mode="r", load_to_ram=True
    )
    import pandas as pd

    df = pd.read_parquet(
        repo / "artifacts/preprocessing/processed/bindingdb/threshold_90/train.parquet",
        columns=["smiles", "uniprot"],
    )
    df = df[df["uniprot"].isin(protein.pid_to_row)].sample(b, random_state=0)
    z_m = torch.from_numpy(
        np.stack([binder.get_mean(s) for s in df["smiles"]], dtype=np.float32)
    ).to(device)
    z_p = stack_z_p(df["uniprot"].tolist(), protein, device)
    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(0, decoy.count, (b * n_decoys * 3,), generator=gen)
    dec = decoy._gather(idx).view(b, n_decoys * 3, decoy.dim).to(device)
    return {
        "binder_z_m": z_m,
        "decoy_z_m": dec,
        "z_p": z_p,
        "binder_smiles": df["smiles"].tolist(),
    }


def _one_step(
    module: EBMLitModule,
    batch: dict,
    *,
    lambda_sink: float,
    lambda_sink_warmup: int,
    cross_target_margin: float,
    step: int,
    fp32_head: bool = False,
) -> StepStats:
    hp = module.hparams
    module.zero_grad(set_to_none=True)
    module.train()

    z_m_pos = module._encode_binders(batch)
    z_p = batch["z_p"].to(module.device)
    z_m_dec = batch["decoy_z_m"].to(module.device)
    if hp.hard_mining_mult > 1:
        z_m_dec = module._mine_hard_negatives(z_m_dec, z_p)

    head_ctx = (
        torch.autocast("cuda", enabled=False)
        if fp32_head
        else contextlib.nullcontext()
    )
    with head_ctx:
        e_pos = module.head(z_m_pos, z_p)
        z_p_dec = z_p.unsqueeze(1).expand(-1, z_m_dec.shape[1], -1)
        e_dec = module.head(z_m_dec, z_p_dec)

        l_info, info_log = module.info_loss(e_pos, e_dec)
        l_sink, sink_log = module.sink_loss(e_pos, e_dec)
        lam = lambda_sink_schedule(step, lambda_sink_warmup, lambda_sink)
        total = l_info + lam * l_sink

        bs = z_p.shape[0]
        perm = torch.randperm(bs, device=module.device)
        if torch.equal(perm, torch.arange(bs, device=module.device)):
            perm = torch.roll(perm, shifts=1)
        e_wrong = module.head(z_m_pos, z_p[perm])
        l_ct, ct_log = cross_target_margin_loss(
            e_pos, e_wrong, margin=cross_target_margin
        )
        total = total + hp.lambda_neg * l_ct
    total.backward()

    g_protein = _grad_norm(module.head.protein_proj.parameters())
    g_mlp = _grad_norm(module.head.energy_mlp.parameters())

    with torch.no_grad():
        h_p = module.head.protein_proj(z_p[:4])
        gamma, beta = h_p.chunk(2, dim=-1)
        gamma_live = float(gamma.abs().mean().item())
        beta_live = float(beta.abs().mean().item())

    return StepStats(
        step=step,
        loss=float(total.detach()),
        infonce=info_log["infonce/loss"],
        sinkhorn=sink_log["sinkhorn/loss"],
        cross_target=ct_log["cross_target/loss"],
        viol=ct_log["cross_target/violation_rate"],
        top1=info_log["infonce/top1"],
        binder_e=float(e_pos.detach().mean()),
        decoy_e=float(e_dec.detach().mean()),
        gap=float((e_wrong.detach() - e_pos.detach()).mean()),
        gamma_norm=gamma_live,
        beta_norm=beta_live,
        grad_protein=g_protein,
        grad_mlp=g_mlp,
    )


def _build_module(ssl_run_id: str, device: torch.device, d_adapter: int) -> EBMLitModule:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    ckpt = repo / f"artifacts/adapter/checkpoints/{ssl_run_id}/last.ckpt"
    enc = load_encoder_from_ckpt(ckpt)
    mod = EBMLitModule(
        enc,
        d_adapter=d_adapter,
        d_protein=1280,
        n_decoys=64,
        learning_rate=3e-4,
        num_steps=500,
        warmup_steps=50,
        temperature=0.1,
        lambda_sink=1.0,
        lambda_sink_warmup=10_000,
        lambda_neg=1.0,
        cross_target_p=1.0,
        cross_target_margin=2.0,
        hard_mining_mult=3,
        hard_skip_frac=0.05,
        seed=0,
    )
    mod.to(device)
    mod.on_fit_start()
    return mod


def _run_micro(
    name: str,
    module: EBMLitModule,
    batch: dict,
    *,
    steps: int,
    lambda_sink: float = 1.0,
    lambda_sink_warmup: int = 10_000,
    cross_target_margin: float = 2.0,
    film_identity: bool = True,
    fp32_head: bool = False,
) -> list[StepStats]:
    mod = copy.deepcopy(module)
    opt = torch.optim.AdamW(mod.head.parameters(), lr=3e-4)
    if not film_identity:
        last = mod.head.protein_proj[-1]
        nn.init.xavier_uniform_(last.weight)
        nn.init.zeros_(last.bias)

    out: list[StepStats] = []
    print(f"\n=== {name} ===")
    for s in range(steps + 1):
        if s > 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        st = _one_step(
            mod,
            batch,
            lambda_sink=lambda_sink,
            lambda_sink_warmup=lambda_sink_warmup,
            cross_target_margin=cross_target_margin,
            step=s,
            fp32_head=fp32_head,
        )
        out.append(st)
        if s % max(1, steps // 5) == 0:
            print(
                f"  step {s:4d} loss={st.loss:6.2f} top1={st.top1:.3f} viol={st.viol:.2f} "
                f"gap={st.gap:+.3f} γ={st.gamma_norm:.4f} β={st.beta_norm:.4f} "
                f"g_prot={st.grad_protein:.2e} g_mlp={st.grad_mlp:.2e}"
            )
    return out


def _probe_init_deadlock(head: EnergyHead, batch: dict, device: torch.device) -> None:
    """At FiLM=identity, cross_target / InfoNCE cannot reach protein_proj."""
    z_m = batch["binder_z_m"].to(device)
    z_p = batch["z_p"].to(device)
    dec = batch["decoy_z_m"][:, :64].to(device)
    z_p_exp = z_p.unsqueeze(1).expand(-1, dec.shape[1], -1)

    for label in ("cross_target", "infonce", "sinkhorn"):
        head.zero_grad()
        e_pos = head(z_m, z_p)
        if label == "cross_target":
            e_wrong = head(z_m, z_p.roll(1, 0))
            loss, _ = cross_target_margin_loss(e_pos, e_wrong, margin=2.0)
        else:
            e_dec = head(dec, z_p_exp)
            if label == "infonce":
                loss, _ = InfoNCEEnergyLoss(0.1)(e_pos, e_dec)
            else:
                loss, _ = SinkhornEnergyLoss()(e_pos, e_dec)
        loss.backward()
        gp = _grad_norm(head.protein_proj.parameters())
        gm = _grad_norm(head.energy_mlp.parameters())
        print(f"  init grad [{label:14s}] protein_proj={gp:.2e} energy_mlp={gm:.2e}")

    z_p2 = z_p[:4].detach().clone().requires_grad_(True)
    e = head(z_m[:4], z_p2).sum()
    head.zero_grad()
    e.backward()
    dz = float(z_p2.grad.abs().mean()) if z_p2.grad is not None else 0.0
    print(f"  dE/dz_p at FiLM=identity: {dz:.2e} (expect 0)")


def _probe_prior_scale(head: EnergyHead, batch: dict, device: torch.device) -> None:
    """Sinkhorn prior mu_neg=1.0 vs typical energy scale."""
    z_m = batch["binder_z_m"][:8].to(device)
    z_p = batch["z_p"][:8].to(device)
    dec = batch["decoy_z_m"][:8, :64].to(device)
    with torch.no_grad():
        e_pos = head(z_m, z_p)
        e_dec = head(dec, z_p.unsqueeze(1).expand(-1, dec.shape[1], -1))
        all_e = torch.cat([e_pos.unsqueeze(1), e_dec], dim=1)
        print(
            f"  energy scale: binder mean={e_pos.mean():.3f} std={e_pos.std():.3f} "
            f"decoy mean={e_dec.mean():.3f} | prior mu_neg=1.0 sigma=0.3"
        )
        sink = SinkhornEnergyLoss()
        _, log = sink(e_pos, e_dec)
        print(f"  sinkhorn loss at init: {log['sinkhorn/loss']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ssl-run-id", default="nsw2w2z5")
    p.add_argument("--d-adapter", type=int, default=256)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-decoys", type=int, default=64)
    p.add_argument("--bf16", action="store_true")
    p.add_argument(
        "--fp32-head",
        action="store_true",
        help="run energy head+losses outside autocast (Lightning bf16-mixed fix)",
    )
    p.add_argument("--real-data", action="store_true", help="use precomputed z_m stores")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(
        f"device={device} ssl_run_id={args.ssl_run_id} d_adapter={args.d_adapter} "
        f"real={args.real_data} n_decoys={args.n_decoys} bf16={args.bf16} "
        f"fp32_head={args.fp32_head}"
    )

    module = _build_module(args.ssl_run_id, device, args.d_adapter)
    if args.n_decoys != 64:
        module.hparams.n_decoys = args.n_decoys
    if args.real_data:
        batch = _real_batch(args.ssl_run_id, device, b=32, n_decoys=args.n_decoys)
    else:
        batch = {
            "binder_z_m": torch.randn(32, args.d_adapter, device=device),
            "decoy_z_m": torch.randn(32, args.n_decoys * 3, args.d_adapter, device=device),
            "z_p": torch.randn(32, 1280, device=device),
            "binder_smiles": [f"C{i}" for i in range(32)],
        }

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext()

    print(f"\n--- init deadlock probe ({'real' if args.real_data else 'synthetic'} batch) ---")
    _probe_init_deadlock(module.head, batch, device)
    _probe_prior_scale(module.head, batch, device)

    def run(name: str, **kw) -> None:
        kw.setdefault("fp32_head", args.fp32_head)
        with ctx:
            _run_micro(name, module, batch, steps=args.steps, **kw)

    run("baseline (margin=2, λ_sink=1)")
    run("ablation: λ_sink=0", lambda_sink=0.0)
    run("ablation: margin=0.2", cross_target_margin=0.2)


if __name__ == "__main__":
    main()
