#!/usr/bin/env python3
"""Write an untrained (backbone-pretrained, random adapter) Lightning ckpt.

Uses ``encoder_config`` from a finished Stage-2 run so dims/layers match the
baseline, then rebuilds from the InVirtuo DDiT + fresh adapter. That is the
leave-one-out for Stage-2 NT-Xent: same architecture, no SSL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lattice_lab.backbone.discrete_flow import build_discrete_flow_encoder
from lattice_lab.models.builders import safe_torch_load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-ckpt", type=Path, required=True, help="finished Stage-2 last.ckpt")
    ap.add_argument("--out", type=Path, required=True, help="output last.ckpt path")
    ap.add_argument(
        "--backbone",
        type=Path,
        default=Path("artifacts/checkpoints/invirtuo_gen.ckpt"),
    )
    args = ap.parse_args()

    raw = safe_torch_load(args.from_ckpt, weights_only=False)
    cfg = dict(raw.get("encoder_config") or {})
    if not cfg:
        raise SystemExit(f"no encoder_config in {args.from_ckpt}")

    enc = build_discrete_flow_encoder(
        ckpt_path=str(args.backbone),
        tokenizer_path=str(cfg["tokenizer_path"]),
        backbone_layer_start=int(cfg["backbone_layer_start"]),
        backbone_layer_end=int(cfg["backbone_layer_end"]),
        d_adapter=int(cfg["d_adapter"]),
        adapter_n_layers=int(cfg["adapter_n_layers"]),
        adapter_pool=str(cfg.get("adapter_pool", "attn")),
        adapter_dual_pool=bool(cfg.get("adapter_dual_pool", False)),
        adapter_proj_dim=int(cfg.get("adapter_proj_dim", 256)),
        encode_time=float(cfg.get("encode_time", 0.98)),
        learnable_time=bool(cfg.get("learnable_time", True)),
        freeze_backbone=True,
        token_id_min=int(cfg.get("token_id_min", 4)),
        n_layer=int(cfg.get("n_layer", 12)),
        n_head=int(cfg.get("n_head", 12)),
        n_embd=int(cfg.get("n_embd", 768)),
        dropout=float(cfg.get("dropout", 0.1)),
        n_conds=int(cfg.get("n_conds", 0)),
        device="cpu",
    )
    state = {f"encoder.{k}": v.cpu() for k, v in enc.state_dict().items()}
    out_cfg = dict(getattr(enc, "build_config", {}) or cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state,
            "encoder_config": out_cfg,
            "fragment_merge": bool(raw.get("fragment_merge", False)),
        },
        args.out,
    )
    print(f"wrote no-SSL adapter → {args.out}")


if __name__ == "__main__":
    main()
