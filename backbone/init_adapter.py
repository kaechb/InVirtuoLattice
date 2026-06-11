"""Write a randomly-initialized adapter checkpoint — a Stage-2-free stand-in.

This is the drop-in alternative to Stage 2 (``lattice.training.train_adapter``,
the SimCLR SSL run). It produces an ``adapter_*.pt`` in exactly the same format
— a dict with an ``adapter_state_dict`` — but leaves the adapter at its random
initialization instead of contrastively pretraining it.

Why this exists: a randomly-initialized adapter is a random projection of
FragMol's (already strong) frozen hidden states, which is a perfectly usable
``z_m``. Pairing this checkpoint with ``--finetune-adapter`` lets the EBM
binding loss — not an SSL proxy — shape the adapter. Running the pipeline once
with this checkpoint and once with the Stage-2-pretrained ``adapter_v1.pt``,
then comparing LIT-PCBA, tells you whether Stage-2 pretraining is worth keeping.

Important: the adapter that encodes the Stage-4 decoy ``z_m`` pool and the
adapter used at Stage-5 training must be the *same* one. So for the
random-init arm, precompute the Stage-4 pools with *this* checkpoint before
training the EBM with it.

The constructor defaults match ``train_ebm.build_encoder`` (d_fragmol=768,
n_fragmol_layers=4, d_adapter=512, n_layers=4) so the state dict loads without
a shape mismatch. Override them only if you also changed them there.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from lattice_lab.backbone.adapter import Adapter, AdapterConfig

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True,
                        help="destination .pt path for the random-init adapter")
    parser.add_argument("--d-fragmol", type=int, default=768)
    parser.add_argument("--n-fragmol-layers", type=int, default=4)
    parser.add_argument("--d-adapter", type=int, default=512)
    parser.add_argument("--n-adapter-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    torch.manual_seed(args.seed)
    cfg = AdapterConfig(
        d_fragmol=args.d_fragmol,
        n_fragmol_layers=args.n_fragmol_layers,
        d_adapter=args.d_adapter,
        n_layers=args.n_adapter_layers,
    )
    adapter = Adapter(cfg)
    n_params = sum(p.numel() for p in adapter.parameters())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "cfg": {
                "random_init": True,
                "seed": args.seed,
                "d_fragmol": args.d_fragmol,
                "n_fragmol_layers": args.n_fragmol_layers,
                "d_adapter": args.d_adapter,
                "n_adapter_layers": args.n_adapter_layers,
            },
        },
        args.output,
    )
    logger.info(
        "wrote random-init adapter (%d params, seed=%d) → %s",
        n_params, args.seed, args.output,
    )


if __name__ == "__main__":
    main()
