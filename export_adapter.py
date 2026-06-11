"""Export a Lightning adapter checkpoint → legacy ``adapter_v1.pt`` format.

The adapter SSL training (``lattice_lab.train experiment=adapter_ssl``) saves a
Lightning ``.ckpt`` whose ``state_dict`` holds ``encoder.adapter.*`` keys. The
downstream decoy precompute and EBM stages instead expect a plain
``{"adapter_state_dict": {...}}`` file (the historical ``adapter_v1.pt``). This
one-shot converter bridges the two:

    python -m lattice_lab.export_adapter \
        --ckpt   lattice_lab/logs/train/<run>/checkpoints/last.ckpt \
        --output artifacts/adapter/checkpoints_ssl2/adapter_v1.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_adapter(ckpt_path: str | Path, output_path: str | Path) -> Path:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    prefix = "encoder.adapter."
    adapter_state = {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }
    if not adapter_state:
        raise ValueError(
            f"no 'encoder.adapter.*' keys found in {ckpt_path}; "
            "is this an AdapterLitModule checkpoint?"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter_state_dict": adapter_state}, output_path)
    return output_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Lightning adapter .ckpt")
    ap.add_argument("--output", required=True, help="destination adapter_v1.pt")
    args = ap.parse_args()
    out = export_adapter(args.ckpt, args.output)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
