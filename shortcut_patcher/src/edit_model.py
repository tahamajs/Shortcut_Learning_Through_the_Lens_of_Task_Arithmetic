from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from utils import load_state_dict


def apply_task_edit(
    state_dict: Dict[str, torch.Tensor],
    task_vector: Dict[str, torch.Tensor],
    alpha: float,
    layers: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    editable = set(layers) if layers else None
    out = {}
    for name, param in state_dict.items():
        if editable is None or name in editable:
            out[name] = param + alpha * task_vector[name]
        else:
            out[name] = param.clone()
    return out


def parse_layers(layer_csv: str | None) -> list[str] | None:
    if not layer_csv:
        return None
    return [name.strip() for name in layer_csv.split(",") if name.strip()]


def main(args: argparse.Namespace) -> None:
    model_state = load_state_dict(Path(args.model_ckpt))
    task_vec = torch.load(args.task_vector, map_location="cpu")
    edited = apply_task_edit(model_state, task_vec, args.alpha, layers=parse_layers(args.layers))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": edited, "alpha": args.alpha}, out)
    print(f"Saved edited model to {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply task-vector edit to a model checkpoint.")
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--task-vector", required=True)
    p.add_argument("--alpha", type=float, default=-1.0)
    p.add_argument("--layers", default=None, help="Optional comma-separated parameter names.")
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
