from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from utils import load_state_dict


def apply_task_edit(
    state_dict: Dict[str, torch.Tensor],
    task_vectors: list[Dict[str, torch.Tensor]],
    alphas: list[float],
    layers: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    editable = set(layers) if layers else None
    out = {}
    for name, param in state_dict.items():
        updated = param.clone()
        if editable is None or name in editable:
            for vec, alpha in zip(task_vectors, alphas):
                if name in vec:
                    updated = updated + alpha * vec[name]
        out[name] = updated
    return out


def parse_layers(layer_csv: str | None) -> list[str] | None:
    if not layer_csv:
        return None
    return [name.strip() for name in layer_csv.split(",") if name.strip()]


def main(args: argparse.Namespace) -> None:
    model_state = load_state_dict(Path(args.model_ckpt))

    if len(args.alpha) != len(args.task_vector):
        raise ValueError("--alpha must be provided once per --task-vector")

    vectors = [torch.load(path, map_location="cpu") for path in args.task_vector]
    edited = apply_task_edit(model_state, vectors, args.alpha, layers=parse_layers(args.layers))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": edited, "alphas": args.alpha, "task_vectors": args.task_vector}, out)

    meta = {
        "model_ckpt": args.model_ckpt,
        "task_vectors": args.task_vector,
        "alphas": args.alpha,
        "layers": parse_layers(args.layers),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved edited model to {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply task-vector edit to a model checkpoint.")
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--task-vector", action="append", required=True, help="Pass multiple times for composition.")
    p.add_argument("--alpha", action="append", type=float, required=True, help="One alpha per task vector.")
    p.add_argument("--layers", default=None, help="Optional comma-separated parameter names.")
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
