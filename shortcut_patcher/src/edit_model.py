from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional

import torch


def load_state_dict_from_ckpt(path: Path) -> Dict[str, torch.Tensor]:
    """
    Loads either:
      - a checkpoint dict with key 'state_dict'
      - a raw state_dict dict
    """
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        # assume it's already a state_dict-like mapping
        return payload
    raise ValueError(f"Unrecognized checkpoint format at {path}")


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict checkpoint at {path}, got {type(payload)}")
    return payload


def parse_layers(spec: Optional[str]) -> Optional[List[str]]:
    """
    Comma-separated list of layer name patterns (supports glob like 'net.0.*').
    If None/empty => edit all layers.
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    return [s.strip() for s in spec.split(",") if s.strip()]


def matches_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) or (p in name) for p in patterns)


def apply_task_edit(
    model_state: Dict[str, torch.Tensor],
    task_vector: Dict[str, torch.Tensor],
    alpha: float,
    layers: Optional[List[str]] = None,
    strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Returns edited state dict:
      w_new = w + alpha * v

    layers:
      - None => apply to all keys found in both dicts
      - list of patterns => only apply to keys matching any pattern

    strict:
      - True => require that every edited key exists in task_vector
      - False => silently skip missing keys
    """
    edited: Dict[str, torch.Tensor] = {}
    missing = []

    for k, w in model_state.items():
        do_edit = True
        if layers is not None:
            do_edit = matches_any(k, layers)

        if do_edit:
            if k not in task_vector:
                if strict:
                    missing.append(k)
                    edited[k] = w
                else:
                    edited[k] = w
            else:
                v = task_vector[k]
                # ensure compatible dtype
                if v.dtype != w.dtype:
                    v = v.to(dtype=w.dtype)
                edited[k] = w + float(alpha) * v
        else:
            edited[k] = w

    if strict and missing:
        raise KeyError(
            "Some parameters were selected for editing but are missing from task_vector:\n"
            + "\n".join(missing[:50])
            + (f"\n... and {len(missing)-50} more" if len(missing) > 50 else "")
        )

    return edited


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply task arithmetic edits to a model checkpoint.")
    p.add_argument("--model-ckpt", required=True, help="Path to model checkpoint (.pt) with state_dict")
    p.add_argument("--task-vector", required=True, help="Path to task vector (.pt) state_dict-like")
    p.add_argument("--alpha", type=float, required=True, help="Scaling factor for task vector")
    p.add_argument("--layers", type=str, default=None, help="Comma-separated layer patterns to edit (glob supported)")
    p.add_argument("--strict", action="store_true", help="Fail if any selected layer missing in task vector")
    p.add_argument("--output", required=True, help="Output checkpoint path")
    return p


def main(args: argparse.Namespace) -> None:
    model_path = Path(args.model_ckpt)
    vec_path = Path(args.task_vector)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_payload = load_payload(model_path)
    model_state = load_state_dict_from_ckpt(model_path)
    vec_state = load_state_dict_from_ckpt(vec_path)

    layer_patterns = parse_layers(args.layers)

    edited_state = apply_task_edit(
        model_state=model_state,
        task_vector=vec_state,
        alpha=args.alpha,
        layers=layer_patterns,
        strict=args.strict,
    )

    # Preserve original checkpoint structure so the rest of your pipeline keeps working
    new_payload = dict(model_payload)
    new_payload["state_dict"] = edited_state
    new_payload["edit"] = {
        "task_vector": str(vec_path),
        "alpha": float(args.alpha),
        "layers": layer_patterns,
        "strict": bool(args.strict),
    }

    torch.save(new_payload, out_path)
    print(f"Saved edited model to {out_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())