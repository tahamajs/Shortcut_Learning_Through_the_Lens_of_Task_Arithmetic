from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import torch

from utils import load_state_dict


def compute_task_vector(pretrained_path: Path, finetuned_path: Path, include_regex: str | None = None) -> Dict[str, torch.Tensor]:
    w_pre = load_state_dict(pretrained_path)
    w_ft = load_state_dict(finetuned_path)

    if w_pre.keys() != w_ft.keys():
        missing = set(w_pre).symmetric_difference(set(w_ft))
        raise ValueError(f"State dict keys do not match. Diff size={len(missing)}")

    matcher = re.compile(include_regex) if include_regex else None
    out: Dict[str, torch.Tensor] = {}
    for k in w_pre.keys():
        if matcher and not matcher.search(k):
            continue
        out[k] = w_ft[k] - w_pre[k]
    return out


def random_like_vector(reference_vector: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in reference_vector.items():
        rnd = torch.randn(tensor.shape, generator=gen, dtype=tensor.dtype)
        rnd = rnd / (rnd.norm() + 1e-8)
        out[name] = rnd * tensor.norm()
    return out


def main(args: argparse.Namespace) -> None:
    vector = compute_task_vector(Path(args.pretrained), Path(args.finetuned), include_regex=args.include_regex)
    if args.random_like:
        vector = random_like_vector(vector, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, out)
    print(f"Saved task vector with {len(vector)} tensors to {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute task vector (delta weights).")
    p.add_argument("--pretrained", required=True)
    p.add_argument("--finetuned", required=True)
    p.add_argument("--include-regex", default=None, help="Only include parameter keys matching this regex.")
    p.add_argument("--random-like", action="store_true", help="Save a random vector matched to per-layer norms.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
