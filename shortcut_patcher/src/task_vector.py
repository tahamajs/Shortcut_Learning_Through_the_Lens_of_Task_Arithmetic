from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from utils import load_state_dict


def compute_task_vector(pretrained_path: Path, finetuned_path: Path) -> Dict[str, torch.Tensor]:
    w_pre = load_state_dict(pretrained_path)
    w_ft = load_state_dict(finetuned_path)

    if w_pre.keys() != w_ft.keys():
        missing = set(w_pre).symmetric_difference(set(w_ft))
        raise ValueError(f"State dict keys do not match. Diff size={len(missing)}")

    return {k: w_ft[k] - w_pre[k] for k in w_pre.keys()}


def main(args: argparse.Namespace) -> None:
    vector = compute_task_vector(Path(args.pretrained), Path(args.finetuned))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, out)
    print(f"Saved task vector with {len(vector)} tensors to {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute task vector (delta weights).")
    p.add_argument("--pretrained", required=True)
    p.add_argument("--finetuned", required=True)
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
