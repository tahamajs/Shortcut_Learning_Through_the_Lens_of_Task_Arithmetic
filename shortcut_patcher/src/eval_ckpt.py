from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import torch

from utils import build_dataloaders, evaluate, load_pretrained, write_json


def _infer_num_classes_from_state_dict(model_name: str, state_dict: Dict[str, torch.Tensor]) -> int:
    if model_name == "resnet18" and "fc.weight" in state_dict:
        return int(state_dict["fc.weight"].shape[0])
    if model_name == "synthetic-mlp" and "net.4.weight" in state_dict:
        return int(state_dict["net.4.weight"].shape[0])
    return 10


def load_checkpoint_into_model(model_name: str, ckpt_path: Path) -> torch.nn.Module:
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    n_classes = _infer_num_classes_from_state_dict(model_name, state_dict)
    model = load_pretrained(model_name, n_classes=n_classes)
    model.load_state_dict(state_dict, strict=False)
    return model


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    _, _, test_loader = build_dataloaders(
        args.task,
        args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        group_balanced=False,
    )

    results = {}
    for item in args.ckpt:
        name, ckpt = item.split("=", 1)
        model = load_checkpoint_into_model(args.model, Path(ckpt)).to(device)
        metrics = evaluate(model, test_loader, device)
        results[name] = metrics

    out = Path(args.output_json)
    write_json(out, results)

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "accuracy", "loss", "worst_group_accuracy"])
        for name, metrics in results.items():
            writer.writerow([name, metrics.get("accuracy"), metrics.get("loss"), metrics.get("worst_group_accuracy")])

    print(f"Wrote {out} and {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one or more checkpoints and save metrics.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="synthetic-mlp")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ckpt", action="append", required=True, help="name=path format, can repeat.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
