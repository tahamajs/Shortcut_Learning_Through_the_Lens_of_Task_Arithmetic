from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils import build_dataloaders, evaluate, load_pretrained, save_checkpoint, set_seed, write_json


def warmup_cosine_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())


def train_on_task(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = load_pretrained(args.model).to(device)
    train_loader, val_loader, test_loader = build_dataloaders(args.task, args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: warmup_cosine_lambda(s, args.warmup_steps, args.max_steps))
    criterion = nn.CrossEntropyLoss()

    out = Path(args.output)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / "pretrained.pt", model, step=0)

    curves: List[Dict[str, Any]] = []
    step = 0
    while step < args.max_steps:
        for x, y in train_loader:
            x = x.to(device)
            y_cls = y["y"].to(device) if isinstance(y, dict) else y.to(device)
            model.train()
            logits = model(x)
            loss = criterion(logits, y_cls)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == args.max_steps:
                val_metrics = evaluate(model, val_loader, device)
                curves.append({"step": step, **val_metrics})

            if step % args.snapshot_every == 0 or step == args.max_steps:
                val_metrics = evaluate(model, val_loader, device)
                save_checkpoint(out / "snapshots" / f"ckpt_{step:05d}.pt", model, step=step, metrics=val_metrics)

            if step >= args.max_steps:
                break

    test_metrics = evaluate(model, test_loader, device)
    save_checkpoint(out / "final.pt", model, step=step, metrics=test_metrics)
    write_json(out / "metrics.json", {"test": test_metrics, "steps": step, "curve": curves})
    print(f"Done. Metrics: {test_metrics}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune a model on a task and log trajectories.")
    p.add_argument("--task", required=True)
    p.add_argument("--model", default="synthetic-mlp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_on_task(args)
