from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import build_dataloaders, evaluate, load_pretrained, save_checkpoint, set_seed, write_json


def train_on_task(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = load_pretrained(args.model).to(device)
    train_loader, val_loader, test_loader = build_dataloaders(args.task, args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps))
    criterion = nn.CrossEntropyLoss()

    out = Path(args.output)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / "pretrained.pt", model, step=0)

    step = 0
    while step < args.max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.snapshot_every == 0 or step == args.max_steps:
                val_metrics = evaluate(model, val_loader, device)
                save_checkpoint(out / "snapshots" / f"ckpt_{step:05d}.pt", model, step=step, metrics=val_metrics)

            if step >= args.max_steps:
                break

    test_metrics = evaluate(model, test_loader, device)
    save_checkpoint(out / "final.pt", model, step=step, metrics=test_metrics)
    write_json(out / "metrics.json", {"test": test_metrics, "steps": step})
    print(f"Done. Metrics: {test_metrics}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune a model on a task and log trajectories.")
    p.add_argument("--task", required=True)
    p.add_argument("--model", default="synthetic-mlp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_on_task(args)
