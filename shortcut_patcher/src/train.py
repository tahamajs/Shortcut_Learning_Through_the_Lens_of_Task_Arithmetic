from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils import build_dataloaders, evaluate, load_pretrained, save_checkpoint, set_seed, write_json


# -----------------------------
# Logging
# -----------------------------
def setup_logging(log_file: Path | None = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=handlers,
        force=True,  # ensure reconfig works if called multiple times
    )


# -----------------------------
# LR schedule: warmup + cosine decay
# -----------------------------
def warmup_cosine_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """
    Returns LR multiplier in (0..1].

    - Linear warmup for warmup_steps
    - Cosine decay to 0 afterwards
    """
    if total_steps <= 0:
        return 1.0
    if warmup_steps < 0:
        warmup_steps = 0

    if step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))

    denom = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / denom  # 0..1
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# -----------------------------
# Helpers: label extraction
# -----------------------------
def _extract_label(batch_y: Any, label_key: str | None = None, label_index: int = 0) -> torch.Tensor:
    """
    Extract training label tensor from dataloader output.

    Supports:
      - Tensor y
      - dict labels like {"y":..., "group":..., ...}
      - tuple/list labels like (y, aux...)
    """
    if torch.is_tensor(batch_y):
        return batch_y

    if isinstance(batch_y, dict):
        if label_key is None:
            for k in ("y", "label", "labels", "class", "target"):
                if k in batch_y:
                    return batch_y[k]
            raise ValueError(f"Label dict has no known key. Keys={list(batch_y.keys())}. "
                             f"Pass --label-key to choose one.")
        if label_key not in batch_y:
            raise ValueError(f"--label-key '{label_key}' not found. Keys={list(batch_y.keys())}")
        return batch_y[label_key]

    if isinstance(batch_y, (tuple, list)):
        if not (0 <= label_index < len(batch_y)):
            raise ValueError(f"--label-index {label_index} out of range for label tuple of len={len(batch_y)}")
        y0 = batch_y[label_index]
        if not torch.is_tensor(y0):
            raise ValueError(f"Selected label is not a tensor. type={type(y0)}")
        return y0

    raise ValueError(f"Unsupported label type: {type(batch_y)}")


def _normalize_labels_for_ce(y: torch.Tensor) -> torch.Tensor:
    """
    CrossEntropy expects y: (N,) long with values in [0..C-1].
    """
    if y.ndim == 2:  # one-hot / probs
        return y.argmax(dim=1).long()
    if y.ndim == 1 and y.dtype.is_floating_point:
        return y.round().long()
    return y.long()


def infer_num_classes(loader, label_key: str | None = None, label_index: int = 0, max_batches: int = 10) -> int:
    ys: List[torch.Tensor] = []
    for i, (_, batch_y) in enumerate(loader):
        y = _extract_label(batch_y, label_key=label_key, label_index=label_index)
        y = _normalize_labels_for_ce(y.detach().cpu())
        ys.append(y)
        if i + 1 >= max_batches:
            break

    if not ys:
        raise ValueError("Could not infer num_classes: no labels read from loader.")

    y_all = torch.cat(ys, dim=0)
    n = int(y_all.max().item()) + 1
    if n <= 1:
        raise ValueError(f"Inferred num_classes={n} invalid. y min/max = {int(y_all.min())}/{int(y_all.max())}")
    return n


def _get_model_num_classes(model: nn.Module) -> int | None:
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        last = model.net[-1]
        if isinstance(last, nn.Linear):
            return int(last.out_features)
    return None


def patch_synthetic_mlp_head(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        last = model.net[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            model.net[-1] = nn.Linear(in_features, num_classes)
            return model
    raise ValueError("Expected synthetic-mlp with model.net as nn.Sequential and last layer nn.Linear")


def _select_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    return torch.device("cpu")


def _sanity_print_once(x: torch.Tensor, y_raw: Any, y: torch.Tensor, logits: torch.Tensor) -> None:
    print("[sanity] x:", tuple(x.shape), x.dtype, "min/max:", float(x.min()), float(x.max()))
    if torch.is_tensor(y_raw):
        print("[sanity] y_raw tensor:", tuple(y_raw.shape), y_raw.dtype)
    else:
        print("[sanity] y_raw type:", type(y_raw))

    uniq = torch.unique(y.detach().cpu())
    show = uniq[:50].tolist()
    print("[sanity] y (used for CE):", tuple(y.shape), y.dtype, "min/max:", int(y.min()), int(y.max()))
    print("[sanity] y unique (up to 50):", show, "count:", uniq.numel())
    print("[sanity] logits:", tuple(logits.shape), logits.dtype)


# -----------------------------
# Training
# -----------------------------
def train_on_task(args: argparse.Namespace) -> None:
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "snapshots").mkdir(parents=True, exist_ok=True)

    setup_logging(outdir / "train.log")
    logging.info("starting training task=%s model=%s seed=%d", args.task, args.model, args.seed)

    set_seed(args.seed)
    device = _select_device(args.cpu)
    logging.info("device=%s", device)

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(
        args.task, args.batch_size, spurious_strength=args.spurious_strength
    )

    # Model (initially)
    model = load_pretrained(args.model).to(device)

    # Choose which label to learn
    label_key = args.label_key
    label_index = args.label_index

    # Infer classes based on the chosen label
    inferred_classes = infer_num_classes(train_loader, label_key=label_key, label_index=label_index)
    current_classes = _get_model_num_classes(model)

    if current_classes is not None and current_classes != inferred_classes:
        logging.info("patching head: model_out=%d -> inferred_num_classes=%d", current_classes, inferred_classes)
        model = patch_synthetic_mlp_head(model, inferred_classes).to(device)
    else:
        logging.info("model head classes=%s inferred=%d", str(current_classes), inferred_classes)

    # Optimizer + warmup/cosine schedule
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: warmup_cosine_lambda(s, args.warmup_steps, args.max_steps),
    )
    criterion = nn.CrossEntropyLoss()

    # Save "pretrained" checkpoint (post head patch, to keep shapes consistent)
    save_checkpoint(outdir / "pretrained.pt", model, step=0)

    curves: List[Dict[str, Any]] = []
    step = 0
    printed_sanity = False

    while step < args.max_steps:
        for x, y_raw in train_loader:
            if step >= args.max_steps:
                break

            x = x.to(device)
            y = _extract_label(y_raw, label_key=label_key, label_index=label_index).to(device)
            y = _normalize_labels_for_ce(y)

            model.train()
            logits = model(x)

            if (not printed_sanity) and args.sanity:
                _sanity_print_once(x, y_raw, y, logits)
                printed_sanity = True

            # Safety checks
            if logits.ndim != 2:
                raise ValueError(f"logits must be 2D (N,C). Got shape={tuple(logits.shape)}")
            if y.ndim != 1:
                raise ValueError(f"labels must be 1D (N,). Got shape={tuple(y.shape)}")
            if logits.shape[0] != y.shape[0]:
                raise ValueError(f"batch size mismatch: logits N={logits.shape[0]} vs labels N={y.shape[0]}")
            if y.min().item() < 0 or y.max().item() >= logits.shape[1]:
                raise ValueError(
                    f"Label out of range for CE: y min/max={int(y.min())}/{int(y.max())}, "
                    f"num_classes(logits)={logits.shape[1]}. "
                    f"Likely training on wrong label field (e.g. group IDs). Use --label-key y."
                )

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1

            # Val logging curve
            if step % args.log_every == 0 or step == args.max_steps:
                val_metrics = evaluate(model, val_loader, device)
                curves.append({"step": step, **val_metrics})
                logging.info(
                    "step %d lr=%.3g train_loss=%.4f val_acc=%.4f val_loss=%.4f",
                    step,
                    float(optimizer.param_groups[0]["lr"]),
                    float(loss.item()),
                    float(val_metrics.get("accuracy", 0.0)),
                    float(val_metrics.get("loss", 0.0)),
                )

            # Snapshots
            if step % args.snapshot_every == 0 or step == args.max_steps:
                val_metrics = evaluate(model, val_loader, device)
                save_checkpoint(
                    outdir / "snapshots" / f"ckpt_{step:05d}.pt",
                    model,
                    step=step,
                    metrics=val_metrics,
                )

    # Final test
    test_metrics = evaluate(model, test_loader, device)
    save_checkpoint(outdir / "final.pt", model, step=step, metrics=test_metrics)

    # Single metrics.json (with curve preserved)
    write_json(outdir / "metrics.json", {"test": test_metrics, "steps": step, "curve": curves})
    logging.info("done. test=%s", test_metrics)


# -----------------------------
# CLI
# -----------------------------
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

    p.add_argument("--spurious-strength", type=float, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", required=True)

    # label selection
    p.add_argument(
        "--label-key",
        type=str,
        default=None,
        help="If labels are dict, train on this key (e.g. y, group, spurious). Default: auto-pick common keys.",
    )
    p.add_argument(
        "--label-index",
        type=int,
        default=0,
        help="If labels are tuple/list, train on this index (default 0).",
    )
    p.add_argument("--sanity", action="store_true", help="Print one batch x/y/logits diagnostics at start.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_on_task(args)

