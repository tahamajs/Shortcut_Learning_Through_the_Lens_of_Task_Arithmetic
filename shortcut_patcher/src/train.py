from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils import build_dataloaders, evaluate, load_pretrained, save_checkpoint, set_seed, write_json


def setup_logging(log_file: Path | None = None) -> None:
    """Configure logging to both console and file.

    If ``log_file`` is provided the file handler appends to it; otherwise only
    stdout is used.  The training routine will call ``logging.info`` instead of
    ``print`` so messages show up in both places.
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", handlers=handlers)



def warmup_cosine_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())


from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import (
    build_dataloaders,
    evaluate,
    load_pretrained,
    save_checkpoint,
    set_seed,
    write_json,
)


# -----------------------------
# Helpers: label extraction
# -----------------------------
def _extract_label(
    batch_y: Any, label_key: str | None = None, label_index: int = 0
) -> torch.Tensor:
    """
    Extract the training label tensor from whatever the dataloader yields.

    Supports:
      - Tensor y
      - (y, aux...) tuples/lists
      - dicts like {"y": y, "group": g, ...}

    Args:
      label_key: If batch_y is dict-like, use this key (e.g. "y", "label", "class", "group").
      label_index: If batch_y is tuple/list, take batch_y[label_index].

    Returns:
      A torch.Tensor label.
    """
    # Already a tensor
    if torch.is_tensor(batch_y):
        return batch_y

    # Dict-like
    if isinstance(batch_y, dict):
        if label_key is None:
            # try common keys
            for k in ("y", "label", "labels", "class", "target", "group", "spurious"):
                if k in batch_y:
                    return batch_y[k]
            raise ValueError(
                f"batch_y is a dict but no label_key provided and no known keys found. Keys={list(batch_y.keys())}"
            )
        if label_key not in batch_y:
            raise ValueError(f"label_key='{label_key}' not found in batch_y keys={list(batch_y.keys())}")
        return batch_y[label_key]

    # Tuple/list
    if isinstance(batch_y, (tuple, list)):
        if not (0 <= label_index < len(batch_y)):
            raise ValueError(f"label_index={label_index} out of range for batch_y of length {len(batch_y)}")
        y0 = batch_y[label_index]
        if not torch.is_tensor(y0):
            raise ValueError(f"Selected label is not a tensor: type={type(y0)}")
        return y0

    raise ValueError(f"Unsupported batch_y type: {type(batch_y)}")


def _normalize_labels_for_ce(y: torch.Tensor) -> torch.Tensor:
    """
    Make labels compatible with nn.CrossEntropyLoss.

    CE expects:
      - logits shape: (N, C)
      - labels shape: (N,) with dtype long and values in [0..C-1]
    """
    # If y is one-hot or probabilities (N, C), convert to class index
    if y.ndim == 2:
        return y.argmax(dim=1).long()

    # If y is float but 1D, assume it's class indices in float form or binary 0/1 floats
    if y.ndim == 1 and y.dtype.is_floating_point:
        return y.round().long()

    return y.long()


def infer_num_classes(
    loader, label_key: str | None = None, label_index: int = 0, max_batches: int = 10
) -> int:
    """
    Infer number of classes from the labels produced by a loader.

    NOTE: This is only reliable if the loader emits true class labels, not group IDs.
    """
    ys = []
    for i, (_, batch_y) in enumerate(loader):
        y = _extract_label(batch_y, label_key=label_key, label_index=label_index)
        y = y.detach().cpu()
        y = _normalize_labels_for_ce(y)
        ys.append(y)
        if i + 1 >= max_batches:
            break

    y_all = torch.cat(ys, dim=0)
    if y_all.numel() == 0:
        raise ValueError("Could not infer classes: empty labels from loader.")

    # If labels are class indices, max+1 is class count (works if all classes appear)
    n = int(y_all.max().item()) + 1
    if n <= 1:
        raise ValueError(f"Inferred num_classes={n} which is invalid. y min/max = {int(y_all.min())}/{int(y_all.max())}")
    return n


def patch_synthetic_mlp_head(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Replace last linear layer of synthetic-mlp (model.net[-1]) to match num_classes.
    """
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        last = model.net[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            model.net[-1] = nn.Linear(in_features, num_classes)
            return model
    raise ValueError("Expected synthetic-mlp with model.net as nn.Sequential and last layer nn.Linear")


def _get_model_num_classes(model: nn.Module) -> int | None:
    """
    Try to infer current output dimension of the model (for synthetic-mlp).
    """
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        last = model.net[-1]
        if isinstance(last, nn.Linear):
            return int(last.out_features)
    return None


def _sanity_print_once(x: torch.Tensor, y_raw: Any, y: torch.Tensor, logits: torch.Tensor) -> None:
    """
    Prints one-time diagnostics to catch label/shape mismatches.
    """
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
    setup_logging(outdir / "train.log")
    logging.info("starting training task=%s model=%s", args.task, args.model)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = load_pretrained(args.model).to(device)
    train_loader, val_loader, test_loader = build_dataloaders(args.task, args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: warmup_cosine_lambda(s, args.warmup_steps, args.max_steps))

    train_loader, val_loader, test_loader = build_dataloaders(
        args.task, args.batch_size, args.spurious_strength
    )

    # Decide which label to train on (very important for shortcut/group datasets!)
    label_key = args.label_key
    label_index = args.label_index

    # Infer classes from the chosen label
    inferred_classes = infer_num_classes(
        train_loader, label_key=label_key, label_index=label_index, max_batches=10
    )

    current_classes = _get_model_num_classes(model)
    if current_classes is None:
        print("[warn] Could not infer model output classes; skipping head patch.")
        num_classes = inferred_classes
    else:
        num_classes = inferred_classes
        if current_classes != num_classes:
            logging.info("patching head: model_out=%d -> inferred_num_classes=%d", current_classes, num_classes)
            model = patch_synthetic_mlp_head(model, num_classes).to(device)
        else:
            logging.info("model head already matches num_classes=%d", num_classes)

    # Optimizer AFTER head patch
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps))
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

    # Save checkpoint of pretrained (or patched-pretrained)
    save_checkpoint(out / "pretrained.pt", model, step=0)

    step = 0
    printed_sanity = False

    while step < args.max_steps:
        for x, y_raw in train_loader:
            x = x.to(device)
            y = _extract_label(y_raw, label_key=label_key, label_index=label_index).to(device)
            y = _normalize_labels_for_ce(y)

            model.train()
            logits = model(x)

            if (not printed_sanity) and args.sanity:
                _sanity_print_once(x, y_raw, y, logits)
                printed_sanity = True

            # Safety check: ensure logits and labels match
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
                    f"This often means you're training on group IDs, not class labels."
                )

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # log to terminal
            if step % args.log_every == 0 or step == args.max_steps:
                val_metrics = evaluate(model, val_loader, device)
                curves.append({"step": step, **val_metrics})
                logging.info("step %d train_loss=%.4f val_acc=%.4f val_loss=%.4f", step,
                             loss.item(), val_metrics.get('accuracy', 0), val_metrics.get('loss', 0))

            if step % args.snapshot_every == 0 or step == args.max_steps:
                val_metrics = evaluate(model, val_loader, device)
                save_checkpoint(out / "snapshots" / f"ckpt_{step:05d}.pt", model, step=step, metrics=val_metrics)

            if step >= args.max_steps:
                break

    test_metrics = evaluate(model, test_loader, device)
    save_checkpoint(out / "final.pt", model, step=step, metrics=test_metrics)
    write_json(out / "metrics.json", {"test": test_metrics, "steps": step, "curve": curves})
    write_json(out / "metrics.json", {"test": test_metrics, "steps": step})
    logging.info("Done. Metrics: %s", test_metrics)


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
    p.add_argument(
        "--spurious-strength",
        type=float,
        default=None,
        help="override the synthetic shortcut strength used by the dataset",
    )

    # NEW: choose which label field is used for training (critical for shortcut datasets)
    p.add_argument(
        "--label-key",
        type=str,
        default=None,
        help="If dataloader yields a dict for labels, train on this key (e.g. y, class, group).",
    )
    p.add_argument(
        "--label-index",
        type=int,
        default=0,
        help="If dataloader yields a tuple/list for labels, train on this index (default 0).",
    )
    p.add_argument(
        "--sanity",
        action="store_true",
        help="Print one batch x/y/logits diagnostics at start.",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_on_task(args)
