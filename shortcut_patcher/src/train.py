from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

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
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return int(model.fc.out_features)
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        last = model.net[-1]
        if isinstance(last, nn.Linear):
            return int(last.out_features)
    return None


def patch_model_head(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        last = model.net[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            model.net[-1] = nn.Linear(in_features, num_classes)
            return model
    raise ValueError("Could not patch model head for this architecture")


def _extract_group(batch_y: Any) -> Optional[torch.Tensor]:
    if isinstance(batch_y, dict) and "group" in batch_y:
        g = batch_y["group"]
        if torch.is_tensor(g):
            return g.long()
    return None


def _collect_group_ids(dataset) -> Optional[List[int]]:
    if hasattr(dataset, "group") and torch.is_tensor(getattr(dataset, "group")):
        return [int(v) for v in dataset.group.detach().cpu().tolist()]

    if hasattr(dataset, "rows"):
        rows = getattr(dataset, "rows")
        if rows and isinstance(rows[0], dict) and ("y" in rows[0]) and ("place" in rows[0]):
            y_map = getattr(dataset, "y_map", None)
            gids: List[int] = []
            for r in rows:
                y_raw = int(r["y"])
                y = int(y_map[y_raw]) if isinstance(y_map, dict) else y_raw
                place = int(r["place"])
                gids.append(2 * y + place)
            return gids

    return None


def _build_group_balanced_loader(loader: DataLoader, batch_size: int) -> Optional[DataLoader]:
    gids = _collect_group_ids(loader.dataset)
    if not gids:
        return None

    counts = Counter(gids)
    if len(counts) < 2:
        return None

    sample_weights = torch.tensor([1.0 / counts[g] for g in gids], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
    )


def _groupdro_loss(
    per_sample_loss: torch.Tensor,
    groups: torch.Tensor,
    adv_probs: Dict[int, float],
    eta: float,
) -> tuple[torch.Tensor, Dict[int, float]]:
    device = per_sample_loss.device
    present_groups = torch.unique(groups.detach()).tolist()

    # initialize unseen groups in adv distribution
    for g in present_groups:
        gi = int(g)
        if gi not in adv_probs:
            adv_probs[gi] = 1.0

    # compute mean loss per present group
    group_losses: Dict[int, torch.Tensor] = {}
    for g in present_groups:
        gi = int(g)
        mask = groups == gi
        group_losses[gi] = per_sample_loss[mask].mean()

    # multiplicative updates on present groups
    for gi, gl in group_losses.items():
        adv_probs[gi] *= float(torch.exp(eta * gl.detach()).item())

    # normalize
    z = sum(adv_probs.values()) + 1e-12
    for gi in list(adv_probs.keys()):
        adv_probs[gi] /= z

    # robust objective over present groups (with normalized present mass)
    present_mass = sum(adv_probs[int(g)] for g in present_groups) + 1e-12
    obj = torch.zeros((), device=device)
    for gi, gl in group_losses.items():
        w = adv_probs[gi] / present_mass
        obj = obj + float(w) * gl

    return obj, adv_probs


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
        args.task,
        args.batch_size,
        spurious_strength=args.spurious_strength,
        num_workers=args.num_workers,
    )

    if args.group_balanced_sampler:
        maybe_balanced = _build_group_balanced_loader(train_loader, args.batch_size)
        if maybe_balanced is not None:
            train_loader = maybe_balanced
            logging.info("enabled group-balanced sampler")
        else:
            logging.info("group-balanced sampler requested but unavailable for this dataset")

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
        model = patch_model_head(model, inferred_classes).to(device)
    else:
        logging.info("model head classes=%s inferred=%d", str(current_classes), inferred_classes)

    # Optimizer + warmup/cosine schedule
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: warmup_cosine_lambda(s, args.warmup_steps, args.max_steps),
    )
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Save "pretrained" checkpoint (post head patch, to keep shapes consistent)
    save_checkpoint(outdir / "pretrained.pt", model, step=0)

    curves: List[Dict[str, Any]] = []
    step = 0
    printed_sanity = False
    best_val = -1.0
    best_step = 0
    best_metrics: Dict[str, Any] = {}
    adv_probs: Dict[int, float] = {}

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

            per_loss = criterion(logits, y)
            if args.method == "groupdro":
                g = _extract_group(y_raw)
                if g is not None:
                    g = g.to(device)
                    loss, adv_probs = _groupdro_loss(per_loss, g, adv_probs, args.groupdro_eta)
                else:
                    loss = per_loss.mean()
            else:
                loss = per_loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1

            # Val logging curve
            if step % args.log_every == 0 or step == args.max_steps:
                val_metrics = evaluate(model, val_loader, device)
                objective_metric = float(val_metrics.get("worst_group_accuracy", val_metrics.get("accuracy", 0.0)))
                curves.append({
                    "step": step,
                    "method": args.method,
                    **val_metrics,
                })
                logging.info(
                    "step %d lr=%.3g train_loss=%.4f val_acc=%.4f val_worst=%.4f val_loss=%.4f",
                    step,
                    float(optimizer.param_groups[0]["lr"]),
                    float(loss.item()),
                    float(val_metrics.get("accuracy", 0.0)),
                    float(val_metrics.get("worst_group_accuracy", float("nan"))),
                    float(val_metrics.get("loss", 0.0)),
                )

                if objective_metric > best_val:
                    best_val = objective_metric
                    best_step = step
                    best_metrics = dict(val_metrics)
                    save_checkpoint(outdir / "best.pt", model, step=step, metrics=val_metrics)

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

    best_test_metrics: Dict[str, Any] = {}
    if (outdir / "best.pt").exists():
        best_payload = torch.load(outdir / "best.pt", map_location="cpu")
        best_sd = best_payload["state_dict"] if isinstance(best_payload, dict) and "state_dict" in best_payload else best_payload
        model.load_state_dict(best_sd, strict=False)
        best_test_metrics = evaluate(model, test_loader, device)

    # Single metrics.json (with curve preserved)
    write_json(
        outdir / "metrics.json",
        {
            "method": args.method,
            "test": test_metrics,
            "best": {
                "step": best_step,
                "val": best_metrics,
                "test": best_test_metrics,
            },
            "steps": step,
            "curve": curves,
        },
    )
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
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--method", choices=["erm", "groupdro"], default="groupdro")
    p.add_argument("--groupdro-eta", type=float, default=0.1)
    p.add_argument("--group-balanced-sampler", action="store_true")

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

