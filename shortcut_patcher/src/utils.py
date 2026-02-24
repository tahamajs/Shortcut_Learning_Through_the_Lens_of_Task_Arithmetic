from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TaskConfig:
    task: str
    model: str
    seed: int = 42
    batch_size: int = 128
    max_steps: int = 2000


class ShortcutDataset(Dataset):
    """Synthetic dataset with class label + spurious/group metadata."""

    def __init__(
        self,
        n_samples: int = 4096,
        n_features: int = 256,
        n_classes: int = 10,
        spurious_strength: float = 1.0,
    ) -> None:
        self.x = torch.randn(n_samples, n_features)
        self.y = torch.randint(0, n_classes, (n_samples,))

        s = torch.bernoulli(torch.full((n_samples,), 0.5)).long()  # spurious attr
        # make spurious correlated with class parity
        parity = (self.y % 2).long()
        self.s = torch.where(torch.rand(n_samples) < spurious_strength / (spurious_strength + 1.0), parity, s)
        self.group = self.y * 2 + self.s

        shortcut_signal = torch.nn.functional.one_hot(self.s, 2).float() @ torch.randn(2, min(16, n_features))
        core_signal = torch.nn.functional.one_hot(self.y, n_classes).float() @ torch.randn(n_classes, min(32, n_features))
        self.x[:, : core_signal.shape[1]] += core_signal
        self.x[:, : shortcut_signal.shape[1]] += spurious_strength * shortcut_signal

    def __len__(self) -> int:
        return self.y.numel()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        label = {
            "y": self.y[idx],
            "spurious": self.s[idx],
            "group": self.group[idx],
        }
        return self.x[idx], label
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synthetic_classification_dataset(
    n_samples: int = 4096,
    n_features: int = 256,
    n_classes: int = 10,
    spurious_strength: float = 1.0,
    *,
    shortcut_matrix: torch.Tensor | None = None,
    seed: int | None = None,
) -> TensorDataset:
    # Make data generation reproducible if desired
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    x = torch.randn(n_samples, n_features, generator=g)
    y = torch.randint(0, n_classes, (n_samples,), generator=g)

    k = min(32, n_features)

    # FIX: keep the shortcut mapping consistent across splits
    if shortcut_matrix is None:
        shortcut_matrix = torch.randn(n_classes, k, generator=g)

    shortcut_signal = torch.nn.functional.one_hot(y, n_classes).float() @ shortcut_matrix
    x[:, :k] += spurious_strength * shortcut_signal

    return TensorDataset(x, y)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int = 256, hidden: int = 512, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MODEL_REGISTRY = {
    "synthetic-mlp": MLPClassifier,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained(model_name: str, in_dim: int = 256, n_classes: int = 10) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](in_dim=in_dim, n_classes=n_classes)


def build_dataloaders(task_name: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if task_name.lower().endswith("shortcut") or task_name in {"Waterbirds", "CelebA"}:
        strength = 1.5
    else:
        strength = 0.7

    train = ShortcutDataset(4096, spurious_strength=strength)
    val = ShortcutDataset(1024, spurious_strength=strength)
    test = ShortcutDataset(1024, spurious_strength=strength)
def load_pretrained(model_name: str, in_dim: int = 256, n_classes: int = 10) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[model_name](in_dim=in_dim, n_classes=n_classes)
    return model


def build_dataloaders(
    task_name: str, batch_size: int, spurious_strength: float | None = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if spurious_strength is None:
        strength = 1.5 if "shortcut" in task_name.lower() else 0.7
    else:
        strength = spurious_strength

    n_features = 256
    n_classes = 10
    k = min(32, n_features)

    # FIX: one shared shortcut mapping across train/val/test
    shortcut_matrix = torch.randn(n_classes, k)

    # Optional: also keep data generation reproducible per split
    train = synthetic_classification_dataset(4096, n_features=n_features, n_classes=n_classes,
                                            spurious_strength=strength, shortcut_matrix=shortcut_matrix, seed=0)
    val   = synthetic_classification_dataset(1024, n_features=n_features, n_classes=n_classes,
                                            spurious_strength=strength, shortcut_matrix=shortcut_matrix, seed=1)
    test  = synthetic_classification_dataset(1024, n_features=n_features, n_classes=n_classes,
                                            spurious_strength=strength, shortcut_matrix=shortcut_matrix, seed=2)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


def _to_device_labels(y: Any, device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor] | None]:
    if isinstance(y, dict):
        y_dict = {k: v.to(device) for k, v in y.items()}
        return y_dict["y"], y_dict
    return y.to(device), None


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    group_correct: Dict[int, int] = {}
    group_total: Dict[int, int] = {}

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_cls, y_meta = _to_device_labels(y, device)
            logits = model(x)
            loss = ce(logits, y_cls)
            pred = logits.argmax(dim=-1)

            correct += (pred == y_cls).sum().item()
            total += y_cls.numel()
            loss_sum += loss.item() * y_cls.numel()

            if y_meta is not None and "group" in y_meta:
                groups = y_meta["group"].detach().cpu().numpy().astype(int)
                matches = (pred == y_cls).detach().cpu().numpy().astype(int)
                for g, m in zip(groups, matches):
                    group_total[g] = group_total.get(g, 0) + 1
                    group_correct[g] = group_correct.get(g, 0) + int(m)

    metrics: Dict[str, Any] = {
        "accuracy": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
    }

    if group_total:
        group_acc = {str(k): group_correct[k] / group_total[k] for k in sorted(group_total)}
        metrics["group_accuracy"] = group_acc
        metrics["worst_group_accuracy"] = min(group_acc.values())

    return metrics


def save_checkpoint(path: Path, model: nn.Module, step: int, metrics: Dict[str, float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "state_dict": model.state_dict(),
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if "state_dict" in payload:
        return payload["state_dict"]
    return payload


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def flatten_state_dict(state_dict: Dict[str, torch.Tensor], include_keys: Iterable[str] | None = None) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for k, v in state_dict.items():
        if include_keys is not None and k not in include_keys:
            continue
        vectors.append(v.detach().cpu().float().reshape(-1).numpy())
    return np.concatenate(vectors)
