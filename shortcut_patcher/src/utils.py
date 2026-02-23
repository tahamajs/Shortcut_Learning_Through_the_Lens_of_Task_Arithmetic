from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TaskConfig:
    task: str
    model: str
    seed: int = 42
    batch_size: int = 128
    max_steps: int = 2000


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
) -> TensorDataset:
    x = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    # Add a synthetic spurious shortcut signal in the first feature dimensions.
    shortcut_signal = torch.nn.functional.one_hot(y, n_classes).float() @ torch.randn(n_classes, min(32, n_features))
    x[:, : shortcut_signal.shape[1]] += spurious_strength * shortcut_signal
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


def load_pretrained(model_name: str, in_dim: int = 256, n_classes: int = 10) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[model_name](in_dim=in_dim, n_classes=n_classes)
    return model


def build_dataloaders(task_name: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    strength = 1.5 if "shortcut" in task_name.lower() else 0.7
    train = synthetic_classification_dataset(4096, spurious_strength=strength)
    val = synthetic_classification_dataset(1024, spurious_strength=strength)
    test = synthetic_classification_dataset(1024, spurious_strength=strength)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += loss.item() * y.numel()
    return {"accuracy": correct / max(total, 1), "loss": loss_sum / max(total, 1)}


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


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def flatten_state_dict(state_dict: Dict[str, torch.Tensor], include_keys: Iterable[str] | None = None) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for k, v in state_dict.items():
        if include_keys is not None and k not in include_keys:
            continue
        vectors.append(v.detach().cpu().float().reshape(-1).numpy())
    return np.concatenate(vectors)
