from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Optional torchvision only needed for real-image dataset / resnet model
try:
    from PIL import Image
    from torchvision import transforms, models
except Exception:
    Image = None
    transforms = None
    models = None

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
    from torchvision import transforms
except Exception:
    Image = None
    transforms = None


def _resolve_img_path(cub_root: Path, rel_path: str) -> Path:
    """
    Tries:
      data/cub/<rel_path>
      data/cub/images/<rel_path>
    """
    rel_path = rel_path.lstrip("/")  # just in case
    p1 = cub_root / rel_path
    if p1.exists():
        return p1
    p2 = cub_root / "images" / rel_path
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find image: {p1} OR {p2}")


class WaterbirdsFromMetadata(Dataset):
    """
    Reads Waterbirds-style metadata.csv with columns:
      img_id,img_filename,y,split,place,place_filename

    Returns:
      x: image tensor [3,224,224]
      label dict: {"y": y, "spurious": place, "group": 2*y + place}
    """

    def __init__(self, cub_root: Path, split: int, transform=None) -> None:
        if Image is None:
            raise RuntimeError("Need pillow + torchvision to load image datasets.")

        self.cub_root = cub_root
        self.transform = transform

        meta = cub_root / "metadata.csv"
        if not meta.exists():
            raise FileNotFoundError(f"Expected {meta}")

        # Read all rows
        rows: List[Dict[str, str]] = []
        with meta.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        # Filter by split
        self.rows = [r for r in rows if int(r["split"]) == int(split)]
        if not self.rows:
            raise ValueError(f"No rows found for split={split} in {meta}")

        # Optional: remap y to 0..C-1 if it's not already (safe for weird label encodings)
        ys = sorted({int(r["y"]) for r in rows})
        self.y_map = {y: i for i, y in enumerate(ys)}  # stable mapping across splits

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r = self.rows[idx]

        img_path = _resolve_img_path(self.cub_root, r["img_filename"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            x = self.transform(img)
        else:
            x = transforms.ToTensor()(img)

        y_raw = int(r["y"])
        y = self.y_map[y_raw]  # ensure 0..C-1
        place = int(r["place"])  # spurious attribute (0/1 usually)

        group = 2 * y + place  # works for binary y; for multi-class it's still a valid grouping scheme

        label = {
            "y": torch.tensor(y, dtype=torch.long),
            "spurious": torch.tensor(place, dtype=torch.long),
            "group": torch.tensor(group, dtype=torch.long),
        }
        return x, label
# -----------------------------
# Config
# -----------------------------
@dataclass
class TaskConfig:
    task: str
    model: str
    seed: int = 42
    batch_size: int = 128
    max_steps: int = 2000


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Synthetic datasets (fallback)
# -----------------------------
def synthetic_classification_dataset(
    n_samples: int = 4096,
    n_features: int = 256,
    n_classes: int = 10,
    spurious_strength: float = 1.0,
    *,
    shortcut_matrix: torch.Tensor | None = None,
    seed: int | None = None,
) -> TensorDataset:
    g = None
    if seed is not None:
        g = torch.Generator().manual_seed(int(seed))

    x = torch.randn(n_samples, n_features, generator=g)
    y = torch.randint(0, n_classes, (n_samples,), generator=g)

    k = min(32, n_features)
    if shortcut_matrix is None:
        shortcut_matrix = torch.randn(n_classes, k, generator=g)

    shortcut_signal = torch.nn.functional.one_hot(y, n_classes).float() @ shortcut_matrix
    x[:, :k] += spurious_strength * shortcut_signal
    return TensorDataset(x, y)


class ShortcutDataset(Dataset):
    """Synthetic dataset with dict labels: y/spurious/group."""
    def __init__(
        self,
        n_samples: int = 4096,
        n_features: int = 256,
        n_classes: int = 10,
        spurious_strength: float = 1.0,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        g = None
        if seed is not None:
            g = torch.Generator().manual_seed(int(seed))

        self.x = torch.randn(n_samples, n_features, generator=g)
        self.y = torch.randint(0, n_classes, (n_samples,), generator=g)

        corr_prob = spurious_strength / (spurious_strength + 1.0)
        parity = (self.y % 2).long()

        if g is not None:
            s_rand = torch.bernoulli(torch.full((n_samples,), 0.5)).long()
            choose_corr = torch.rand(n_samples, generator=g) < corr_prob
        else:
            s_rand = torch.bernoulli(torch.full((n_samples,), 0.5)).long()
            choose_corr = torch.rand(n_samples) < corr_prob

        self.s = torch.where(choose_corr, parity, s_rand)
        self.group = self.y * 2 + self.s

        core_k = min(32, n_features)
        spur_k = min(16, n_features)
        core_matrix = torch.randn(n_classes, core_k, generator=g)
        spur_matrix = torch.randn(2, spur_k, generator=g)

        core_signal = torch.nn.functional.one_hot(self.y, n_classes).float() @ core_matrix
        spur_signal = torch.nn.functional.one_hot(self.s, 2).float() @ spur_matrix

        self.x[:, :core_k] += core_signal
        self.x[:, :spur_k] += spurious_strength * spur_signal

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int):
        return self.x[idx], {"y": self.y[idx], "spurious": self.s[idx], "group": self.group[idx]}


# -----------------------------
# Real dataset loader (CSV + images in data/cub)
# -----------------------------
class CSVSplittedImageDataset(Dataset):
    """
    Reads: root / metadata.csv with columns:
      path, split, y, (optional) spurious, (optional) group

    Images are read from: root / path
    """
    def __init__(self, root: Path, split: int, transform=None) -> None:
        if Image is None:
            raise RuntimeError("PIL/torchvision not available. Install pillow+torchvision to use real image datasets.")

        self.root = root
        self.transform = transform
        meta = root / "metadata.csv"
        if not meta.exists():
            raise FileNotFoundError(f"Expected metadata at {meta}")

        rows: List[Dict[str, str]] = []
        with meta.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        # filter split
        keep = []
        for r in rows:
            if "split" not in r:
                raise ValueError("metadata.csv must contain a 'split' column (0=train,1=val,2=test).")
            if int(r["split"]) == int(split):
                keep.append(r)

        if not keep:
            raise ValueError(f"No rows found for split={split} in {meta}")

        self.rows = keep

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        rel = r.get("path")
        if rel is None:
            raise ValueError("metadata.csv must have a 'path' column.")
        img_path = self.root / rel
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            x = self.transform(img)
        else:
            x = transforms.ToTensor()(img)

        y = int(r["y"])
        sp = int(r.get("spurious", "0"))
        group = int(r.get("group", str(2 * y + sp)))

        label = {
            "y": torch.tensor(y, dtype=torch.long),
            "spurious": torch.tensor(sp, dtype=torch.long),
            "group": torch.tensor(group, dtype=torch.long),
        }
        return x, label


# -----------------------------
# Models
# -----------------------------
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


def _make_resnet18(num_classes: int) -> nn.Module:
    if models is None:
        raise RuntimeError("torchvision not available. Install torchvision to use resnet18.")
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


MODEL_REGISTRY = {
    "synthetic-mlp": ("vector", MLPClassifier),
    "resnet18": ("image", _make_resnet18),
}


def load_pretrained(model_name: str, in_dim: int = 256, n_classes: int = 10) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    kind, ctor = MODEL_REGISTRY[model_name]
    if model_name == "synthetic-mlp":
        return ctor(in_dim=in_dim, n_classes=n_classes)
    # resnet18
    return ctor(n_classes)


# -----------------------------
# Dataloaders (synthetic OR real)
# -----------------------------
def build_dataloaders(
    task_name: str,
    batch_size: int,
    spurious_strength: float | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    name = task_name.lower()

    # ✅ Prefer real dataset if Waterbirds metadata exists
    cub_root = Path("data") / "cub"
    meta = cub_root / "metadata.csv"
    if meta.exists() and ("waterbirds" in name or name.endswith("shortcut")):
        if transforms is None:
            raise RuntimeError("Need torchvision installed for image transforms.")

        tfm = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        train = WaterbirdsFromMetadata(cub_root, split=0, transform=tfm)
        val   = WaterbirdsFromMetadata(cub_root, split=1, transform=tfm)
        test  = WaterbirdsFromMetadata(cub_root, split=2, transform=tfm)

        return (
            DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0),
        )

    # 🔁 Otherwise fall back to your synthetic logic
    if spurious_strength is None:
        strength = 1.5 if "shortcut" in name else 0.7
    else:
        strength = float(spurious_strength)

    if name.endswith("shortcut"):
        train = ShortcutDataset(4096, spurious_strength=strength, seed=0)
        val = ShortcutDataset(1024, spurious_strength=strength, seed=1)
        test = ShortcutDataset(1024, spurious_strength=strength, seed=2)
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size, shuffle=False),
            DataLoader(test, batch_size=batch_size, shuffle=False),
        )

    # non-shortcut synthetic
    n_features = 256
    n_classes = 10
    k = min(32, n_features)
    shortcut_matrix = torch.randn(n_classes, k)
    train = synthetic_classification_dataset(4096, n_features, n_classes, strength, shortcut_matrix=shortcut_matrix, seed=0)
    val   = synthetic_classification_dataset(1024, n_features, n_classes, strength, shortcut_matrix=shortcut_matrix, seed=1)
    test  = synthetic_classification_dataset(1024, n_features, n_classes, strength, shortcut_matrix=shortcut_matrix, seed=2)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )

# -----------------------------
# Evaluation (group-aware)
# -----------------------------
def _to_device_labels(y: Any, device: torch.device) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    if isinstance(y, dict):
        y_dict = {k: v.to(device) for k, v in y.items()}
        return y_dict["y"], y_dict
    return y.to(device), None


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0

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
                groups = y_meta["group"].detach().cpu().tolist()
                matches = (pred == y_cls).detach().cpu().tolist()
                for g, m in zip(groups, matches):
                    g = int(g)
                    group_total[g] = group_total.get(g, 0) + 1
                    group_correct[g] = group_correct.get(g, 0) + int(m)

    metrics: Dict[str, Any] = {
        "accuracy": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
    }

    if group_total:
        group_acc = {str(k): group_correct[k] / group_total[k] for k in sorted(group_total)}
        metrics["group_accuracy"] = group_acc
        metrics["worst_group_accuracy"] = float(min(group_acc.values()))

    return metrics


# -----------------------------
# Checkpoints + utilities
# -----------------------------
def save_checkpoint(path: Path, model: nn.Module, step: int, metrics: Dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": int(step), "state_dict": model.state_dict(), "metrics": metrics or {}}, path)


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Bad checkpoint format at {path}")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def flatten_state_dict(state_dict: Dict[str, torch.Tensor], include_keys: Iterable[str] | None = None) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for k, v in state_dict.items():
        if include_keys is not None and k not in include_keys:
            continue
        vecs.append(v.detach().cpu().float().reshape(-1).numpy())
    return np.concatenate(vecs) if vecs else np.zeros((0,), dtype=np.float32)