from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def infer_steps_from_snapshots(snapshot_dir: Path) -> np.ndarray:
    ckpts = sorted(snapshot_dir.glob("ckpt_*.pt"))
    steps = []
    for ckpt in ckpts:
        stem = ckpt.stem
        try:
            steps.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    if not steps:
        return np.arange(0)
    return np.array(steps)


def _align_xy(steps: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(steps) == len(y):
        return steps, y
    if len(steps) == 0:
        return np.arange(len(y)), y
    idx = np.linspace(0, len(steps) - 1, num=len(y)).round().astype(int)
    return steps[idx], y


def plot_trajectory_2d(points: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(points.shape[0])
    plt.figure(figsize=(6, 5))
    plt.scatter(points[:, 0], points[:, 1], c=t, cmap="viridis", s=30)
    plt.plot(points[:, 0], points[:, 1], alpha=0.5)
    plt.colorbar(label="Snapshot index")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Weight trajectory PCA")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_accuracy_curve(steps: np.ndarray, target: np.ndarray, control: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sx, target = _align_xy(steps, target)
    sx2, control = _align_xy(steps, control)

    plt.figure(figsize=(6, 4))
    plt.plot(sx, target, label="Target accuracy")
    plt.plot(sx2, control, label="Control accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if args.mode == "trajectory":
        points = np.load(args.points)
        plot_trajectory_2d(points, output)
    else:
        target = np.load(args.target)
        control = np.load(args.control)

        if args.steps:
            steps = np.load(args.steps)
        elif args.snapshot_dir:
            steps = infer_steps_from_snapshots(Path(args.snapshot_dir))
        else:
            steps = np.arange(max(len(target), len(control)))

        plot_accuracy_curve(steps, target, control, output)

    print(f"Saved figure to {output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualization helpers for shortcut-patcher project.")
    p.add_argument("--mode", choices=["trajectory", "accuracy"], required=True)
    p.add_argument("--points")
    p.add_argument("--steps", default=None)
    p.add_argument("--snapshot-dir", default=None, help="Infer step numbers from snapshot filenames.")
    p.add_argument("--target")
    p.add_argument("--control")
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
