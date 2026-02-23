from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    plt.figure(figsize=(6, 4))
    plt.plot(steps, target, label="Target accuracy")
    plt.plot(steps, control, label="Control accuracy")
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
        steps = np.load(args.steps)
        target = np.load(args.target)
        control = np.load(args.control)
        plot_accuracy_curve(steps, target, control, output)

    print(f"Saved figure to {output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualization helpers for shortcut-patcher project.")
    p.add_argument("--mode", choices=["trajectory", "accuracy"], required=True)
    p.add_argument("--points")
    p.add_argument("--steps")
    p.add_argument("--target")
    p.add_argument("--control")
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
