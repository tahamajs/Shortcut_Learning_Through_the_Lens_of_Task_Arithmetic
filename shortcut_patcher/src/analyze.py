from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from utils import flatten_state_dict, load_state_dict


def load_snapshots(snapshot_dir: Path) -> np.ndarray:
    files = sorted(snapshot_dir.glob("ckpt_*.pt"))
    if not files:
        raise FileNotFoundError(f"No snapshots found in {snapshot_dir}")
    vectors: List[np.ndarray] = []
    for file in files:
        state = load_state_dict(file)
        vectors.append(flatten_state_dict(state))
    return np.stack(vectors)


def run_pca(weight_snapshots: np.ndarray, n_components: int = 2) -> PCA:
    return PCA(n_components=n_components).fit(weight_snapshots)


def run_cca(x: np.ndarray, y: np.ndarray, n_components: int = 10) -> CCA:
    cca = CCA(n_components=min(n_components, x.shape[1], y.shape[1]))
    cca.fit(x, y)
    return cca


def train_probe(features: np.ndarray, labels: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=500).fit(features, labels)
    return clf.score(features, labels)


def gradient_alignment(trajectory: np.ndarray, final_task_vector: np.ndarray) -> np.ndarray:
    deltas = trajectory[1:] - trajectory[:-1]
    norm_v = np.linalg.norm(final_task_vector) + 1e-8
    sims = []
    for delta in deltas:
        sims.append(float(np.dot(delta, final_task_vector) / ((np.linalg.norm(delta) + 1e-8) * norm_v)))
    return np.array(sims)


def main(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "pca":
        arr = load_snapshots(Path(args.trajectory))
        pca = run_pca(arr, n_components=args.n_components)
        payload = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
        out.write_text(str(payload))
        if args.save_model:
            joblib.dump(pca, out.with_suffix(".joblib"))

    elif args.method == "alignment":
        arr = load_snapshots(Path(args.trajectory))
        final = flatten_state_dict(load_state_dict(Path(args.final_vector)))
        sims = gradient_alignment(arr, final)
        np.save(out.with_suffix(".npy"), sims)
        out.write_text(f"mean_alignment={float(np.mean(sims)):.4f}\n")

    elif args.method == "probe":
        data = np.load(args.features)
        labels = np.load(args.labels)
        score = train_probe(data, labels)
        out.write_text(f"probe_train_acc={score:.4f}\n")

    elif args.method == "cca":
        x = np.load(args.features)
        y = np.load(args.features_b)
        cca = run_cca(x, y, args.n_components)
        x_c, y_c = cca.transform(x, y)
        corr = [float(np.corrcoef(x_c[:, i], y_c[:, i])[0, 1]) for i in range(x_c.shape[1])]
        out.write_text(f"cca_correlations={corr}\n")

    else:
        raise ValueError(f"Unknown method {args.method}")

    print(f"Wrote analysis output to {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze weight trajectories and representations.")
    p.add_argument("--method", choices=["pca", "alignment", "probe", "cca"], required=True)
    p.add_argument("--trajectory")
    p.add_argument("--final-vector")
    p.add_argument("--features")
    p.add_argument("--features-b")
    p.add_argument("--labels")
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
