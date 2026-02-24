from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unrecognized checkpoint format at {path}")


def _flatten_state_dict(
    state_dict: Dict[str, torch.Tensor],
    include_keys: Optional[List[str]] = None,
) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for k, v in state_dict.items():
        if include_keys is not None and k not in include_keys:
            continue
        vecs.append(v.detach().cpu().float().reshape(-1).numpy())
    if not vecs:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(vecs, axis=0)


def _list_snapshots(traj_dir: Path) -> List[Path]:
    # Accept ckpt_00050.pt etc
    snaps = list(traj_dir.glob("ckpt_*.pt"))
    def step_of(p: Path) -> int:
        m = re.search(r"ckpt_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    snaps = [p for p in snaps if step_of(p) >= 0]
    snaps.sort(key=step_of)
    return snaps


def run_pca(traj_dir: Path, n_components: int, output: Path) -> None:
    from sklearn.decomposition import PCA  # local import so file loads even if sklearn missing

    snaps = _list_snapshots(traj_dir)
    if not snaps:
        raise FileNotFoundError(f"No snapshots found in {traj_dir} (expected ckpt_*.pt)")

    X = []
    steps = []
    for p in snaps:
        sd = _load_state_dict(p)
        X.append(_flatten_state_dict(sd))
        m = re.search(r"ckpt_(\d+)\.pt$", p.name)
        steps.append(int(m.group(1)) if m else -1)

    # Stack
    X = np.stack(X, axis=0)  # (T, D)

    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

    result = {
        "n_snapshots": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "steps": steps,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "projection": Z.tolist(),  # useful for plotting later
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote analysis output to {output}")


def run_alignment(traj_dir: Path, final_vector_path: Path, output: Path) -> None:
    snaps = _list_snapshots(traj_dir)
    if len(snaps) < 2:
        raise ValueError("Need at least 2 snapshots for alignment.")

    v_sd = _load_state_dict(final_vector_path)
    v = _flatten_state_dict(v_sd)
    v_norm = np.linalg.norm(v) + 1e-12

    alignments: List[float] = []
    steps: List[int] = []

    prev = _flatten_state_dict(_load_state_dict(snaps[0]))
    prev_step = int(re.search(r"ckpt_(\d+)\.pt$", snaps[0].name).group(1))

    for p in snaps[1:]:
        cur = _flatten_state_dict(_load_state_dict(p))
        cur_step = int(re.search(r"ckpt_(\d+)\.pt$", p.name).group(1))

        delta = cur - prev
        d_norm = np.linalg.norm(delta) + 1e-12
        cos = float(np.dot(delta, v) / (d_norm * v_norm))

        alignments.append(cos)
        steps.append(cur_step)

        prev = cur
        prev_step = cur_step

    result = {
        "mean_alignment": float(np.mean(alignments)) if alignments else 0.0,
        "steps": steps,
        "alignment": alignments,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote analysis output to {output}")


def run_probe(features_path: Path, labels_path: Path, output: Path, save_model: bool) -> None:
    from sklearn.linear_model import LogisticRegression
    import joblib

    X = np.load(features_path)
    y = np.load(labels_path)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    acc = float(clf.score(X, y))

    result: Dict[str, Any] = {"train_accuracy": acc, "n": int(len(y)), "dim": int(X.shape[1])}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    if save_model:
        joblib.dump(clf, output.with_suffix(".joblib"))

    print(f"Wrote analysis output to {output}")


def run_cca(features_a: Path, features_b: Path, n_components: int, output: Path) -> None:
    from sklearn.cross_decomposition import CCA

    A = np.load(features_a)
    B = np.load(features_b)
    cca = CCA(n_components=n_components)
    A_c, B_c = cca.fit_transform(A, B)

    # correlation per component
    corrs = []
    for i in range(n_components):
        a = A_c[:, i]
        b = B_c[:, i]
        corr = float(np.corrcoef(a, b)[0, 1])
        corrs.append(corr)

    result = {"n_components": n_components, "corrs": corrs, "mean_corr": float(np.mean(corrs))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote analysis output to {output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze weight trajectories and representations.")
    p.add_argument("--method", choices=["pca", "alignment", "probe", "cca"], required=True)

    p.add_argument("--trajectory", type=str, default=None, help="Directory containing ckpt_*.pt snapshots")
    p.add_argument("--final-vector", type=str, default=None, help="Path to final task vector .pt")

    p.add_argument("--features", type=str, default=None, help="Features .npy for probe or cca A")
    p.add_argument("--features-b", type=str, default=None, help="Features .npy for cca B")
    p.add_argument("--labels", type=str, default=None, help="Labels .npy for probe")

    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--output", type=str, required=True)
    return p


def main(args: argparse.Namespace) -> None:
    method = args.method
    out = Path(args.output)

    if method == "pca":
        if args.trajectory is None:
            raise ValueError("--trajectory is required for pca")
        run_pca(Path(args.trajectory), args.n_components, out)

    elif method == "alignment":
        if args.trajectory is None or args.final_vector is None:
            raise ValueError("--trajectory and --final-vector are required for alignment")
        run_alignment(Path(args.trajectory), Path(args.final_vector), out)

    elif method == "probe":
        if args.features is None or args.labels is None:
            raise ValueError("--features and --labels are required for probe")
        run_probe(Path(args.features), Path(args.labels), out, args.save_model)

    elif method == "cca":
        if args.features is None or args.features_b is None:
            raise ValueError("--features and --features-b are required for cca")
        run_cca(Path(args.features), Path(args.features_b), args.n_components, out)

    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    main(build_parser().parse_args())