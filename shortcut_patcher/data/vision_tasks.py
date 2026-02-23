from __future__ import annotations

import argparse
import json
from pathlib import Path

VISION_TASKS = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
    "Waterbirds",
    "CelebA",
]


def main(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tasks": VISION_TASKS,
        "note": "Populate raw datasets manually or integrate HuggingFace datasets in production.",
    }
    (out / "vision_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out / 'vision_manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create vision dataset manifest.")
    parser.add_argument("--output", default="data")
    main(parser.parse_args())
