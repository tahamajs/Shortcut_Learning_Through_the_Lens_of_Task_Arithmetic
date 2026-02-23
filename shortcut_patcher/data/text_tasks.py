from __future__ import annotations

import argparse
import json
from pathlib import Path

TEXT_TASKS = {
    "toxicity": {
        "source": "CivilComments",
        "threshold": 0.8,
    },
    "control": {
        "source": "WikiText-103",
    },
}


def main(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "text_manifest.json").write_text(json.dumps(TEXT_TASKS, indent=2))
    print(f"Wrote {out / 'text_manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create text dataset manifest.")
    parser.add_argument("--output", default="data")
    main(parser.parse_args())
