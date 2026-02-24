#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_NAME="waterbirds"
LOG_DIR="experiments/logs/${RUN_NAME}"
RES_DIR="experiments/results/${RUN_NAME}"
mkdir -p "$LOG_DIR" "$RES_DIR"

python data/vision_tasks.py --output data
python data/text_tasks.py --output data

# run training and copy stdout/stderr to a logfile so we can follow progress
python src/train.py \
  --task WaterbirdsShortcut \
  --model synthetic-mlp \
  --seed 42 \
  --max-steps 300 \
  --snapshot-every 50 \
  --log-every 50 \
  --output "$LOG_DIR" 2>&1 | tee "$LOG_DIR/train.log"

python - <<'PY'
from pathlib import Path
import json
import numpy as np

log_dir = Path("experiments/logs/waterbirds")
res_dir = Path("experiments/results/waterbirds")
res_dir.mkdir(parents=True, exist_ok=True)

steps = []
for ckpt in sorted((log_dir / "snapshots").glob("ckpt_*.pt")):
    try:
        steps.append(int(ckpt.stem.split("_")[-1]))
    except ValueError:
        continue
np.save(res_dir / "steps.npy", np.array(steps))

curve = json.loads((log_dir / "metrics.json").read_text()).get("curve", [])
acc = np.array([row.get("accuracy", 0.0) for row in curve], dtype=float)
worst = np.array([row.get("worst_group_accuracy", 0.0) for row in curve], dtype=float)
np.save(res_dir / "target_acc.npy", acc)
np.save(res_dir / "control_acc.npy", worst if len(worst) else acc)
print("Saved steps/curve arrays")
PY

python src/task_vector.py \
  --pretrained "$LOG_DIR/pretrained.pt" \
  --finetuned "$LOG_DIR/final.pt" \
  --output "$RES_DIR/task_vector.pt"

python src/task_vector.py \
  --pretrained "$LOG_DIR/pretrained.pt" \
  --finetuned "$LOG_DIR/final.pt" \
  --random-like \
  --output "$RES_DIR/random_like_vector.pt"

python src/edit_model.py \
  --model-ckpt "$LOG_DIR/pretrained.pt" \
  --task-vector "$RES_DIR/task_vector.pt" \
  --alpha -1.0 \
  --output "$RES_DIR/forget.pt"

python src/edit_model.py \
  --model-ckpt "$LOG_DIR/pretrained.pt" \
  --task-vector "$RES_DIR/task_vector.pt" \
  --alpha 1.0 \
  --output "$RES_DIR/add.pt"

python src/edit_model.py \
  --model-ckpt "$LOG_DIR/pretrained.pt" \
  --task-vector "$RES_DIR/task_vector.pt" \
  --alpha 0.5 \
  --output "$RES_DIR/half.pt"

python src/edit_model.py \
  --model-ckpt "$LOG_DIR/pretrained.pt" \
  --task-vector "$RES_DIR/random_like_vector.pt" \
  --alpha 1.0 \
  --output "$RES_DIR/random_baseline.pt"

python src/eval_ckpt.py \
  --task WaterbirdsShortcut \
  --model synthetic-mlp \
  --ckpt pretrained="$LOG_DIR/pretrained.pt" \
  --ckpt finetuned="$LOG_DIR/final.pt" \
  --ckpt forget="$RES_DIR/forget.pt" \
  --ckpt add="$RES_DIR/add.pt" \
  --ckpt half="$RES_DIR/half.pt" \
  --ckpt random="$RES_DIR/random_baseline.pt" \
  --output-json "$RES_DIR/edited_metrics.json" \
  --output-csv "$RES_DIR/edited_metrics.csv"

python src/analyze.py \
  --method pca \
  --trajectory "$LOG_DIR/snapshots" \
  --n-components 2 \
  --output "$RES_DIR/pca_summary.txt"

python src/visualize.py \
  --mode accuracy \
  --steps "$RES_DIR/steps.npy" \
  --target "$RES_DIR/target_acc.npy" \
  --control "$RES_DIR/control_acc.npy" \
  --output "$RES_DIR/accuracy_curve.png"
  --output experiments/logs/waterbirds

python src/task_vector.py \
  --pretrained experiments/logs/waterbirds/pretrained.pt \
  --finetuned experiments/logs/waterbirds/final.pt \
  --output experiments/results/waterbirds_vector.pt

python src/edit_model.py \
  --model-ckpt experiments/logs/waterbirds/pretrained.pt \
  --task-vector experiments/results/waterbirds_vector.pt \
  --alpha -1.0 \
  --output experiments/results/waterbirds_forget.pt

python src/analyze.py \
  --method pca \
  --trajectory experiments/logs/waterbirds/snapshots \
  --n-components 2 \
  --output experiments/results/pca_summary.txt

echo "Pipeline complete."
