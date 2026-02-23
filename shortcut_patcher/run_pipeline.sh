#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python data/vision_tasks.py --output data
python data/text_tasks.py --output data

python src/train.py \
  --task WaterbirdsShortcut \
  --model synthetic-mlp \
  --seed 42 \
  --max-steps 300 \
  --snapshot-every 50 \
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
