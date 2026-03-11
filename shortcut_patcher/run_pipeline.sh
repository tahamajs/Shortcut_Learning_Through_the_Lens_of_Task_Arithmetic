#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-waterbirds}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-100}"
NUM_WORKERS="${NUM_WORKERS:-2}"
METHOD="${METHOD:-groupdro}"
GROUPDRO_ETA="${GROUPDRO_ETA:-0.1}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

LOG_DIR="experiments/logs/${RUN_NAME}"
RES_DIR="experiments/results/${RUN_NAME}"

# clean old outputs (prevents mixing old snapshots/metrics)
rm -rf "$LOG_DIR" "$RES_DIR"
mkdir -p "$LOG_DIR" "$RES_DIR"

# save *all* stdout/stderr from this script
PIPELINE_LOG="$RES_DIR/pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "==================== PIPELINE START ===================="
echo "[info] RUN_NAME=$RUN_NAME"
echo "[info] SEED=$SEED"
echo "[info] MAX_STEPS=$MAX_STEPS"
echo "[info] METHOD=$METHOD"
echo "[info] LOG_DIR=$LOG_DIR"
echo "[info] RES_DIR=$RES_DIR"
echo "[info] PIPELINE_LOG=$PIPELINE_LOG"

# ------------------------------------------------------------------
# 1) Data manifests (scaffold)
# ------------------------------------------------------------------
echo ""
echo "==> Generating data manifests"
python data/vision_tasks.py --output data
python data/text_tasks.py --output data

# ------------------------------------------------------------------
# 2) Train (logs snapshots + metrics.json with curve)
# ------------------------------------------------------------------
echo ""
echo "==> Training"
python src/train.py \
  --task WaterbirdsShortcut \
  --model resnet18 \
  --seed "$SEED" \
  --method "$METHOD" \
  --groupdro-eta "$GROUPDRO_ETA" \
  --group-balanced-sampler \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$MAX_STEPS" \
  --snapshot-every "$SNAPSHOT_EVERY" \
  --log-every "$LOG_EVERY" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --output "$LOG_DIR"

# ------------------------------------------------------------------
# 3) Build steps.npy + target/control accuracy arrays from metrics.json
# ------------------------------------------------------------------
echo ""
echo "==> Building steps + curves arrays"
export LOG_DIR RES_DIR RUN_NAME
python - <<'PY'
import os, json
from pathlib import Path
import numpy as np

log_dir = Path(os.environ["LOG_DIR"])
res_dir = Path(os.environ["RES_DIR"])
run_name = os.environ.get("RUN_NAME", "run")

snap_dir = log_dir / "snapshots"
metrics_path = log_dir / "metrics.json"

if not snap_dir.exists():
    raise FileNotFoundError(f"missing snapshots dir: {snap_dir}")
if not metrics_path.exists():
    raise FileNotFoundError(f"missing metrics.json: {metrics_path}")

# curve points (y-axis for plots)
metrics = json.loads(metrics_path.read_text())
curve = metrics.get("curve", [])

steps = np.array([int(row.get("step", 0)) for row in curve], dtype=np.int64)
if len(steps) == 0:
  # fallback to snapshot-derived x-axis
  snap_steps = []
  for ckpt in sorted(snap_dir.glob("ckpt_*.pt")):
    try:
      snap_steps.append(int(ckpt.stem.split("_")[-1]))
    except ValueError:
      pass
  steps = np.array(sorted(set(snap_steps)), dtype=np.int64)
np.save(res_dir / "steps.npy", steps)

target_acc = np.array([row.get("accuracy", 0.0) for row in curve], dtype=float)
control_acc = np.array(
    [row.get("worst_group_accuracy", row.get("accuracy", 0.0)) for row in curve],
    dtype=float
)

np.save(res_dir / "target_acc.npy", target_acc)
np.save(res_dir / "control_acc.npy", control_acc)

print("[saved] steps.npy length:", len(steps), "values:", steps.tolist())
print("[saved] target_acc.npy length:", len(target_acc), "values:", target_acc.tolist())
print("[saved] control_acc.npy length:", len(control_acc), "values:", control_acc.tolist())

# Safety check to catch plotting errors early
if len(steps) != len(target_acc):
    print("[warn] steps and target_acc lengths differ!")
  print("       plotting code will auto-align lengths, but consider matching log/snapshot frequency.")
PY

# ------------------------------------------------------------------
# 4) Task vectors (real + random baseline)
# ------------------------------------------------------------------
echo ""
echo "==> Computing task vectors"
python src/task_vector.py \
  --pretrained "$LOG_DIR/pretrained.pt" \
  --finetuned "$LOG_DIR/final.pt" \
  --output "$RES_DIR/task_vector.pt"

python src/task_vector.py \
  --pretrained "$LOG_DIR/pretrained.pt" \
  --finetuned "$LOG_DIR/final.pt" \
  --random-like \
  --output "$RES_DIR/random_like_vector.pt"

# ------------------------------------------------------------------
# 5) Apply edits
# ------------------------------------------------------------------
echo ""
echo "==> Applying edits"
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

# ------------------------------------------------------------------
# 6) Evaluate all checkpoints (writes json + csv)
# ------------------------------------------------------------------
echo ""
echo "==> Evaluating checkpoints"
python src/eval_ckpt.py \
  --task WaterbirdsShortcut \
  --model resnet18 \
  --ckpt pretrained="$LOG_DIR/pretrained.pt" \
  --ckpt finetuned="$LOG_DIR/final.pt" \
  --ckpt forget="$RES_DIR/forget.pt" \
  --ckpt add="$RES_DIR/add.pt" \
  --ckpt half="$RES_DIR/half.pt" \
  --ckpt random="$RES_DIR/random_baseline.pt" \
  --output-json "$RES_DIR/edited_metrics.json" \
  --output-csv "$RES_DIR/edited_metrics.csv"

# ------------------------------------------------------------------
# 7) PCA analysis on trajectory
# (your analyze.py writes JSON content; use .json extension)
# ------------------------------------------------------------------
echo ""
echo "==> PCA analysis"
python src/analyze.py \
  --method pca \
  --trajectory "$LOG_DIR/snapshots" \
  --n-components 2 \
  --output "$RES_DIR/pca_summary.json"

# ------------------------------------------------------------------
# 8) Plot accuracy curve
# ------------------------------------------------------------------
echo ""
echo "==> Plotting accuracy curve"
python src/visualize.py \
  --mode accuracy \
  --steps "$RES_DIR/steps.npy" \
  --target "$RES_DIR/target_acc.npy" \
  --control "$RES_DIR/control_acc.npy" \
  --output "$RES_DIR/accuracy_curve.png"

echo ""
echo "==================== PIPELINE COMPLETE ✅ ===================="
echo "[out] $RES_DIR"
echo "  - pipeline.log:        $RES_DIR/pipeline.log"
echo "  - edited_metrics.csv:  $RES_DIR/edited_metrics.csv"
echo "  - edited_metrics.json: $RES_DIR/edited_metrics.json"
echo "  - pca_summary.json:    $RES_DIR/pca_summary.json"
echo "  - accuracy_curve.png:  $RES_DIR/accuracy_curve.png"