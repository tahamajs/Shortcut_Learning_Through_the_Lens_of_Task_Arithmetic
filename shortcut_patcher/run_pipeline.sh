#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-waterbirds}"
DATA_ROOT="${DATA_ROOT:-}"
MAX_STEPS="${MAX_STEPS:-1200}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-200}"
EVAL_EVERY="${EVAL_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
echo "[info] DATA_ROOT=${DATA_ROOT:-<auto>}"
echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] LOG_DIR=$LOG_DIR"
echo "[info] RES_DIR=$RES_DIR"
echo "[info] PIPELINE_LOG=$PIPELINE_LOG"

# ------------------------------------------------------------------
# 1) Data manifests (scaffold)
# ------------------------------------------------------------------
echo ""
echo "==> Generating data manifests"
"$PYTHON_BIN" data/vision_tasks.py --output data
"$PYTHON_BIN" data/text_tasks.py --output data

# ------------------------------------------------------------------
# 2) Train (logs snapshots + metrics.json with curve)
# ------------------------------------------------------------------
echo ""
echo "==> Training"
"$PYTHON_BIN" src/train.py \
  --task WaterbirdsShortcut \
  --model resnet18 \
  --seed 42 \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$MAX_STEPS" \
  --snapshot-every "$SNAPSHOT_EVERY" \
  --log-every "$EVAL_EVERY" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --num-workers "$NUM_WORKERS" \
  --group-balanced \
  --robust-objective group_dro \
  --group-dro-eta 0.2 \
  --grad-clip 1.0 \
  --label-smoothing 0.05 \
  ${DATA_ROOT:+--data-root "$DATA_ROOT"} \
  --output "$LOG_DIR"

# ------------------------------------------------------------------
# 3) Build steps.npy + target/control accuracy arrays from metrics.json
# ------------------------------------------------------------------
echo ""
echo "==> Building steps + curves arrays"
export LOG_DIR RES_DIR RUN_NAME
"$PYTHON_BIN" - <<'PY'
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

# snapshot steps (x-axis for plots)
steps = []
for ckpt in sorted(snap_dir.glob("ckpt_*.pt")):
    try:
        steps.append(int(ckpt.stem.split("_")[-1]))
    except ValueError:
        pass
steps = np.array(sorted(set(steps)), dtype=np.int64)
np.save(res_dir / "steps.npy", steps)

# curve points (y-axis for plots)
metrics = json.loads(metrics_path.read_text())
curve = metrics.get("curve", [])

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
    print("       This usually means you logged snapshots and curve at different frequencies.")
PY

# ------------------------------------------------------------------
# 4) Task vectors (real + random baseline)
# ------------------------------------------------------------------
echo ""
echo "==> Computing task vectors"
"$PYTHON_BIN" src/task_vector.py \
  --pretrained "$LOG_DIR/pretrained.pt" \
  --finetuned "$LOG_DIR/final.pt" \
  --output "$RES_DIR/task_vector.pt"

"$PYTHON_BIN" src/task_vector.py \
  --pretrained "$LOG_DIR/pretrained.pt" \
  --finetuned "$LOG_DIR/final.pt" \
  --random-like \
  --output "$RES_DIR/random_like_vector.pt"

# ------------------------------------------------------------------
# 5) Apply edits
# ------------------------------------------------------------------
echo ""
echo "==> Applying edits"
"$PYTHON_BIN" src/edit_model.py \
  --model-ckpt "$LOG_DIR/pretrained.pt" \
  --task-vector "$RES_DIR/task_vector.pt" \
  --alpha -0.5 \
  --output "$RES_DIR/forget_half.pt"

"$PYTHON_BIN" src/edit_model.py \
  --model-ckpt "$LOG_DIR/pretrained.pt" \
  --task-vector "$RES_DIR/task_vector.pt" \
  --alpha -1.0 \
  --output "$RES_DIR/forget.pt"

"$PYTHON_BIN" src/edit_model.py \
  --model-ckpt "$LOG_DIR/pretrained.pt" \
  --task-vector "$RES_DIR/task_vector.pt" \
  --alpha 0.25 \
  --output "$RES_DIR/quarter.pt"

"$PYTHON_BIN" src/edit_model.py \
  --model-ckpt "$LOG_DIR/pretrained.pt" \
  --task-vector "$RES_DIR/task_vector.pt" \
  --alpha 1.0 \
  --output "$RES_DIR/add.pt"

"$PYTHON_BIN" src/edit_model.py \
  --model-ckpt "$LOG_DIR/pretrained.pt" \
  --task-vector "$RES_DIR/task_vector.pt" \
  --alpha 0.5 \
  --output "$RES_DIR/half.pt"

"$PYTHON_BIN" src/edit_model.py \
  --model-ckpt "$LOG_DIR/pretrained.pt" \
  --task-vector "$RES_DIR/random_like_vector.pt" \
  --alpha 1.0 \
  --output "$RES_DIR/random_baseline.pt"

# ------------------------------------------------------------------
# 6) Evaluate all checkpoints (writes json + csv)
# ------------------------------------------------------------------
echo ""
echo "==> Evaluating checkpoints"
"$PYTHON_BIN" src/eval_ckpt.py \
  --task WaterbirdsShortcut \
  --model resnet18 \
  ${DATA_ROOT:+--data-root "$DATA_ROOT"} \
  --num-workers "$NUM_WORKERS" \
  --ckpt pretrained="$LOG_DIR/pretrained.pt" \
  --ckpt finetuned="$LOG_DIR/final.pt" \
  --ckpt forget_half="$RES_DIR/forget_half.pt" \
  --ckpt forget="$RES_DIR/forget.pt" \
  --ckpt quarter="$RES_DIR/quarter.pt" \
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
"$PYTHON_BIN" src/analyze.py \
  --method pca \
  --trajectory "$LOG_DIR/snapshots" \
  --n-components 2 \
  --output "$RES_DIR/pca_summary.json"

# ------------------------------------------------------------------
# 8) Plot accuracy curve
# ------------------------------------------------------------------
echo ""
echo "==> Plotting accuracy curve"
"$PYTHON_BIN" src/visualize.py \
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