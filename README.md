# Shortcut Learning Through the Lens of Task Arithmetic

This repository contains a runnable `shortcut_patcher/` implementation for:

- Fine-tuning with snapshot logging and warmup+cosine scheduling.
- Group-robust training options for shortcut-prone datasets (group-balanced sampling + GroupDRO).
- Task-vector computation (`w_finetuned - w_pretrained`) including random-norm-matched baselines.
- Task arithmetic edits with multi-vector composition and metadata logging.
- Analysis utilities (PCA, CCA, probes, gradient alignment, activation extraction).
- Robust evaluation outputs (accuracy + worst-group accuracy) and CSV/JSON export.
- A complete LaTeX paper draft under `shortcut_patcher/paper/`.
This repository now includes a complete implementation scaffold in `shortcut_patcher/` for:

- Fine-tuning models with checkpoint/snapshot logging.
- Computing task vectors (`w_finetuned - w_pretrained`).
- Applying task arithmetic edits (negation/addition/scaling, optional layer filtering).
- Running trajectory analysis (PCA, CCA, probe accuracy, gradient alignment).
- Producing reproducibility assets (`environment.yml`, `Dockerfile`, `run_pipeline.sh`).
- Building a full LaTeX paper draft in `shortcut_patcher/paper/`.

## Quickstart

```bash
cd shortcut_patcher
bash run_pipeline.sh
```

Optional runtime overrides:

```bash
DATA_ROOT=/absolute/path/to/waterbirds_v1.0 \
MAX_STEPS=1200 SNAPSHOT_EVERY=200 EVAL_EVERY=100 \
BATCH_SIZE=64 NUM_WORKERS=2 \
bash run_pipeline.sh
```

Outputs are written to:

- `experiments/logs/waterbirds/`
- `experiments/results/waterbirds/` (includes `edited_metrics.csv`, `edited_metrics.json`, plots, vectors)

## Better robustness defaults (already enabled in pipeline)

- Group-balanced sampling for Waterbirds groups.
- GroupDRO objective during training.
- Stronger image augmentation and ImageNet normalization.
- Additional edit variants (`alpha=-0.5`, `0.25`, `0.5`, `1.0`, `-1.0`) for ablations.

## Paper

```bash
cd shortcut_patcher/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
