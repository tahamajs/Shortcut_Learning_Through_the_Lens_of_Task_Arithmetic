# Shortcut Learning Through the Lens of Task Arithmetic

This repository contains a runnable `shortcut_patcher/` implementation for:

- Fine-tuning with snapshot logging and warmup+cosine scheduling.
- Task-vector computation (`w_finetuned - w_pretrained`) including random-norm-matched baselines.
- Task arithmetic edits with multi-vector composition and metadata logging.
- Analysis utilities (PCA, CCA, probes, gradient alignment, activation extraction).
- Robust evaluation outputs (accuracy + worst-group accuracy) and CSV/JSON export.
- A complete LaTeX paper draft under `shortcut_patcher/paper/`.

## Quickstart

```bash
cd shortcut_patcher
bash run_pipeline.sh
```

Outputs are written to:

- `experiments/logs/waterbirds/`
- `experiments/results/waterbirds/` (includes `edited_metrics.csv`, `edited_metrics.json`, plots, vectors)

## Paper

```bash
cd shortcut_patcher/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
