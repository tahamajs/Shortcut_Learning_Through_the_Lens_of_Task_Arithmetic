# Shortcut Learning Through the Lens of Task Arithmetic

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

## Paper

```bash
cd shortcut_patcher/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
