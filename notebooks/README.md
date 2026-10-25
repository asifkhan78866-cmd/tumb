# Notebooks

This folder holds the research notebook the paper is written from — the Colab
work exported as `.ipynb`.

It is **not** the same implementation as `backend/`. The notebook is TensorFlow /
Keras and works from BraTS2023 downloaded through `kagglehub`; the backend is
PyTorch and serves the four methods described in [../README.md](../README.md).
They share the subject, not the code, and a number from one should never be
quoted as a number from the other.

## Running them in VS Code

1. Install the **Jupyter** extension (`ms-toolsai.jupyter`).
2. Open a notebook → **Select Kernel** → **Python Environments** → `.venv/bin/python`.
3. The notebook needs TensorFlow, which this project does not install (the
   backend is PyTorch). For a Colab notebook, either run it in Colab where those
   packages already exist, or add them here first:

   ```bash
   .venv/bin/pip install tensorflow kagglehub fuzzy-c-means
   ```

   On Apple Silicon use `tensorflow-macos` instead of `tensorflow`.

## What used to be here

Four generated notebooks, one per backend method, plus an overview. They read
the recorded metrics and ran the trained checkpoints, but they described the
backend rather than the paper, so they were removed along with their generator
(`backend/cli/build_notebooks.py`). Both are recoverable from git history:

```bash
git log --oneline --diff-filter=D -- notebooks/
git checkout <commit>^ -- notebooks/ backend/cli/build_notebooks.py
```

To see what the backend's trained models actually produce, use the terminal
runner instead — it shows every method's prediction and images side by side:

```bash
.venv/bin/python -m backend.cli.run_methods data/bri/archive/Testing/meningioma/Te-me_0014.jpg
```

`train_unet_brats_gpu.ipynb` is older and unrelated to either set.
