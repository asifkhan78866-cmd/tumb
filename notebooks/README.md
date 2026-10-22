# Notebooks

One notebook per method, plus an overview. Each is saved **with its outputs**, so
the images and numbers are visible without running anything — open a file and
scroll.

| notebook | method |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | All four side by side: what is trained, the dataset, the held-out results, the same image through every method |
| [01_method1_pretrained_transfer.ipynb](01_method1_pretrained_transfer.ipynb) | Pre-trained CNNs + transfer learning (GoogleNet + VGG16 + ResNet-50 + AlexNet feature fusion) |
| [02_method2_zfnet_redfox.ipynb](02_method2_zfnet_redfox.ipynb) | Red Fox optimised ZFNet, and the SVM on Red Fox–selected features |
| [03_method3_unet_convlstm_sfla.ipynb](03_method3_unet_convlstm_sfla.ipynb) | U-Net → ROI → ConvLSTM → SFLA, the SFLA feature selection, and the 3D U-Net |
| [04_method4_mri_spect_fusion.ipynb](04_method4_mri_spect_fusion.ipynb) | MRI–SPECT fusion: the multi-class segmentation, the feature stage, and the fusion transform |

`train_unet_brats_gpu.ipynb` is older and unrelated to this set.

## Running them in VS Code

1. Install the **Jupyter** extension (`ms-toolsai.jupyter`) if it isn't already.
2. Open a notebook, click **Select Kernel** (top right) → **Python Environments** →
   `.venv/bin/python` from this project. Everything it imports lives in that venv.
3. **Run All**. A full pass takes a minute or two; nothing in them trains.

If the kernel picker doesn't list the venv, run this once and reopen VS Code:

```bash
cd /Users/syedasif/tumb && .venv/bin/python -m ipykernel install --user --name tumb --display-name "Python (tumb)"
```

## Regenerating them

The notebooks are generated from the registry and the recorded evaluations, so
they cannot drift from the code:

```bash
python -m backend.cli.build_notebooks            # rewrite the notebooks
python -m backend.cli.build_notebooks --execute  # rewrite and run them, saving outputs
```

Re-run with `--execute` after retraining anything, otherwise the saved outputs
describe the previous checkpoint.

## What they do not do

Nothing here trains a model or computes a test metric. Every accuracy is read
from the evaluation its training run recorded, and every image is produced by
the checkpoint currently on disk. Each notebook ends with the caveats that
method recorded — they are part of the result, not a footnote.
