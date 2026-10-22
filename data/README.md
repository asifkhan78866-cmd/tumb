# data/

The datasets themselves are **not** in this repository — they are downloaded:

```bash
scripts/download_dataset.sh                                   # the 4-class MRI set (Kaggle)
python -m backend.utils.dataset_download --kind segmentation  # BraTS, needs a Kaggle account
```

The Decathlon release of the BraTS volumes needs no account and is what the
volumetric and multi-class models here were trained on:

```bash
curl -L -o data/msd_brain/Task01_BrainTumour.tar \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar -xf data/msd_brain/Task01_BrainTumour.tar -C data/msd_brain
```

## What *is* committed

A handful of example images, so the documented commands and the notebooks run
on a fresh clone before anything is downloaded:

| path | from |
|---|---|
| `bri/archive/Testing/{glioma,meningioma,notumor,pituitary}/…` | [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — one or two slices per class, from its **Testing** split |
| `spect_mri/{MRI,SPECT}/10017.png` | one registered MRI–SPECT pair, for the fusion stage |

## The full paired MRI–SPECT set

`spect_mri.zip` (12 MB) holds all **357 registered pairs over 15 cases** — the
data Method 4's fusion stage was measured on. It is committed because it is
small enough to ship and there is no download script for it:

```bash
unzip -qo data/spect_mri.zip -d data/      # -> data/spect_mri/{MRI,SPECT}
```

`-o` overwrites without asking: the clone already holds one pair from this same
set, and unzip would otherwise stop on a prompt.

The archive carries `__MACOSX/` resource-fork entries from the Mac it was made
on; every loader here skips dot-files, so they are noise rather than a problem.
After unpacking, `PAIRED_MRI_SPECT_PATH` finds it with no configuration, and
the fusion evaluation runs on all 357 pairs:

```bash
python -m backend.methods.method2.training.evaluate_fusion
```

They are examples for demonstrations. Every accuracy in this project was
measured on the full held-out split, not on these.

```bash
python -m backend.cli.run_methods data/bri/archive/Testing/meningioma/Te-me_0014.jpg
python -m backend.cli.run_methods data/bri/archive/Testing/glioma/Te-gl_0012.jpg \
    --methods 4 --spect data/spect_mri/SPECT/10017.png
```
