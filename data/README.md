# data/

One command puts everything in place:

```bash
scripts/unpack_data.sh          # unpack what the repository ships
scripts/unpack_data.sh --msd    # also fetch the 7.1 GB Decathlon volumes
```

## What travels with the repository

Nine sample images, 230 KB: one or two slices per class and one MRI–SPECT pair.
Enough for every documented command and every method to run — nothing else is
needed to *use* the trained models.

The datasets themselves are not committed. They were, briefly, as 160 MB of
archives, and every clone had to carry them whether or not it ever trained
anything; they were removed from the history for that reason.

| Dataset | Needed for | Size | Get it |
|---|---|---|---|
| 4-class MRI, 7,023 images | training any classifier | 149 MB | `scripts/download_dataset.sh`, or drop `bri.zip` into `data/` |
| 357 MRI–SPECT pairs | reproducing the fusion metrics | 12 MB | drop `spect_mri.zip` into `data/` |
| Decathlon (BraTS) volumes | training segmentation or the 3D U-Net | 7.1 GB | `scripts/unpack_data.sh --msd` |

Put any archive you were given into `data/` and run `scripts/unpack_data.sh`;
it finds `bri.zip`, `bri.zip.part-*` or `spect_mri.zip` and unpacks whichever
are there.

## What does not, and why

**The Decathlon (BraTS) volumes — 7.1 GB unpacked, 484 annotated volumes.** Far
past what a git repository should carry. They are needed only to *retrain*
segmentation or the 3D U-Net; every trained model in this repository already
works without them. The download is public — no Kaggle account, no token:

```bash
scripts/unpack_data.sh --msd
# or by hand:
curl -L -o data/msd_brain/Task01_BrainTumour.tar \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar -xf data/msd_brain/Task01_BrainTumour.tar -C data/msd_brain
```

Method 3's 2D U-Net was trained on the figshare (Cheng) T1-contrast masks,
which are a separate download again — see the segmentation run card for which
release.

## Where the data came from

| Dataset | Source | Licence |
|---|---|---|
| Brain Tumor MRI Dataset (4-class) | [Kaggle · masoudnickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — itself a compilation of figshare, SARTAJ and Br35H | CC0 / CC BY per source |
| Medical Segmentation Decathlon, Task01 | [medicaldecathlon.com](http://medicaldecathlon.com/) (BraTS) | CC BY-SA 4.0 |
| Paired MRI–SPECT slices | Whole Brain Atlas–style registered pairs | research use |

Redistributed here for reproducibility, with attribution. If you intend to use
any of it beyond that, check the licence at the source.

## Reproducing the numbers

```bash
scripts/unpack_data.sh                     # unpacks whatever archives are in data/
python -m backend.methods.method2.training.evaluate_fusion
```

`-o` overwrites without asking: the clone already holds one pair from this same
set, and unzip would otherwise stop on a prompt. The archive carries
`__MACOSX/` resource-fork entries from the Mac it was made on; every loader
here skips dot-files, so they are noise rather than a problem.
