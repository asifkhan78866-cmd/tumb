# data/

One command puts everything in place:

```bash
scripts/unpack_data.sh          # unpack what the repository ships
scripts/unpack_data.sh --msd    # also fetch the 7.1 GB Decathlon volumes
```

## What travels with the repository

| Archive | Unpacks to | Size | What it is |
|---|---|---|---|
| `bri.zip.part-aa` + `-ab` | `bri/archive/{Training,Testing}` | 149 MB | the 4-class MRI dataset, 7,023 images — what every classifier was trained and tested on |
| `spect_mri.zip` | `spect_mri/{MRI,SPECT}` | 12 MB | 357 registered MRI–SPECT pairs over 15 cases, behind Method 4's fusion numbers |
| (loose) | `bri/archive/Testing/…`, `spect_mri/…` | 230 KB | nine sample images, so the documented commands run before anything is unpacked |

The MRI dataset is split in two because GitHub refuses a single file over
100 MB. `cat data/bri.zip.part-* > data/bri.zip` reassembles it, which is what
the script does.

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
unzip -qo data/spect_mri.zip -d data/      # -> data/spect_mri/{MRI,SPECT}
python -m backend.methods.method2.training.evaluate_fusion
```

`-o` overwrites without asking: the clone already holds one pair from this same
set, and unzip would otherwise stop on a prompt. The archive carries
`__MACOSX/` resource-fork entries from the Mac it was made on; every loader
here skips dot-files, so they are noise rather than a problem.
