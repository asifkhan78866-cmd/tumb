# Project structure — what a clone has, and what it does not

A clone is **281 files, ~140 MB**, and it can predict immediately: the trained
checkpoints, the metrics they were measured with, and a few sample images are
all committed. What is *not* committed is everything that can be downloaded or
regenerated — the datasets (gigabytes), the Python environment, the caches.

On a machine that has only what was pushed, this tells you what you have:

```bash
python -m backend.cli.doctor
```

It prints every component, whether it is present, and the one command that fixes
each missing one.

---

## The tree, as cloned

Sizes are what arrives with `git clone`.

```
tumb/
│
├── README.md                     the project, its architecture, its results
├── STRUCTURE.md                  this file
├── DOCKER.md                     sending the app to another machine
├── docker-compose.yml            backend + frontend containers
├── .env.example                  every setting, documented, with defaults
├── scripts/
│   ├── setup.sh                  creates .venv, installs everything, reports readiness
│   └── download_dataset.sh       fetches the 4-class MRI dataset
│
├── backend/
│   ├── main.py                   FastAPI application
│   ├── config.py                 paths, device, seeds, dataset discovery
│   ├── requirements.txt          every Python dependency
│   ├── Dockerfile                CPU torch + the trained models
│   │
│   ├── api/                      ── HTTP layer ───────────────────────────────
│   │   ├── methods_routes.py       /api/methods · /api/predict · /api/metrics
│   │   ├── gallery_routes.py       plots, dataset samples, fusion panels
│   │   └── routes.py               legacy routes (delegate to Method 3)
│   │
│   ├── cli/                      ── terminal tools ───────────────────────────
│   │   ├── run_methods.py          every method on one image, with its images
│   │   ├── render.py               drawing images in a terminal
│   │   ├── build_notebooks.py      generates notebooks/ from the registry
│   │   └── doctor.py               what this copy can run
│   │
│   ├── methods/                  ── the four pipelines ───────────────────────
│   │   ├── registry.py             identity, pipeline, datasets, weights per method
│   │   ├── common/                 shared machinery (see below)
│   │   ├── method1/                U-Net + ConvLSTM + SFLA      → displayed 3
│   │   ├── method2/                MRI–SPECT fusion             → displayed 4
│   │   ├── method3/                transfer learning + fusion   → displayed 1
│   │   └── method4/                ZFNet + Red Fox              → displayed 2
│   │
│   ├── models/                     U-Net, 3D U-Net, ConvLSTM
│   ├── utils/                      Grad-CAM, metrics, plots, PDF reports
│   ├── training/                   shared training helpers (seeding, early stop)
│   │
│   ├── weights/        ~130 MB   ── TRAINED MODELS, COMMITTED ────────────────
│   │   ├── method1/                ConvLSTM 2.3 MB · U-Net 30 MB · 3D U-Net 5.4 MB
│   │   │                           · feature_svm.joblib (SFLA-selected SVM)
│   │   ├── method3/                fusion head 4.8 MB  (backbones: see below)
│   │   ├── method4/                ZFNet 54 MB · rfo_svm.joblib 0.2 MB
│   │   ├── method2_dcn.pth         2.2 MB  the DCN that serves Method 4
│   │   └── method2_segmentation.pth 30 MB  multi-class U-Net
│   │
│   └── logs/             ~3 MB   ── RECORDED EVALUATIONS, COMMITTED ──────────
│       ├── method*_metrics.json    what each method measured, and its warnings
│       ├── method*_*_runcard.json  data, split, seed, geometry, git revision
│       ├── method*_cls_*.png       confusion matrices, ROC curves, curves
│       ├── method1/                SFLA results, ROI comparison, 3D metrics
│       ├── method2/fusion_samples/ MRI | SPECT | fused panels
│       └── method4/                Red Fox results, feature-selection plot
│
├── frontend/                     ── Next.js 15 dashboard ─────────────────────
│   ├── app/                        6 pages: home, upload, metrics, compare,
│   │                               history, about
│   ├── components/                 prediction card, galleries, status cards
│   └── lib/api.ts                  typed client for every endpoint
│
├── notebooks/            ~5 MB   one per method + overview, saved with outputs
│
├── data/                 230 KB  ── SAMPLES ONLY ─────────────────────────────
│   ├── README.md                   where the real datasets come from
│   ├── bri/archive/Testing/        7 slices, one or two per class
│   └── spect_mri/{MRI,SPECT}/      one registered MRI–SPECT pair
│
└── tests/                        240 tests
```

### `backend/methods/common/` — the machinery the four share

| File | Responsibility |
|---|---|
| `bri_data.py` | the one split every MRI method uses, and the duplicate removal |
| `splits.py` | patient/volume-level grouping, leak checks |
| `checkpoint.py` | writes and validates a checkpoint's identity |
| `metrics_store.py` | one metrics file per method, one section per stage |
| `runcard.py` | the record a training run leaves behind |
| `feature_selection.py` | the search space and SVM wrapper both metaheuristics use |
| `classifier_engine.py` | serving for the two whole-image methods |
| `msd_brain.py` | the Decathlon (BraTS) reader: volumes, slices, label remapping |
| `shared_segmentation.py` | lending Method 3's mask to methods that have none |

---

## What is **not** in a clone

| Missing | Size | Consequence | How to get it |
|---|---|---|---|
| `.venv/` | ~3 GB | nothing runs | `scripts/setup.sh` |
| `frontend/node_modules/` | ~400 MB | the dashboard cannot start | `cd frontend && npm install` |
| **ImageNet backbones** | ~800 MB | **Method 1 only**: downloads them on first use | automatic, needs internet once |
| `data/bri/archive` (full) | ~165 MB | predictions fine; **training** impossible | `scripts/download_dataset.sh` |
| Decathlon (BraTS) volumes | ~7.6 GB | cannot retrain segmentation or the 3D U-Net | see [data/README.md](data/README.md) |
| Paired MRI–SPECT (357) | 13 MB | fusion runs on the one sample pair only | see [data/README.md](data/README.md) |
| `outputs/`, `backend/predictions/` | — | created on first run | automatic |
| `backend/.model_cache/` | varies | preprocessing recomputed once | automatic |

**The one that surprises people:** Method 1 ships its trained *head* (4.8 MB),
not the four ImageNet backbones it reads features from. Those are third-party
weights torchvision downloads once into `~/.cache/torch/hub/checkpoints`. Until
that download happens, Method 1 needs an internet connection — the other three
methods work entirely offline. `backend.cli.doctor` reports whether the cache is
already populated.

---

## Generated at runtime, never committed

| Path | Written by | Safe to delete |
|---|---|---|
| `outputs/<timestamp>_<image>/` | `backend.cli.run_methods` | yes |
| `backend/predictions/` | every prediction (originals, masks, overlays) | yes |
| `backend/uploads/` | the upload endpoint | yes |
| `backend/.model_cache/` | preprocessing and feature caches | yes — rebuilt on demand |
| `backend/logs/*.log` | training runs | yes |
| `.venv/`, `frontend/.next/` | setup and builds | yes |

---

## Where each method's files live

| | Method 1 (`method3`) | Method 2 (`method4`) | Method 3 (`method1`) | Method 4 (`method2`) |
|---|---|---|---|---|
| Code | `backend/methods/method3/` | `backend/methods/method4/` | `backend/methods/method1/` | `backend/methods/method2/` |
| Weights | `weights/method3/` | `weights/method4/` | `weights/method1/` | `weights/method2_*.pth` |
| Metrics | `logs/method3_metrics.json` | `logs/method4_metrics.json` | `logs/method1_metrics.json` | `logs/method2_metrics.json` |
| Notebook | `notebooks/01_…` | `notebooks/02_…` | `notebooks/03_…` | `notebooks/04_…` |

The internal ids (`method1`…`method4`) are recorded inside every checkpoint and
never change; the display numbers are what the UI shows. They do not line up,
which is deliberate — renaming the ids would invalidate every checkpoint.

---

## First run on a machine that has only the clone

```bash
scripts/setup.sh                      # 1 · environment            (~10 min)
python -m backend.cli.doctor          # 2 · what is present
python -m backend.cli.run_methods data/bri/archive/Testing/meningioma/Te-me_0014.jpg
```

Expect Methods 2, 3 and 4 to answer immediately, and Method 1 to pause on its
first prediction while torchvision fetches the backbones.
