# 🧠 NeuroSeg AI — Four-Method Brain Tumor Analysis

A full-stack research platform running **four brain-tumour classification methods**
behind one FastAPI backend and one Next.js dashboard.

> ⚠️ **Research / decision-support only.** This is not a certified medical device,
> it has not been clinically validated, and it must not be used to diagnose, treat,
> or make any care decision for a patient. Every output requires interpretation by a
> qualified radiologist.

---

## The four methods

| # | Method | Internal id | Model | Optimisation |
|---|---|---|---|---|
| 1 | Deep Learning Pre-trained Models + Transfer Learning-Based Brain Tumor Classification | `method3` | ImageNet-pretrained EfficientNet-B0 and ResNet-50, fine-tuned; best on validation kept | backbone selection on validation macro-F1 |
| 2 | Red Fox Optimized ZFNet-Based Brain Tumor Classification | `method4` | ZFNet (BatchNorm, 1-channel) trained from scratch | Red Fox Optimization over lr, weight decay, dropout, FC width, batch size |
| 3 | 3D U-Net–ConvLSTM–SFLA-Based Anomaly Segmentation & Classification | `method1` | U-Net (2D, untrained) → ROI → ConvLSTM | Shuffled Frog Leaping Algorithm |
| 4 | MRI–SPECT Multimodal Fusion-Based Brain Tumor Classification | `method2` | MRI + SPECT branches → fusion → dense CNN (**MRI branch trained; SPECT branch and fusion untrained**) | — |

Internal ids never change because they are recorded inside checkpoints; the UI
shows the numbers above.

### Results (all four share one split and the same 1,311 test images)

| | Method 1 · Transfer learning | Method 2 · RFO ZFNet | Method 3 · U-Net–ConvLSTM–SFLA | Method 4 · Fusion |
|---|---|---|---|---|
| Selected model | ResNet-50 (val macro-F1 0.992 vs EfficientNet-B0 0.980) | ZFNet, RFO params | ConvLSTM, SFLA params | DCN, MRI branch only |
| Test accuracy | **98.70 %** | **97.18 %** | **96.41 %** | **89.09 %** |
| Macro precision | 98.62 % | 97.07 % | 96.39 % | 88.91 % |
| Macro recall / sensitivity | 98.78 % | 97.20 % | 96.59 % | 88.62 % |
| Macro specificity | 99.58 % | 99.07 % | 98.82 % | 96.35 % |
| Macro F1 | 98.69 % | 97.12 % | 96.43 % | 88.71 % |
| Macro AUC (one-vs-rest) | 99.85 % | 99.82 % | 99.70 % | 97.88 % |

Each test set was evaluated **once**, after model selection on validation. The
split is the dataset's own `Training/`/`Testing/` folders with validation carved
from `Training/`; exact copies of test images were removed from training. It is
an **image-level** split — the dataset has no patient identifiers — so these
numbers do not demonstrate patient-level generalisation. Method 4's number is
MRI-only classification by its MRI branch; it says nothing about MRI–SPECT fusion.

### What is and is not trained

| Component | State |
|---|---|
| Method 1 (`weights/method3/best_classifier.pth`, ResNet-50) | Trained (94 MB). |
| Method 2 (`weights/method4/best_classifier.pth`, ZFNet) | Trained (57 MB). |
| Method 3 classifier (`weights/method1/best_classifier.pth`) | Trained, committed. Whole-slice geometry. |
| Method 3 U-Net | **Not trained.** Needs BraTS; segmentation reports itself unavailable. |
| Method 4 MRI branch (`weights/method2_dcn.pth`, DenseNet-BC style DCN) | Trained on the shared MRI split (2.3 MB). |
| Method 4 SPECT branch and fusion | **Not trained** — no SPECT or paired MRI–SPECT data. No fusion is performed; results say so. The AI assessment is only used if the MRI-branch checkpoint is absent. |

An untrained method never returns a guessed class. A number that was not
measured shows as `null` in the API and **N/A** in the UI.

---

## 📁 Folder layout

```
tumb/
├── .env.example                  # every variable, documented
├── backend/
│   ├── main.py                   # FastAPI app (legacy + /api routes)
│   ├── config.py                 # global config: paths, device, seeds, limits
│   ├── api/
│   │   ├── routes.py             # legacy method-unaware routes (delegate to Method 1)
│   │   ├── methods_routes.py     # /api/methods, /api/predict/{id}, /api/metrics/{id}, …
│   │   └── schemas.py
│   ├── methods/
│   │   ├── registry.py           # ← the only place that knows which methods exist
│   │   ├── common/
│   │   │   ├── splits.py         # patient/volume-level splitting
│   │   │   ├── checkpoint.py     # tagged checkpoints + method isolation
│   │   │   ├── metrics_store.py  # per-method, per-stage metrics (merge, never clobber)
│   │   │   ├── runcard.py        # training provenance records
│   │   │   └── schemas.py        # the shared prediction envelope
│   │   ├── method1/
│   │   │   ├── config.py  transforms.py  datasets.py  pipeline.py  inference.py
│   │   │   ├── models/           # re-exports UNet + ConvLSTMClassifier
│   │   │   ├── training/         # train_segmentation.py, train_classifier.py
│   │   │   └── optimization/     # sfla.py (the algorithm), run_sfla.py (the entry point)
│   │   └── method2/
│   │       ├── config.py  transforms.py  datasets.py  features.py  pipeline.py  inference.py
│   │       ├── models/           # dcn.py (DenseConvNetClassifier), segmentation.py
│   │       └── training/         # train_segmentation.py, train_classifier.py
│   ├── models/                   # shared architectures (UNet, ConvLSTM)
│   ├── utils/                    # losses, metrics, gradcam, report, dataset_download
│   ├── weights/ dataset/ logs/ predictions/
│   └── requirements.txt
├── frontend/
│   ├── app/                      # landing, upload, compare, history, metrics, about
│   ├── components/               # method-selector, architecture-card, warning-list, …
│   └── lib/api.ts                # typed, method-aware API client
├── tests/                        # 88 tests
└── docker-compose.yml
```

Adding a third method means adding one `MethodSpec` and one package. No dispatch
table elsewhere changes — the API, the UI and the PDF report all iterate the registry.

---

## 🚀 Quick start

```bash
# 1. Configure
cp .env.example .env          # fill in Kaggle credentials only if you will download data

# 2. Backend
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload          # http://localhost:8000/docs

# 3. Frontend
cd frontend && npm install
npm run dev                                # http://localhost:3000
```

The app runs immediately. Method 1 will classify (its classifier ships) and will
tell you that segmentation is unavailable; Method 2 will tell you it is untrained.

---

## 🔑 Environment variables

All live in `.env` (git-ignored). Full documentation is in `.env.example`.

> **Security:** `KAGGLE_KEY` and `HF_TOKEN` are read **only** by the backend process.
> Never give them a `NEXT_PUBLIC_` prefix — Next.js inlines every `NEXT_PUBLIC_*`
> value into the browser bundle, publishing the secret to anyone who opens the page.

| Group | Variables |
|---|---|
| **App** | `APP_ENV` · `BACKEND_HOST` · `BACKEND_PORT` · `FRONTEND_ORIGIN` · `ALLOWED_ORIGINS` |
| **Data** | `DATA_ROOT` · `BRI_DATASET_PATH` · `BRATS_DATASET_PATH` · `SPECT_DATASET_PATH` |
| **Download** | `KAGGLE_USERNAME` · `KAGGLE_KEY` · `HF_TOKEN` *(optional)* |
| **Weights** | `METHOD1_UNET_WEIGHTS` · `METHOD1_CONVLSTM_WEIGHTS` · `METHOD2_SEGMENTATION_WEIGHTS` · `METHOD2_DCN_WEIGHTS` |
| **Modality** | `METHOD2_MODALITY` · `METHOD2_SPEC_MODALITY_LABEL` |
| **Optimization** | `SFLA_ENABLED` · `SFLA_SEED` · `SFLA_POPULATION` · `SFLA_ITERATIONS` · `SFLA_MEMEPLEXES` · `SFLA_LOCAL_ITERATIONS` |
| **Runtime** | `DEVICE` · `MODEL_CACHE_DIR` · `MAX_UPLOAD_MB` · `RANDOM_SEED` · `TRAINING_API_ENABLED` |

---

## 📦 Datasets

### MRI dataset for Methods 1–3 (not stored in this repository)

The images (Kaggle *Brain Tumor MRI Dataset*, 7,023 JPGs, ~165 MB) are **not
committed**: they are third-party data with their own redistribution terms. Get
them with one command once a Kaggle API token is configured
(https://www.kaggle.com/settings → *Create New API Token*; put `KAGGLE_USERNAME`
and `KAGGLE_KEY` in `.env`, or `kaggle.json` in `~/.kaggle/`):

```bash
scripts/download_dataset.sh
```

**Already have the images?** You do not need to move or rename anything. With
`BRI_DATASET_PATH` unset, the backend looks for a folder that directly contains
`Training/` and `Testing/` (each with the four class folders) in, in order:
`data/bri/archive`, `data/bri`, `data/archive`, `archive`, `dataset/archive`,
`backend/dataset/brain-tumor-mri-dataset`. An unzipped `archive/` in the project
root is found automatically. Set `BRI_DATASET_PATH` only to point elsewhere — an
explicit value is always obeyed, so a wrong path fails loudly instead of training
on a directory nobody chose.

It downloads `masoudnickparvar/brain-tumor-mri-dataset`, places it at
`data/bri/archive/{Training,Testing}/{glioma,meningioma,notumor,pituitary}`
(the default `BRI_DATASET_PATH`), verifies both splits and all four classes, and
prints the inventory. `--dry-run` shows what would happen; an existing copy is
never replaced without `--force`. Expected counts: 5,712 Training / 1,311 Testing.

### All datasets

Nothing downloads implicitly. Ask for it:

```bash
python -m backend.utils.dataset_download --list                  # status only, no network
python -m backend.utils.dataset_download --kind classification   # BRI 4-class
python -m backend.utils.dataset_download --kind segmentation     # BraTS
python -m backend.utils.dataset_download --kind both --dry-run   # show the plan
python -m backend.utils.dataset_download --kind both --force     # replace existing (prints what)
```

The downloader verifies the extracted files, prints the exact destination, never
overwrites without `--force`, and never logs any part of a credential.

The **SPECT** dataset is not auto-downloadable — point `SPECT_DATASET_PATH` at a
directory of class-named folders (`normal/`, `glioma/`, `meningioma/`,
`pituitary/`). Method 2 discovers its labels from those folder names and fails
with an explicit error if they do not match, rather than guessing.

---

## 🏋️ Training

### Method 3 — U-Net–ConvLSTM–SFLA (id `method1`)

```bash
# 1. Segmentation (volume-level splits, held-out test set)
python -m backend.methods.method1.training.train_segmentation --epochs 50 --batch-size 16

# 2. Dataset check (expects BRI_DATASET_PATH=./data/bri/archive with Training/ and Testing/)
python -m backend.methods.method1.inventory

# 3. Validation-only baseline (never loads Testing/)
python -m backend.methods.method1.training.train_classifier --skip-test --epochs 25 --patience 5

# 4. SFLA hyper-parameter search (train/val only — Testing/ is never loaded)
python -m backend.methods.method1.optimization.run_sfla --population 12 --memeplexes 3 \
    --iterations 10 --local-iterations 2 --proxy-epochs 3 --seed 42
#    ...then set SFLA_ENABLED=true in .env to make training use the result.
#    --dry-run optimises an analytic objective, so you can exercise it with no dataset.

# 5. Final classifier; evaluates Testing/ exactly once at the end
SFLA_ENABLED=true python -m backend.methods.method1.training.train_classifier --epochs 60 --patience 10
```

Without U-Net weights the classifier trains and serves the whole-slice geometry
(`m1-v1-legacy`), and says so. Preprocessed arrays are cached in
`backend/.model_cache/`, so only the first run pays for the non-local-means
denoise. Training copies byte-identical to a `Testing/` image (134 in the Kaggle
release) are dropped from the training pool; `Testing/` itself is untouched.

### Method 1 — transfer learning (id `method3`)

```bash
python -m backend.methods.method3.training.train_classifier            # EfficientNet-B0 + ResNet-50
python -m backend.methods.method3.training.train_classifier --skip-test # validation only
```

Head warm-up with the backbone frozen, then cosine fine-tuning; each backbone's
best weights are saved as soon as it finishes, and an identical rerun reuses them.

### Method 2 — Red Fox Optimized ZFNet (id `method4`)

```bash
python -m backend.methods.method4.optimization.run_rfo --population 6 --iterations 3 --proxy-epochs 2
python -m backend.methods.method4.training.train_classifier --epochs 40 --patience 8
```

The search reads train/validation only and writes `backend/logs/method4/rfo_results.json`,
which training then uses.

### Method 4 — MRI–SPECT fusion (id `method2`)


```bash
# 1. Multi-class segmentation head (BraTS — the only real multi-region masks)
python -m backend.methods.method2.training.train_segmentation --epochs 40 --batch-size 16

# 2. DCN classifier
python -m backend.methods.method2.training.train_classifier --modality spect --epochs 40
#    --modality mri is an explicit fallback using BRI. The resulting model is an MRI
#    model, is labelled as such in its checkpoint, and its numbers do not describe
#    SPECT performance. The two modalities are never merged into one training set.
```

Every run writes a **run card** to `backend/logs/` recording dataset, split
strategy, seed, preprocessing, image size, architecture, optimizer, epochs, best
epoch, metrics, checkpoint path, inference time, version, git revision and warnings.

### How splits work

Splits are **patient/volume-level**, always three-way (train / val / **held-out
test**), deterministic for a given `RANDOM_SEED`, and independent of file
discovery order. Every slice of a BraTS volume lands in exactly one part, and a
run refuses to start if the split produces overlapping groups.

Early stopping watches validation. **The test split is opened exactly once**, after
training finishes — and never by the SFLA search.

---

## 🔌 API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/methods` | List both methods with their live trained/untrained state |
| `GET` | `/api/methods/{method_id}` | Full detail: pipeline, datasets, weights, run cards, SFLA result |
| `POST` | `/api/predict/{method_id}` | Run that method's pipeline on an upload |
| `POST` | `/api/train/{method_id}?stage=…` | Safe training endpoint (see below) |
| `GET` | `/api/metrics/{method_id}` | That method's metrics — never mixed with the other's |
| `GET` | `/api/metrics` | One row per method, for the Compare page |
| `GET` | `/api/report/{method_id}/{id}` | Method-aware PDF report |
| `GET` | `/health`, `/model-info`, `/history`, `/upload` | Legacy routes (delegate to Method 1) |
| `GET` | `/docs` | Swagger UI |

Every prediction answers with the same envelope:

```json
{
  "method_id": "method2",
  "method_name": "Method 2 — Multi-class Segmentation + SPECT Feature Stage + DCN",
  "prediction": null,
  "confidence": null,
  "class_probabilities": {},
  "segmentation_available": false,
  "segmentation_mask_url": null,
  "heatmap_url": null,
  "processing_time_s": 0.0024,
  "model_version": "untrained",
  "dataset_context": "SPECT Image Dataset (modality); BraTS 2023 (segmentation)",
  "modality": "SPECT",
  "warnings": ["Method 2's DCN is not trained.", "…"],
  "details": { "feature_stage": { "dimension": 34, "names": [...], "values": [...] } }
}
```

`prediction` is `null` when no classifier ran — it is never filled with a guess.

**`POST /api/train`** returns the exact command to run and refuses to launch
anything unless `TRAINING_API_ENABLED=true`. An unauthenticated endpoint that
starts a multi-hour GPU job is a denial-of-service primitive.

---

## 🖥️ Frontend

- **Analyse** — `[ Method 1 ] [ Method 2 ]` toggle above the dropzone. Each option
  shows whether that method actually has trained weights. Switching methods clears
  the previous result, so one method's mask can never appear under the other's heading.
- **Compare** — both pipelines side by side, **without ranking them**. Dice, IoU,
  accuracy, precision, recall/sensitivity, specificity, F1, AUC and inference time,
  with `N/A` wherever a metric was not computed. No aggregate score is produced:
  the methods use different datasets, modalities and splits.
- **Metrics** — one method at a time, never merged.
- **History** — filterable by method; each entry carries the method that produced it.

---

## 🧪 Tests

```bash
pip install pytest httpx
pytest
```

88 tests covering the method registry, both response schemas, checkpoint
isolation, preprocessing consistency, missing-weight safety, SFLA reproducibility,
patient-level splitting, metric calculations and API method switching. Tests that
need torch skip cleanly when it is absent.

---

## 🐳 Docker

```bash
docker compose up --build
# frontend → http://localhost:3000
# backend  → http://localhost:8000/docs
```

`NEXT_PUBLIC_API_URL` is a **build argument**, not a runtime variable — Next.js
inlines it into the browser bundle at build time. Change it in
`docker-compose.yml` under `frontend.build.args` and rebuild.

---

## ⚠️ SPECT vs PECT

The reference diagram for Method 2 writes *Photon Emission Computed Tomography
(PECT)*; the dataset and this implementation are **SPECT** (Single Photon Emission
Computed Tomography). **These are not interchangeable terms.**

`METHOD2_MODALITY=SPECT` is the real modality and is what the UI, the API and the
PDF report display. `METHOD2_SPEC_MODALITY_LABEL=PECT` exists only so the research
specification's own wording can be rendered where it is explicitly required, and
it is always shown alongside the real modality — never as a claim that the two are
the same thing.

---

## 📋 Known limitations

- **Method 2 ships untrained.** No checkpoint, no metrics. This is reported
  explicitly everywhere rather than being papered over.
- **Method 1's U-Net is not included.** Segmentation is unavailable until trained.
  In `APP_ENV=development` a clearly-labelled `PLACEHOLDER` image may be rendered in
  the mask panel; it is never presented as a model output.
- **Method 1's 96.4% test accuracy is image-level, not patient-level.** It is measured
  once on the dataset's own `Testing/` folder, with validation carved from `Training/`
  only. The BRI dataset has no patient identifiers, so slices from one patient may
  sit on both sides and the figure may be optimistic. Exact duplicates are handled
  (grouped across train/val, dropped from training when they duplicate a test image),
  but near-duplicates are not.
- **The trained classifier uses whole slices, not U-Net ROI crops**, because no U-Net
  checkpoint exists yet. Training and serving use the same geometry.
- **Method 2's segmentation head trains on BraTS MRI** because that is the only
  multi-region annotated source available, while its classifier may run on SPECT.
  This cross-modality step is recorded in the checkpoint and surfaced as a warning.
- **`DenseConvNetClassifier` is a DenseNet-BC adaptation**, not a novel architecture.
  The adaptations (single-channel input, compact stem, modality-feature branch) are
  documented in `backend/methods/method2/models/dcn.py`.
- **Binary tumour/no-tumour datasets are rejected**, not remapped onto four classes.

---

## 📄 License

Released for research and educational purposes. Verify dataset licenses (BraTS /
Kaggle / SPECT source) before any redistribution.
