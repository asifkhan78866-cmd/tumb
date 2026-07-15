# 🧠 NeuroSeg AI — Brain Tumor Segmentation & Classification

A production-ready, full-stack deep-learning system that **segments** brain
tumors with a U-Net and **classifies** them with a ConvLSTM, wrapped in a FastAPI
backend and a modern Next.js 15 medical dashboard.

> ⚠️ **Research & educational use only.** This is not a certified medical device
> and must not be used for clinical diagnosis.

---

## ✨ Features

| Area | Highlights |
|------|-----------|
| **Segmentation** | 2D U-Net · encoder/decoder · skip connections · BatchNorm · Dropout · Dice + BCE loss · Adam · LR scheduler · mixed precision · early stopping · TensorBoard · checkpointing (`best_unet.pth`) |
| **Classification** | ConvLSTM (Conv + BatchNorm + ConvLSTM + Dense + Softmax) · 4 classes: Glioma / Meningioma / Pituitary / No Tumor (`best_classifier.pth`) |
| **Explainability** | Grad-CAM heatmap overlay on every prediction |
| **Data** | Automatic Kaggle download (BraTS 2023 → fallbacks) · integrity check · dataset summary |
| **API** | FastAPI · `/upload`, `/health`, `/model-info`, `/train-status`, `/metrics`, `/history`, `/report/{id}` · Swagger at `/docs` |
| **Frontend** | Next.js 15 · TypeScript · TailwindCSS · shadcn-style UI · Framer Motion · dark mode · drag-and-drop upload · progress bar · confidence meter · charts · history · PDF report · toasts · loading skeletons |
| **Reports** | One-click PDF (image, mask, Grad-CAM, class, confidence, date, model) |
| **Deploy** | Dockerfiles + `docker-compose.yml` |

---

## 📁 Project structure

```
tumb/
├── backend/
│   ├── api/            # FastAPI routes + Pydantic schemas
│   ├── models/         # UNet, ConvLSTM classifier
│   ├── training/       # train_segmentation, train_classifier, evaluate, common
│   ├── utils/          # preprocessing, dataset, dataset_download, losses, metrics, gradcam, report
│   ├── services/       # inference engine + JSON stores
│   ├── weights/        # best_unet.pth, best_classifier.pth (generated)
│   ├── dataset/        # auto-downloaded data (generated)
│   ├── predictions/    # generated images + PDF reports
│   ├── logs/           # TensorBoard, plots, metrics.json
│   ├── config.py       # central config (paths, hyper-params, device)
│   ├── main.py         # FastAPI app entrypoint
│   ├── predict.py      # CLI single-image inference
│   └── requirements.txt
├── frontend/
│   ├── app/            # landing, upload, history, metrics, about (App Router)
│   ├── components/     # navbar, dropzone, prediction card, charts, ui/*
│   ├── hooks/          # useTrainStatus
│   ├── lib/            # api client, utils
│   └── services/       # prediction service
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run the API immediately** (works even before training — falls back to
randomly-initialised weights so you can exercise the full UI):

```bash
# from the repo root:
uvicorn backend.main:app --reload
# → http://localhost:8000  ·  Swagger at http://localhost:8000/docs
```

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env.local          # points NEXT_PUBLIC_API_URL at the backend
npm run dev
# → http://localhost:3000
```

Upload an MRI on the **Upload** page and you'll get the original scan, the
segmentation mask, a Grad-CAM overlay, the predicted class, confidence and
inference time — plus a downloadable PDF report.

---

## 🔑 API keys (Kaggle)

Dataset downloading needs a free **Kaggle API token**:

1. Go to <https://www.kaggle.com/settings> → **Create New API Token** (downloads
   `kaggle.json` with your `username` and `key`).
2. Put those values in `backend/.env`:

   ```bash
   cp backend/.env.example backend/.env    # if you haven't already
   # then edit backend/.env:
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key
   ```

`backend/.env` is git-ignored. (Alternatively, drop `kaggle.json` at
`~/.kaggle/kaggle.json` and `chmod 600` it — either works.)

## 📦 Dataset (automatic)

> **Accuracy matters here:** classification labels and segmentation masks come
> from **different** datasets, so the system downloads both for best results:
>
> | Task | Dataset | Provides |
> |------|---------|----------|
> | Classification | `masoudnickparvar/brain-tumor-mri-dataset` | 4 labelled classes (glioma/meningioma/pituitary/notumor) |
> | Segmentation | `awsaf49/brats2020-training-data` (BraTS) | Ground-truth tumor masks |

```bash
python -m backend.utils.dataset_download                    # both (default)
python -m backend.utils.dataset_download --kind classification
python -m backend.utils.dataset_download --kind segmentation --force
```

Both extract into `backend/dataset/`; the script verifies integrity and prints a
summary (patients, classes, image/mask counts) also written to
`backend/dataset/summary.json`. The training scripts auto-download the dataset
they need if it's missing (`train_classifier` → classification data,
`train_segmentation` → BraTS).

### For the best / most accurate results

- **Classification** already uses the properly labelled 4-class dataset above.
- **Segmentation**: BraTS ships real masks — train on those (not the Otsu
  fallback). BraTS is ~2–15 GB and benefits hugely from a **GPU**.
- Train longer with the defaults (`SEG_EPOCHS=50`, `CLS_EPOCHS=40`) or raise them
  in `backend/.env`; early stopping prevents overfitting.
- Run `evaluate.py` afterwards to populate real metrics on the dashboard.

---

## 🏋️ Training

```bash
# from the repo root (recommended):
python -m backend.training.train_segmentation --epochs 50 --batch-size 16
python -m backend.training.train_classifier   --epochs 40 --batch-size 16
python -m backend.training.evaluate           # writes metrics.json + plots

# …or from inside backend/ using the shims:
cd backend
python train_segmentation.py
python train_classifier.py
python evaluate.py
```

Each run prints per-epoch **loss, Dice, accuracy, precision, recall, F1,
specificity, sensitivity**, saves the best checkpoint, and writes loss/Dice/
accuracy curves, a **confusion matrix** and **ROC curves** to `backend/logs/`.
Live TensorBoard:

```bash
tensorboard --logdir backend/logs
```

Training progress is exposed to the frontend via `GET /train-status`.

> **Note on segmentation masks:** BraTS provides ground-truth masks (used
> directly). If a mask-free fallback dataset is downloaded, the pipeline derives
> weak tumor masks via Otsu thresholding so segmentation remains trainable
> end-to-end — swap in real masks for clinical-grade results.

---

## 🔌 API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload an MRI → segmentation + classification result |
| `GET`  | `/health` | Service + weight-load status |
| `GET`  | `/model-info` | Architectures, classes, parameter counts |
| `GET`  | `/train-status` | Live training progress |
| `GET`  | `/metrics` | Aggregate evaluation metrics |
| `GET`  | `/history?limit=` | Recent predictions |
| `GET`  | `/report/{id}` | Download a PDF report |
| `GET`  | `/docs` | Swagger UI |

**`/upload` response:**

```json
{
  "prediction_id": "a1b2c3d4e5f6",
  "class": "Glioma",
  "confidence": 98.2,
  "inference_time": "0.32 sec",
  "segmentation_mask": "/predictions/a1b2c3d4e5f6_mask.png",
  "original_image": "/predictions/a1b2c3d4e5f6_original.png",
  "gradcam_overlay": "/predictions/a1b2c3d4e5f6_overlay.png",
  "probabilities": { "Glioma": 98.2, "Meningioma": 1.1, "No Tumor": 0.4, "Pituitary": 0.3 }
}
```

### CLI inference

```bash
python -m backend.predict path/to/mri.png --report
```

---

## 🐳 Docker

```bash
docker compose up --build
# frontend → http://localhost:3000
# backend  → http://localhost:8000/docs
```

Weights, dataset and predictions are mounted as volumes so they persist across
restarts. For GPU training, base the backend image on `nvidia/cuda` and install
the matching PyTorch build.

---

## 🖥️ Hardware

Runs on **CPU by default** and automatically uses **CUDA** (with mixed precision)
when a GPU is available — no code changes needed. Device is reported at
`GET /health` and on the Metrics page.

---

## 🧪 Tech stack

**Backend:** Python · PyTorch · OpenCV · NiBabel · FastAPI · Uvicorn · ReportLab · TensorBoard
**Frontend:** Next.js 15 · React 19 · TypeScript · TailwindCSS · Framer Motion · Recharts · Sonner · next-themes

---

## 📄 License

Released for research and educational purposes. Verify dataset licenses (BraTS /
Kaggle) before any redistribution.
