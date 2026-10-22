"""Generate the notebooks under ``notebooks/``.

    python -m backend.cli.build_notebooks            # write them
    python -m backend.cli.build_notebooks --execute  # write and run them, so the
                                                     # outputs are saved in the file

One notebook per method, plus an overview. They are *generated* rather than
hand-written because every heading, metric and caveat in them comes from the
registry and the recorded evaluations — a notebook that quoted its own numbers
would drift from the code the moment anything was retrained.

Each method notebook follows the same order: what the method is, the data it
was trained on, the preprocessing its checkpoint records, the model as loaded,
one prediction with every image it produces, the recorded held-out evaluation,
whatever stage is particular to that method, and the caveats that come with the
numbers. Nothing in them trains anything; the training commands are printed,
not run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

from backend import config
from backend.methods.registry import get_method, list_methods

NOTEBOOK_DIR = config.ROOT_DIR / "notebooks"
KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}

# Every notebook starts here: the project on sys.path, so a cell can import the
# same code the API runs rather than a copy of it.
BOOTSTRAP = '''import sys, warnings
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "backend").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np

from backend import config
from backend.methods.registry import get_method, load_engine, runtime_status
from backend.methods.common.metrics_store import read_metrics, metrics_for

plt.rcParams["figure.dpi"] = 110
print("project root :", ROOT)
print("device       :", config.DEVICE)
print("dataset      :", config.BRI_DATASET_PATH)'''

SHOW_HELPERS = '''def show(images, titles, width=4.2):
    """Draw images side by side with their titles."""
    images = [i for i in images if i is not None]
    fig, axes = plt.subplots(1, len(images), figsize=(width * len(images), width))
    for ax, image, title in zip(np.atleast_1d(axes), images, titles):
        ax.imshow(image if image.ndim == 2 else image[..., ::-1], cmap="gray" if image.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def read(path):
    import cv2
    return None if path is None else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def table(rows, headers):
    """A plain-text table, so the notebook reads the same with or without pandas."""
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))'''


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def notebook(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    return nb


# --------------------------------------------------------------------------- #
# Per-method extras: the stage that makes each method itself
# --------------------------------------------------------------------------- #
def extra_cells(method_id: str) -> list:
    if method_id == "method4":
        return [
            md("""
## 7 · Red Fox feature selection

The network's own softmax layer is not what classifies here. A second Red Fox
search chose a subset of ZFNet's feature vector and the SVM's hyper-parameters,
scored on **validation** macro-F1; the head that won on validation is the one
served. Both heads were then evaluated once on the test set.
"""),
            code('''search = json.loads((config.LOGS_DIR / "method4/rfo_feature_selection.json").read_text())
print(f"searched      : {search['features_available_to_search']} features + 3 SVM hyper-parameters")
print(f"kept          : {search['features_selected']} features")
print(f"SVM           : {search['svm']}")
print(f"evaluations   : {search['evaluations']} in {search['duration_s']:.0f}s")
print(f"served head   : {search['served_head']}  ({search['head_choice']})")
print()
table([[k, f"{v['accuracy']:.4f}", f"{v['f1']:.4f}"] for k, v in search["validation"].items()],
      ["head", "validation acc", "validation F1"])
print()
table([[k, f"{v['accuracy']:.4f}", f"{v['f1']:.4f}"] for k, v in search["test"].items()],
      ["head", "TEST accuracy", "TEST macro F1"])'''),
            code('''show([read(config.LOGS_DIR / "method4/rfo_feature_selection.png")],
     ["Red Fox fitness per iteration (validation)"], width=7)'''),
        ]

    if method_id == "method1":
        return [
            md("""
## 7 · SFLA feature selection, and the geometry that lost

Two searches, both scored on validation only. The first chose the ConvLSTM's
hyper-parameters; the second chose which of its 128 features an SVM should
classify. Separately, the ROI-cropped geometry the architecture diagram shows
was trained to convergence and **lost** to whole slices on validation, so whole
slices are what the served checkpoint uses.
"""),
            code('''search = json.loads((config.LOGS_DIR / "method1/sfla_feature_selection.json").read_text())
print(f"kept        : {search['features_selected']} of {search['features_available_to_search']} features")
print(f"SVM         : {search['svm']}")
print(f"evaluations : {search['evaluations']} in {search['duration_s']:.0f}s")
print(f"served head : {search['served_head']}")
print()
table([[k, f"{v['f1']:.4f}"] for k, v in search["validation"].items()], ["head", "validation F1"])
table([[k, f"{v['accuracy']:.4f}", f"{v['f1']:.4f}"] for k, v in search["test"].items()],
      ["head", "TEST accuracy", "TEST macro F1"])'''),
            code('''geometry = json.loads((config.LOGS_DIR / "method1/roi_geometry_comparison.json").read_text())
print(geometry["question"])
print(f"decided on  : {geometry['decided_on']}\\n")
table([[name, c["transform_id"], f"{c['validation']['f1']:.4f}", c["best_epoch"]]
       for name, c in geometry["candidates"].items()],
      ["candidate", "transform", "validation F1", "best epoch"])
print(f"\\nchosen: {geometry['decision']}")
for note in geometry["notes"]:
    print(" •", note)'''),
            md("""
### The volumetric model

A 3D U-Net was trained on 484 Decathlon (BraTS) volumes, split by patient. It is
**not** what segments an upload — an upload is one slice and a 3D model needs the
slices around it — so its Dice is reported on its own.
"""),
            code('''volumetric = read_metrics("method1").get("segmentation_3d") or {}
if volumetric:
    table([[volumetric["architecture"], volumetric["dice"], volumetric["decision_threshold"],
            volumetric["volumes"], volumetric["predicted_to_true_volume_ratio"]]],
          ["model", "TEST Dice", "threshold", "volumes", "predicted/true volume"])
    print("\\nthreshold chosen on validation:")
    for row in json.loads((config.LOGS_DIR / "method1/segmentation3d_metrics.json").read_text())["validation_threshold_sweep"]:
        print(f"   {row['threshold']:.2f} -> Dice {row['dice']:.4f}  (volume ratio {row['predicted_to_true_volume_ratio']})")
else:
    print("No volumetric model trained on this machine.")'''),
            code('''show([read(config.LOGS_DIR / "method1/sfla_feature_selection.png"),
      read(config.LOGS_DIR / "method1/seg3d_dice.png")],
     ["SFLA fitness per iteration (validation)", "3D U-Net patch Dice per epoch"], width=5.5)'''),
        ]

    if method_id == "method2":
        return [
            md("""
## 7 · The fusion stage

This is the stage the method is named after, and it is deterministic — there is
nothing in it to train. The SPECT image goes to YCrCb, its colour channels pass
through untouched because the colour *is* the functional reading, and only the
luminance is fused with the MRI through a Laplacian pyramid.

Measured on all 357 registered pairs against the trivial averaging baseline.
"""),
            code('''fusion = read_metrics("method2")["fusion"]
print(f"{fusion['pairs']} pairs over {fusion['cases']} cases · {fusion['ms_per_pair']} ms/pair\\n")
table([[k, f"{fusion['metrics'][k]['mean']:.4f}", f"{fusion['average_baseline'][k]['mean']:.4f}"]
       for k in ("qabf", "spatial_frequency", "entropy", "ssim_mri", "mutual_information_spect")],
      ["metric", "Laplacian fusion", "averaging baseline"])
print("\\nQ_abf is the share of both sources' gradient information kept (0-1).")'''),
            code('''from backend.methods.method2 import fusion as F

name = sorted(p.name for p in (config.PAIRED_MRI_SPECT_PATH / "MRI").glob("*.png"))[10]
mri = read(config.PAIRED_MRI_SPECT_PATH / "MRI" / name)
spect = read(config.PAIRED_MRI_SPECT_PATH / "SPECT" / name)
pair = F.fuse(mri, spect)
show([mri, spect, (pair.fused_rgb * 255).astype("uint8")],
     [f"MRI ({name})", "SPECT (perfusion)", "fused"])
for k, v in F.metrics(pair.mri, pair.spect_luminance, pair.fused_luminance).items():
    print(f"  {k:<26}{v}")'''),
            md("""
### Why no accuracy for the fusion

The paired set carries no diagnosis labels, so nothing can be trained or scored
for classification on it. The four-class prediction above comes from the trained
MRI branch, and this notebook never claims otherwise.
"""),
        ]

    if method_id == "method3":
        return [
            md("""
## 7 · What the four backbones each contribute

The served checkpoint is the fusion ensemble: GoogleNet, VGG16, ResNet-50 and
AlexNet are frozen, each reduced to a 64-d branch vector, and a meta-learner
weighs the concatenation. Only the branch heads and the meta-learner were
trained, which is also why no augmentation was used — an augmented copy would
need its own pass through all four backbones.
"""),
            code('''meta = json.loads((config.WEIGHTS_DIR / "method3/model_metadata.json").read_text())
print("architecture :", meta["architecture"])
print("frozen       :", ", ".join(meta["frozen_backbones"]))
print("trainable    :", f"{meta['trainable_parameters']:,} parameters")
print("augmentation :", meta["augmentation"])
print("best epoch   :", meta["best_epoch"], "of", meta["epochs_run"])
print()
table([[k, f"{meta['validation_metrics'][k]:.4f}", f"{meta['test_metrics'][k]:.4f}"]
       for k in ("accuracy", "precision", "recall", "f1", "auc")],
      ["metric", "validation", "TEST"])'''),
        ]
    return []


# --------------------------------------------------------------------------- #
def method_notebook(method_id: str) -> nbf.NotebookNode:
    spec = get_method(method_id)
    cells = [
        md(f"""
# Method {spec.display_number} · {spec.short_name}

{spec.summary}

**Modality:** {spec.modality}
**Classifier:** {spec.classifier_model}
**Segmentation:** {spec.segmentation_model}
**Optimisation:** {spec.optimization or "—"}

Nothing here trains anything. Every number is read from a recorded evaluation,
every image is produced by the checkpoint on disk, and the training commands are
printed rather than run.
"""),
        md("## 1 · Setup"),
        code(BOOTSTRAP),
        code("import json\n\n" + SHOW_HELPERS),
        code(f'''METHOD = "{method_id}"
spec = get_method(METHOD)
status = runtime_status(spec)

print(f"Method {{spec.display_number}} — {{spec.display_name}}\\n")
for stage in spec.pipeline:
    print(f"  {{stage.id:<16}}{{stage.label:<28}}{{stage.description}}")
print()
table([[role, w["architecture"], "yes" if w["present"] else "NO", w["model_version"] or "—"]
       for role, w in status["weights"].items()],
      ["role", "architecture", "checkpoint", "version"])'''),
        md("""
## 2 · The data

The images below come from the dataset's own **Testing** folder — the held-out
split the metrics further down were measured on. They are examples, not
predictions.
"""),
        code('''root = config.BRI_DATASET_PATH / "Testing"
samples = {}
for class_key in spec.class_names:
    folder = root / ({"normal": "notumor"}.get(class_key, class_key))
    files = sorted(folder.glob("*.jpg"))[:1] if folder.is_dir() else []
    if files:
        samples[spec.class_labels[class_key]] = files[0]

show([read(p) for p in samples.values()], list(samples), width=3.2)
table([[label, path.name] for label, path in samples.items()], ["class", "example file"])'''),
        md("""
## 3 · Preprocessing

The geometry is read from the checkpoint, not chosen here: a model is served the
same transform it was trained with.
"""),
        code('''engine = load_engine(METHOD)
engine.load()
transform = status["weights"].get("classification", {}).get("transform_id")
print("transform recorded in the checkpoint:", transform or "—")

import importlib
transforms = importlib.import_module(spec.transforms_module)
spec_obj = transforms.get_spec(transform) if hasattr(transforms, "get_spec") else transforms.DEFAULT_SPEC
for k, v in spec_obj.to_dict().items():
    print(f"  {k:<16}{v}")'''),
        md("""
## 4 · The model as loaded

Built from the checkpoint's own record of its hyper-parameters, then the stored
weights are loaded into it. A checkpoint that belonged to another method, or to
other classes, is refused rather than coerced.
"""),
        code('''model = getattr(engine, "model", None) or getattr(engine, "cls_model", None)
print(type(model).__name__)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
print()
for key, value in (engine.meta if hasattr(engine, "meta") else engine.cls_meta).items():
    if key != "extra":
        print(f"  {key:<18}{value}")'''),
        md("""
## 5 · One prediction, with everything it produced

The class, the probabilities, and every image this method writes: the
preprocessed input it actually read, the segmentation mask, the Grad-CAM
heatmap, and — for the fusion method — the fused image.
"""),
        code('''image_path = list(samples.values())[0]
result = engine.predict(image_path.read_bytes())

print(f"input      : {image_path}")
print(f"prediction : {result.prediction}   {result.confidence}%")
print(f"model      : {result.model_version}   ({result.inference_time_s}s)\\n")
for name, value in sorted(result.probabilities.items(), key=lambda kv: -kv[1]):
    print(f"  {name:<14}{'█' * int(value / 4):<25}{value:6.2f}%")'''),
        code('''panels = [("input", result.original_path), ("mask", result.mask_path),
          ("Grad-CAM", result.overlay_path), ("fused", getattr(result, "fused_path", None))]
panels = [(label, path) for label, path in panels if path]
show([read(p) for _, p in panels], [label for label, _ in panels])

note = result.details.get("segmentation_note")
if note:
    print("mask:", note)
if result.details.get("gradcam_note"):
    print("Grad-CAM:", result.details["gradcam_note"])'''),
        md("""
## 6 · The recorded held-out evaluation

These numbers are not computed here. They were measured once, by the training
run, on the test split — and are read back from the metrics file so this
notebook cannot quietly disagree with the API.
"""),
        code('''metrics = metrics_for(METHOD)
table([[k, f"{metrics[k]:.4f}" if metrics[k] is not None else "—"]
       for k in ("accuracy", "precision", "recall", "specificity", "f1", "auc", "dice", "iou")],
      ["metric", "held-out test"])
print()
print(metrics["split"])'''),
        code(f'''show([read(config.LOGS_DIR / "{method_id}_cls_confusion_matrix.png"),
      read(config.LOGS_DIR / "{method_id}_cls_roc.png")],
     ["Confusion matrix (held-out test)", "ROC curves (one-vs-rest)"], width=5.5)'''),
        *extra_cells(method_id),
        md("""
## 8 · Caveats

Every warning the method records travels with its numbers. They are part of the
result, not a footnote to it.
"""),
        code('''seen = set()
for section, values in read_metrics(METHOD).items():
    if not isinstance(values, dict):
        continue
    for warning in values.get("warnings", []) or []:
        if warning not in seen:
            seen.add(warning)
            print(f"[{section}] {warning}\\n")'''),
        md(f"""
## 9 · Reproducing this

```bash
{chr(10).join(spec.training_entrypoints)}
```

Or run every method on one image from a terminal:

```bash
python -m backend.cli.run_methods data/bri/archive/Testing/glioma/Te-gl_0012.jpg
```
"""),
    ]
    return notebook(cells)


def overview_notebook() -> nbf.NotebookNode:
    cells = [
        md("""
# NeuroSeg AI · all four methods

Four separately trained pipelines over the same brain-MRI dataset, each
evaluated **once** on the same held-out test images. This notebook is the
summary; one notebook per method sits beside it.

| notebook | method |
|---|---|
| `01_method1_pretrained_transfer.ipynb` | Pre-trained CNNs + transfer learning (feature fusion) |
| `02_method2_zfnet_redfox.ipynb` | Red Fox optimised ZFNet (+ feature selection → SVM) |
| `03_method3_unet_convlstm_sfla.ipynb` | U-Net + ConvLSTM + SFLA (+ 3D U-Net) |
| `04_method4_mri_spect_fusion.ipynb` | MRI–SPECT fusion |
"""),
        md("## 1 · Setup"),
        code(BOOTSTRAP),
        code("import json\n\n" + SHOW_HELPERS),
        md("""
## 2 · What is trained on this machine

A method with no checkpoint reports no prediction rather than a guess, so this
table is worth reading before any number below it.
"""),
        code('''from backend.methods.registry import list_methods

rows = []
for spec in list_methods():
    status = runtime_status(spec)
    rows.append([spec.display_number, spec.short_name,
                 "yes" if status["classifier_available"] else "NO",
                 "yes" if status["segmentation_available"] else ("—" if not status["has_segmentation_stage"] else "NO"),
                 spec.modality])
table(sorted(rows), ["#", "method", "classifier", "segmentation", "modality"])'''),
        md("""
## 3 · The dataset

The same four-class MRI set behind every method, with its own Training/Testing
split. The test folder below is what every accuracy in this project was measured
on.
"""),
        code('''root = config.BRI_DATASET_PATH
rows = []
for part in ("Training", "Testing"):
    for folder in sorted((root / part).iterdir()):
        if folder.is_dir():
            rows.append([part, folder.name, len(list(folder.glob("*.jpg")))])
table(rows, ["split", "class", "images"])

examples = [next((root / "Testing" / c).glob("*.jpg")) for c in
            ("glioma", "meningioma", "notumor", "pituitary")]
show([read(p) for p in examples], ["glioma", "meningioma", "no tumour", "pituitary"], width=3.2)'''),
        md("""
## 4 · Held-out results, side by side

One test evaluation per method, on the same images. They are **not** an
ensemble and their numbers are not averaged: four independent pipelines, four
independent results.
"""),
        code('''rows = []
for spec in list_methods():
    m = metrics_for(spec.method_id)
    rows.append([spec.display_number, spec.short_name,
                 f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "—",
                 f"{m['f1']:.4f}" if m["f1"] is not None else "—",
                 f"{m['auc']:.4f}" if m["auc"] is not None else "—",
                 f"{m['dice']:.4f}" if m["dice"] is not None else "—"])
table(sorted(rows), ["#", "method", "accuracy", "macro F1", "AUC", "seg Dice"])'''),
        code('''fig, ax = plt.subplots(figsize=(8, 3.4))
specs = sorted(list_methods(), key=lambda s: s.display_number)
values = [(metrics_for(s.method_id)["accuracy"] or 0) * 100 for s in specs]
bars = ax.barh([f"{s.display_number}. {s.short_name}" for s in specs], values, color="#3b82f6")
ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=9)
ax.set_xlim(0, 108)
ax.set_xlabel("held-out test accuracy (%)")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()'''),
        md("""
## 5 · The same image through every method

Disagreement here is information, not noise: these are different architectures
trained separately, so where they disagree no vote among them is evidence.
"""),
        code('''image_path = next((config.BRI_DATASET_PATH / "Testing/meningioma").glob("*.jpg"))
image_bytes = image_path.read_bytes()
print("input:", image_path.name)

results = {}
for spec in sorted(list_methods(), key=lambda s: s.display_number):
    engine = load_engine(spec.method_id)
    results[spec.display_number] = (spec, engine.predict(image_bytes))

table([[n, s.short_name, r.prediction or "no prediction",
        f"{r.confidence:.2f}%" if r.confidence else "—", f"{r.inference_time_s:.2f}s"]
       for n, (s, r) in results.items()],
      ["#", "method", "prediction", "confidence", "time"])'''),
        code('''for number, (spec, result) in results.items():
    panels = [("input", result.original_path), ("mask", result.mask_path),
              ("Grad-CAM", result.overlay_path)]
    panels = [(label, path) for label, path in panels if path]
    print(f"Method {number} · {spec.short_name} → {result.prediction} ({result.confidence}%)")
    show([read(p) for _, p in panels], [label for label, _ in panels], width=3.4)'''),
        md("""
## 6 · Everything these results come with

The warnings are recorded by the training runs themselves and are part of each
result.
"""),
        code('''for spec in sorted(list_methods(), key=lambda s: s.display_number):
    warnings_seen = []
    for section, values in read_metrics(spec.method_id).items():
        if isinstance(values, dict):
            warnings_seen += [w for w in (values.get("warnings") or []) if w not in warnings_seen]
    print(f"── Method {spec.display_number}: {spec.short_name} " + "─" * 30)
    for warning in warnings_seen[:4]:
        print(" •", warning)
    print()'''),
    ]
    return notebook(cells)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the per-method notebooks")
    ap.add_argument("--execute", action="store_true",
                    help="run each notebook and save its outputs into the file")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    out = Path(args.outdir).expanduser() if args.outdir else NOTEBOOK_DIR
    out.mkdir(parents=True, exist_ok=True)

    names = {"method3": "01_method1_pretrained_transfer",
             "method4": "02_method2_zfnet_redfox",
             "method1": "03_method3_unet_convlstm_sfla",
             "method2": "04_method4_mri_spect_fusion"}
    written = [(out / "00_overview.ipynb", overview_notebook())]
    for spec in sorted(list_methods(), key=lambda s: s.display_number):
        written.append((out / f"{names[spec.method_id]}.ipynb", method_notebook(spec.method_id)))

    for path, nb in written:
        nbf.write(nb, path)
        # --outdir may point anywhere, so the path is only shortened when it can be.
        try:
            shown = path.relative_to(config.ROOT_DIR)
        except ValueError:
            shown = path
        print(f"[notebooks] wrote {shown} ({len(nb.cells)} cells)")

    if args.execute:
        from nbclient import NotebookClient

        for path, nb in written:
            print(f"[notebooks] executing {path.name} …", flush=True)
            client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                    resources={"metadata": {"path": str(config.ROOT_DIR)}},
                                    allow_errors=False)
            client.execute()
            nbf.write(nb, path)
            print(f"[notebooks]   saved with outputs -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
