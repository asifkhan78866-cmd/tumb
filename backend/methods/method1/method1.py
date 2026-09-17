"""Reference snippet 1/4 — the GoogleNet (InceptionV3) branch of a Keras ensemble.

**This file is documentation, not part of the pipeline.** Nothing here runs:
the code is held in a string, the module imports no TensorFlow, and no other
module imports this one. It exists so the ensemble design can be read next to
the method it resembles.

Where it sits in the whole ensemble
-----------------------------------
The original snippet builds one Keras model with four parallel branches whose
features are concatenated and fed to a meta-learner. It is split across the
four method packages, one fragment each:

* ``method1/method1.py`` — GoogleNet / InceptionV3 branch (this file)
* ``method2/method2.py`` — VGGNet / VGG16 branch
* ``method3/method3.py`` — ResNet50 branch
* ``method4/method4.py`` — AlexNet branch, the fusion layer and the meta-learner

How it relates to this method
-----------------------------
This package is the U-Net → ROI → ConvLSTM → SFLA method (displayed as Method 3),
written in PyTorch. The snippet below is a *different* approach: a frozen
ImageNet backbone used as a fixed feature extractor, with only the small dense
head learning. This method instead segments the tumour first and classifies the
region with a recurrent model.

Differences that would have to be resolved before any of this could be used
here — which is why it is kept as reference only:

* **Framework** — TensorFlow/Keras; this project is PyTorch end to end.
* **Classes** — the ensemble ends in ``num_classes=2`` (benign/malignant). Every
  method here predicts four classes: glioma, meningioma, no tumour, pituitary.
* **Input** — ``(224, 224, 3)`` RGB; this method feeds 128×128 single-channel
  slices whose exact geometry is recorded in the checkpoint.
* **Frozen backbone** — ``base_inception.trainable = False`` trains only the
  head. The trained transfer-learning method in this repository warms up the
  head and then fine-tunes the whole backbone, which measured better.
"""

from __future__ import annotations

__all__ = ["REFERENCE_SNIPPET"]

# A string on purpose: reading this module has no side effects.
REFERENCE_SNIPPET = '''
# ============================================
# Branch 1: GoogleNet (InceptionV3)
# ============================================
print("  🧠 Adding GoogleNet branch...")
base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
base_inception.trainable = False

x1 = base_inception(inputs)
x1 = layers.GlobalAveragePooling2D()(x1)
x1 = layers.Dense(256, activation='relu')(x1)
x1 = layers.Dropout(0.5)(x1)
x1 = layers.Dense(128, activation='relu')(x1)
googlenet_out = layers.Dense(64, activation='relu', name='googlenet_features')(x1)
'''

# Line by line:
#   InceptionV3(...)          Inception-v3 with ImageNet weights, classifier head
#                             removed, so the branch outputs a feature map.
#   trainable = False         Freezes the backbone: its weights never update, so
#                             only the dense layers below are learned.
#   GlobalAveragePooling2D    Collapses the (H, W, C) map to one value per channel,
#                             giving a fixed-length vector regardless of input size.
#   Dense(256) → Dropout(0.5) Compresses those features; dropout regularises a head
#     → Dense(128)            trained on a small dataset.
#   Dense(64, name=...)       The branch's 64-dimensional output. The name matters:
#                             the fusion layer in method4/method4.py concatenates
#                             this tensor with the other three branches' outputs.
