"""Reference snippet 3/4 — the ResNet50 branch of a Keras ensemble.

**This file is documentation, not part of the pipeline.** Nothing here runs:
the code is held in a string, the module imports no TensorFlow, and no other
module imports this one.

Where it sits in the whole ensemble
-----------------------------------
* ``method1/method1.py`` — GoogleNet / InceptionV3 branch
* ``method2/method2.py`` — VGGNet / VGG16 branch
* ``method3/method3.py`` — ResNet50 branch (this file)
* ``method4/method4.py`` — AlexNet branch, the fusion layer and the meta-learner

How it relates to this method
-----------------------------
This package is the transfer-learning method (displayed as Method 1), and it is
the closest match in the repository: it also fine-tunes ImageNet-pretrained
backbones, and **ResNet-50 is the backbone it actually selected** (validation
macro-F1 0.992 against EfficientNet-B0's 0.980), reaching 98.70% on the held-out
test set.

Two differences are worth noting against the trained implementation in
``training/train_classifier.py``:

* **Frozen vs fine-tuned.** The snippet sets ``base_resnet.trainable = False``,
  so ImageNet features are used as-is. This method freezes the backbone only for
  a short head warm-up, then unfreezes it and fine-tunes with a lower learning
  rate and cosine decay — brain MRI is far from ImageNet's natural images, so
  adapting the backbone matters.
* **Ensemble vs selection.** The snippet keeps all four branches and learns to
  combine them. This method trains candidate backbones separately and keeps the
  single best on validation, then tests that one once. That is cheaper to serve
  and leaves one model to explain with Grad-CAM, at the cost of any accuracy an
  ensemble might have added.

Also: Keras rather than PyTorch, ``num_classes=2`` rather than four classes, and
RGB 224×224 input rather than this method's grayscale-replicated-to-3-channel
224×224 tensors.
"""

from __future__ import annotations

__all__ = ["REFERENCE_SNIPPET"]

# A string on purpose: reading this module has no side effects.
REFERENCE_SNIPPET = '''
# ============================================
# Branch 3: ResNet (ResNet50)
# ============================================
print("  🧠 Adding ResNet branch...")
base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
base_resnet.trainable = False

x3 = base_resnet(inputs)
x3 = layers.GlobalAveragePooling2D()(x3)
x3 = layers.Dense(256, activation='relu')(x3)
x3 = layers.BatchNormalization()(x3)
x3 = layers.Dropout(0.5)(x3)
x3 = layers.Dense(128, activation='relu')(x3)
resnet_out = layers.Dense(64, activation='relu', name='resnet_features')(x3)
'''

# Line by line:
#   ResNet50(...)             50 layers of residual blocks; the skip connections
#                             are what let a network this deep train at all.
#   trainable = False         Frozen backbone (see the caveat above).
#   GlobalAveragePooling2D    (7, 7, 2048) → 2048 values.
#   BatchNormalization()      Present only in this branch — it standardises the
#                             256-unit activations, which steadies training when
#                             the four branches' feature scales differ.
#   Dropout(0.5) → Dense(128) The shared head shape.
#     → Dense(64)
#   name='resnet_features'    Referenced by the fusion layer in method4/method4.py.
