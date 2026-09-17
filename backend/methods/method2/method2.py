"""Reference snippet 2/4 — the VGGNet (VGG16) branch of a Keras ensemble.

**This file is documentation, not part of the pipeline.** Nothing here runs:
the code is held in a string, the module imports no TensorFlow, and no other
module imports this one.

Where it sits in the whole ensemble
-----------------------------------
* ``method1/method1.py`` — GoogleNet / InceptionV3 branch
* ``method2/method2.py`` — VGGNet / VGG16 branch (this file)
* ``method3/method3.py`` — ResNet50 branch
* ``method4/method4.py`` — AlexNet branch, the fusion layer and the meta-learner

How it relates to this method
-----------------------------
This package is the MRI–SPECT multimodal fusion method (displayed as Method 4).
Both designs fuse several feature sources before one classifier, but they fuse
different things:

* This method fuses **modalities** — an MRI branch and a SPECT branch describing
  the same patient. Only its MRI branch is trained, because no paired MRI–SPECT
  dataset exists, so no fusion is actually performed.
* The snippet fuses **architectures** — four networks reading the *same* image.
  That is an ensemble, not multimodal fusion: it needs only one dataset and
  cannot add information a second modality would bring.

Other differences: TensorFlow/Keras rather than PyTorch, ``num_classes=2``
(benign/malignant) rather than this project's four classes, and a frozen
backbone rather than a trained-from-scratch dense network.

Note on cost: VGG16 is the heaviest branch of the four (about 138M parameters in
the full network, ~14.7M in this convolutional base), and freezing it means that
cost buys no task-specific learning.
"""

from __future__ import annotations

__all__ = ["REFERENCE_SNIPPET"]

# A string on purpose: reading this module has no side effects.
REFERENCE_SNIPPET = '''
# ============================================
# Branch 2: VGGNet (VGG16)
# ============================================
print("  🧠 Adding VGGNet branch...")
base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
base_vgg.trainable = False

x2 = base_vgg(inputs)
x2 = layers.GlobalAveragePooling2D()(x2)
x2 = layers.Dense(256, activation='relu')(x2)
x2 = layers.Dropout(0.5)(x2)
x2 = layers.Dense(128, activation='relu')(x2)
vgg_out = layers.Dense(64, activation='relu', name='vgg_features')(x2)
'''

# Line by line:
#   VGG16(...)                A plain stack of 3×3 convolutions and max-pools —
#                             no residual or inception blocks. include_top=False
#                             drops its 1000-class ImageNet classifier.
#   trainable = False         Frozen: used purely as a fixed feature extractor.
#   GlobalAveragePooling2D    (7, 7, 512) → 512 values, one per channel.
#   Dense(256) → Dropout(0.5) The same head shape as every other branch, so the
#     → Dense(128) → Dense(64) four outputs can be concatenated as equals.
#   name='vgg_features'       Referenced by the fusion layer in method4/method4.py.
