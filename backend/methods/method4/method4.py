"""Reference snippet 4/4 — the AlexNet branch, the fusion layer and the meta-learner.

**This file is documentation, not part of the pipeline.** Nothing here runs:
the code is held in strings, the module imports no TensorFlow, and no other
module imports this one.

Where it sits in the whole ensemble
-----------------------------------
* ``method1/method1.py`` — GoogleNet / InceptionV3 branch
* ``method2/method2.py`` — VGGNet / VGG16 branch
* ``method3/method3.py`` — ResNet50 branch
* ``method4/method4.py`` — AlexNet branch, fusion and meta-learner (this file)

This file holds the part that makes the four branches one model: the
concatenation and the dense stack that learns how to weigh them.

How it relates to this method
-----------------------------
This package is the Red Fox optimised ZFNet method (displayed as Method 2), and
the AlexNet branch below is its nearest relative: ZFNet *is* a re-tuned AlexNet
(smaller 7×7 stride-2 stem instead of 11×11 stride-4, so less is thrown away in
the first layer). Both are trained from scratch rather than pretrained.

Differences against ``models.py`` in this package:

* **Stem** — the snippet keeps AlexNet's 11×11 stride-4 convolution; the ZFNet
  here uses 7×7 stride-2, which is the change Zeiler & Fergus introduced.
* **Head** — the snippet pools globally and uses small dense layers; ZFNet keeps
  the two wide fully connected layers, whose width Red Fox Optimization searches
  (1024 was chosen).
* **Normalisation** — both replace AlexNet's local response normalisation with
  BatchNorm, which is the modern equivalent.
* **Hyper-parameters** — the snippet hard-codes dropout 0.5 and lr 1e-4. This
  method searched lr, weight decay, dropout, fully connected width and batch size
  with RFO on validation macro-F1 only, then trained once with the result.

On the fusion part: this is *architecture* fusion (four networks, one image), not
the *modality* fusion the MRI–SPECT method describes (one network, two imaging
modalities of the same patient). An ensemble like this could be built from the
four trained methods in this repository — but their measured accuracies would no
longer apply to the combination, which would need its own validation run and a
single held-out test evaluation of its own.
"""

from __future__ import annotations

__all__ = ["ALEXNET_BRANCH", "FUSION_AND_META_LEARNER", "MODEL_ASSEMBLY", "REFERENCE_SNIPPET"]

# Strings on purpose: reading this module has no side effects.
ALEXNET_BRANCH = '''
# ============================================
# Branch 4: AlexNet (Custom)
# ============================================
print("  🧠 Adding AlexNet branch...")
x4 = layers.Conv2D(96, (11, 11), strides=4, activation='relu')(inputs)
x4 = layers.MaxPooling2D((3, 3), strides=2)(x4)
x4 = layers.BatchNormalization()(x4)
x4 = layers.Conv2D(256, (5, 5), padding='same', activation='relu')(x4)
x4 = layers.MaxPooling2D((3, 3), strides=2)(x4)
x4 = layers.BatchNormalization()(x4)
x4 = layers.Conv2D(384, (3, 3), padding='same', activation='relu')(x4)
x4 = layers.Conv2D(384, (3, 3), padding='same', activation='relu')(x4)
x4 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x4)
x4 = layers.MaxPooling2D((3, 3), strides=2)(x4)
x4 = layers.GlobalAveragePooling2D()(x4)
x4 = layers.Dense(256, activation='relu')(x4)
x4 = layers.Dropout(0.5)(x4)
x4 = layers.Dense(128, activation='relu')(x4)
alexnet_out = layers.Dense(64, activation='relu', name='alexnet_features')(x4)
'''

# The five convolutions are AlexNet's: 96 @ 11×11/4, 256 @ 5×5, then 384, 384, 256
# @ 3×3, with overlapping 3×3 stride-2 pooling between them. Unlike the other
# three branches it has no pretrained weights, so it starts from random values and
# must learn everything from the training set.

FUSION_AND_META_LEARNER = '''
# ============================================
# Fusion Layer: Combine all branches
# ============================================
print("  🔗 Fusing all branches...")

# Concatenate features from all models
concatenated_features = Concatenate(name='feature_fusion')([googlenet_out, vgg_out, resnet_out, alexnet_out])

# Meta-learner (learns optimal combination)
meta_layer_1 = layers.Dense(256, activation='relu', name='meta_learner_1')(concatenated_features)
meta_layer_1 = layers.BatchNormalization()(meta_layer_1)
meta_layer_1 = layers.Dropout(0.5)(meta_layer_1)

meta_layer_2 = layers.Dense(128, activation='relu', name='meta_learner_2')(meta_layer_1)
meta_layer_2 = layers.BatchNormalization()(meta_layer_2)
meta_layer_2 = layers.Dropout(0.3)(meta_layer_2)

meta_layer_3 = layers.Dense(64, activation='relu', name='meta_learner_3')(meta_layer_2)
meta_layer_3 = layers.Dropout(0.2)(meta_layer_3)

# Output layer (Benign/Malignant classification)
outputs = layers.Dense(num_classes, activation='softmax', name='ensemble_output')(meta_layer_3)
'''

# Concatenate joins the four 64-dimensional branch outputs into one 256-dimensional
# vector. The meta-learner (256 → 128 → 64, with BatchNorm and decreasing dropout)
# is what "learns the optimal combination": nothing weights the branches explicitly,
# the dense layers simply learn which features help. Because three backbones are
# frozen, these layers plus the branch heads and AlexNet are the only trained parts.
# The softmax output has num_classes=2 — benign/malignant, not this project's four
# classes, so the labels would have to be redefined before the code could be used here.

MODEL_ASSEMBLY = '''
# Input layer
inputs = tf.keras.Input(shape=input_shape)

...  # the four branches, then the fusion block above

# Create the unified model
unified_model = KerasModel(inputs=inputs, outputs=outputs, name='Unified_Brain_Tumor_Ensemble')

unified_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
'''

# One Input feeds all four branches, so every network sees the same image, and the
# whole thing trains as a single model. 'sparse_categorical_crossentropy' expects
# integer labels; 'accuracy' alone hides per-class behaviour, which is what the
# metrics in this repository (per-class precision, recall, specificity, F1, plus the
# confusion matrix) are there to expose — on an imbalanced set, accuracy can look
# healthy while one class is being missed.

REFERENCE_SNIPPET = ALEXNET_BRANCH + FUSION_AND_META_LEARNER + MODEL_ASSEMBLY
