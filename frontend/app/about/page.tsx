"use client";

import { AlertTriangle, BookOpen, Database, GitCompareArrows, ShieldAlert } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <h1 className="text-3xl font-bold">About this project</h1>
        <p className="mt-2 text-muted-foreground">
          Two independent brain-tumour analysis pipelines, kept deliberately separate.
        </p>
      </div>

      <Card className="border-destructive/40 bg-destructive/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <ShieldAlert className="h-5 w-5" /> Not a medical device
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p>
            This is a <strong>research and decision-support</strong> project. It is not a
            certified medical device, it has not been clinically validated, and it must
            not be used to diagnose, treat, or make any care decision for a patient.
          </p>
          <p className="text-muted-foreground">
            Every output requires interpretation by a qualified radiologist.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitCompareArrows className="h-5 w-5 text-primary" /> The two methods
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <div>
            <p className="font-semibold">Method 1 — U-Net + ConvLSTM + SFLA</p>
            <p className="text-muted-foreground">
              A 2D U-Net produces a binary whole-tumour mask; the mask&apos;s bounding box
              is cropped and passed to a ConvLSTM classifier for four-way typing
              (Normal / Glioma / Meningioma / Pituitary). Classifier hyper-parameters are
              selected by a Shuffled Frog Leaping Algorithm search that reads only the
              training and validation splits. Grad-CAM explains each prediction.
            </p>
          </div>
          <div>
            <p className="font-semibold">Method 2 — Multi-class segmentation + SPECT + DCN</p>
            <p className="text-muted-foreground">
              Grayscale conversion and median/bilateral filtering feed a four-way
              segmentation (background, necrotic core, edema, enhancing tumour). A
              modality-specific feature stage turns those regions into per-region uptake
              statistics and an intensity histogram, which a densely connected
              convolutional network uses alongside the image channels.
            </p>
          </div>
          <p className="rounded-lg border bg-muted/30 p-3 text-xs text-muted-foreground">
            The two methods share no weights, no datasets, no preprocessing and no
            metrics. The Compare page shows them side by side without ranking them,
            because they are evaluated on different data and different splits.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-primary" /> SPECT vs PECT
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p>
            The reference diagram for Method 2 labels its modality stage{" "}
            <em>Photon Emission Computed Tomography (PECT)</em>, while the dataset
            actually used is <strong>SPECT</strong> (Single Photon Emission Computed
            Tomography).
          </p>
          <p className="text-muted-foreground">
            These are not interchangeable terms, so this implementation carries the real
            modality in <code className="rounded bg-muted px-1">METHOD2_MODALITY</code> and
            shows &ldquo;SPECT&rdquo; everywhere by default. The specification&apos;s own wording is
            rendered only where a run explicitly asks for it, and never as a claim that
            the two are the same thing.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" /> Datasets
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>
            <strong className="text-foreground">BraTS 2023 / 2020 mirror</strong> — the only
            source here with real multi-region tumour annotation. Used for Method 1&apos;s
            binary segmentation and Method 2&apos;s multi-class head.
          </p>
          <p>
            <strong className="text-foreground">Brain Tumor MRI Dataset (BRI)</strong> —
            genuinely four-class, used for Method 1&apos;s classifier. Binary
            tumour/no-tumour datasets are rejected rather than remapped onto four labels.
          </p>
          <p>
            <strong className="text-foreground">SPECT study set</strong> — drives Method
            2&apos;s modality branch. Its class folders are discovered at training time and
            must match the configured class list; they are never assumed.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-500" /> Known limitations
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <ul className="list-inside list-disc space-y-1.5">
            <li>
              Method 2 ships <strong>untrained</strong>. No checkpoint and no metrics are
              included, and the interface says so rather than showing placeholder numbers.
            </li>
            <li>
              A stage without trained weights reports itself unavailable. Masks from an
              untrained network are never presented as results.
            </li>
            <li>
              The BRI dataset carries no patient identifiers, so its split groups by
              filename stem only; scores from it may still be optimistic.
            </li>
            <li>
              Method 2&apos;s segmentation head trains on BraTS MRI because it is the only
              multi-region annotated source, while its classifier may run on SPECT. That
              cross-modality step is recorded in the checkpoint, not hidden.
            </li>
            <li>
              Metrics that were never computed display as <strong>N/A</strong>, never as 0.
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
