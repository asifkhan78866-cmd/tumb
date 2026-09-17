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
            <GitCompareArrows className="h-5 w-5 text-primary" /> The four methods
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <div>
            <p className="font-semibold">
              Method 1 — Deep Learning Pre-trained Models + Transfer Learning-Based Brain Tumor Classification
            </p>
            <p className="text-muted-foreground">
              ImageNet-pretrained EfficientNet-B0 and ResNet-50 are each
              fine-tuned on brain MRI (head warm-up, then full fine-tuning). The backbone with
              the best validation macro-F1 becomes the model and is tested once.
            </p>
          </div>
          <div>
            <p className="font-semibold">Method 2 — Red Fox Optimized ZFNet-Based Brain Tumor Classification</p>
            <p className="text-muted-foreground">
              A ZFNet trained from scratch on single-channel MRI. Red Fox Optimization (global
              search, local search, habitat reproduction) chooses its learning rate, weight
              decay, dropout, fully connected width and batch size using validation data only.
            </p>
          </div>
          <div>
            <p className="font-semibold">Method 3 — 3D U-Net–ConvLSTM–SFLA-Based Anomaly Segmentation &amp; Classification</p>
            <p className="text-muted-foreground">
              A U-Net segments the tumour, its bounding box is cropped, and a ConvLSTM
              classifies it; a Shuffled Frog Leaping Algorithm tunes the classifier. The U-Net
              is implemented in 2D on slices; a volumetric 3D U-Net needs BraTS volumes and is
              not trained, so the classifier currently runs on whole slices.
            </p>
          </div>
          <div>
            <p className="font-semibold">Method 4 — MRI–SPECT Multimodal Fusion-Based Brain Tumor Classification</p>
            <p className="text-muted-foreground">
              An MRI branch and a SPECT branch whose features are fused before a densely
              connected classifier. Fusion needs MRI and SPECT scans of the same patients; no
              such paired dataset is available, so the fusion network is not trained. Uploads
              are read by an AI vision model instead, and every such result is labelled as an
              unvalidated AI assessment.
            </p>
          </div>
          <p className="rounded-lg border bg-muted/30 p-3 text-xs text-muted-foreground">
            Methods 1–3 are trained and tested on the same MRI split, so their test numbers
            describe the same images. The methods share no weights, and nothing is averaged
            into a single score.
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
            The reference diagram for the SPECT branch (Method 4) labels its modality stage{" "}
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
            source here with real multi-region tumour annotation. Needed for Method 3&apos;s
            U-Net and Method 4&apos;s segmentation head; not downloaded yet.
          </p>
          <p>
            <strong className="text-foreground">Brain Tumor MRI Dataset (BRI)</strong> —
            genuinely four-class, used by Methods 1, 2 and 3 with one shared split. Binary
            tumour/no-tumour datasets are rejected rather than remapped onto four labels.
          </p>
          <p>
            <strong className="text-foreground">SPECT study set</strong> — would drive Method
            4&apos;s SPECT branch. Its class folders are discovered at training time and
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
              Method 4&apos;s fusion network is <strong>untrained</strong> (no paired MRI–SPECT
              data). Its results come from an AI vision model and are labelled as such; it has
              no metrics.
            </li>
            <li>
              A stage without trained weights reports itself unavailable. Masks from an
              untrained network are never presented as results.
            </li>
            <li>
              The BRI dataset carries no patient identifiers, so its split is image-level;
              exact duplicates are handled, but scores may still be optimistic.
            </li>
            <li>
              Method 3&apos;s U-Net is 2D and untrained, so its classifier runs on whole
              slices rather than tumour crops. The title&apos;s 3D U-Net needs volumetric data.
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
