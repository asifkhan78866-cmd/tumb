"use client";

import { motion } from "framer-motion";
import { Brain, Database, Cpu, Layers, FileText, ShieldCheck } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const SECTIONS = [
  {
    icon: Database,
    title: "Dataset",
    body: "Trained on the BraTS 2023 dataset, with automatic fallback to the Brain MRI / Brain Tumor MRI Kaggle datasets. Downloading is fully automated via the Kaggle API.",
  },
  {
    icon: Layers,
    title: "Segmentation — U-Net",
    body: "A 2D U-Net with encoder/decoder, skip connections, batch normalisation and dropout. Trained with a combined Dice + BCE loss, Adam, LR scheduling, mixed precision and early stopping.",
  },
  {
    icon: Cpu,
    title: "Classification — ConvLSTM",
    body: "Segmented tumor regions are cropped and passed through convolutional feature extractors, a ConvLSTM and dense + softmax layers to classify Glioma, Meningioma, Pituitary or No Tumor.",
  },
  {
    icon: FileText,
    title: "Explainability & Reports",
    body: "Grad-CAM heatmaps highlight the regions driving each prediction, and a one-click PDF report bundles the scan, mask, class, confidence and timestamp.",
  },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
          <Brain className="h-7 w-7" />
        </div>
        <h1 className="text-3xl font-bold">About NeuroSeg AI</h1>
        <p className="mt-3 text-muted-foreground">
          An end-to-end deep-learning system for brain tumor segmentation and
          classification, built with PyTorch, FastAPI and Next.js.
        </p>
      </motion.div>

      <div className="space-y-4">
        {SECTIONS.map((s, i) => (
          <motion.div
            key={s.title}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.08 }}
          >
            <Card>
              <CardHeader className="flex flex-row items-center gap-3 pb-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <s.icon className="h-5 w-5" />
                </div>
                <CardTitle>{s.title}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                {s.body}
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      <Card className="border-amber-500/30 bg-amber-500/10">
        <CardContent className="flex items-start gap-3 p-5 text-sm">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-amber-500" />
          <p>
            <strong>Disclaimer:</strong> This project is for research and
            educational purposes only. It is not a certified medical device and
            must not be used to make clinical decisions.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
