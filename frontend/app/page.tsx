"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  Brain,
  ScanLine,
  Layers,
  Activity,
  ShieldCheck,
  Sparkles,
  ArrowRight,
} from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const FEATURES = [
  {
    icon: ScanLine,
    title: "Four independent methods",
    desc: "Separate models, weights and metrics per method — nothing is shared or averaged between them.",
  },
  {
    icon: Layers,
    title: "Transfer learning to segmentation",
    desc: "Fine-tuned pre-trained CNNs, a from-scratch ZFNet, a U-Net + ConvLSTM pipeline and an MRI–SPECT fusion design.",
  },
  {
    icon: Sparkles,
    title: "Explainability & optimization",
    desc: "Grad-CAM heatmaps, plus real Red Fox and Shuffled Frog Leaping hyper-parameter searches on validation data.",
  },
  {
    icon: Activity,
    title: "Honest metrics",
    desc: "One shared split, a test set evaluated once, and \u201cN/A\u201d wherever a metric was never computed.",
  },
];

const METHOD_FLOWS = [
  {
    name: "Method 1",
    subtitle: "Deep Learning Pre-trained Models + Transfer Learning",
    steps: ["Input", "Resize 224", "ImageNet CNNs", "Fine-tune", "Select on Val", "Classes"],
  },
  {
    name: "Method 2",
    subtitle: "Red Fox Optimized ZFNet",
    steps: ["Input", "Resize 224", "ZFNet", "Red Fox Optimization", "Classes"],
  },
  {
    name: "Method 3",
    subtitle: "3D U-Net–ConvLSTM–SFLA Segmentation & Classification",
    steps: ["Input", "Preprocess", "U-Net", "ROI Crop", "ConvLSTM", "SFLA", "Classes"],
  },
  {
    name: "Method 4",
    subtitle: "MRI–SPECT Multimodal Fusion",
    steps: ["MRI branch", "SPECT branch", "Feature Fusion", "Dense CNN", "Classes"],
  },
];

export default function LandingPage() {
  return (
    <div className="space-y-24">
      {/* Hero */}
      <section className="medical-grid relative overflow-hidden rounded-3xl border px-6 py-20 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mx-auto max-w-3xl"
        >
          <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-lg">
            <Brain className="h-8 w-8" />
          </div>
          <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
            Brain Tumor Analysis
            <span className="block bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Four Methods, One Platform
            </span>
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground">
            Four independent deep-learning pipelines for brain-tumour analysis, each
            with its own models, datasets and metrics — and an interface that tells
            you exactly which parts have actually been trained.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <Link href="/upload" className={buttonVariants({ size: "lg" })}>
              Upload an MRI <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/compare"
              className={buttonVariants({ variant: "outline", size: "lg" })}
            >
              Compare Methods
            </Link>
          </div>
        </motion.div>
      </section>

      {/* Features */}
      <section>
        <h2 className="mb-10 text-center text-3xl font-bold">
          What&apos;s under the hood
        </h2>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.1 }}
            >
              <Card className="h-full">
                <CardContent className="space-y-3 p-6">
                  <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <f.icon className="h-6 w-6" />
                  </div>
                  <h3 className="font-semibold">{f.title}</h3>
                  <p className="text-sm text-muted-foreground">{f.desc}</p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Pipelines */}
      <section className="space-y-10 rounded-3xl border bg-muted/30 px-6 py-14">
        <div className="text-center">
          <h2 className="text-3xl font-bold">Two inference pipelines</h2>
          <p className="mx-auto mt-3 max-w-2xl text-sm text-muted-foreground">
            Pick one on the Analyse page. They are evaluated separately and never
            compared by a single score.
          </p>
        </div>
        {METHOD_FLOWS.map((method, mi) => (
          <div key={method.name} className="space-y-4">
            <div className="text-center">
              <span className="text-xs font-semibold uppercase tracking-wide text-primary">
                {method.name}
              </span>
              <p className="text-sm font-medium">{method.subtitle}</p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-3">
              {method.steps.map((step, i) => (
                <motion.div
                  key={step}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: mi * 0.1 + i * 0.06 }}
                  className="flex items-center gap-3"
                >
                  <div className="rounded-full border bg-background px-4 py-2 text-sm font-medium shadow-sm">
                    {step}
                  </div>
                  {i < method.steps.length - 1 && (
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        ))}
        <div className="text-center">
          <Link href="/compare" className={buttonVariants({ variant: "outline" })}>
            Compare all methods <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-5 text-sm">
        <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-amber-500" />
        <p>
          <strong>Research &amp; educational use only.</strong> This tool is not a
          medical device and must not be used for clinical diagnosis. Always
          consult a qualified radiologist.
        </p>
      </section>
    </div>
  );
}
