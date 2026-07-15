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
    title: "U-Net Segmentation",
    desc: "Pixel-accurate tumor masks from a deep encoder–decoder with skip connections.",
  },
  {
    icon: Layers,
    title: "ConvLSTM Classification",
    desc: "Four-way tumor typing: Glioma, Meningioma, Pituitary or No Tumor.",
  },
  {
    icon: Sparkles,
    title: "Grad-CAM Explainability",
    desc: "Heatmaps reveal exactly where the model is looking on each scan.",
  },
  {
    icon: Activity,
    title: "Live Metrics",
    desc: "Dice, sensitivity, specificity, precision, recall and inference time.",
  },
];

const PIPELINE = [
  "Upload MRI",
  "Preprocessing",
  "U-Net Mask",
  "Crop Tumor",
  "ConvLSTM",
  "Prediction",
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
            Brain Tumor Segmentation
            <span className="block bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              &amp; Classification AI
            </span>
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground">
            A production-grade deep-learning pipeline — U-Net segmentation
            followed by a ConvLSTM classifier — with Grad-CAM explainability and
            downloadable clinical-style reports.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <Link href="/upload" className={buttonVariants({ size: "lg" })}>
              Upload an MRI <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/metrics"
              className={buttonVariants({ variant: "outline", size: "lg" })}
            >
              View Metrics
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

      {/* Pipeline */}
      <section className="rounded-3xl border bg-muted/30 px-6 py-14">
        <h2 className="mb-10 text-center text-3xl font-bold">
          Inference pipeline
        </h2>
        <div className="flex flex-wrap items-center justify-center gap-3">
          {PIPELINE.map((step, i) => (
            <motion.div
              key={step}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08 }}
              className="flex items-center gap-3"
            >
              <div className="rounded-full border bg-background px-5 py-2 text-sm font-medium shadow-sm">
                {step}
              </div>
              {i < PIPELINE.length - 1 && (
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
              )}
            </motion.div>
          ))}
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
