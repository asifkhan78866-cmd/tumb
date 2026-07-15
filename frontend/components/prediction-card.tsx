"use client";

import { motion } from "framer-motion";
import { Clock, Download, Activity } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ConfidenceMeter } from "@/components/confidence-meter";
import { assetUrl, reportUrl, type PredictionResult } from "@/lib/api";

function ImagePanel({ label, src }: { label: string; src: string }) {
  return (
    <div className="flex flex-col items-center gap-2">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={assetUrl(src)}
        alt={label}
        className="aspect-square w-full rounded-lg border object-cover"
      />
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
    </div>
  );
}

export function PredictionCard({ result }: { result: PredictionResult }) {
  const badgeVariant =
    result.class === "No Tumor" ? "success" : "warning";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" /> Prediction Result
          </CardTitle>
          <a
            href={reportUrl(result.prediction_id)}
            target="_blank"
            rel="noreferrer"
            className={buttonVariants({ variant: "outline", size: "sm" })}
          >
            <Download className="h-4 w-4" /> PDF Report
          </a>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <ImagePanel label="Original MRI" src={result.original_image} />
            <ImagePanel label="Segmentation Mask" src={result.segmentation_mask} />
            <ImagePanel label="Grad-CAM" src={result.gradcam_overlay} />
          </div>

          <div className="flex flex-col items-center gap-6 rounded-xl border bg-muted/30 p-6 sm:flex-row sm:justify-between">
            <div className="flex flex-col items-center gap-2 sm:items-start">
              <span className="text-sm text-muted-foreground">Predicted Class</span>
              <Badge variant={badgeVariant} className="px-4 py-1 text-base">
                {result.class}
              </Badge>
              <div className="mt-2 flex items-center gap-1 text-sm text-muted-foreground">
                <Clock className="h-4 w-4" />
                Inference time: {result.inference_time}
              </div>
            </div>
            <ConfidenceMeter value={result.confidence} />
          </div>

          <div className="space-y-3">
            <h4 className="text-sm font-semibold">Class Probabilities</h4>
            {Object.entries(result.probabilities).map(([name, prob]) => (
              <div key={name} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span>{name}</span>
                  <span className="tabular-nums text-muted-foreground">
                    {prob.toFixed(1)}%
                  </span>
                </div>
                <Progress value={prob} />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
