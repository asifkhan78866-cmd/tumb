"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Loader2, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { UploadDropzone } from "@/components/upload-dropzone";
import { PredictionCard } from "@/components/prediction-card";
import { uploadImage, type PredictionResult } from "@/lib/api";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<PredictionResult | null>(null);

  async function handlePredict() {
    if (!file) {
      toast.error("Please select an MRI image first.");
      return;
    }
    setLoading(true);
    setResult(null);
    setProgress(0);
    try {
      const res = await uploadImage(file, setProgress);
      setResult(res);
      toast.success(`Prediction: ${res.class} (${res.confidence.toFixed(1)}%)`);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Prediction failed. Is the backend running?"
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold">Upload MRI</h1>
        <p className="mt-2 text-muted-foreground">
          Drop a brain MRI slice to run segmentation and classification.
        </p>
      </div>

      <Card>
        <CardContent className="space-y-6 p-6">
          <UploadDropzone onFile={setFile} disabled={loading} />

          {loading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Uploading &amp; analysing…</span>
                <span className="tabular-nums">{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          )}

          <Button
            onClick={handlePredict}
            disabled={!file || loading}
            size="lg"
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Analysing…
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" /> Run Prediction
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {loading && (
        <Card>
          <CardContent className="space-y-4 p-6">
            <div className="grid grid-cols-3 gap-4">
              <Skeleton className="aspect-square w-full" />
              <Skeleton className="aspect-square w-full" />
              <Skeleton className="aspect-square w-full" />
            </div>
            <Skeleton className="h-24 w-full" />
          </CardContent>
        </Card>
      )}

      {result && !loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <PredictionCard result={result} />
        </motion.div>
      )}
    </div>
  );
}
