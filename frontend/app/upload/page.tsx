"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Loader2, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { UploadDropzone } from "@/components/upload-dropzone";
import { PredictionCard } from "@/components/prediction-card";
import { MethodSelector } from "@/components/method-selector";
import { ArchitectureCard } from "@/components/architecture-card";
import { ModelStatusCard } from "@/components/model-status-card";
import { WarningList } from "@/components/warning-list";
import {
  errorMessage,
  getMethod,
  getMethods,
  predictWithMethod,
  type MethodDetail,
  type MethodId,
  type MethodPrediction,
  type MethodSummary,
} from "@/lib/api";

export default function UploadPage() {
  const [methods, setMethods] = useState<MethodSummary[] | null>(null);
  const [selected, setSelected] = useState<MethodId>("method3");
  const [detail, setDetail] = useState<MethodDetail | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<MethodPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMethods().then(setMethods).catch((e) => setError(errorMessage(e)));
  }, []);

  // Fetch the selected method's detail, and drop any result belonging to the
  // method we just switched away from — showing one method's mask under another
  // method's heading would be exactly the kind of mixing this UI must prevent.
  useEffect(() => {
    let active = true;
    setDetail(null);
    setResult(null);
    getMethod(selected)
      .then((d) => active && setDetail(d))
      .catch((e) => active && setError(errorMessage(e)));
    return () => {
      active = false;
    };
  }, [selected]);

  async function handlePredict() {
    if (!file) {
      toast.error("Please select an image first.");
      return;
    }
    setLoading(true);
    setResult(null);
    setProgress(0);
    try {
      const res = await predictWithMethod(selected, file, setProgress);
      setResult(res);
      if (res.prediction && res.details?.result_source === "ai_assessment") {
        toast.warning(`AI assessment (not a trained model): ${res.prediction}`);
      } else if (res.prediction) {
        toast.success(
          `${res.method_name.split("—")[0].trim()}: ${res.prediction}` +
            (res.confidence !== null ? ` (${res.confidence.toFixed(1)}%)` : "")
        );
      } else {
        toast.warning("Analysis ran, but this method has no trained classifier.");
      }
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Prediction failed. Is the backend running?"
      );
    } finally {
      setLoading(false);
    }
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6 text-destructive">
          Failed to reach the backend: {error}
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold">Analyse a Scan</h1>
        <p className="mt-2 text-muted-foreground">
          Choose a method, then upload an image to run that pipeline.
        </p>
      </div>

      <Card>
        <CardContent className="space-y-6 p-6">
          {methods ? (
            <MethodSelector
              methods={methods}
              selected={selected}
              onSelect={setSelected}
              disabled={loading}
            />
          ) : (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Skeleton className="h-24 w-full rounded-xl" />
              <Skeleton className="h-24 w-full rounded-xl" />
            </div>
          )}

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

          <Button onClick={handlePredict} disabled={!file || loading} size="lg" className="w-full">
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Analysing…
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" /> Run {detail?.short_name ?? "Prediction"}
              </>
            )}
          </Button>

          {detail && !detail.trained && !detail.ai_assessment_available && (
            <WarningList
              warnings={detail.warnings}
              title="This method is not fully trained"
            />
          )}
        </CardContent>
      </Card>

      {detail ? (
        <>
          <ModelStatusCard method={detail} />
          <ArchitectureCard method={detail} />
        </>
      ) : (
        <Skeleton className="h-64 w-full rounded-xl" />
      )}

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
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
          <PredictionCard result={result} />
        </motion.div>
      )}
    </div>
  );
}
