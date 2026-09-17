"use client";

import { motion } from "framer-motion";
import { Activity, Bot, Clock, Download, EyeOff } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ConfidenceMeter } from "@/components/confidence-meter";
import { WarningList } from "@/components/warning-list";
import { assetUrl, reportUrl, type MethodPrediction } from "@/lib/api";

/**
 * One image panel. When the corresponding stage did not run we render an
 * explicit "unavailable" placeholder rather than an empty or misleading frame —
 * a blank square next to a confidence score reads as a negative finding.
 */
function ImagePanel({
  label,
  src,
  unavailableReason,
}: {
  label: string;
  src: string | null;
  unavailableReason?: string;
}) {
  if (!src) {
    return (
      <div className="flex flex-col items-center gap-2">
        <div className="flex aspect-square w-full flex-col items-center justify-center gap-2 rounded-lg border border-dashed bg-muted/30 p-3 text-center">
          <EyeOff className="h-5 w-5 text-muted-foreground" />
          <span className="text-[11px] leading-tight text-muted-foreground">
            {unavailableReason ?? "Not available"}
          </span>
        </div>
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
      </div>
    );
  }
  const isPlaceholder = src.toLowerCase().includes("placeholder");
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-full">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={assetUrl(src)}
          alt={label}
          className="aspect-square w-full rounded-lg border object-cover"
        />
        {isPlaceholder && (
          <span className="absolute left-1 top-1 rounded bg-amber-500 px-1.5 py-0.5 text-[9px] font-bold uppercase text-white">
            Placeholder
          </span>
        )}
      </div>
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
    </div>
  );
}

export function PredictionCard({ result }: { result: MethodPrediction }) {
  const hasPrediction = result.prediction !== null;
  const ai = result.details?.result_source === "ai_assessment" ? result.details.ai_assessment : null;
  // Compare against the stable class key, not the display label.
  const badgeVariant = result.prediction_key === "notumor" || result.prediction_key === "normal"
    ? "success"
    : "warning";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <Card>
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" /> Prediction Result
            </CardTitle>
            <p className="pt-1 text-xs text-muted-foreground">
              {result.method_name} · {result.modality} · model {result.model_version}
            </p>
          </div>
          <a
            href={reportUrl(result.method_id, result.prediction_id)}
            target="_blank"
            rel="noreferrer"
            className={buttonVariants({ variant: "outline", size: "sm" })}
          >
            <Download className="h-4 w-4" /> PDF Report
          </a>
        </CardHeader>

        <CardContent className="space-y-6">
          {ai && <AIAssessmentBanner ai={ai} />}

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <ImagePanel label="Original" src={result.original_image_url} />
            <ImagePanel
              label={
                result.segmentation_available
                  ? "Segmentation Mask"
                  : result.details?.segmentation_in_method === false
                    ? "Segmentation"
                    : "Segmentation (unavailable)"
              }
              src={result.segmentation_mask_url}
              unavailableReason={result.details?.segmentation_note ?? "Segmentation model not trained"}
            />
            <ImagePanel
              label="Grad-CAM"
              src={result.heatmap_url}
              unavailableReason={
                result.details?.explainability ?? "No heatmap for this result"
              }
            />
          </div>

          {hasPrediction ? (
            <div className="flex flex-col items-center gap-6 rounded-xl border bg-muted/30 p-6 sm:flex-row sm:justify-between">
              <div className="flex flex-col items-center gap-2 sm:items-start">
                <span className="text-sm text-muted-foreground">
                  {ai ? "AI model's assessment" : "Predicted Class"}
                </span>
                <Badge variant={badgeVariant} className="px-4 py-1 text-base">
                  {result.prediction}
                </Badge>
                <div className="mt-2 flex items-center gap-1 text-sm text-muted-foreground">
                  <Clock className="h-4 w-4" />
                  {result.processing_time_s.toFixed(2)} s
                </div>
              </div>
              {result.confidence !== null && <ConfidenceMeter value={result.confidence} />}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed bg-muted/30 p-6 text-center">
              <p className="font-semibold">No classification produced</p>
              <p className="mt-1 text-sm text-muted-foreground">
                {ai
                  ? "The AI model's assessment was indeterminate, so no class is reported."
                  : "This method has no trained classifier, so no class is reported. See the caveats below for the training command."}
              </p>
            </div>
          )}

          {Object.keys(result.class_probabilities).length > 0 && (
            <div className="space-y-3">
              <h4 className="text-sm font-semibold">
                {ai ? "AI-estimated likelihoods (uncalibrated)" : "Class Probabilities"}
              </h4>
              {Object.entries(result.class_probabilities).map(([name, prob]) => (
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
          )}

          {ai && <AIFindings ai={ai} />}

          {result.details?.feature_stage && (
            <FeatureStage details={result.details} />
          )}

          <WarningList warnings={result.warnings} />

          <p className="border-t pt-4 text-[11px] leading-relaxed text-muted-foreground">
            Research / decision-support output only. Not a clinical diagnosis and not a
            certified medical device. Dataset context: {result.dataset_context}
          </p>
        </CardContent>
      </Card>
    </motion.div>
  );
}

/** Shown first on any AI-assessed result so it cannot be mistaken for the DCN. */
function AIAssessmentBanner({ ai }: { ai: Record<string, any> }) {
  return (
    <div className="flex gap-3 rounded-xl border border-violet-500/40 bg-violet-500/10 p-4">
      <Bot className="mt-0.5 h-5 w-5 shrink-0 text-violet-600 dark:text-violet-400" />
      <div className="space-y-1 text-sm">
        <p className="font-semibold text-violet-700 dark:text-violet-300">
          AI assessment — not a trained model result
        </p>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {ai.reason ?? "This method's model is not trained, so the image was read by an AI model."}{" "}
          Model: {ai.model}. It has no measured accuracy
          on this project&apos;s data, the percentages are its own uncalibrated
          estimates, and it is not a diagnosis.
        </p>
      </div>
    </div>
  );
}

function AIFindings({ ai }: { ai: Record<string, any> }) {
  return (
    <div className="rounded-xl border bg-muted/20 p-4 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="font-semibold">AI model findings</h4>
        <span className="rounded bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground">
          observed {ai.observed_modality} · quality {ai.image_quality}
        </span>
      </div>
      {ai.key_findings?.length > 0 && (
        <ul className="mt-2 list-disc space-y-0.5 pl-5 text-xs">
          {ai.key_findings.map((f: string, i: number) => (
            <li key={i}>{f}</li>
          ))}
        </ul>
      )}
      {ai.rationale && (
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{ai.rationale}</p>
      )}
    </div>
  );
}

/** The SPECT branch's modality-specific descriptor, shown so the stage is inspectable. */
function FeatureStage({ details }: { details: Record<string, any> }) {
  const stage = details.feature_stage;
  const top: [string, number][] = (stage.names as string[])
    .map((n: string, i: number) => [n, stage.values[i]] as [string, number])
    .filter(([, v]) => v !== 0)
    .slice(0, 8);

  return (
    <div className="rounded-xl border bg-muted/20 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-sm font-semibold">
          {details.modality} feature stage
          <span className="ml-2 text-xs font-normal text-muted-foreground">
            {stage.dimension} descriptors
          </span>
        </h4>
        <span
          className={
            stage.computed_from_segmentation
              ? "rounded bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-600 dark:text-emerald-400"
              : "rounded bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold text-amber-600 dark:text-amber-400"
          }
        >
          {stage.computed_from_segmentation ? "from segmentation" : "no regions — all zero"}
        </span>
      </div>
      {top.length > 0 ? (
        <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
          {top.map(([name, value]) => (
            <div key={name} className="flex flex-col">
              <span className="truncate text-[10px] text-muted-foreground" title={name}>
                {name}
              </span>
              <span className="tabular-nums">{value.toFixed(4)}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">
          Every descriptor is zero because no tumour regions were segmented.
        </p>
      )}
      {details.spec_modality_label && details.spec_modality_label !== details.modality && (
        <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
          Terminology: the research specification writes &ldquo;{details.spec_modality_label}&rdquo;;
          this implementation and its dataset are {details.modality}. They are not
          synonyms, so the real modality is shown here.
        </p>
      )}
    </div>
  );
}
