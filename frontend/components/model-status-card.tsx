"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, CircleSlash, Cpu } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { getMethodMetrics, pct, type MethodDetail, type MethodMetrics } from "@/lib/api";

/**
 * What is actually loaded for one method: classifier and segmentation status,
 * the checkpoint file, and — only once a held-out evaluation has been written by
 * training — the measured test metrics. Nothing here is a hard-coded number.
 */
export function ModelStatusCard({ method }: { method: MethodDetail }) {
  const [metrics, setMetrics] = useState<MethodMetrics | null>(null);

  useEffect(() => {
    let active = true;
    setMetrics(null);
    getMethodMetrics(method.method_id)
      .then((m) => active && setMetrics(m))
      .catch(() => active && setMetrics(null));
    return () => {
      active = false;
    };
  }, [method.method_id]);

  const measured = metrics?.evaluated && metrics.accuracy !== null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Method {method.display_number}
        </p>
        <CardTitle className="flex items-center gap-2 text-base">
          <Cpu className="h-4 w-4 text-primary" />
          {method.short_name}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-xs">
        <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
          <Item label="Classifier status">
            <Status ok={method.classifier_available} />
          </Item>
          <Item label="Segmentation status">
            {method.has_segmentation_stage ? (
              <Status ok={method.segmentation_available} />
            ) : (
              <span className="text-muted-foreground">Not part of this method</span>
            )}
          </Item>
          <Item label="Checkpoint">
            <span className="font-mono">{method.classifier_checkpoint ?? "—"}</span>
            {method.classifier_model_version && (
              <span className="block text-muted-foreground">
                version {method.classifier_model_version}
              </span>
            )}
          </Item>
          <Item label="Classes">{method.class_labels.join(" · ")}</Item>
          {method.ai_assessment_available && (
            <Item label="Results from">
              <span className="font-semibold text-violet-600 dark:text-violet-400">
                AI model assessment — unvalidated
              </span>
            </Item>
          )}
        </dl>

        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            Held-out test metrics
          </p>
          {measured ? (
            <>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <Metric label="Accuracy" value={pct(metrics!.accuracy)} />
                <Metric label="Macro F1" value={pct(metrics!.f1)} />
                <Metric label="Sensitivity" value={pct(metrics!.sensitivity)} />
                <Metric label="Specificity" value={pct(metrics!.specificity)} />
              </div>
              <p className="mt-2 text-[11px] leading-snug text-muted-foreground">{metrics!.split}</p>
            </>
          ) : (
            <p className="text-muted-foreground">Not evaluated — no test metrics recorded.</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function Item({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd className="leading-snug">{children}</dd>
    </div>
  );
}

function Status({ ok }: { ok: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex w-fit items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold",
        ok
          ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
          : "bg-muted text-muted-foreground"
      )}
    >
      {ok ? <CheckCircle2 className="h-3 w-3" /> : <CircleSlash className="h-3 w-3" />}
      {ok ? "TRAINED" : "NOT TRAINED"}
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border px-2.5 py-2">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className="text-sm font-semibold tabular-nums">{value}</p>
    </div>
  );
}
