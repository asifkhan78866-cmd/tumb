"use client";

import { useEffect, useState } from "react";
import { GitCompareArrows, Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ArchitectureCard } from "@/components/architecture-card";
import { WarningList } from "@/components/warning-list";
import {
  errorMessage,
  getAllMetrics,
  getMethod,
  pct,
  seconds,
  type MethodDetail,
  type MethodMetrics,
} from "@/lib/api";

/**
 * Side-by-side comparison of the two pipelines.
 *
 * Deliberately NOT a ranking: no winner is declared, no "best" is highlighted
 * and no score is aggregated. The two methods use different datasets, different
 * modalities and different evaluation splits, so a single ordering would be
 * meaningless. Metrics that were never computed render as "N/A".
 */
const METRIC_ROWS: { label: string; key: keyof MethodMetrics; kind: "pct" | "sec" }[] = [
  { label: "Dice", key: "dice", kind: "pct" },
  { label: "IoU", key: "iou", kind: "pct" },
  { label: "Accuracy", key: "accuracy", kind: "pct" },
  { label: "Precision", key: "precision", kind: "pct" },
  { label: "Recall / Sensitivity", key: "recall", kind: "pct" },
  { label: "Specificity", key: "specificity", kind: "pct" },
  { label: "F1", key: "f1", kind: "pct" },
  { label: "AUC", key: "auc", kind: "pct" },
  { label: "Inference time", key: "avg_inference_time_s", kind: "sec" },
];

export default function ComparePage() {
  const [metrics, setMetrics] = useState<MethodMetrics[] | null>(null);
  const [details, setDetails] = useState<MethodDetail[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      getAllMetrics(),
      Promise.all([getMethod("method1"), getMethod("method2")]),
    ])
      .then(([m, d]) => {
        setMetrics(m);
        setDetails(d);
      })
      .catch((e) => setError(errorMessage(e)));
  }, []);

  if (error) {
    return (
      <Card>
        <CardContent className="space-y-3 p-6">
          <p className="font-semibold text-destructive">Cannot load data from the backend</p>
          <pre className="whitespace-pre-wrap rounded-lg border bg-muted/40 p-3 text-xs text-muted-foreground">
            {error}
          </pre>
        </CardContent>
      </Card>
    );
  }

  if (!metrics || !details) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-72" />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Skeleton className="h-80 w-full rounded-xl" />
          <Skeleton className="h-80 w-full rounded-xl" />
        </div>
        <Skeleton className="h-96 w-full rounded-xl" />
      </div>
    );
  }

  const allWarnings = metrics.flatMap((m) =>
    m.warnings.map((w) => `${m.method_id}: ${w}`)
  );

  return (
    <div className="space-y-8">
      <div>
        <h1 className="flex items-center gap-2 text-3xl font-bold">
          <GitCompareArrows className="h-7 w-7 text-primary" />
          Compare Methods
        </h1>
        <p className="mt-2 max-w-3xl text-muted-foreground">
          The two pipelines side by side. This page reports facts only — it does not
          rank the methods or pick a winner.
        </p>
      </div>

      <div className="flex items-start gap-3 rounded-xl border bg-muted/30 p-4 text-sm">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
        <p className="leading-relaxed text-muted-foreground">
          The methods are evaluated on <strong>different datasets, different
          modalities and different splits</strong>, so their numbers are not directly
          comparable and are never aggregated into a single score. A metric that has
          not been computed shows as <strong>N/A</strong> rather than zero.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {details.map((d) => (
          <ArchitectureCard key={d.method_id} method={d} />
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Measured metrics</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full min-w-[540px] border-collapse text-sm">
            <thead>
              <tr className="border-b">
                <th className="py-2 text-left font-semibold">Metric</th>
                {metrics.map((m) => (
                  <th key={m.method_id} className="py-2 text-left font-semibold">
                    {m.method_id === "method1" ? "Method 1" : "Method 2"}
                    <span className="block text-[11px] font-normal text-muted-foreground">
                      {m.evaluated ? m.split : "not evaluated"}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRIC_ROWS.map((row) => (
                <tr key={row.key as string} className="border-b last:border-0">
                  <td className="py-2 pr-4 text-muted-foreground">{row.label}</td>
                  {metrics.map((m) => {
                    const raw = m[row.key] as number | null;
                    const display =
                      row.kind === "sec" ? seconds(raw) : pct(raw, 2);
                    return (
                      <td
                        key={m.method_id}
                        className={
                          raw === null
                            ? "py-2 pr-4 text-muted-foreground/60"
                            : "py-2 pr-4 font-medium tabular-nums"
                        }
                      >
                        {display}
                      </td>
                    );
                  })}
                </tr>
              ))}
              <tr className="border-t">
                <td className="py-2 pr-4 text-muted-foreground">Model version</td>
                {metrics.map((m) => (
                  <td key={m.method_id} className="py-2 pr-4 text-xs">
                    {m.model_version ?? "—"}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </CardContent>
      </Card>

      <WarningList warnings={allWarnings} title="What these numbers do and do not mean" />

      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Research / decision-support output only. Neither method is a certified medical
        device, and neither has been clinically validated.
      </p>
    </div>
  );
}
