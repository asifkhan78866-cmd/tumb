"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, Crosshair, Gauge, Percent, Target, Timer, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { StatCard } from "@/components/stat-card";
import { MethodSelector } from "@/components/method-selector";
import { WarningList } from "@/components/warning-list";
import {
  errorMessage,
  getHealth,
  getMethod,
  getMethodMetrics,
  getMethods,
  pct,
  seconds,
  type Health,
  type MethodDetail,
  type MethodId,
  type MethodMetrics,
  type MethodSummary,
} from "@/lib/api";

export default function MetricsPage() {
  const [methods, setMethods] = useState<MethodSummary[] | null>(null);
  const [selected, setSelected] = useState<MethodId>("method1");
  const [metrics, setMetrics] = useState<MethodMetrics | null>(null);
  const [detail, setDetail] = useState<MethodDetail | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getMethods(), getHealth()])
      .then(([m, h]) => {
        setMethods(m);
        setHealth(h);
      })
      .catch((e) => setError(errorMessage(e)));
  }, []);

  // Never leave one method's numbers on screen while another is selected.
  useEffect(() => {
    let active = true;
    setMetrics(null);
    setDetail(null);
    Promise.all([getMethodMetrics(selected), getMethod(selected)])
      .then(([m, d]) => {
        if (!active) return;
        setMetrics(m);
        setDetail(d);
      })
      .catch((e) => active && setError(errorMessage(e)));
    return () => {
      active = false;
    };
  }, [selected]);

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

  // Only metrics that were actually computed become bars — a missing metric is
  // absent from the chart rather than drawn as a zero-height bar.
  const chartData = metrics
    ? (
        [
          ["Accuracy", metrics.accuracy],
          ["Dice", metrics.dice],
          ["IoU", metrics.iou],
          ["Sensitivity", metrics.sensitivity],
          ["Specificity", metrics.specificity],
          ["Precision", metrics.precision],
          ["Recall", metrics.recall],
          ["F1", metrics.f1],
          ["AUC", metrics.auc],
        ] as [string, number | null][]
      )
        .filter(([, v]) => v !== null)
        .map(([name, v]) => ({ name, value: (v as number) * 100 }))
    : [];

  const colors = ["#14b8a6", "#0ea5e9", "#8b5cf6", "#f59e0b", "#ec4899", "#10b981", "#6366f1", "#f43f5e", "#84cc16"];

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Model Metrics</h1>
          <p className="mt-2 text-muted-foreground">
            Per-method evaluation results. Metrics are never mixed between methods.
          </p>
        </div>
        {health && (
          <div className="rounded-lg border bg-muted/30 px-4 py-2 text-sm">
            Device: <span className="font-semibold">{health.device}</span>
          </div>
        )}
      </div>

      {methods ? (
        <MethodSelector methods={methods} selected={selected} onSelect={setSelected} />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Skeleton className="h-24 w-full rounded-xl" />
          <Skeleton className="h-24 w-full rounded-xl" />
        </div>
      )}

      {!metrics || !detail ? (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-28 w-full rounded-xl" />
          ))}
        </div>
      ) : (
        <>
          <div className="rounded-lg border bg-muted/30 px-4 py-3 text-sm">
            <span className="font-semibold">{detail.display_name}</span>
            <span className="block pt-1 text-xs text-muted-foreground">
              Split: {metrics.evaluated ? metrics.split : "not evaluated"} · model{" "}
              {metrics.model_version ?? "untrained"}
            </span>
          </div>

          {!metrics.evaluated && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-sm">
              <strong>This method has not been evaluated.</strong> No metrics are
              reported for it — the values below read N/A rather than 0. Train it with:
              <code className="mt-2 block rounded bg-background/60 px-2 py-1 text-xs">
                {detail.training_entrypoints[0]}
              </code>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard label="Accuracy" value={pct(metrics.accuracy)} icon={Target} delay={0} />
            <StatCard label="Dice Score" value={pct(metrics.dice)} icon={Crosshair} delay={0.05} />
            <StatCard label="Sensitivity" value={pct(metrics.sensitivity)} icon={Activity} delay={0.1} />
            <StatCard label="Specificity" value={pct(metrics.specificity)} icon={Gauge} delay={0.15} />
            <StatCard label="Precision" value={pct(metrics.precision)} icon={Percent} delay={0.2} />
            <StatCard label="Recall" value={pct(metrics.recall)} icon={TrendingUp} delay={0.25} />
            <StatCard label="F1 Score" value={pct(metrics.f1)} icon={Target} delay={0.3} />
            <StatCard
              label="Inference"
              value={seconds(metrics.avg_inference_time_s)}
              icon={Timer}
              delay={0.35}
            />
          </div>

          {chartData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Performance Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="name" tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }} />
                      <Tooltip
                        formatter={(v: number) => [`${v.toFixed(1)}%`, "Score"]}
                        contentStyle={{
                          background: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: 8,
                          color: "hsl(var(--foreground))",
                        }}
                      />
                      <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                        {chartData.map((_, i) => (
                          <Cell key={i} fill={colors[i % colors.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Architecture &amp; weights</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-1 gap-4 text-sm sm:grid-cols-2">
              {Object.entries(detail.weights).map(([role, w]) => (
                <div key={role} className="space-y-1">
                  <p className="font-semibold capitalize">{role}</p>
                  <p className="text-muted-foreground">{w.architecture}</p>
                  <p className={w.present ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"}>
                    {w.present ? "weights loaded" : "weights missing"}
                  </p>
                  <p className="break-all text-[11px] text-muted-foreground">{w.env_var}</p>
                </div>
              ))}
            </CardContent>
          </Card>

          <WarningList warnings={metrics.warnings} title="How to read these numbers" />
        </>
      )}
    </div>
  );
}
