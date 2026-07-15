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
import {
  Activity,
  Crosshair,
  Gauge,
  Percent,
  Target,
  Timer,
  TrendingUp,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { StatCard } from "@/components/stat-card";
import {
  getHealth,
  getMetrics,
  getModelInfo,
  type Health,
  type Metrics,
  type ModelInfo,
} from "@/lib/api";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getMetrics(), getModelInfo(), getHealth()])
      .then(([m, i, h]) => {
        setMetrics(m);
        setInfo(i);
        setHealth(h);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <Card>
        <CardContent className="p-6 text-destructive">
          Failed to load metrics: {error}. Make sure the backend is running.
        </CardContent>
      </Card>
    );
  }

  if (!metrics || !info) {
    return (
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-28 w-full rounded-xl" />
        ))}
      </div>
    );
  }

  const chartData = [
    { name: "Accuracy", value: metrics.accuracy * 100 },
    { name: "Dice", value: metrics.dice * 100 },
    { name: "Sensitivity", value: metrics.sensitivity * 100 },
    { name: "Specificity", value: metrics.specificity * 100 },
    { name: "Precision", value: metrics.precision * 100 },
    { name: "Recall", value: metrics.recall * 100 },
    { name: "F1", value: metrics.f1 * 100 },
  ];
  const colors = ["#14b8a6", "#0ea5e9", "#8b5cf6", "#f59e0b", "#ec4899", "#10b981", "#6366f1"];

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Model Metrics</h1>
          <p className="mt-2 text-muted-foreground">
            Evaluation results on the held-out validation split.
          </p>
        </div>
        {health && (
          <div className="rounded-lg border bg-muted/30 px-4 py-2 text-sm">
            Device: <span className="font-semibold">{health.device}</span> ·
            Seg weights:{" "}
            <span className={health.seg_weights_loaded ? "text-emerald-500" : "text-amber-500"}>
              {health.seg_weights_loaded ? "loaded" : "random"}
            </span>{" "}
            · Cls weights:{" "}
            <span className={health.cls_weights_loaded ? "text-emerald-500" : "text-amber-500"}>
              {health.cls_weights_loaded ? "loaded" : "random"}
            </span>
          </div>
        )}
      </div>

      {metrics.note && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-sm">
          {metrics.note}
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
          value={metrics.avg_inference_time_s.toFixed(3)}
          suffix=" s"
          icon={Timer}
          delay={0.35}
        />
      </div>

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

      <Card>
        <CardHeader>
          <CardTitle>Architecture</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-4 text-sm sm:grid-cols-2">
          <div className="space-y-1">
            <p className="font-semibold">Segmentation</p>
            <p className="text-muted-foreground">{info.segmentation_model}</p>
            <p className="text-muted-foreground">
              {info.parameters.segmentation.toLocaleString()} parameters
            </p>
          </div>
          <div className="space-y-1">
            <p className="font-semibold">Classification</p>
            <p className="text-muted-foreground">{info.classification_model}</p>
            <p className="text-muted-foreground">
              {info.parameters.classification.toLocaleString()} parameters
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
