"use client";

import { motion } from "framer-motion";
import {
  ArrowRight,
  Boxes,
  Brain,
  Crosshair,
  Filter,
  ImageIcon,
  Layers,
  LineChart,
  Sparkles,
  Workflow,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { MethodDetail, PipelineStage } from "@/lib/api";

/** One icon + accent per pipeline stage kind, shared by both methods. */
const KIND_STYLE: Record<string, { icon: any; className: string }> = {
  input: { icon: ImageIcon, className: "border-sky-500/40 bg-sky-500/10 text-sky-600 dark:text-sky-400" },
  preprocess: { icon: Filter, className: "border-violet-500/40 bg-violet-500/10 text-violet-600 dark:text-violet-400" },
  segmentation: { icon: Crosshair, className: "border-teal-500/40 bg-teal-500/10 text-teal-600 dark:text-teal-400" },
  feature: { icon: Boxes, className: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400" },
  classifier: { icon: Layers, className: "border-fuchsia-500/40 bg-fuchsia-500/10 text-fuchsia-600 dark:text-fuchsia-400" },
  optimization: { icon: Sparkles, className: "border-lime-500/40 bg-lime-500/10 text-lime-700 dark:text-lime-400" },
  output: { icon: Brain, className: "border-rose-500/40 bg-rose-500/10 text-rose-600 dark:text-rose-400" },
  eval: { icon: LineChart, className: "border-slate-500/40 bg-slate-500/10 text-slate-600 dark:text-slate-400" },
};

function StageChip({ stage, index }: { stage: PipelineStage; index: number }) {
  const style = KIND_STYLE[stage.kind] ?? KIND_STYLE.input;
  const Icon = style.icon;
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.04 }}
      className="flex items-center gap-2"
    >
      <div
        title={stage.description}
        className={cn(
          "flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium",
          style.className
        )}
      >
        <Icon className="h-3.5 w-3.5 shrink-0" />
        <span className="whitespace-nowrap">{stage.label}</span>
      </div>
    </motion.div>
  );
}

/**
 * Architecture / flow diagram for one method. Stage labels come from the
 * backend registry, so the picture cannot drift from the implementation.
 */
export function ArchitectureCard({ method }: { method: MethodDetail }) {
  const stages = method.pipeline_stages.filter((s) => s.kind !== "eval");
  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Workflow className="h-4 w-4 text-primary" />
          {method.short_name}
        </CardTitle>
        <p className="pt-1 text-xs text-muted-foreground">{method.summary}</p>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-wrap items-center gap-2">
          {stages.map((stage, i) => (
            <div key={stage.id} className="flex items-center gap-2">
              <StageChip stage={stage} index={i} />
              {i < stages.length - 1 && (
                <ArrowRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground/50" />
              )}
            </div>
          ))}
        </div>

        <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-xs sm:grid-cols-2">
          <Row label="Modality" value={method.modality} />
          <Row label="Classes" value={method.class_labels.join(" · ")} />
          <Row label="Segmentation" value={method.segmentation_model} />
          <Row label="Classifier" value={method.classifier_model} />
          {method.optimization && <Row label="Optimization" value={method.optimization} />}
        </dl>

        <div>
          <p className="mb-2 text-xs font-semibold">Datasets</p>
          <ul className="space-y-1.5">
            {method.datasets.map((d) => (
              <li key={d.name + d.role} className="flex flex-wrap items-center gap-2 text-xs">
                <span
                  className={cn(
                    "rounded px-1.5 py-0.5 text-[10px] font-semibold",
                    d.compatible === false
                      ? "bg-muted text-muted-foreground line-through"
                      : "bg-primary/10 text-primary"
                  )}
                >
                  {d.role}
                </span>
                <span className="font-medium">{d.name}</span>
                <span
                  className={cn(
                    "text-[10px]",
                    d.present ? "text-emerald-600 dark:text-emerald-400" : "text-muted-foreground"
                  )}
                >
                  {d.present ? "present" : "not downloaded"}
                </span>
                {d.notes && (
                  <span className="w-full text-[11px] leading-snug text-muted-foreground">
                    {d.notes}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd className="leading-snug">{value}</dd>
    </div>
  );
}
