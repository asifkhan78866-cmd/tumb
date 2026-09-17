"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Bot, CheckCircle2, CircleSlash } from "lucide-react";
import { cn } from "@/lib/utils";
import type { MethodId, MethodSummary } from "@/lib/api";

/**
 * The method toggle. Each option shows whether that method actually has trained
 * weights, so switching never silently changes what the numbers below mean.
 */
export function MethodSelector({
  methods,
  selected,
  onSelect,
  disabled,
}: {
  methods: MethodSummary[];
  selected: MethodId;
  onSelect: (id: MethodId) => void;
  disabled?: boolean;
}) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      {methods.map((m, i) => {
        const active = m.method_id === selected;
        return (
          <button
            key={m.method_id}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(m.method_id)}
            aria-pressed={active}
            className={cn(
              "relative overflow-hidden rounded-xl border p-4 text-left transition-all",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
              active
                ? "border-primary bg-primary/5 shadow-sm"
                : "border-border hover:border-primary/40 hover:bg-muted/40",
              disabled && "cursor-not-allowed opacity-60"
            )}
          >
            {active && (
              <motion.div
                layoutId="method-underline"
                className="absolute inset-x-0 bottom-0 h-0.5 bg-primary"
              />
            )}
            <div className="flex items-start justify-between gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Method {i + 1}
              </span>
              <StatusChip method={m} />
            </div>
            <p className="mt-1 font-semibold leading-tight">{m.short_name}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              {m.modality} · {m.num_classes} classes
            </p>
          </button>
        );
      })}
    </div>
  );
}

function StatusChip({ method }: { method: MethodSummary }) {
  if (method.trained) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
        <CheckCircle2 className="h-3 w-3" /> Trained
      </span>
    );
  }
  if (method.classifier_available) {
    return (
      <span
        title="Classifier trained; segmentation not trained"
        className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-600 dark:text-emerald-400"
      >
        <CheckCircle2 className="h-3 w-3" /> Classifier trained
      </span>
    );
  }
  if (method.ai_assessment_available) {
    return (
      <span
        title="No trained classifier; uploads are assessed by an AI model"
        className="inline-flex items-center gap-1 rounded-full bg-violet-500/10 px-2 py-0.5 text-[10px] font-semibold text-violet-600 dark:text-violet-400"
      >
        <Bot className="h-3 w-3" /> AI assessment
      </span>
    );
  }
  if (method.segmentation_available) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold text-amber-600 dark:text-amber-400">
        <AlertTriangle className="h-3 w-3" /> Partial
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground">
      <CircleSlash className="h-3 w-3" /> Not trained
    </span>
  );
}
