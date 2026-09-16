"use client";

import { AlertTriangle } from "lucide-react";

/**
 * Warnings attached to a result travel from the training run all the way to the
 * screen. They are rendered, never collapsed away, because most of them qualify
 * what a number actually means.
 */
export function WarningList({
  warnings,
  title = "Caveats",
}: {
  warnings: string[];
  title?: string;
}) {
  if (!warnings?.length) return null;
  return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
      <p className="flex items-center gap-2 text-sm font-semibold text-amber-700 dark:text-amber-400">
        <AlertTriangle className="h-4 w-4 shrink-0" />
        {title}
      </p>
      <ul className="mt-2 space-y-1.5">
        {warnings.map((w, i) => (
          <li key={i} className="text-xs leading-relaxed text-amber-900/90 dark:text-amber-200/90">
            • {w}
          </li>
        ))}
      </ul>
    </div>
  );
}
