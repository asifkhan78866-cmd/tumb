"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export function ConfidenceMeter({ value }: { value: number }) {
  const color =
    value >= 85
      ? "text-emerald-500"
      : value >= 60
      ? "text-amber-500"
      : "text-rose-500";
  const stroke =
    value >= 85 ? "#10b981" : value >= 60 ? "#f59e0b" : "#f43f5e";
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="relative flex h-32 w-32 items-center justify-center">
      <svg className="h-32 w-32 -rotate-90" viewBox="0 0 120 120">
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="hsl(var(--muted))"
          strokeWidth="10"
        />
        <motion.circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={stroke}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className={cn("text-2xl font-bold", color)}>
          {value.toFixed(1)}%
        </span>
        <span className="text-xs text-muted-foreground">confidence</span>
      </div>
    </div>
  );
}
