"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Clock, Download, EyeOff, ImageOff } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { assetUrl, getHistory, reportUrl, type HistoryItem, type MethodId } from "@/lib/api";

const FILTERS: { id: MethodId | "all"; label: string }[] = [
  { id: "all", label: "All methods" },
  { id: "method1", label: "Method 1" },
  { id: "method2", label: "Method 2" },
];

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [filter, setFilter] = useState<MethodId | "all">("all");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setItems(null);
    getHistory(50, filter === "all" ? undefined : filter)
      .then((r) => active && setItems(r))
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [filter]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Prediction History</h1>
        <p className="mt-2 text-muted-foreground">
          Recent predictions, labelled with the method that produced them.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            onClick={() => setFilter(f.id)}
            className={cn(
              "rounded-full border px-3 py-1.5 text-sm font-medium transition-colors",
              filter === f.id
                ? "border-primary bg-primary/10 text-primary"
                : "text-muted-foreground hover:border-primary/40 hover:text-foreground"
            )}
          >
            {f.label}
          </button>
        ))}
      </div>

      {error && (
        <Card>
          <CardContent className="flex items-center gap-3 p-6 text-destructive">
            <ImageOff className="h-5 w-5" /> {error}
          </CardContent>
        </Card>
      )}

      {!items && !error && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-72 w-full rounded-xl" />
          ))}
        </div>
      )}

      {items && items.length === 0 && (
        <Card>
          <CardContent className="p-10 text-center text-muted-foreground">
            No predictions yet. Head to the Analyse page to run one.
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {items?.map((item, i) => {
          const isNormal =
            item.prediction_key === "notumor" || item.prediction_key === "normal";
          return (
            <motion.div
              key={item.prediction_id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.04, 0.4) }}
            >
              <Card className="h-full overflow-hidden">
                <CardHeader className="flex flex-row items-start justify-between gap-2 pb-3">
                  <div className="space-y-1">
                    <CardTitle className="text-base">
                      {item.class ? (
                        <Badge variant={isNormal ? "success" : "warning"}>{item.class}</Badge>
                      ) : (
                        <Badge variant="outline">No classification</Badge>
                      )}
                    </CardTitle>
                    <span className="block text-[11px] text-muted-foreground">
                      {item.method_name ?? "Method 1"}
                    </span>
                  </div>
                  {item.confidence !== null && item.confidence !== undefined && (
                    <span className="text-sm font-medium tabular-nums text-primary">
                      {item.confidence.toFixed(1)}%
                    </span>
                  )}
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-2">
                    <Thumb src={item.original_image} alt="Original" />
                    <Thumb
                      src={item.segmentation_available ? item.segmentation_mask : null}
                      alt="Mask"
                      fallback="No segmentation"
                    />
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Clock className="h-3 w-3" /> {item.inference_time}
                    </span>
                    <span>{new Date(item.timestamp).toLocaleString()}</span>
                  </div>
                  <a
                    href={reportUrl(item.method_id, item.prediction_id)}
                    target="_blank"
                    rel="noreferrer"
                    className={buttonVariants({ variant: "outline", size: "sm", className: "w-full" })}
                  >
                    <Download className="h-4 w-4" /> Report
                  </a>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

function Thumb({
  src,
  alt,
  fallback,
}: {
  src: string | null | undefined;
  alt: string;
  fallback?: string;
}) {
  if (!src) {
    return (
      <div className="flex aspect-square w-full flex-col items-center justify-center gap-1 rounded-md border border-dashed bg-muted/30 p-2 text-center">
        <EyeOff className="h-4 w-4 text-muted-foreground" />
        <span className="text-[10px] leading-tight text-muted-foreground">
          {fallback ?? "Unavailable"}
        </span>
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={assetUrl(src)}
      alt={alt}
      className="aspect-square w-full rounded-md border object-cover"
    />
  );
}
