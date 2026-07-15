"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Clock, ImageOff, Download } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { buttonVariants } from "@/components/ui/button";
import {
  assetUrl,
  getHistory,
  reportUrl,
  type HistoryItem,
} from "@/lib/api";

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHistory(50)
      .then(setItems)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Prediction History</h1>
        <p className="mt-2 text-muted-foreground">
          The most recent predictions made through this system.
        </p>
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
            No predictions yet. Head to the Upload page to analyse an MRI.
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {items?.map((item, i) => (
          <motion.div
            key={item.prediction_id}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
          >
            <Card className="overflow-hidden">
              <CardHeader className="flex flex-row items-center justify-between pb-3">
                <CardTitle className="text-base">
                  <Badge variant={item.class === "No Tumor" ? "success" : "warning"}>
                    {item.class}
                  </Badge>
                </CardTitle>
                <span className="text-sm font-medium tabular-nums text-primary">
                  {item.confidence.toFixed(1)}%
                </span>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={assetUrl(item.original_image)}
                    alt="Original"
                    className="aspect-square w-full rounded-md border object-cover"
                  />
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={assetUrl(item.segmentation_mask)}
                    alt="Mask"
                    className="aspect-square w-full rounded-md border object-cover"
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" /> {item.inference_time}
                  </span>
                  <span>{new Date(item.timestamp).toLocaleString()}</span>
                </div>
                <a
                  href={reportUrl(item.prediction_id)}
                  target="_blank"
                  rel="noreferrer"
                  className={buttonVariants({
                    variant: "outline",
                    size: "sm",
                    className: "w-full",
                  })}
                >
                  <Download className="h-4 w-4" /> Report
                </a>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
