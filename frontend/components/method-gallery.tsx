"use client";

import { useEffect, useState } from "react";
import { Images, LineChart } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  assetUrl,
  getDatasetSamples,
  getMethodPlots,
  type DatasetSamples,
  type MethodId,
  type PlotImage,
} from "@/lib/api";

/**
 * Two galleries for one method: example images from the dataset it was tested
 * on, and the plots its training run produced. Both come from the backend, so a
 * plot that was never produced simply does not appear — nothing is illustrated
 * with a picture from another method or another run.
 */
export function MethodGallery({ methodId }: { methodId: MethodId }) {
  const [plots, setPlots] = useState<PlotImage[] | null>(null);
  const [samples, setSamples] = useState<DatasetSamples | null>(null);

  useEffect(() => {
    let active = true;
    setPlots(null);
    setSamples(null);
    getMethodPlots(methodId)
      .then((p) => active && setPlots(p))
      .catch(() => active && setPlots([]));
    getDatasetSamples(methodId, 3)
      .then((d) => active && setSamples(d))
      .catch(() => active && setSamples(null));
    return () => {
      active = false;
    };
  }, [methodId]);

  const byClass = (samples?.images ?? []).reduce<Record<string, { label: string; urls: string[] }>>(
    (acc, img) => {
      acc[img.class_key] ??= { label: img.class_label, urls: [] };
      acc[img.class_key].urls.push(img.url);
      return acc;
    },
    {}
  );

  return (
    <>
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Images className="h-4 w-4 text-primary" /> Dataset samples
          </CardTitle>
          {samples && (
            <p className="pt-1 text-xs text-muted-foreground">
              {samples.dataset} · {samples.split} split · {samples.note}
            </p>
          )}
        </CardHeader>
        <CardContent>
          {!samples ? (
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="aspect-square w-full rounded-lg" />
              ))}
            </div>
          ) : !samples.present ? (
            <p className="text-sm text-muted-foreground">{samples.note}</p>
          ) : (
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
              {Object.entries(byClass).map(([key, { label, urls }]) => (
                <div key={key} className="space-y-2">
                  <p className="text-xs font-semibold">{label}</p>
                  <div className="grid grid-cols-3 gap-1.5">
                    {urls.map((url) => (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        key={url}
                        src={assetUrl(url)}
                        alt={`${label} example from the test split`}
                        loading="lazy"
                        className="aspect-square w-full rounded border object-cover"
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <LineChart className="h-4 w-4 text-primary" /> Training &amp; evaluation plots
          </CardTitle>
          <p className="pt-1 text-xs text-muted-foreground">
            Produced by this method&apos;s own training run. Confusion matrix and ROC come
            from the single held-out test evaluation; the curves are per epoch.
          </p>
        </CardHeader>
        <CardContent>
          {plots === null ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {[0, 1].map((i) => (
                <Skeleton key={i} className="aspect-[4/3] w-full rounded-lg" />
              ))}
            </div>
          ) : plots.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No plots recorded for this method yet — they are written by a training run.
            </p>
          ) : (
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
              {plots.map((plot) => (
                <figure key={plot.key} className="space-y-1.5">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={assetUrl(plot.url)}
                    alt={plot.label}
                    loading="lazy"
                    className="w-full rounded-lg border bg-white"
                  />
                  <figcaption className="text-xs">
                    <span className="font-semibold">{plot.label}</span>
                    {plot.description && (
                      <span className="block text-muted-foreground">{plot.description}</span>
                    )}
                  </figcaption>
                </figure>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}
