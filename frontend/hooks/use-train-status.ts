"use client";

import { useEffect, useState } from "react";
import { getTrainStatus, type TrainStatus } from "@/lib/api";

/**
 * Poll the backend's training progress while mounted.
 *
 * Training runs are launched from the CLI (see each method's
 * `training_entrypoints`); they write progress to a JSON file that
 * `GET /train-status` serves. This hook is how a page can follow a run that is
 * already in flight.
 */
export function useTrainStatus(intervalMs = 5000) {
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const tick = () =>
      getTrainStatus()
        .then((s) => {
          if (!active) return;
          setStatus(s);
          setError(null);
        })
        .catch((e: unknown) => {
          if (!active) return;
          setError(e instanceof Error ? e.message : String(e));
        });
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { status, error };
}
