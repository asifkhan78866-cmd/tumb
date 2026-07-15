"use client";

import { useEffect, useState } from "react";
import { getTrainStatus, type TrainStatus } from "@/lib/api";

/** Poll the backend training status every `intervalMs` while mounted. */
export function useTrainStatus(intervalMs = 5000) {
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const tick = () =>
      getTrainStatus()
        .then((s) => active && setStatus(s))
        .catch((e) => active && setError(e.message));
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { status, error };
}
