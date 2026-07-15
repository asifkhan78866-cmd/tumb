// Higher-level prediction service wrapping the raw API client.
import {
  getHistory,
  uploadImage,
  type HistoryItem,
  type PredictionResult,
} from "@/lib/api";

export async function predict(
  file: File,
  onProgress?: (pct: number) => void
): Promise<PredictionResult> {
  return uploadImage(file, onProgress);
}

export async function recentPredictions(limit = 12): Promise<HistoryItem[]> {
  return getHistory(limit);
}
