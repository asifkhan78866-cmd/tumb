// Higher-level prediction service wrapping the raw API client.
//
// Every call is method-scoped: there is no way to request a prediction without
// naming the method that should produce it.
import {
  getHistory,
  predictWithMethod,
  type HistoryItem,
  type MethodId,
  type MethodPrediction,
} from "@/lib/api";

export async function predict(
  methodId: MethodId,
  file: File,
  onProgress?: (pct: number) => void
): Promise<MethodPrediction> {
  return predictWithMethod(methodId, file, onProgress);
}

export async function recentPredictions(
  limit = 12,
  methodId?: MethodId
): Promise<HistoryItem[]> {
  return getHistory(limit, methodId);
}
