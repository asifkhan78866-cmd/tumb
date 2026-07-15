// Typed client for the FastAPI backend.
// In dev, requests go through the Next.js rewrite proxy at /api/* (see
// next.config.mjs) so images and JSON share one origin.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Backend host used to resolve relative image URLs (/predictions/...).
export const ASSET_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface PredictionResult {
  prediction_id: string;
  class: string;
  confidence: number;
  inference_time: string;
  segmentation_mask: string;
  original_image: string;
  gradcam_overlay: string;
  probabilities: Record<string, number>;
}

export interface HistoryItem extends PredictionResult {
  timestamp: string;
}

export interface Metrics {
  accuracy: number;
  dice: number;
  sensitivity: number;
  specificity: number;
  precision: number;
  recall: number;
  f1: number;
  avg_inference_time_s: number;
  note?: string;
  confusion_matrix?: number[][];
  per_class?: {
    precision: number[];
    recall: number[];
    f1: number[];
    specificity: number[];
  };
}

export interface Health {
  status: string;
  device: string;
  seg_weights_loaded: boolean;
  cls_weights_loaded: boolean;
}

export interface ModelInfo {
  segmentation_model: string;
  classification_model: string;
  classes: string[];
  image_size: number;
  device: string;
  parameters: { segmentation: number; classification: number };
}

export interface TrainStatus {
  state: string;
  message: string;
  seg: Record<string, number>;
  cls: Record<string, number>;
  updated_at: string | null;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function assetUrl(path: string): string {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${ASSET_BASE}${path}`;
}

export async function uploadImage(
  file: File,
  onProgress?: (pct: number) => void
): Promise<PredictionResult> {
  // Use XHR for upload progress support.
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`API ${xhr.status}: ${xhr.responseText}`));
      }
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.send(form);
  });
}

export const getHealth = () => req<Health>("/health");
export const getModelInfo = () => req<ModelInfo>("/model-info");
export const getMetrics = () => req<Metrics>("/metrics");
export const getTrainStatus = () => req<TrainStatus>("/train-status");
export const getHistory = (limit = 50) =>
  req<HistoryItem[]>(`/history?limit=${limit}`);
export const reportUrl = (id: string) => `${API_BASE}/report/${id}`;
