// Typed client for the FastAPI backend.
//
// Requests go directly to NEXT_PUBLIC_API_URL; the backend sets CORS for the
// configured frontend origin. NEXT_PUBLIC_* values are inlined at BUILD time by
// Next.js, so changing the API URL requires a rebuild — see the Docker notes in
// the README.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Backend host used to resolve relative image URLs (/predictions/...).
export const ASSET_BASE = API_BASE;

export type MethodId = "method1" | "method2";

export interface PipelineStage {
  id: string;
  label: string;
  kind: string;
  description: string;
}

export interface DatasetSummary {
  key?: string;
  name: string;
  role: string;
  source?: string;
  present: boolean;
  path?: string;
  compatible?: boolean;
  notes?: string;
}

export interface MethodSummary {
  method_id: MethodId;
  display_name: string;
  short_name: string;
  summary: string;
  modality: string;
  num_classes: number;
  class_labels: string[];
  pipeline: string[];
  segmentation_model: string;
  classifier_model: string;
  optimization: string | null;
  trained: boolean;
  segmentation_available: boolean;
  classifier_available: boolean;
  warnings: string[];
}

export interface MethodDetail extends MethodSummary {
  pipeline_stages: PipelineStage[];
  datasets: DatasetSummary[];
  weights: Record<string, { architecture: string; env_var: string; path: string; present: boolean }>;
  preprocessing: Record<string, unknown>;
  training_entrypoints: string[];
  run_cards: Record<string, Record<string, unknown>>;
  optimization_result: Record<string, unknown> | null;
}

/** The uniform prediction envelope every method answers with. */
export interface MethodPrediction {
  method_id: MethodId;
  method_name: string;
  /** null when no classifier ran — never a guess. */
  prediction: string | null;
  prediction_key: string | null;
  confidence: number | null;
  class_probabilities: Record<string, number>;
  segmentation_available: boolean;
  segmentation_mask_url: string | null;
  heatmap_url: string | null;
  original_image_url: string | null;
  processing_time_s: number;
  model_version: string;
  dataset_context: string;
  modality: string;
  warnings: string[];
  prediction_id: string;
  details: Record<string, any>;
}

/** Metrics for one method. Every field is null when it was not computed. */
export interface MethodMetrics {
  method_id: MethodId;
  method_name: string;
  evaluated: boolean;
  split: string;
  dice: number | null;
  iou: number | null;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  sensitivity: number | null;
  specificity: number | null;
  f1: number | null;
  auc: number | null;
  avg_inference_time_s: number | null;
  confusion_matrix: number[][] | null;
  class_labels: string[];
  per_class: Record<string, number[]> | null;
  model_version: string | null;
  dataset_context: string;
  warnings: string[];
}

export interface HistoryItem {
  prediction_id: string;
  method_id?: MethodId;
  method_name?: string;
  timestamp: string;
  class: string | null;
  prediction_key?: string | null;
  confidence: number | null;
  inference_time: string;
  original_image: string | null;
  segmentation_mask: string | null;
  gradcam_overlay: string | null;
  segmentation_available?: boolean;
  probabilities: Record<string, number>;
  warnings?: string[];
}

/** Legacy training-progress payload written by the training scripts. */
export interface TrainStatus {
  state: string;
  message: string;
  seg: Record<string, number>;
  cls: Record<string, number>;
  updated_at: string | null;
}

export interface Health {
  status: string;
  device: string;
  seg_weights_loaded: boolean;
  cls_weights_loaded: boolean;
  warnings: string[];
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* keep statusText */
    }
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export function assetUrl(path: string | null | undefined): string {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${ASSET_BASE}${path}`;
}

/** Percentage formatter that renders "N/A" for metrics that were never computed. */
export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(digits)}%`;
}

export function seconds(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return `${value.toFixed(digits)} s`;
}

export const getMethods = () => req<MethodSummary[]>("/api/methods");
export const getMethod = (id: MethodId) => req<MethodDetail>(`/api/methods/${id}`);
export const getMethodMetrics = (id: MethodId) => req<MethodMetrics>(`/api/metrics/${id}`);
export const getAllMetrics = () => req<MethodMetrics[]>("/api/metrics");
export const getHealth = () => req<Health>("/health");
export const getTrainStatus = () => req<TrainStatus>("/train-status");
export const getHistory = (limit = 50, methodId?: MethodId) =>
  req<HistoryItem[]>(
    `/history?limit=${limit}${methodId ? `&method_id=${methodId}` : ""}`
  );
export const reportUrl = (methodId: MethodId | undefined, id: string) =>
  `${API_BASE}/api/report/${methodId ?? "method1"}/${id}`;

/** Upload an image to one method's pipeline, with upload-progress callbacks. */
export function predictWithMethod(
  methodId: MethodId,
  file: File,
  onProgress?: (pct: number) => void
): Promise<MethodPrediction> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/predict/${methodId}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        let detail = xhr.responseText;
        try {
          detail = JSON.parse(xhr.responseText).detail ?? detail;
        } catch {
          /* keep raw text */
        }
        reject(new Error(`API ${xhr.status}: ${detail}`));
      }
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.send(form);
  });
}
