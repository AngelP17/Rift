export type MetricPayload = {
  model_type: string;
  metrics: {
    pr_auc?: number;
    ece?: number;
    brier?: number;
    recall_at_1pct_fpr?: number;
    review_rate?: number;
  };
};

export type DashboardSummary = {
  version: string;
  git_commit: string;
  refreshed_at: string;
  current_model: {
    run_id: string;
    artifact_path?: string;
  } | null;
  current_metrics: {
    model_type: string;
    sector_profile?: string;
    time_split?: boolean;
    metrics: MetricPayload["metrics"];
  } | null;
  etl_runs: Record<string, unknown>[];
  fairness_audits: Record<string, unknown>[];
  drift_reports: Record<string, unknown>[];
  federated_runs: Record<string, unknown>[];
  prepared_datasets: { summary?: Record<string, unknown> }[];
  recent_audits: Record<string, unknown>[];
  run_history: Array<{ run_id: string; pr_auc: number }>;
  kpis: Record<string, number>;
};

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: {
      Accept: "application/json"
    },
    next: { revalidate: 0 }
  });

  if (!response.ok) {
    throw new Error(`Request failed for ${path}: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  getMetrics: () => request<MetricPayload>("/metrics/latest"),
  getDashboardSummary: () => request<DashboardSummary>("/dashboard/summary")
};
