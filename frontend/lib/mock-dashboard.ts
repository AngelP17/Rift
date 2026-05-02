import { DashboardSummary, MetricPayload } from "@/lib/api";

export const mockMetrics: MetricPayload = {
  model_type: "graphsage_xgb_time_split",
  metrics: {
    pr_auc: 0.912,
    ece: 0.031,
    brier: 0.087,
    recall_at_1pct_fpr: 0.472,
    review_rate: 0.068
  }
};

export const mockDashboardSummary: DashboardSummary = {
  version: "0.1.0-demo",
  git_commit: "local-preview",
  refreshed_at: new Date(Date.now() - 7 * 60_000).toISOString(),
  current_model: {
    run_id: "run_graphsage_20260502_1148",
    artifact_path: "artifacts/models/graphsage_xgb_20260502.pkl"
  },
  current_metrics: {
    model_type: "graphsage_xgb_time_split",
    sector_profile: "fintech-card-present",
    time_split: true,
    metrics: mockMetrics.metrics
  },
  etl_runs: [
    {
      run_id: "etl_20260502_1154",
      source_system: "card_processor_midwest",
      rows_valid: 18472,
      rows_invalid: 31,
      duplicates_removed: 119
    },
    {
      run_id: "etl_20260502_1042",
      source_system: "treasury_disbursements",
      rows_valid: 11208,
      rows_invalid: 17,
      duplicates_removed: 84
    },
    {
      run_id: "etl_20260501_2316",
      source_system: "marketplace_settlements",
      rows_valid: 26341,
      rows_invalid: 58,
      duplicates_removed: 213
    }
  ],
  fairness_audits: [
    {
      audit_id: "fair_7c19",
      sensitive_column: "channel",
      demographic_parity_difference: 0.037,
      disparate_impact_ratio: 0.91
    },
    {
      audit_id: "fair_6af2",
      sensitive_column: "region",
      demographic_parity_difference: 0.044,
      disparate_impact_ratio: 0.88
    }
  ],
  drift_reports: [
    {
      report_id: "drift_20260502_1201",
      drift_score: 0.128,
      is_drift: true,
      retrain_triggered: false
    },
    {
      report_id: "drift_20260501_1800",
      drift_score: 0.073,
      is_drift: false,
      retrain_triggered: false
    }
  ],
  federated_runs: [
    {
      run_id: "fed_round_42",
      client_column: "issuer_region",
      client_count: 8,
      rounds: 3
    },
    {
      run_id: "fed_round_41",
      client_column: "merchant_vertical",
      client_count: 11,
      rounds: 4
    }
  ],
  prepared_datasets: [
    {
      summary: {
        dataset_id: "ieee_cis_shadow_eval",
        adapter: "ieee_cis",
        rows_prepared: 21436,
        auto_etl_run_id: "etl_20260502_1042"
      }
    },
    {
      summary: {
        dataset_id: "gov_disbursement_synthetic",
        adapter: "government_payments",
        rows_prepared: 11391,
        auto_etl_run_id: "etl_20260501_2316"
      }
    }
  ],
  recent_audits: [
    {
      decision_id: "DEC_84AF31",
      model_run_id: "run_graphsage_20260502_1148",
      decision: "manual_review",
      calibrated_probability: 0.782,
      confidence: "high"
    },
    {
      decision_id: "DEC_84AE92",
      model_run_id: "run_graphsage_20260502_1148",
      decision: "approve",
      calibrated_probability: 0.041,
      confidence: "medium"
    },
    {
      decision_id: "DEC_84AD76",
      model_run_id: "run_graphsage_20260502_1148",
      decision: "block",
      calibrated_probability: 0.914,
      confidence: "high"
    }
  ],
  run_history: [
    { run_id: "run_1138", pr_auc: 0.842 },
    { run_id: "run_1141", pr_auc: 0.861 },
    { run_id: "run_1144", pr_auc: 0.879 },
    { run_id: "run_1148", pr_auc: 0.912 }
  ],
  kpis: {
    etl_runs: 18,
    fairness_audits: 6,
    drift_reports: 4,
    federated_runs: 5,
    recent_audits: 47
  }
};
