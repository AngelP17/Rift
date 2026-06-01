"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ColumnDef } from "@tanstack/react-table";
import { motion } from "framer-motion";
import { ArrowsClockwise, GitBranch, LockKey } from "@phosphor-icons/react";
import { DataTable } from "@/components/dashboard/data-table";
import { KpiCard } from "@/components/dashboard/kpi-card";
import { OperationsBreakdownChart } from "@/components/dashboard/operations-breakdown-chart";
import { PerformanceTrendChart } from "@/components/dashboard/performance-trend-chart";
import { StatePanel } from "@/components/shared/state-panel";
import { useDashboardSummary, useMetrics } from "@/hooks/use-dashboard-data";
import { DashboardSummary } from "@/lib/api";
import { cn, formatDecimal, formatNumber, formatPercent, relativeTime, titleCase } from "@/lib/utils";

type DashboardRow = Record<string, unknown>;
type KpiTone = "good" | "warn" | "bad" | "neutral";

function useMounted(): boolean {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}

function statusTone(value: number, target: number, direction: "up" | "down"): KpiTone {
  if (direction === "up") {
    if (value >= target) return "good";
    if (value >= target * 0.82) return "warn";
    return "bad";
  }
  if (value <= target) return "good";
  if (value <= target * 1.5) return "warn";
  return "bad";
}

function makeColumns(keys: string[]): ColumnDef<DashboardRow>[] {
  return keys.map((key) => ({
    accessorKey: key,
    header: titleCase(key),
    cell: ({ getValue }) => {
      const value = getValue();
      if (typeof value === "number") {
        return value.toLocaleString("en-US", { maximumFractionDigits: 4 });
      }
      return String(value ?? "—");
    }
  }));
}

function normalizePreparedDatasets(summary?: DashboardSummary) {
  return summary?.prepared_datasets.map((item) => item.summary ?? {}) ?? [];
}

function useTables(summary?: DashboardSummary) {
  return useMemo(() => {
    const preparedDatasets = normalizePreparedDatasets(summary);
    return [
      {
        title: "Latest ETL Runs",
        subtitle: "Sortable lineage records from the ETL pipeline.",
        data: summary?.etl_runs ?? [],
        columns: makeColumns(["run_id", "source_system", "rows_valid", "rows_invalid", "duplicates_removed"]),
        emptyState: {
          title: "No ETL runs yet",
          description: "Run the auditable ETL on a CSV, JSON, or Parquet source to populate lineage, validation, and dedup records.",
          command: "rift etl run --source <path> --source-system <name> --dataset-name <id>"
        }
      },
      {
        title: "Recent Fairness Audits",
        subtitle: "Governance metrics with demographic parity and impact ratios.",
        data: summary?.fairness_audits ?? [],
        columns: makeColumns([
          "audit_id",
          "sensitive_column",
          "demographic_parity_difference",
          "disparate_impact_ratio"
        ]),
        emptyState: {
          title: "No fairness audits yet",
          description: "Score a labeled dataset with the current model and group outcomes by a sensitive column such as channel or region.",
          command: "rift fairness audit --sensitive-column channel"
        }
      },
      {
        title: "Recent Drift Reports",
        subtitle: "Monitor distribution drift and automatic retrain triggers.",
        data: summary?.drift_reports ?? [],
        columns: makeColumns(["report_id", "drift_score", "is_drift", "retrain_triggered"]),
        emptyState: {
          title: "No drift reports yet",
          description: "Compare a reference snapshot to the current data using the local drift detector. Threshold crossings can trigger retraining.",
          command: "rift monitor drift --reference-path <ref> --current-path <cur>"
        }
      },
      {
        title: "Federated Training Runs",
        subtitle: "Client-aware round summaries for collaborative training.",
        data: summary?.federated_runs ?? [],
        columns: makeColumns(["run_id", "client_column", "client_count", "rounds"]),
        emptyState: {
          title: "No federated runs yet",
          description: "Run the local FedAvg-style simulator to partition the dataset by a client column and aggregate per-round metrics.",
          command: "rift federated train --client-column channel --rounds 3"
        }
      },
      {
        title: "Prepared Public Datasets",
        subtitle: "Canonicalized datasets ready for ETL and model evaluation.",
        data: preparedDatasets,
        columns: makeColumns(["dataset_id", "adapter", "rows_prepared", "auto_etl_run_id"]),
        emptyState: {
          title: "No prepared datasets yet",
          description: "Use a public dataset adapter to normalize, validate, and stage rows for ETL and model evaluation.",
          command: "rift dataset prepare --adapter ieee_cis --source <path>"
        }
      },
      {
        title: "Recent Audit Decisions",
        subtitle: "Latest replayable decisions from the DuckDB audit store.",
        data: summary?.recent_audits ?? [],
        columns: makeColumns([
          "decision_id",
          "model_run_id",
          "decision",
          "calibrated_probability",
          "confidence"
        ]),
        emptyState: {
          title: "No audit decisions yet",
          description: "Score a transaction with the current model run. Every decision is recorded with payload, features, and a deterministic hash.",
          command: "rift predict --tx <path>"
        }
      }
    ];
  }, [summary]);
}

export default function DashboardPage() {
  const summaryQuery = useDashboardSummary();
  const metricsQuery = useMetrics();
  const mounted = useMounted();

  const summary = summaryQuery.data;
  const metrics = metricsQuery.data?.metrics ?? summary?.current_metrics?.metrics;
  const usesDemoData = Boolean(summaryQuery.error || metricsQuery.error);
  const coverage = 1 - Number(metrics?.review_rate ?? 0);
  const tables = useTables(summary);

  const snapshotLabel = mounted && summary?.refreshed_at
    ? `Snapshot ${relativeTime(summary.refreshed_at)}`
    : summary?.refreshed_at
      ? "Snapshot pending"
      : "Waiting for API response";

  const operationsData = useMemo(
    () => [
      { label: "ETL", value: summary?.kpis.etl_runs ?? 0, fill: "#6ea8fe" },
      { label: "Fairness", value: summary?.kpis.fairness_audits ?? 0, fill: "#8e44ad" },
      { label: "Drift", value: summary?.kpis.drift_reports ?? 0, fill: "#f39c12" },
      { label: "Federated", value: summary?.kpis.federated_runs ?? 0, fill: "#27ae60" },
      { label: "Audits", value: summary?.kpis.recent_audits ?? 0, fill: "#e74c3c" },
      { label: "Datasets", value: normalizePreparedDatasets(summary).length, fill: "#53c2ff" }
    ],
    [summary]
  );

  const kpis: Array<{
    label: string;
    value: number;
    detail: string;
    tone: KpiTone;
    formatter: (value: number) => string;
  }> = [
    {
      label: "PR-AUC",
      value: Number(metrics?.pr_auc ?? 0),
      detail: "Higher is better. Tracks the precision-recall lift the hybrid model keeps over a flat tabular baseline.",
      tone: statusTone(Number(metrics?.pr_auc ?? 0), 0.85, "up"),
      formatter: (value: number) => formatPercent(value, 1)
    },
    {
      label: "Expected Calibration Error",
      value: Number(metrics?.ece ?? 0),
      detail: "Lower is better. Measures how closely the calibrated probability reflects the observed fraud rate.",
      tone: statusTone(Number(metrics?.ece ?? 0), 0.05, "down"),
      formatter: (value: number) => formatDecimal(value)
    },
    {
      label: "Brier Score",
      value: Number(metrics?.brier ?? 0),
      detail: "Lower is better. Quadratic penalty for confidence and calibration quality on the latest run.",
      tone: statusTone(Number(metrics?.brier ?? 0), 0.12, "down"),
      formatter: (value: number) => formatDecimal(value)
    },
    {
      label: "Coverage",
      value: coverage,
      detail: "Approximate share of decisions that do not require an analyst, derived from the current review rate.",
      tone: statusTone(coverage, 0.95, "up"),
      formatter: (value: number) => formatPercent(value, 1)
    },
    {
      label: "ETL Runs",
      value: Number(summary?.kpis.etl_runs ?? 0),
      detail: "Recent ETL executions available for audit and lineage inspection.",
      tone: "neutral",
      formatter: (value: number) => formatNumber(Math.round(value))
    },
    {
      label: "Drift Reports",
      value: Number(summary?.kpis.drift_reports ?? 0),
      detail: "Distribution shifts tracked against the active reference window. Threshold crossings can trigger retraining.",
      tone: Number(summary?.kpis.drift_reports ?? 0) > 0 ? "warn" : "neutral",
      formatter: (value: number) => formatNumber(Math.round(value))
    },
    {
      label: "Recorded Audits",
      value: Number(summary?.kpis.recent_audits ?? 0),
      detail: "Replayable decisions with markdown and JSON payloads stored in the local DuckDB audit ledger.",
      tone: "neutral",
      formatter: (value: number) => formatNumber(Math.round(value))
    }
  ];

  const showNoModelState =
    !usesDemoData &&
    !summaryQuery.isLoading &&
    summary?.current_model === null &&
    summary?.current_metrics === null;

  return (
    <main className="min-h-[100dvh] px-4 py-5 text-ink md:px-6 xl:px-8">
      <div className="mx-auto max-w-[1480px]">
        <nav className="mb-5 rounded-full border border-white/10 bg-slate-950/72 px-4 py-3 shadow-glass backdrop-blur-2xl">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <Link className="flex items-center gap-3 text-sm font-semibold tracking-tight" href="/">
              <span className="grid h-9 w-9 place-items-center rounded-full bg-white text-slate-950">
                <GitBranch size={18} weight="bold" />
              </span>
              <span>Rift Evidence Console</span>
            </Link>
            <Link className="text-sm text-muted transition hover:text-ink" href="/">
              Landing
            </Link>
          </div>
        </nav>

        <motion.header
          className="glass-edge relative mb-6 overflow-hidden rounded-[28px] border border-[color:var(--color-line)] bg-slate-950/72 px-6 py-6 md:px-8 md:py-7"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: [0.25, 0.46, 0.45, 0.94] }}
        >
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_76%_10%,rgba(110,168,254,0.22),transparent_32%),linear-gradient(135deg,rgba(15,23,42,0.72),rgba(2,6,23,0.92))]" />
          <div className="absolute inset-0 bg-[linear-gradient(rgba(110,168,254,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(110,168,254,0.05)_1px,transparent_1px)] bg-[size:72px_72px] opacity-40" />
          <div className="relative grid gap-7 xl:grid-cols-[minmax(0,1fr)_430px] xl:items-end">
            <div className="max-w-4xl">
              <h1 className="font-display text-[clamp(2.6rem,5vw,4.75rem)] leading-[0.94] tracking-[-0.06em]">
                Evidence console for the fraud graph.
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-8 text-muted md:text-lg">
                Review graph risk, calibration drift, replayable decisions, and data lineage from the same local snapshot.
              </p>
            </div>

            <div className="grid gap-3">
              <div className="border-l border-white/10 bg-slate-950/44 px-5 py-4">
                <div className="flex items-center justify-between gap-4">
                  <div className="text-sm text-muted">Current evidence chain</div>
                  <LockKey className="h-4 w-4 text-emerald-300" weight="duotone" />
                </div>
                <div className="mt-3 break-all font-mono text-base text-ink">{summary?.current_model?.run_id ?? "No active run"}</div>
                <div className="mt-2 text-sm text-muted">{summary?.current_metrics?.model_type ?? "No model metadata"}</div>
              </div>
              <div className="border-l border-white/10 bg-slate-950/44 px-5 py-4">
                <div className="flex items-center justify-between gap-4 text-sm text-muted">
                  <span>Refresh status</span>
                  <span
                    className={cn(
                      "inline-flex items-center gap-2 rounded-full px-2.5 py-1 text-xs",
                      usesDemoData
                        ? "bg-amber-400/10 text-amber-300"
                        : summaryQuery.isValidating || metricsQuery.isValidating
                        ? "bg-accent/10 text-accent"
                        : "bg-emerald-400/10 text-emerald-300"
                    )}
                  >
                    <ArrowsClockwise
                      className={cn(
                        "h-3 w-3",
                        summaryQuery.isValidating || metricsQuery.isValidating ? "animate-spin" : ""
                      )}
                    />
                    {usesDemoData
                      ? "Demo telemetry"
                      : summaryQuery.isValidating || metricsQuery.isValidating
                      ? "Updating"
                      : "Live"}
                  </span>
                </div>
                <div className="mt-3 text-sm text-ink" suppressHydrationWarning>
                  {snapshotLabel}
                </div>
                <div className="mt-2 text-sm text-muted">
                  {usesDemoData
                    ? "FastAPI not reachable. Realistic demo data is rendered so the console stays useful for screenshots."
                    : "Connected to the FastAPI service."}
                </div>
              </div>
            </div>
          </div>
        </motion.header>

        <section className="mb-6 grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
          {kpis.map((kpi, index) => (
            <KpiCard key={kpi.label} delay={index * 0.04} {...kpi} />
          ))}
        </section>

        {showNoModelState ? (
          <section className="mb-6">
            <StatePanel
              title="No trained model is registered yet"
              description="Run a training pass to register a model, persist metrics under .rift/runs/, and unlock replay, model card, and audit exports."
              tone="empty"
            >
              <code className="rounded-full border border-white/10 bg-slate-950/80 px-3 py-1 font-mono text-xs text-ink">
                rift generate --txns 5000 --users 500 --merchants 120 --fraud-rate 0.03
              </code>
              <code className="rounded-full border border-white/10 bg-slate-950/80 px-3 py-1 font-mono text-xs text-ink">
                rift train --model graphsage_xgb --time-split
              </code>
            </StatePanel>
          </section>
        ) : null}

        <section className="mb-6 grid gap-5 xl:grid-cols-[1.25fr,0.95fr]">
          <PerformanceTrendChart data={summary?.run_history ?? []} />
          <OperationsBreakdownChart data={operationsData} />
        </section>

        <section className="grid gap-5 xl:grid-cols-2">
          {tables.map((table) => (
            <DataTable
              columns={table.columns}
              data={table.data as DashboardRow[]}
              emptyState={table.emptyState}
              key={table.title}
              subtitle={table.subtitle}
              title={table.title}
            />
          ))}
        </section>
      </div>
    </main>
  );
}
