"use client";

import { useMemo } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { motion } from "framer-motion";
import { Activity, DatabaseZap, RefreshCcw, ShieldAlert, Table2, Workflow } from "lucide-react";
import { DataTable } from "@/components/dashboard/data-table";
import { KpiCard } from "@/components/dashboard/kpi-card";
import { OperationsBreakdownChart } from "@/components/dashboard/operations-breakdown-chart";
import { PerformanceTrendChart } from "@/components/dashboard/performance-trend-chart";
import { useDashboardSummary, useMetrics } from "@/hooks/use-dashboard-data";
import { DashboardSummary } from "@/lib/api";
import { cn, formatDecimal, formatNumber, formatPercent, relativeTime, titleCase } from "@/lib/utils";

type DashboardRow = Record<string, unknown>;
type KpiTone = "good" | "warn" | "bad" | "neutral";

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
        columns: makeColumns(["run_id", "source_system", "rows_valid", "rows_invalid", "duplicates_removed"])
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
        ])
      },
      {
        title: "Recent Drift Reports",
        subtitle: "Monitor distribution drift and automatic retrain triggers.",
        data: summary?.drift_reports ?? [],
        columns: makeColumns(["report_id", "drift_score", "is_drift", "retrain_triggered"])
      },
      {
        title: "Federated Training Runs",
        subtitle: "Client-aware round summaries for collaborative training.",
        data: summary?.federated_runs ?? [],
        columns: makeColumns(["run_id", "client_column", "client_count", "rounds"])
      },
      {
        title: "Prepared Public Datasets",
        subtitle: "Canonicalized datasets ready for ETL and model evaluation.",
        data: preparedDatasets,
        columns: makeColumns(["dataset_id", "adapter", "rows_prepared", "auto_etl_run_id"])
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
        ])
      }
    ];
  }, [summary]);
}

export default function DashboardPage() {
  const summaryQuery = useDashboardSummary();
  const metricsQuery = useMetrics();

  const summary = summaryQuery.data;
  const metrics = metricsQuery.data?.metrics ?? summary?.current_metrics?.metrics;
  const coverage = 1 - Number(metrics?.review_rate ?? 0);
  const tables = useTables(summary);

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
      detail: "Target above 85% with graph-aware lift over flat tabular baselines.",
      tone: statusTone(Number(metrics?.pr_auc ?? 0), 0.85, "up"),
      formatter: (value: number) => formatPercent(value, 1)
    },
    {
      label: "Expected Calibration Error",
      value: Number(metrics?.ece ?? 0),
      detail: "Lower is better. Tracks how closely probabilities reflect observed fraud rates.",
      tone: statusTone(Number(metrics?.ece ?? 0), 0.05, "down"),
      formatter: (value: number) => formatDecimal(value)
    },
    {
      label: "Brier Score",
      value: Number(metrics?.brier ?? 0),
      detail: "Quadratic scoring penalty for calibration and confidence quality.",
      tone: statusTone(Number(metrics?.brier ?? 0), 0.12, "down"),
      formatter: (value: number) => formatDecimal(value)
    },
    {
      label: "Coverage",
      value: coverage,
      detail: "Approximate non-review coverage derived from the current review rate.",
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
      detail: "Distribution shifts tracked against the active reference window.",
      tone: Number(summary?.kpis.drift_reports ?? 0) > 0 ? "warn" : "neutral",
      formatter: (value: number) => formatNumber(Math.round(value))
    },
    {
      label: "Recorded Audits",
      value: Number(summary?.kpis.recent_audits ?? 0),
      detail: "Replayable decisions with markdown and JSON payloads in the audit store.",
      tone: "neutral",
      formatter: (value: number) => formatNumber(Math.round(value))
    }
  ];

  return (
    <main className="min-h-screen px-4 py-5 text-ink md:px-6 xl:px-8">
      <div className="mx-auto max-w-[1480px]">
        <motion.header
          className="surface mb-6 rounded-[36px] border border-[color:var(--color-line)] px-6 py-6 md:px-8 md:py-7"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: [0.25, 0.46, 0.45, 0.94] }}
          style={{ backgroundImage: "var(--gradient-hero)" }}
        >
          <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-3xl">
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-[color:var(--color-line)] bg-white/[0.03] px-4 py-2 text-[11px] uppercase tracking-[0.22em] text-muted">
                <Workflow className="h-4 w-4 text-accent" />
                React Operations Console
              </div>
              <h1 className="font-display text-[clamp(2.6rem,5vw,4.75rem)] leading-[0.94] tracking-[-0.06em]">
                Real-time fraud operations, rebuilt as a product surface.
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-8 text-muted md:text-lg">
                Next.js, SWR, Recharts, Framer Motion, and TanStack Table now sit on top of the existing FastAPI endpoints so the dashboard behaves like a modern SPA instead of a static admin page.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 xl:min-w-[360px]">
              <div className="rounded-[24px] border border-[color:var(--color-line)] bg-slate-950/55 p-4">
                <div className="text-[11px] uppercase tracking-[0.22em] text-muted">Current Run</div>
                <div className="mt-3 font-mono text-base text-ink">{summary?.current_model?.run_id ?? "No active run"}</div>
                <div className="mt-2 text-sm text-muted">{summary?.current_metrics?.model_type ?? "No model metadata"}</div>
              </div>
              <div className="rounded-[24px] border border-[color:var(--color-line)] bg-slate-950/55 p-4">
                <div className="flex items-center justify-between gap-4 text-[11px] uppercase tracking-[0.22em] text-muted">
                  <span>Live Refresh</span>
                  <span
                    className={cn(
                      "inline-flex items-center gap-2 rounded-full px-2.5 py-1 text-[10px]",
                      summaryQuery.isValidating || metricsQuery.isValidating
                        ? "bg-accent/10 text-accent"
                        : "bg-emerald-400/10 text-emerald-300"
                    )}
                  >
                    <RefreshCcw className={cn("h-3 w-3", summaryQuery.isValidating || metricsQuery.isValidating ? "animate-spin" : "")} />
                    {summaryQuery.isValidating || metricsQuery.isValidating ? "Updating" : "Live"}
                  </span>
                </div>
                <div className="mt-3 text-sm text-ink">
                  {summary?.refreshed_at ? `Snapshot ${relativeTime(summary.refreshed_at)}` : "Waiting for API response"}
                </div>
                <div className="mt-2 text-sm text-muted">Polling `/dashboard/summary` and `/metrics/latest` every 30 seconds.</div>
              </div>
            </div>
          </div>
        </motion.header>

        <section className="mb-6 grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
          {kpis.map((kpi, index) => (
            <KpiCard key={kpi.label} delay={index * 0.04} {...kpi} />
          ))}
        </section>

        <section className="mb-6 grid gap-5 xl:grid-cols-[1.25fr,0.95fr]">
          <PerformanceTrendChart data={summary?.run_history ?? []} />
          <OperationsBreakdownChart data={operationsData} />
        </section>

        <motion.section
          className="mb-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4"
          initial={{ opacity: 0, y: 22 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.12, ease: [0.25, 0.46, 0.45, 0.94] }}
        >
          {[
            {
              icon: ShieldAlert,
              title: "Audit drill-downs",
              body: "Every row opens a modal with the full JSON payload from the summary snapshot."
            },
            {
              icon: Table2,
              title: "Searchable tables",
              body: "Global filtering and sorting are powered by TanStack Table across all six operational views."
            },
            {
              icon: Activity,
              title: "Smooth motion system",
              body: "Framer Motion handles card reveals, hover states, and modal transitions with the same easing system as the new landing page."
            },
            {
              icon: DatabaseZap,
              title: "Same source of truth",
              body: "The SPA consumes the existing FastAPI endpoints rather than inventing a second backend."
            }
          ].map((item) => (
            <article className="surface rounded-[28px] border border-[color:var(--color-line)] p-5" key={item.title}>
              <item.icon className="h-5 w-5 text-accent" />
              <h2 className="mt-4 font-display text-2xl tracking-[-0.04em]">{item.title}</h2>
              <p className="mt-3 text-sm leading-7 text-muted">{item.body}</p>
            </article>
          ))}
        </motion.section>

        <section className="grid gap-5 xl:grid-cols-2">
          {tables.map((table) => (
            <DataTable
              columns={table.columns}
              data={table.data as DashboardRow[]}
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
