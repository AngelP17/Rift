"use client";

import { motion } from "framer-motion";
import {
  Area,
  AreaChart,
  Brush,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { formatPercent } from "@/lib/utils";

type PerformanceTrendChartProps = {
  data: Array<{ run_id: string; pr_auc: number }>;
};

export function PerformanceTrendChart({ data }: PerformanceTrendChartProps) {
  return (
    <motion.section
      className="surface rounded-[32px] border border-[color:var(--color-line)] p-6"
      initial={{ opacity: 0, y: 22 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: [0.25, 0.46, 0.45, 0.94] }}
    >
      <div className="mb-6 flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.22em] text-muted">Performance Trend</p>
          <h3 className="mt-2 font-display text-[2rem] tracking-[-0.05em] text-ink">PR-AUC history with zoomable brush</h3>
        </div>
        <p className="max-w-xl text-sm leading-7 text-muted">
          Recharts replaces the previous Chart.js sparkline with interactive tooltips, area gradients, and a brush for zooming into individual runs.
        </p>
      </div>

      <div className="h-[340px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="prauc-fill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="rgba(110,168,254,0.55)" />
                <stop offset="100%" stopColor="rgba(110,168,254,0.04)" />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
            <XAxis dataKey="run_id" stroke="#92a3c7" tick={{ fontSize: 11 }} tickFormatter={(value) => value.replace("run_", "")} />
            <YAxis domain={[0, 1]} stroke="#92a3c7" tick={{ fontSize: 11 }} tickFormatter={(value: number) => formatPercent(value, 0)} />
            <Tooltip
              contentStyle={{
                background: "rgba(8, 13, 27, 0.96)",
                borderRadius: 18,
                border: "1px solid rgba(110, 168, 254, 0.18)"
              }}
              formatter={(value: number) => [formatPercent(value, 2), "PR-AUC"]}
            />
            <Area
              type="monotone"
              dataKey="pr_auc"
              stroke="#6ea8fe"
              strokeWidth={3}
              fill="url(#prauc-fill)"
              activeDot={{ r: 6, fill: "#2fbf71", stroke: "#0b1020" }}
            />
            <Brush
              dataKey="run_id"
              height={28}
              stroke="#6ea8fe"
              fill="rgba(110, 168, 254, 0.08)"
              travellerWidth={10}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.section>
  );
}
