"use client";

import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type OperationsBreakdownChartProps = {
  data: Array<{ label: string; value: number; fill: string }>;
};

export function OperationsBreakdownChart({ data }: OperationsBreakdownChartProps) {
  return (
    <motion.section
      className="surface rounded-[32px] border border-[color:var(--color-line)] p-6"
      initial={{ opacity: 0, y: 22 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, delay: 0.08, ease: [0.25, 0.46, 0.45, 0.94] }}
    >
      <div className="mb-6 flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.22em] text-muted">System Activity</p>
          <h3 className="mt-2 font-display text-[2rem] tracking-[-0.05em] text-ink">Operational workload snapshot</h3>
        </div>
        <p className="max-w-xl text-sm leading-7 text-muted">
          Use the bar chart to compare ETL, fairness, drift, federated, dataset, and audit activity without jumping between tables.
        </p>
      </div>

      <div className="h-[340px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
            <XAxis dataKey="label" stroke="#92a3c7" tick={{ fontSize: 11 }} />
            <YAxis stroke="#92a3c7" tick={{ fontSize: 11 }} />
            <Tooltip
              cursor={{ fill: "rgba(255,255,255,0.03)" }}
              contentStyle={{
                background: "rgba(8, 13, 27, 0.96)",
                borderRadius: 18,
                border: "1px solid rgba(110, 168, 254, 0.18)"
              }}
            />
            <Bar dataKey="value" radius={[14, 14, 0, 0]}>
              {data.map((entry) => (
                <Cell key={entry.label} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </motion.section>
  );
}
