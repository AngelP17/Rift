"use client";

import { motion } from "framer-motion";
import { AnimatedNumber } from "@/components/shared/animated-number";
import { cn } from "@/lib/utils";

type KpiCardProps = {
  label: string;
  value: number;
  detail: string;
  tone: "good" | "warn" | "bad" | "neutral";
  formatter: (value: number) => string;
  delay?: number;
};

const toneStyles: Record<KpiCardProps["tone"], string> = {
  good: "border-emerald-400/25 bg-emerald-400/5",
  warn: "border-amber-400/25 bg-amber-400/5",
  bad: "border-rose-400/25 bg-rose-400/5",
  neutral: "border-[color:var(--color-line)] bg-white/[0.02]"
};

export function KpiCard({ label, value, detail, tone, formatter, delay = 0 }: KpiCardProps) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
      whileHover={{ y: -4, scale: 1.01 }}
      className={cn(
        "surface rounded-[28px] border p-5 transition-transform duration-300 ease-fluid will-change-transform",
        toneStyles[tone]
      )}
    >
      <div className="text-[11px] uppercase tracking-[0.22em] text-muted">{label}</div>
      <div className="mt-4 font-mono text-4xl font-semibold tracking-[-0.04em] text-ink">
        <AnimatedNumber value={value} format={formatter} />
      </div>
      <p className="mt-3 text-sm leading-7 text-muted">{detail}</p>
    </motion.article>
  );
}
