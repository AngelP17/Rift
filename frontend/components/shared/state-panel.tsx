"use client";

import { Warning, WarningCircle } from "@phosphor-icons/react";
import { ReactNode } from "react";
import { cn } from "@/lib/utils";

type StatePanelProps = {
  title: string;
  description: string;
  tone?: "empty" | "error" | "loading";
  children?: ReactNode;
  className?: string;
};

const toneStyles: Record<NonNullable<StatePanelProps["tone"]>, string> = {
  empty: "border-white/10 bg-slate-950/42 text-muted",
  error: "border-rose-400/25 bg-rose-400/[0.04] text-rose-100/80",
  loading: "border-white/10 bg-slate-950/42 text-muted"
};

const toneIcon: Record<NonNullable<StatePanelProps["tone"]>, ReactNode> = {
  empty: <WarningCircle className="h-5 w-5 text-muted" weight="duotone" />,
  error: <Warning className="h-5 w-5 text-rose-300" weight="duotone" />,
  loading: <span aria-hidden="true" className="block h-2 w-2 animate-pulse rounded-full bg-accent" />
};

export function StatePanel({
  title,
  description,
  tone = "empty",
  children,
  className
}: StatePanelProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-start gap-3 rounded-[24px] border px-5 py-6",
        toneStyles[tone],
        className
      )}
      data-state={tone}
      role={tone === "error" ? "alert" : undefined}
    >
      <div className="flex items-center gap-2">
        {toneIcon[tone]}
        <h4 className="font-display text-base font-semibold tracking-[-0.04em] text-ink">{title}</h4>
      </div>
      <p className="max-w-xl text-sm leading-6 text-muted">{description}</p>
      {children ? <div className="mt-1 flex flex-wrap gap-2 text-sm">{children}</div> : null}
    </div>
  );
}
