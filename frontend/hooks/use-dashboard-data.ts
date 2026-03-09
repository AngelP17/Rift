"use client";

import useSWR from "swr";
import { api } from "@/lib/api";

const THIRTY_SECONDS = 30_000;

export function useDashboardSummary() {
  return useSWR("dashboard-summary", api.getDashboardSummary, {
    refreshInterval: THIRTY_SECONDS,
    revalidateOnFocus: false
  });
}

export function useMetrics() {
  return useSWR("dashboard-metrics", api.getMetrics, {
    refreshInterval: THIRTY_SECONDS,
    revalidateOnFocus: false
  });
}
