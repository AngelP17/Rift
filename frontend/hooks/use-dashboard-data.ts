"use client";

import useSWR from "swr";
import { api } from "@/lib/api";
import { mockDashboardSummary, mockMetrics } from "@/lib/mock-dashboard";

const THIRTY_SECONDS = 30_000;

export function useDashboardSummary() {
  return useSWR("dashboard-summary", api.getDashboardSummary, {
    fallbackData: mockDashboardSummary,
    refreshInterval: THIRTY_SECONDS,
    revalidateOnFocus: false
  });
}

export function useMetrics() {
  return useSWR("dashboard-metrics", api.getMetrics, {
    fallbackData: mockMetrics,
    refreshInterval: THIRTY_SECONDS,
    revalidateOnFocus: false
  });
}
