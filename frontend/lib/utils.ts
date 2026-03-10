import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatPercent(value: number, maximumFractionDigits = 1) {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumFractionDigits
  }).format(value);
}

export function formatDecimal(value: number, digits = 3) {
  return value.toFixed(digits);
}

export function titleCase(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function relativeTime(timestamp?: string) {
  if (!timestamp) {
    return "Unknown";
  }
  const date = new Date(timestamp);
  const delta = Math.round((date.getTime() - Date.now()) / 1000);
  const formatter = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

  if (Math.abs(delta) < 60) return formatter.format(delta, "second");
  if (Math.abs(delta) < 3600) return formatter.format(Math.round(delta / 60), "minute");
  if (Math.abs(delta) < 86400) return formatter.format(Math.round(delta / 3600), "hour");
  return formatter.format(Math.round(delta / 86400), "day");
}
