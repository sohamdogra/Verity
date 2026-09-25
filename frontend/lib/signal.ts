import type { Band } from "@/lib/api";

export interface Thresholds {
  green_max: number;
  red_min: number;
}

// Mirrors backend/config.py; the live values are fetched from /health.
export const DEFAULT_THRESHOLDS: Thresholds = { green_max: 0.4, red_min: 0.65 };

export function bandFor(score: number, t: Thresholds = DEFAULT_THRESHOLDS): Exclude<Band, "idle"> {
  if (score > t.red_min) return "red";
  if (score >= t.green_max) return "amber";
  return "green";
}

/** Words only — we never show false-precision percentages. */
export const BAND_COPY: Record<Band, { label: string; short: string; emoji: string }> = {
  idle: { label: "Waiting for audio", short: "Waiting", emoji: "⚪" },
  green: { label: "Low signal", short: "Low", emoji: "🟢" },
  amber: { label: "Elevated signal", short: "Elevated", emoji: "🟡" },
  red: { label: "High concern", short: "High", emoji: "🔴" },
};

export const BAND_STYLES: Record<Band, { text: string; bg: string; ring: string; dot: string }> = {
  idle: { text: "text-muted-foreground", bg: "bg-muted", ring: "ring-border", dot: "bg-muted-foreground/40" },
  green: { text: "text-safe", bg: "bg-safe-soft", ring: "ring-safe/30", dot: "bg-safe" },
  amber: { text: "text-caution", bg: "bg-caution-soft", ring: "ring-caution/30", dot: "bg-caution" },
  red: { text: "text-concern", bg: "bg-concern-soft", ring: "ring-concern/30", dot: "bg-concern" },
};

export function stabilityWord(stability: number, samples: number): string {
  if (samples < 2) return "still gathering samples";
  if (stability >= 0.8) return "consistent";
  if (stability >= 0.55) return "fairly consistent";
  return "varied";
}
