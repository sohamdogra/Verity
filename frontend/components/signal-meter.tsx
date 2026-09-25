import { cn } from "@/lib/utils";
import type { Reading } from "@/hooks/use-shield";
import { BAND_COPY, BAND_STYLES, bandFor, type Thresholds } from "@/lib/signal";

/** Three soft zones and a marker. Deliberately no numbers — this is a signal, not a score. */
export function SignalMeter({ value, thresholds }: { value: number | null; thresholds: Thresholds }) {
  const green = thresholds.green_max * 100;
  const amber = (thresholds.red_min - thresholds.green_max) * 100;
  return (
    <div>
      <div className="relative h-5 w-full">
        <div className="flex h-full w-full overflow-hidden rounded-full ring-1 ring-border">
          <div className="h-full bg-safe/25" style={{ width: `${green}%` }} />
          <div className="h-full bg-caution/30" style={{ width: `${amber}%` }} />
          <div className="h-full flex-1 bg-concern/30" />
        </div>
        {value != null && (
          <div
            className="absolute top-1/2 size-7 -translate-x-1/2 -translate-y-1/2 rounded-full border-4 border-card shadow-md transition-[left] duration-700 ease-out"
            style={{ left: `${Math.min(97, Math.max(3, value * 100))}%` }}
          >
            <div className={cn("size-full rounded-full", BAND_STYLES[bandFor(value, thresholds)].dot)} />
          </div>
        )}
      </div>
      <div className="mt-2 flex text-sm text-muted-foreground">
        <span style={{ width: `${green}%` }}>Low</span>
        <span style={{ width: `${amber}%` }} className="text-center">
          Elevated
        </span>
        <span className="flex-1 text-right">High concern</span>
      </div>
    </div>
  );
}

export function SampleDots({ readings, thresholds }: { readings: Reading[]; thresholds: Thresholds }) {
  if (!readings.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-2" aria-label="Recent audio samples">
      <span className="text-sm text-muted-foreground">Recent samples</span>
      {readings.map((r, i) => {
        const band = bandFor(r.score, thresholds);
        return (
          <span
            key={`${r.at}-${i}`}
            title={BAND_COPY[band].label}
            className={cn("size-3.5 rounded-full animate-in zoom-in-50", BAND_STYLES[band].dot)}
          />
        );
      })}
    </div>
  );
}
