"use client";

import { useEffect, useRef } from "react";
import type { Reading } from "@/hooks/use-shield";
import { bandFor, type Thresholds } from "@/lib/signal";
import { COLORMAP } from "@/lib/spectrogram";

const WIDTH = 800; // internal pixels; ~10 s visible
const ROWS = 128;
const PX_PER_SEC = 80;
const MAX_HZ = 8000;
const BAND_HEX = { green: "#3f8a5c", amber: "#b97d12", red: "#b8432f" } as const;

/**
 * A scrolling spectrogram of whatever Verity is listening to, with a verdict strip underneath:
 * each analyzed window lights up under the exact stretch of audio it judged.
 */
export function LiveSpectrogram({
  analyser,
  readings,
  windowSeconds,
  thresholds,
}: {
  analyser: AnalyserNode | null;
  readings: Reading[];
  windowSeconds: number;
  thresholds: Thresholds;
}) {
  const specRef = useRef<HTMLCanvasElement>(null);
  const stripRef = useRef<HTMLCanvasElement>(null);
  const painted = useRef(new Set<number>());
  const readingsRef = useRef(readings);

  useEffect(() => {
    readingsRef.current = readings;
    if (readings.length === 0) painted.current.clear();
  }, [readings]);

  useEffect(() => {
    const spec = specRef.current;
    const strip = stripRef.current;
    const gs = spec?.getContext("2d");
    const gt = strip?.getContext("2d");
    if (!spec || !strip || !gs || !gt) return;
    if (spec.width !== WIDTH) {
      spec.width = WIDTH;
      spec.height = ROWS;
      strip.width = WIDTH;
      strip.height = 1;
      gs.fillStyle = "rgb(250,246,239)";
      gs.fillRect(0, 0, WIDTH, ROWS);
    }
    if (!analyser) return;

    const bins = new Uint8Array(analyser.frequencyBinCount);
    const maxBin = Math.min(bins.length, Math.round((MAX_HZ / (analyser.context.sampleRate / 2)) * bins.length));
    const column = gs.createImageData(1, ROWS);
    let last = performance.now();
    let carry = 0;
    let frame = 0;

    const draw = (now: number) => {
      carry += ((now - last) / 1000) * PX_PER_SEC;
      last = now;
      const dx = Math.floor(carry);
      if (dx > 0) {
        carry -= dx;
        // Scroll both canvases left by dx and paint the newest spectrum on the right edge.
        gs.drawImage(spec, -dx, 0);
        gt.drawImage(strip, -dx, 0);
        gt.clearRect(WIDTH - dx, 0, dx, 1);
        analyser.getByteFrequencyData(bins);
        for (let r = 0; r < ROWS; r++) {
          const bin = Math.floor(((ROWS - 1 - r) / ROWS) * maxBin);
          const idx = bins[bin] * 3;
          const p = r * 4;
          column.data[p] = COLORMAP[idx];
          column.data[p + 1] = COLORMAP[idx + 1];
          column.data[p + 2] = COLORMAP[idx + 2];
          column.data[p + 3] = 255;
        }
        for (let i = 0; i < dx; i++) gs.putImageData(column, WIDTH - dx + i, 0);
      }
      // Paint any new verdicts under the stretch of audio they describe.
      const wallNow = Date.now();
      for (const r of readingsRef.current) {
        if (painted.current.has(r.at)) continue;
        painted.current.add(r.at);
        const xEnd = WIDTH - ((wallNow - r.at) / 1000) * PX_PER_SEC;
        const xStart = xEnd - windowSeconds * PX_PER_SEC;
        gt.fillStyle = BAND_HEX[bandFor(r.score, thresholds)];
        gt.fillRect(Math.floor(xStart) + 1, 0, Math.max(1, Math.floor(xEnd - xStart) - 2), 1);
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [analyser, windowSeconds, thresholds]);

  return (
    <div className="space-y-1.5">
      <div className="relative overflow-hidden rounded-2xl ring-1 ring-border">
        <canvas ref={specRef} className="block h-32 w-full bg-background sm:h-36" aria-label="Live spectrogram of the voice" />
        {!analyser && readings.length === 0 && (
          <div className="absolute inset-0 grid place-items-center px-4 text-center text-sm text-muted-foreground">
            The voice appears here as it&apos;s heard, one slice of sound at a time.
          </div>
        )}
        {analyser && (
          <span className="absolute top-2 right-2 flex items-center gap-1.5 rounded-full bg-card/90 px-2 py-0.5 text-xs font-bold text-primary">
            <span className="size-2 animate-pulse rounded-full bg-concern" /> LIVE
          </span>
        )}
      </div>
      <canvas ref={stripRef} className="block h-3 w-full rounded-full bg-muted [image-rendering:pixelated]" aria-hidden="true" />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>≈10 s ago</span>
        <span>verdict for each {windowSeconds}-second sample</span>
        <span>now</span>
      </div>
    </div>
  );
}
