"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, ScanLine } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ForensicReport } from "@/lib/api";
import { audioBufferTo16kMono, TARGET_RATE } from "@/lib/audio";
import { BAND_COPY, bandFor, type Thresholds } from "@/lib/signal";
import { computeSpectrogram, paintSpectrogram } from "@/lib/spectrogram";

const BAND_HEX = { green: "#3f8a5c", amber: "#b97d12", red: "#b8432f" } as const;

function fmt(t: number) {
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
}

/** Score at time t: average of the timeline windows that cover it. */
function scoreAt(t: number, timeline: NonNullable<ForensicReport["timeline"]>): number | null {
  const hits = timeline.filter((w) => w.score != null && w.start <= t && t < w.end).map((w) => w.score as number);
  return hits.length ? hits.reduce((a, b) => a + b, 0) / hits.length : null;
}

export function EvidenceView({ audio, report, thresholds }: { audio: Blob; report: ForensicReport; thresholds: Thresholds }) {
  const specRef = useRef<HTMLCanvasElement>(null);
  const stripRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [scanned, setScanned] = useState(0); // furthest point the scan has reached (seconds)
  const [url, setUrl] = useState<string | null>(null);
  const duration = report.metadata.duration_s || 1;
  const timeline = useMemo(() => report.timeline ?? [], [report.timeline]);
  const speech = report.speech_segments ?? [];

  // Own the object URL inside the effect so React's dev double-mount can't revoke a URL still in use.
  useEffect(() => {
    const next = URL.createObjectURL(audio);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- the URL must be created and revoked together
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [audio]);

  // Decode and paint the spectrogram once.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const ctx = new AudioContext();
        const buffer = await ctx.decodeAudioData(await audio.arrayBuffer());
        void ctx.close();
        const mono = await audioBufferTo16kMono(buffer);
        if (cancelled || !specRef.current) return;
        paintSpectrogram(specRef.current, computeSpectrogram(mono, TARGET_RATE));
        setReady(true);
      } catch {
        if (!cancelled) setError("This browser couldn't draw the spectrogram for this file.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [audio]);

  // Paint the synthetic-signal strip: revealed up to the scan point, faint beyond it.
  useEffect(() => {
    const canvas = stripRef.current;
    const g = canvas?.getContext("2d");
    if (!canvas || !g) return;
    const width = 600;
    canvas.width = width;
    canvas.height = 1;
    g.clearRect(0, 0, width, 1);
    for (let x = 0; x < width; x++) {
      const t = ((x + 0.5) / width) * duration;
      const s = scoreAt(t, timeline);
      if (s == null) continue;
      g.globalAlpha = t <= scanned ? 1 : 0.28;
      g.fillStyle = BAND_HEX[bandFor(s, thresholds)];
      g.fillRect(x, 0, 1, 1);
    }
  }, [timeline, duration, scanned, thresholds]);

  // Follow playback.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      const el = audioRef.current;
      if (el) {
        setTime(el.currentTime);
        setScanned((s) => Math.max(s, el.currentTime));
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (playing) el.pause();
    else {
      if (el.ended || el.currentTime >= duration - 0.05) el.currentTime = 0;
      void el.play();
    }
  };

  const current = scoreAt(time, timeline);
  const currentBand = current == null ? null : bandFor(current, thresholds);
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / duration) * 100))}%`;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-xl font-semibold">
            <ScanLine className="size-5 text-primary" />
            Voice evidence
          </h2>
          <p className="text-sm text-muted-foreground">
            The sound of the clip over time (spectrogram). Outlined areas are where someone is speaking; the strip shows
            how synthetic each moment looked.
          </p>
        </div>
        <Button size="lg" onClick={toggle} disabled={!ready}>
          {playing ? <Pause /> : <Play />}
          {playing ? "Pause" : scanned > 0 ? "Play again" : "Play & scan"}
        </Button>
      </div>

      <div className="relative overflow-hidden rounded-2xl ring-1 ring-border">
        <canvas ref={specRef} className="block h-44 w-full bg-background [image-rendering:auto] sm:h-52" aria-label="Spectrogram" />
        {speech.map((s, i) => (
          <div
            key={i}
            className="pointer-events-none absolute inset-y-2 rounded-lg border-2 border-primary/60 bg-primary/5"
            style={{ left: pct(s.start), width: `calc(${pct(s.end - s.start)})` }}
          >
            <span className="absolute -top-0.5 left-1 rounded bg-primary px-1 text-[10px] font-bold tracking-wide text-primary-foreground uppercase">
              voice
            </span>
          </div>
        ))}
        {(playing || scanned > 0) && (
          <div
            className="pointer-events-none absolute inset-y-0 w-0.5 bg-foreground shadow-[0_0_12px_2px_rgba(47,107,94,0.6)]"
            style={{ left: pct(time) }}
          />
        )}
        {!ready && !error && (
          <div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">Drawing spectrogram…</div>
        )}
        {error && <div className="absolute inset-0 grid place-items-center p-4 text-sm text-muted-foreground">{error}</div>}
      </div>

      {timeline.length > 0 && (
        <div>
          <div className="relative">
            <canvas ref={stripRef} className="block h-5 w-full rounded-full ring-1 ring-border [image-rendering:pixelated]" />
            {(playing || scanned > 0) && (
              <div className="pointer-events-none absolute -inset-y-1 w-0.5 bg-foreground" style={{ left: pct(time) }} />
            )}
          </div>
          <div className="mt-1 flex justify-between text-xs text-muted-foreground">
            <span>0:00</span>
            <span>Synthetic signal over time</span>
            <span>{fmt(duration)}</span>
          </div>
        </div>
      )}

      <p className="min-h-6 text-base" aria-live="polite">
        {(playing || scanned > 0) && currentBand ? (
          <>
            <span className="font-bold">{fmt(time)}</span>
            {" · "}
            <span style={{ color: BAND_HEX[currentBand] }} className="font-bold">
              {BAND_COPY[currentBand].label}
            </span>
            {currentBand === "red" ? ": synthetic characteristics in this moment" : ""}
          </>
        ) : (
          <span className="text-muted-foreground">Press Play &amp; scan to hear the clip while Verity walks through it.</span>
        )}
      </p>

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="size-3 rounded border-2 border-primary/60" /> speech
        </span>
        {(["green", "amber", "red"] as const).map((b) => (
          <span key={b} className="flex items-center gap-1.5">
            <span className="size-3 rounded-full" style={{ background: BAND_HEX[b] }} /> {BAND_COPY[b].label}
          </span>
        ))}
        {report.timeline_source && <span>· timeline: {report.timeline_source}</span>}
      </div>

      <audio
        ref={audioRef}
        src={url ?? undefined}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => {
          setPlaying(false);
          setScanned(duration);
        }}
        className="hidden"
      />
    </div>
  );
}
