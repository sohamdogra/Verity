"use client";

import { useEffect, useRef, useState } from "react";
import { AlertTriangle, AudioLines, ChevronDown, FileSearch, Loader2, MessageSquareQuote, Upload } from "lucide-react";
import { BandStatus } from "@/components/band-status";
import { EvidenceView } from "@/components/evidence-view";
import { DetectionUnavailable } from "@/components/notices";
import { ProtectActions } from "@/components/protect-actions";
import { SignalMeter } from "@/components/signal-meter";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useFamily } from "@/hooks/use-family";
import { useHealth } from "@/hooks/use-health";
import { api, ApiError, type ForensicReport, type ForensicTechnique } from "@/lib/api";
import { DEMO_CLIPS } from "@/lib/demo-clips";
import { BAND_COPY, BAND_STYLES, bandFor, DEFAULT_THRESHOLDS, type Thresholds } from "@/lib/signal";
import { cn } from "@/lib/utils";

const PROGRESS = [
  "Decoding the recording…",
  "Running several independent AI voice detectors…",
  "Checking background noise, frequency range and edits…",
  "Listening to what was said…",
  "Weighing the evidence…",
];

const CUE_LABELS: Record<string, string> = {
  urgency: "Urgency",
  money: "Money request",
  secrecy: "Secrecy",
  distress: "Emergency story",
  authority: "Claims authority",
};

function Highlighted({ text, phrases }: { text: string; phrases: string[] }) {
  if (!phrases.length) return <>{text}</>;
  const escaped = phrases.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const parts = text.split(new RegExp(`(${escaped.join("|")})`, "gi"));
  return (
    <>
      {parts.map((part, i) =>
        phrases.some((p) => p.toLowerCase() === part.toLowerCase()) ? (
          <mark key={i} className="rounded bg-caution-soft px-1 font-bold text-caution">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

function TechniqueRow({ t, thresholds }: { t: ForensicTechnique; thresholds: Thresholds }) {
  const band = t.score == null ? null : bandFor(t.score, thresholds);
  return (
    <li className="flex gap-3 py-3">
      <span
        title={band ? BAND_COPY[band].label : "Informational"}
        className={cn("mt-1.5 size-3.5 shrink-0 rounded-full", band ? BAND_STYLES[band].dot : "bg-muted-foreground/30")}
      />
      <div className="min-w-0">
        <p className="text-base font-bold break-words">{t.name}</p>
        <p className="text-base text-foreground/80">{t.finding}</p>
        {t.windows && t.windows.length > 1 && (
          <div className="mt-1.5 flex flex-wrap gap-1" aria-label="Per-segment readings">
            {t.windows.map((w, i) => (
              <span key={i} className={cn("h-2 w-4 rounded-full", BAND_STYLES[bandFor(w, thresholds)].dot)} />
            ))}
          </div>
        )}
      </div>
    </li>
  );
}

export default function CheckRecordingPage() {
  const { family } = useFamily();
  const { health } = useHealth();
  const thresholds = health?.thresholds ?? DEFAULT_THRESHOLDS;
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [report, setReport] = useState<ForensicReport | null>(null);
  const [analyzed, setAnalyzed] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(() => setProgress((p) => Math.min(p + 1, PROGRESS.length - 1)), 1800);
    return () => clearInterval(timer);
  }, [busy]);

  const run = async (file: Blob, name: string) => {
    setBusy(name);
    setProgress(0);
    setReport(null);
    setAnalyzed(null);
    setError(null);
    setUnavailable(false);
    try {
      setReport(await api.forensics(file, name, family?.id));
      setAnalyzed(file);
    } catch (e) {
      if (e instanceof ApiError && e.status === 503) setUnavailable(true);
      else setError(e instanceof Error ? e.message : "Couldn't check that recording.");
    } finally {
      setBusy(null);
    }
  };

  const runSample = async (url: string, name: string) => {
    const res = await fetch(url);
    if (!res.ok) return setError(`Couldn't load the sample (${res.status}).`);
    await run(await res.blob(), name);
  };

  const detectors = report?.techniques?.filter((t) => t.kind === "neural_detector") ?? [];
  const signals = report?.techniques?.filter((t) => t.kind !== "neural_detector") ?? [];
  const value = report?.synthetic_likelihood != null ? report.synthetic_likelihood / 100 : null;
  const cues = report?.transcript?.cues ?? [];

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-4xl font-semibold sm:text-5xl">Check a recording</h1>
        <p className="mt-3 max-w-3xl text-lg text-foreground/80">
          Got a worrying voicemail or voice message? Upload it and Verity runs several independent checks: AI voice
          detectors, sound forensics, and a look at what was said. Any format works (WAV, MP3, M4A and more).
        </p>
      </section>

      <Card className="gap-4 p-5 sm:p-6">
        <button
          type="button"
          disabled={!!busy}
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file) void run(file, file.name);
          }}
          className={cn(
            "flex w-full flex-col items-center gap-3 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors",
            dragging ? "border-primary bg-primary/5" : "border-input hover:bg-muted/60",
          )}
        >
          {busy ? <Loader2 className="size-10 animate-spin text-primary" /> : <Upload className="size-10 text-primary" />}
          <span className="font-heading text-2xl font-semibold">
            {busy ? `Checking ${busy}…` : "Choose or drop a recording"}
          </span>
          <span className="text-base text-muted-foreground">
            {busy ? PROGRESS[progress] : "Voicemails, voice notes or call recordings. Nothing is stored."}
          </span>
        </button>
        <input
          ref={input}
          type="file"
          accept="audio/*,.wav,.mp3,.m4a,.aac,.ogg,.opus,.flac,.webm"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void run(file, file.name);
            e.target.value = "";
          }}
        />
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-base text-muted-foreground">Or try a sample:</span>
          {DEMO_CLIPS.map((clip) => (
            <Button key={clip.id} variant="secondary" disabled={!!busy} onClick={() => runSample(clip.url, `${clip.id}.wav`)}>
              <AudioLines />
              {clip.title}
            </Button>
          ))}
        </div>
        {error && <p className="text-base text-destructive">{error}</p>}
      </Card>

      {unavailable && <DetectionUnavailable />}

      {report && report.status !== "ok" && (
        <Card className="p-6 text-lg">{report.summary}</Card>
      )}

      {report?.status === "ok" && report.band && (
        <>
          {report.band === "red" && (
            <section className="rounded-3xl bg-concern-soft p-6 ring-2 ring-concern/40">
              <div className="flex items-start gap-4">
                <AlertTriangle className="mt-1 size-9 shrink-0 text-concern" />
                <div>
                  <p className="text-lg font-bold text-concern">⚠ Synthetic characteristics detected</p>
                  <p className="mt-1 font-heading text-3xl leading-tight font-semibold">
                    Don&apos;t act on this message. Call them back using a number you trust.
                  </p>
                </div>
              </div>
              <div className="mt-5">
                <ProtectActions family={family} band="red" smoothed={value} big />
              </div>
            </section>
          )}

          {analyzed && (
            <Card className="p-5 sm:p-6">
              <EvidenceView audio={analyzed} report={report} thresholds={thresholds} />
            </Card>
          )}

          <div className="grid gap-6 lg:grid-cols-[1.35fr_1fr]">
            <Card className="gap-5 p-5 sm:p-6">
              <BandStatus band={report.band} listening={false} />
              <SignalMeter value={value} thresholds={thresholds} />
              <p className="text-lg">{report.summary}</p>
              {report.manipulation_type && report.manipulation_type !== "none" && (
                <p className="rounded-xl bg-muted px-4 py-3 text-base">
                  <span className="font-bold">Likely manipulation:</span> {report.manipulation_label}
                </p>
              )}
              <div>
                <h2 className="text-xl font-semibold">AI voice detectors</h2>
                <p className="text-sm text-muted-foreground">
                  Independent models trained on different kinds of fake audio. Agreement matters more than any one.
                </p>
                <ul className="divide-y divide-border">
                  {detectors.map((t) => (
                    <TechniqueRow key={t.id} t={t} thresholds={thresholds} />
                  ))}
                </ul>
              </div>
              <div>
                <h2 className="text-xl font-semibold">Sound forensics</h2>
                <ul className="divide-y divide-border">
                  {signals.map((t) => (
                    <TechniqueRow key={t.id} t={t} thresholds={thresholds} />
                  ))}
                </ul>
              </div>
            </Card>

            <div className="space-y-6">
              {report.transcript && (
                <Card className="gap-3 p-5">
                  <h2 className="flex items-center gap-2 text-xl font-semibold">
                    <MessageSquareQuote className="size-5 text-primary" />
                    What was said
                  </h2>
                  <p className="rounded-xl bg-muted/60 p-3 text-base leading-relaxed">
                    {report.transcript.text ? (
                      <Highlighted text={report.transcript.text} phrases={cues.map((c) => c.phrase)} />
                    ) : (
                      <span className="text-muted-foreground">No clear words were recognized.</span>
                    )}
                  </p>
                  {cues.length > 0 ? (
                    <div>
                      <p className="text-base font-bold">Common scam-script signs:</p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {cues.map((c) => (
                          <span key={c.category} className="rounded-full bg-caution-soft px-3 py-1 text-sm font-bold text-caution">
                            {CUE_LABELS[c.category] ?? c.category}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <p className="text-base text-muted-foreground">No common scam-script phrases found.</p>
                  )}
                  <p className="text-sm text-muted-foreground">
                    This checks what was said, not how the voice sounds, so it doesn&apos;t change the signal above.
                  </p>
                </Card>
              )}

              <Card className="gap-2 p-5 text-base">
                <h2 className="flex items-center gap-2 text-xl font-semibold">
                  <FileSearch className="size-5 text-primary" />
                  Remember
                </h2>
                <p>
                  This is a probabilistic signal, not proof. A clean result never means a message is safe. If it asks
                  for money or secrecy, verify first.
                </p>
                {report.band !== "red" && <ProtectActions family={family} band={report.band} smoothed={value} />}
              </Card>

              <details className="group rounded-xl bg-card p-5 ring-1 ring-foreground/10">
                <summary className="flex cursor-pointer list-none items-center justify-between text-lg font-bold">
                  For analysts
                  <ChevronDown className="size-5 transition-transform group-open:rotate-180" />
                </summary>
                <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                  <dt className="text-muted-foreground">Synthetic likelihood</dt>
                  <dd className="font-bold">{Math.round(report.synthetic_likelihood ?? 0)} / 100</dd>
                  <dt className="text-muted-foreground">Fusion</dt>
                  <dd>{report.fusion}</dd>
                  <dt className="text-muted-foreground">Format</dt>
                  <dd>
                    {report.metadata.codec ?? report.metadata.container} · {report.metadata.sample_rate / 1000} kHz ·{" "}
                    {report.metadata.channels} ch · {report.metadata.duration_s}s
                  </dd>
                  {report.container && report.container.riff_chunks.length > 0 && (
                    <>
                      <dt className="text-muted-foreground">Container</dt>
                      <dd>
                        chunks {report.container.riff_chunks.join(", ")}
                        {Object.entries(report.container.tags).map(([k, v]) => ` · ${k}=${v}`).join("")}
                      </dd>
                    </>
                  )}
                  <dt className="text-muted-foreground">Time</dt>
                  <dd>{report.elapsed_seconds}s</dd>
                </dl>
                <ol className="mt-3 list-decimal space-y-1 pl-5 text-sm text-muted-foreground">
                  {report.steps.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ol>
                {!report.calibrated && (
                  <p className="mt-3 text-sm text-muted-foreground">
                    Using the default uncalibrated ensemble. Train a model with scripts/hearsay.py for calibrated scores.
                  </p>
                )}
              </details>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
