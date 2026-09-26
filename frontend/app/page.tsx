"use client";

import Link from "next/link";
import { useRef } from "react";
import { AlertTriangle, FileAudio, Loader2, Mic, MicOff, PhoneCall, Play, Square } from "lucide-react";
import { BandStatus } from "@/components/band-status";
import { ExplanationPanel } from "@/components/explanation-panel";
import { DetectionLoading, DetectionUnavailable, ServerUnreachable } from "@/components/notices";
import { ProtectActions } from "@/components/protect-actions";
import { SampleDots, SignalMeter } from "@/components/signal-meter";
import { LiveSpectrogram } from "@/components/live-spectrogram";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useFamily } from "@/hooks/use-family";
import { useHealth } from "@/hooks/use-health";
import { useShield } from "@/hooks/use-shield";
import { DEMO_CLIPS } from "@/lib/demo-clips";
import { DEFAULT_THRESHOLDS } from "@/lib/signal";

export default function LiveShieldPage() {
  const { health, reachable } = useHealth();
  const { family } = useFamily();
  const thresholds = health?.thresholds ?? DEFAULT_THRESHOLDS;
  const shield = useShield({ familyId: family?.id ?? null, windowSeconds: health?.window_seconds ?? 2.5 });
  const fileInput = useRef<HTMLInputElement>(null);

  const band = shield.result?.band ?? "idle";
  const listening = shield.status === "listening" || shield.status === "starting";
  const detectionDown =
    shield.status === "unavailable" ||
    health?.detection_state === "unavailable" ||
    health?.detection_state === "disabled";
  const detectionLoading = health?.detection_state === "loading";
  const canAnalyze = reachable && !!health && !detectionDown && !detectionLoading;

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-4xl font-semibold sm:text-5xl">Is this call real?</h1>
        <p className="mt-3 max-w-3xl text-lg text-foreground/80">
          Put the call on speaker and let Verity listen. Your strongest protection is verification: a safe-word,
          a private question, or calling back on a number you trust. Verity&apos;s detector is an extra signal on
          top of that.
        </p>
        <Link
          href="/call"
          className="mt-4 inline-flex items-center gap-2 rounded-xl bg-primary/10 px-4 py-2.5 text-lg font-bold text-primary hover:bg-primary/15"
        >
          <PhoneCall className="size-5" /> On a call right now? Use simple call mode
        </Link>
      </section>

      {!reachable && <ServerUnreachable />}
      {reachable && detectionLoading && <DetectionLoading />}
      {reachable && detectionDown && (
        <DetectionUnavailable
          reason={health?.detection_state === "disabled" ? "The voice detector is turned off." : undefined}
        />
      )}

      {band === "red" && (
        <section className="animate-in rounded-3xl bg-concern-soft p-6 ring-2 ring-concern/40 fade-in slide-in-from-top-2">
          <div className="flex items-start gap-4">
            <AlertTriangle className="mt-1 size-9 shrink-0 text-concern" />
            <div>
              <p className="text-lg font-bold text-concern">⚠ Elevated synthetic signal</p>
              <p className="mt-1 font-heading text-3xl leading-tight font-semibold sm:text-4xl">
                Hang up and call them back using a number you trust.
              </p>
              <p className="mt-2 text-base text-foreground/80">
                This is a signal, not proof, but it&apos;s worth checking before you send money or share anything.
              </p>
            </div>
          </div>
          <div className="mt-5">
            <p className="mb-3 text-base font-bold">Recommended actions</p>
            <ProtectActions family={family} band={band} smoothed={shield.result?.smoothed} big />
          </div>
        </section>
      )}

      <div className="grid gap-6 lg:grid-cols-[1.35fr_1fr]">
        <Card className="gap-5 p-5 sm:p-6">
          <BandStatus band={band} listening={listening} />
          <LiveSpectrogram
            analyser={shield.analyser}
            readings={shield.readings}
            windowSeconds={health?.window_seconds ?? 2.5}
            thresholds={thresholds}
          />
          <SignalMeter value={shield.result?.smoothed ?? null} thresholds={thresholds} />
          <SampleDots readings={shield.readings} thresholds={thresholds} />

          <div className="flex flex-wrap items-center gap-3">
            {shield.source === "mic" ? (
              <Button size="xl" variant="outline" onClick={shield.stop}>
                <MicOff />
                Stop listening
              </Button>
            ) : (
              <Button size="xl" onClick={shield.startMic} disabled={!canAnalyze || listening}>
                <Mic />
                Start listening
              </Button>
            )}
            {shield.source === "clip" && (
              <Button size="xl" variant="outline" onClick={shield.stop}>
                <Square />
                Stop sample
              </Button>
            )}
            {shield.status === "starting" && <Loader2 className="size-6 animate-spin text-primary" />}
          </div>
          <div className="min-h-6 text-base text-muted-foreground">
            {listening && shield.sourceLabel && <p>Listening to: {shield.sourceLabel}</p>}
            {listening && shield.quiet && <p>It&apos;s quiet. Verity only checks samples with speech in them.</p>}
            {shield.status === "error" && shield.error && <p className="text-destructive">{shield.error}</p>}
          </div>
        </Card>

        <div className="space-y-6">
          <ExplanationPanel result={shield.result} readings={shield.readings} thresholds={thresholds} />

          <Card className="gap-4 p-5">
            <div>
              <h2 className="text-xl font-semibold">Try a recording</h2>
              <p className="mt-1 text-base text-muted-foreground">
                Samples play out loud and are analyzed by the same detector, one short sample at a time.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
              {DEMO_CLIPS.map((clip) => (
                <Button
                  key={clip.id}
                  variant="secondary"
                  size="lg"
                  className="h-auto justify-start py-3 text-left"
                  disabled={!canAnalyze || listening}
                  onClick={() => shield.startClip({ url: clip.url, label: `${clip.title} sample` })}
                >
                  <Play />
                  <span className="leading-tight whitespace-normal">
                    <span className="block font-bold">{clip.title}</span>
                    <span className="block text-sm text-muted-foreground">{clip.description}</span>
                  </span>
                </Button>
              ))}
            </div>
            <input
              ref={fileInput}
              type="file"
              accept="audio/wav,audio/x-wav,.wav,audio/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void shield.startClip({ file, label: file.name });
                e.target.value = "";
              }}
            />
            <Button
              variant="outline"
              size="lg"
              disabled={!canAnalyze || listening}
              onClick={() => fileInput.current?.click()}
            >
              <FileAudio />
              Upload a WAV recording
            </Button>
          </Card>
        </div>
      </div>

      {band !== "red" && (
        <Card className="gap-4 p-5 sm:p-6">
          <div>
            <h2 className="text-2xl font-semibold">Not sure? Verify, any time.</h2>
            <p className="mt-1 text-base text-muted-foreground">
              Verification works even against a perfect voice clone, and even when detection is unavailable.
            </p>
          </div>
          <ProtectActions family={family} band={band} smoothed={shield.result?.smoothed} />
        </Card>
      )}
    </div>
  );
}
