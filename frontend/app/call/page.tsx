"use client";

import Link from "next/link";
import { AlertTriangle, Loader2, Mic, ShieldCheck, Square } from "lucide-react";
import { DetectionUnavailable, ServerUnreachable } from "@/components/notices";
import { ProtectActions } from "@/components/protect-actions";
import { useFamily } from "@/hooks/use-family";
import { useHealth } from "@/hooks/use-health";
import { useShield } from "@/hooks/use-shield";
import { BAND_COPY, BAND_STYLES } from "@/lib/signal";
import { cn } from "@/lib/utils";

const RING = { idle: "bg-primary", green: "bg-safe", amber: "bg-caution", red: "bg-concern" } as const;

/** Simplified screen for the moment a call is happening: one big button, one big answer. */
export default function CallModePage() {
  const { health, reachable } = useHealth();
  const { family } = useFamily();
  const shield = useShield({ familyId: family?.id ?? null, windowSeconds: health?.window_seconds ?? 2.5 });
  const band = shield.result?.band ?? "idle";
  const listening = shield.source === "mic" && (shield.status === "listening" || shield.status === "starting");
  const down = shield.status === "unavailable" || health?.detection_state === "unavailable" || health?.detection_state === "disabled";
  const ready = reachable && health?.detection_state === "ready";

  return (
    <div className="mx-auto flex max-w-lg flex-col items-center gap-6 text-center">
      <div>
        <h1 className="text-4xl font-semibold sm:text-5xl">On a call right now?</h1>
        <p className="mt-2 text-xl text-foreground/80">Put the call on speaker, then tap the big button.</p>
      </div>

      {!reachable && <ServerUnreachable />}
      {reachable && down && <DetectionUnavailable />}

      <button
        type="button"
        onClick={listening ? shield.stop : shield.startMic}
        disabled={!listening && !ready}
        className={cn(
          "relative grid size-60 place-items-center rounded-full text-white shadow-xl transition-colors duration-500 disabled:opacity-50 sm:size-64",
          RING[band],
        )}
        aria-label={listening ? "Stop listening" : "Start listening"}
      >
        {listening && <span className={cn("absolute inset-0 animate-ping rounded-full opacity-25", RING[band])} />}
        <span className="relative flex flex-col items-center gap-2">
          {shield.status === "starting" ? (
            <Loader2 className="size-16 animate-spin" />
          ) : listening ? (
            <Square className="size-14" />
          ) : (
            <Mic className="size-16" />
          )}
          <span className="text-2xl font-bold">{listening ? "Stop" : "Start listening"}</span>
        </span>
      </button>

      <div aria-live="polite" className="min-h-20">
        <p className={cn("font-heading text-4xl font-semibold", BAND_STYLES[band].text)}>
          {band === "idle" ? (listening ? "Listening…" : "Ready when you are") : BAND_COPY[band].label}
        </p>
        {band === "red" && (
          <p className="mt-2 flex items-center justify-center gap-2 text-2xl font-bold">
            <AlertTriangle className="size-7 text-concern" /> Hang up and call them back.
          </p>
        )}
        {band === "green" && (
          <p className="mt-2 flex items-center justify-center gap-2 text-lg text-foreground/80">
            <ShieldCheck className="size-6 text-safe" /> Still, if they ask for money, verify first.
          </p>
        )}
        {shield.status === "error" && shield.error && <p className="mt-2 text-lg text-destructive">{shield.error}</p>}
      </div>

      <div className="w-full">
        <ProtectActions family={family} band={band} smoothed={shield.result?.smoothed} big stacked />
      </div>

      <Link href="/" className="text-lg text-muted-foreground underline underline-offset-4">
        Exit call mode
      </Link>
    </div>
  );
}
