import { AlertTriangle, Ear, ShieldCheck, ShieldQuestion } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Band } from "@/lib/api";
import { BAND_COPY, BAND_STYLES } from "@/lib/signal";

const ICONS = { idle: Ear, green: ShieldCheck, amber: ShieldQuestion, red: AlertTriangle };

const SUBTITLE: Record<Band, string> = {
  idle: "Start listening, or play a sample recording.",
  green: "Few synthetic characteristics so far. Still verify any request for money.",
  amber: "Some synthetic characteristics. Consider verifying the caller.",
  red: "Synthetic characteristics stayed elevated. Verify before doing anything.",
};

export function BandStatus({ band, listening }: { band: Band; listening: boolean }) {
  const Icon = ICONS[band];
  const style = BAND_STYLES[band];
  return (
    <div className={cn("flex items-center gap-4 rounded-2xl p-4 ring-1 transition-colors duration-500", style.bg, style.ring)}>
      <span className={cn("relative grid size-16 shrink-0 place-items-center rounded-full bg-card", style.text)}>
        {listening && <span className={cn("absolute inset-0 animate-ping rounded-full opacity-20", style.dot)} />}
        <Icon className="size-8" />
      </span>
      <div>
        <p className={cn("font-heading text-3xl font-semibold", style.text)}>
          {band === "idle" && listening ? "Listening…" : BAND_COPY[band].label}
        </p>
        <p className="mt-1 text-base text-foreground/80">
          {band === "idle" && listening ? "Gathering the first audio sample." : SUBTITLE[band]}
        </p>
      </div>
    </div>
  );
}
