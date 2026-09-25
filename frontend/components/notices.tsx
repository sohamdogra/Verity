import { Loader2, PlugZap, ShieldCheck } from "lucide-react";

/** Shown whenever AI detection can't run. Verification never depends on it. */
export function DetectionUnavailable({ reason }: { reason?: string }) {
  return (
    <div className="flex gap-4 rounded-2xl bg-muted p-5 ring-1 ring-border">
      <ShieldCheck className="mt-1 size-7 shrink-0 text-primary" />
      <div>
        <p className="font-heading text-xl font-semibold">Detection unavailable. Verification remains active.</p>
        <p className="mt-1 text-base text-foreground/80">
          {reason ? `${reason} ` : ""}You&apos;re still protected: ask for your safe-word, ask a challenge question, or hang up
          and call back on your trusted number.
        </p>
      </div>
    </div>
  );
}

export function DetectionLoading() {
  return (
    <div className="flex items-center gap-3 rounded-2xl bg-muted px-5 py-4 text-base ring-1 ring-border">
      <Loader2 className="size-5 animate-spin text-primary" />
      Preparing the voice detector (the first run downloads the model). Verification already works.
    </div>
  );
}

export function ServerUnreachable() {
  return (
    <div className="flex gap-4 rounded-2xl bg-concern-soft p-5 ring-1 ring-concern/30">
      <PlugZap className="mt-1 size-7 shrink-0 text-concern" />
      <div>
        <p className="font-heading text-xl font-semibold">Can&apos;t reach the Verity server</p>
        <p className="mt-1 text-base text-foreground/80">
          Start the backend (see README). Even without this app, your family&apos;s safe-word and a call back on a
          number you trust still work.
        </p>
      </div>
    </div>
  );
}
