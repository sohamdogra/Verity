"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { BadgeCheck, BellRing, Loader2, Phone } from "lucide-react";
import { api, type Band, type Family } from "@/lib/api";
import { cn } from "@/lib/utils";
import { buttonVariants } from "@/components/ui/button";

export function TrustedCallLink({ family, className, big }: { family: Family; className?: string; big?: boolean }) {
  return (
    <a
      href={`tel:${family.trusted_phone.replace(/[^\d+]/g, "")}`}
      onClick={() => void api.logEvent({ family_id: family.id, event_type: "callback_started" }).catch(() => {})}
      className={cn(buttonVariants({ variant: "outline", size: big ? "xl" : "lg" }), "h-auto min-h-13 justify-start py-2", className)}
    >
      <Phone />
      <span className="text-left leading-tight">
        <span className="block">Call trusted number</span>
        <span className="block text-base font-bold tracking-wide">{family.trusted_phone}</span>
      </span>
    </a>
  );
}

/** One-tap Verify / Call back / Alert family. */
export function ProtectActions({
  family,
  band,
  smoothed,
  big,
  stacked,
}: {
  family: Family | null;
  band?: Band;
  smoothed?: number | null;
  big?: boolean;
  stacked?: boolean;
}) {
  const router = useRouter();
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const size = big ? "xl" : "lg";

  const alertFamily = async () => {
    setSending(true);
    setError(null);
    try {
      const res = await api.alert({
        family_id: family?.id,
        reason: band === "red" ? "high_concern_signal" : "manual",
        band: band && band !== "idle" ? band : null,
        synthetic_likelihood: smoothed ?? null,
      });
      router.push(`/alert?event=${res.event_id}&delivery=${res.delivery}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not send the alert.");
      setSending(false);
    }
  };

  return (
    <div className="space-y-2">
      <div className={cn("grid gap-3", !stacked && "sm:grid-cols-3")}>
        <Link href="/verify" className={cn(buttonVariants({ size }), "min-h-13")}>
          <BadgeCheck />
          Verify caller
        </Link>
        {family ? (
          <TrustedCallLink family={family} big={big} />
        ) : (
          <Link href="/setup" className={cn(buttonVariants({ variant: "outline", size }), "min-h-13")}>
            <Phone />
            Add trusted number
          </Link>
        )}
        <button
          type="button"
          onClick={alertFamily}
          disabled={sending}
          className={cn(buttonVariants({ variant: "outline", size }), "min-h-13")}
        >
          {sending ? <Loader2 className="animate-spin" /> : <BellRing />}
          Alert family
        </button>
      </div>
      {error && <p className="text-base text-destructive">{error}</p>}
    </div>
  );
}
