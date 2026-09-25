"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { BellRing, CheckCircle2, Loader2 } from "lucide-react";
import { TrustedCallLink } from "@/components/protect-actions";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useFamily } from "@/hooks/use-family";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const DELIVERY_NOTE: Record<string, string> = {
  simulated: "Demo mode: the alert was recorded in Verity. Add ALERT_WEBHOOK_URL to deliver it as a real message.",
  webhook: "Delivered to your family's alert channel.",
  webhook_failed: "We couldn't reach the alert channel, but the alert was recorded. Please call them directly.",
};

export function AlertConfirmation() {
  const params = useSearchParams();
  const router = useRouter();
  const { family, loading } = useFamily();
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const eventId = params.get("event");
  const delivery = params.get("delivery") ?? "simulated";

  const send = async () => {
    setSending(true);
    setError(null);
    try {
      const res = await api.alert({ family_id: family?.id, reason: "manual" });
      router.replace(`/alert?event=${res.event_id}&delivery=${res.delivery}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not send the alert.");
    } finally {
      setSending(false);
    }
  };

  if (loading) return <Loader2 className="mx-auto mt-16 size-10 animate-spin text-primary" />;

  if (!eventId) {
    return (
      <Card className="mx-auto max-w-2xl gap-5 p-8">
        <BellRing className="size-12 text-caution" />
        <h1 className="text-4xl font-semibold">Alert your family</h1>
        <p className="text-lg text-foreground/80">
          {family
            ? `We'll let ${family.alert_contact} know a possibly suspicious call is happening.`
            : "Set up a family alert contact first so we know who to tell."}
        </p>
        {error && <p className="text-base text-destructive">{error}</p>}
        {family ? (
          <Button size="xl" onClick={send} disabled={sending}>
            {sending ? <Loader2 className="animate-spin" /> : <BellRing />}
            Send alert now
          </Button>
        ) : (
          <Link href="/setup" className={buttonVariants({ size: "xl" })}>
            Go to family setup
          </Link>
        )}
      </Card>
    );
  }

  return (
    <Card className="mx-auto max-w-2xl gap-5 p-8">
      <div className="flex items-center gap-3">
        <CheckCircle2 className="size-12 text-safe" />
        <h1 className="text-4xl font-semibold">Alert sent</h1>
      </div>
      <p className="text-xl">
        {family ? (
          <>
            We let <span className="font-bold">{family.alert_contact}</span> know that a possibly suspicious call is
            happening.
          </>
        ) : (
          "The alert was recorded."
        )}
      </p>
      <p className="rounded-xl bg-muted px-4 py-3 text-base text-muted-foreground">
        {DELIVERY_NOTE[delivery] ?? DELIVERY_NOTE.simulated}
      </p>
      <div className="rounded-2xl bg-card p-5 ring-1 ring-border">
        <p className="text-lg font-bold">What to do now</p>
        <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-lg">
          <li>Hang up. You don&apos;t owe the caller anything.</li>
          <li>Call them back using a number you trust.</li>
          <li>Don&apos;t send money, gift cards or codes until you&apos;ve spoken to them directly.</li>
        </ol>
      </div>
      <div className="flex flex-wrap gap-3">
        {family && <TrustedCallLink family={family} />}
        <Link href="/history" className={cn(buttonVariants({ variant: "outline", size: "lg" }), "min-h-13")}>
          View history
        </Link>
        <Link href="/" className={cn(buttonVariants({ variant: "ghost", size: "lg" }), "min-h-13")}>
          Back to Live Shield
        </Link>
      </div>
    </Card>
  );
}
