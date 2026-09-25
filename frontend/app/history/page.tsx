"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  AlertTriangle,
  BellRing,
  CheckCircle2,
  FileSearch,
  Loader2,
  Lock,
  Phone,
  ShieldCheck,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { ServerUnreachable } from "@/components/notices";
import { buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useFamily } from "@/hooks/use-family";
import { api, type VerityEvent } from "@/lib/api";
import { cn } from "@/lib/utils";

const EVENT_COPY: Record<string, { title: string; icon: LucideIcon; tone: string }> = {
  suspicious_voice: { title: "Suspicious voice detected", icon: AlertTriangle, tone: "bg-concern-soft text-concern" },
  verification_passed: { title: "Verification passed", icon: CheckCircle2, tone: "bg-safe-soft text-safe" },
  verification_failed: { title: "Verification failed", icon: XCircle, tone: "bg-caution-soft text-caution" },
  verification_locked: { title: "Verification locked after too many attempts", icon: Lock, tone: "bg-concern-soft text-concern" },
  alert_sent: { title: "Family alert sent", icon: BellRing, tone: "bg-caution-soft text-caution" },
  recording_checked: { title: "Recording checked", icon: FileSearch, tone: "bg-primary/10 text-primary" },
  callback_started: { title: "Called back on the trusted number", icon: Phone, tone: "bg-safe-soft text-safe" },
  family_created: { title: "Family shield turned on", icon: ShieldCheck, tone: "bg-primary/10 text-primary" },
  family_updated: { title: "Family shield updated", icon: ShieldCheck, tone: "bg-primary/10 text-primary" },
};

function dayLabel(date: Date) {
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);
  if (date.toDateString() === today.toDateString()) return "Today";
  if (date.toDateString() === yesterday.toDateString()) return "Yesterday";
  return date.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
}

function groupByDay(events: VerityEvent[]) {
  const groups: { label: string; events: VerityEvent[] }[] = [];
  for (const event of events) {
    const label = dayLabel(new Date(event.timestamp));
    const last = groups[groups.length - 1];
    if (last?.label === label) last.events.push(event);
    else groups.push({ label, events: [event] });
  }
  return groups;
}

export default function HistoryPage() {
  const { family, loading, error: loadError } = useFamily();
  const [events, setEvents] = useState<VerityEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!family) return;
    api.events(family.id).then(setEvents, (e) => setError(e instanceof Error ? e.message : "Could not load history."));
  }, [family]);

  if (loading) return <Loader2 className="mx-auto mt-16 size-10 animate-spin text-primary" />;
  if (loadError) return <ServerUnreachable />;
  if (!family) {
    return (
      <Card className="mx-auto max-w-xl gap-4 p-8 text-center">
        <h1 className="text-3xl font-semibold">No history yet</h1>
        <p className="text-lg text-muted-foreground">Set up your family shield to start keeping a protection history.</p>
        <Link href="/setup" className={buttonVariants({ size: "xl" })}>
          Go to family setup
        </Link>
      </Card>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <section>
        <h1 className="text-4xl font-semibold sm:text-5xl">Protection history</h1>
        <p className="mt-3 text-lg text-foreground/80">Everything Verity noticed and every step your family took.</p>
      </section>

      {error && <p className="text-base text-destructive">{error}</p>}
      {!events && !error && <Loader2 className="mx-auto size-8 animate-spin text-primary" />}
      {events?.length === 0 && <p className="text-lg text-muted-foreground">Nothing yet. That&apos;s a good thing.</p>}

      {events &&
        groupByDay(events).map((group) => (
          <section key={group.label}>
            <h2 className="mb-3 text-2xl font-semibold">{group.label}</h2>
            <ol className="relative space-y-3 border-l-2 border-border pl-6">
              {group.events.map((event) => {
                const copy = EVENT_COPY[event.event_type] ?? {
                  title: event.event_type.replace(/_/g, " "),
                  icon: ShieldCheck,
                  tone: "bg-muted text-muted-foreground",
                };
                const Icon = copy.icon;
                return (
                  <li key={event.id} className="relative">
                    <span
                      className={cn(
                        "absolute top-3 -left-[2.4rem] grid size-8 place-items-center rounded-full ring-4 ring-background",
                        copy.tone,
                      )}
                    >
                      <Icon className="size-4.5" />
                    </span>
                    <Card className="gap-1 p-4">
                      <div className="flex flex-wrap items-baseline justify-between gap-2">
                        <p className="text-lg font-bold">{copy.title}</p>
                        <time className="text-sm text-muted-foreground" dateTime={event.timestamp}>
                          {new Date(event.timestamp).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}
                        </time>
                      </div>
                      {event.notes && <p className="text-base text-foreground/80">{event.notes}</p>}
                    </Card>
                  </li>
                );
              })}
            </ol>
          </section>
        ))}
    </div>
  );
}
