"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { CheckCircle2, KeyRound, Loader2, Lock, MessageCircleQuestion, RefreshCw, XCircle } from "lucide-react";
import { ProtectActions, TrustedCallLink } from "@/components/protect-actions";
import { ServerUnreachable } from "@/components/notices";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useFamily } from "@/hooks/use-family";
import { api, type VerifyResult } from "@/lib/api";
import { cn } from "@/lib/utils";

type Method = "challenge" | "safe_word";

function randomIndex(count: number, not?: number) {
  if (count <= 1) return 0;
  let i = Math.floor(Math.random() * count);
  if (i === not) i = (i + 1) % count;
  return i;
}

export default function VerifyPage() {
  const { family, loading, error: loadError } = useFamily();
  const [method, setMethod] = useState<Method>("challenge");
  const [index, setIndex] = useState<number | null>(null);
  const [answer, setAnswer] = useState("");
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const count = family?.challenges.length ?? 0;
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- pick a random question once loaded
    if (count && index === null) setIndex(randomIndex(count));
  }, [count, index]);

  const reset = (next?: Partial<{ method: Method; index: number }>) => {
    if (next?.method) setMethod(next.method);
    if (next?.index !== undefined) setIndex(next.index);
    setAnswer("");
    setResult(null);
    setError(null);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!family) return;
    setChecking(true);
    setError(null);
    try {
      setResult(
        await api.verify({ family_id: family.id, method, challenge_index: index ?? 0, answer }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not check the answer.");
    } finally {
      setChecking(false);
    }
  };

  if (loading) {
    return <Loader2 className="mx-auto mt-16 size-10 animate-spin text-primary" />;
  }
  if (loadError) return <ServerUnreachable />;
  if (!family) {
    return (
      <Card className="mx-auto max-w-xl gap-4 p-8 text-center">
        <h1 className="text-3xl font-semibold">Set up your family shield first</h1>
        <p className="text-lg text-muted-foreground">
          Choose a safe-word and a few private questions together, so you can check who&apos;s really calling.
        </p>
        <Link href="/setup" className={cn(buttonVariants({ size: "xl" }))}>
          Go to family setup
        </Link>
      </Card>
    );
  }

  const question = index !== null ? family.challenges[index]?.question : null;

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <section>
        <h1 className="text-4xl font-semibold sm:text-5xl">Verify the caller</h1>
        <p className="mt-3 text-lg text-foreground/80">
          Ask something only your real family member would know. This works even against a perfect voice clone.
        </p>
      </section>

      <div className="grid grid-cols-2 gap-3" role="tablist">
        {(
          [
            { id: "challenge", label: "Ask a question", icon: MessageCircleQuestion },
            { id: "safe_word", label: "Ask for the safe-word", icon: KeyRound },
          ] as const
        ).map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            role="tab"
            aria-selected={method === id}
            onClick={() => reset({ method: id })}
            className={cn(
              "flex items-center justify-center gap-2 rounded-2xl px-4 py-4 text-lg ring-1 transition-colors",
              method === id ? "bg-primary font-bold text-primary-foreground ring-primary" : "bg-card ring-border hover:bg-muted",
            )}
          >
            <Icon className="size-6" />
            {label}
          </button>
        ))}
      </div>

      {!result ? (
        <Card className="gap-5 p-6">
          {method === "challenge" ? (
            <div>
              <p className="text-base text-muted-foreground">Ask the caller:</p>
              <p className="mt-1 font-heading text-3xl leading-snug font-semibold">&ldquo;{question}&rdquo;</p>
              {count > 1 && (
                <Button variant="ghost" className="mt-2 -ml-2" onClick={() => reset({ index: randomIndex(count, index ?? undefined) })}>
                  <RefreshCw />
                  Ask a different question
                </Button>
              )}
            </div>
          ) : (
            <div>
              <p className="text-base text-muted-foreground">Say to the caller:</p>
              <p className="mt-1 font-heading text-3xl leading-snug font-semibold">
                &ldquo;What&apos;s our family safe-word?&rdquo;
              </p>
              <p className="mt-2 text-base text-muted-foreground">Don&apos;t hint at it. A real family member won&apos;t need help.</p>
            </div>
          )}
          <form onSubmit={submit} className="space-y-4">
            <label htmlFor="answer" className="block text-lg font-bold">
              Type what they said
            </label>
            <Input
              id="answer"
              autoFocus
              autoComplete="off"
              required
              className="h-14 text-xl"
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
            />
            {error && <p className="text-base text-destructive">{error}</p>}
            <Button type="submit" size="xl" className="w-full" disabled={checking || !answer.trim()}>
              {checking && <Loader2 className="animate-spin" />}
              Check answer
            </Button>
          </form>
          <p className="text-sm text-muted-foreground">
            They can&apos;t answer? Don&apos;t argue. Just hang up and call back on your trusted number.
          </p>
        </Card>
      ) : result.passed ? (
        <Card className="gap-4 bg-safe-soft p-6 ring-safe/30">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="size-10 text-safe" />
            <p className="font-heading text-3xl font-semibold text-safe">Verification succeeded.</p>
          </div>
          <p className="text-lg">
            The caller answered correctly. If anything still feels off, you can always hang up and call them back on
            the number you trust.
          </p>
          <div className="flex flex-wrap gap-3">
            <TrustedCallLink family={family} />
            <Button variant="outline" size="lg" onClick={() => reset()}>
              Verify again
            </Button>
          </div>
        </Card>
      ) : (
        <Card className="gap-4 bg-concern-soft p-6 ring-concern/30">
          <div className="flex items-center gap-3">
            {result.locked ? <Lock className="size-10 text-concern" /> : <XCircle className="size-10 text-concern" />}
            <p className="font-heading text-3xl leading-tight font-semibold text-concern">
              {result.locked ? "Too many attempts." : "We couldn't verify this caller."}
            </p>
          </div>
          <p className="text-xl font-bold">Call them back using your trusted number.</p>
          <p className="text-base text-foreground/80">
            {result.locked
              ? "Verification is paused for a few minutes to stop guessing. Hang up now. A real family member will understand."
              : `A wrong answer doesn't prove it's a scam, but don't send money or share anything. ${result.attempts_remaining} attempt${result.attempts_remaining === 1 ? "" : "s"} left.`}
          </p>
          <ProtectActions family={family} />
          {!result.locked && (
            <Button variant="ghost" size="lg" className="self-start" onClick={() => reset({ index: randomIndex(count, index ?? undefined) })}>
              <RefreshCw />
              Try a different question
            </Button>
          )}
        </Card>
      )}
    </div>
  );
}

