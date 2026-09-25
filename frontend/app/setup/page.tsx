"use client";

import { useEffect, useState } from "react";
import { Eye, EyeOff, Lightbulb, Loader2, Plus, Trash2 } from "lucide-react";
import { FamilyShieldCard } from "@/components/family-shield-card";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useFamily } from "@/hooks/use-family";
import { api, type Family } from "@/lib/api";

interface ChallengeDraft {
  id?: number;
  question: string;
  answer: string;
}

const SUGGESTED = [
  "What did we name the stray cat that visited every summer?",
  "Where did we eat after your graduation?",
  "What's the nickname only Grandma uses for you?",
];

function draftFrom(family: Family | null): {
  name: string;
  safeWord: string;
  trustedPhone: string;
  alertContact: string;
  challenges: ChallengeDraft[];
} {
  return {
    name: family?.name ?? "",
    safeWord: "",
    trustedPhone: family?.trusted_phone ?? "",
    alertContact: family?.alert_contact ?? "",
    challenges: family?.challenges.length
      ? family.challenges.map((c) => ({ id: c.id, question: c.question, answer: "" }))
      : [
          { question: SUGGESTED[0], answer: "" },
          { question: SUGGESTED[1], answer: "" },
        ],
  };
}

function Field({ id, label, hint, children }: { id: string; label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <Label htmlFor={id} className="text-lg font-bold">
        {label}
      </Label>
      {children}
      {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
    </div>
  );
}

export default function SetupPage() {
  const { family, loading, save } = useFamily();
  const [draft, setDraft] = useState(() => draftFrom(null));
  const [showSafeWord, setShowSafeWord] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- hydrate the form once the family loads
    if (!loading) setDraft(draftFrom(family));
  }, [loading, family]);

  const editing = !!family;
  const set = (patch: Partial<typeof draft>) => {
    setSaved(false);
    setDraft((d) => ({ ...d, ...patch }));
  };
  const setChallenge = (i: number, patch: Partial<ChallengeDraft>) =>
    set({ challenges: draft.challenges.map((c, j) => (j === i ? { ...c, ...patch } : c)) });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const next = await api.saveFamily({
        id: family?.id,
        name: draft.name,
        safe_word: draft.safeWord || undefined,
        trusted_phone: draft.trustedPhone,
        alert_contact: draft.alertContact,
        challenges: draft.challenges.map((c) => ({
          id: c.id,
          question: c.question,
          answer: c.answer || undefined,
        })),
      });
      save(next);
      setDraft(draftFrom(next));
      setSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-4xl font-semibold sm:text-5xl">Family setup</h1>
        <p className="mt-3 max-w-3xl text-lg text-foreground/80">
          Agree on these together, in person. A voice clone can copy how someone sounds, but not what only your
          family knows. Secrets are stored hashed and never shown again.
        </p>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Card className="p-5 sm:p-6">
          <form onSubmit={submit} className="space-y-6">
            <Field id="name" label="Family name">
              <Input
                id="name"
                required
                placeholder="The Dogra family"
                value={draft.name}
                onChange={(e) => set({ name: e.target.value })}
              />
            </Field>

            <Field
              id="safe-word"
              label="Safe-word"
              hint={
                editing
                  ? "Leave blank to keep your current safe-word."
                  : "Something memorable but never posted online, like \"blue pelican\". Capitals and punctuation don't matter."
              }
            >
              <div className="flex gap-2">
                <Input
                  id="safe-word"
                  type={showSafeWord ? "text" : "password"}
                  autoComplete="off"
                  required={!editing}
                  placeholder={editing ? "••••••• (unchanged)" : "blue pelican"}
                  value={draft.safeWord}
                  onChange={(e) => set({ safeWord: e.target.value })}
                />
                <Button
                  type="button"
                  variant="outline"
                  size="lg"
                  onClick={() => setShowSafeWord((s) => !s)}
                  aria-label={showSafeWord ? "Hide safe-word" : "Show safe-word"}
                >
                  {showSafeWord ? <EyeOff /> : <Eye />}
                </Button>
              </div>
            </Field>

            <Field
              id="trusted-phone"
              label="Trusted callback number"
              hint="The number you already know is really theirs. On a suspicious call, hang up and dial this."
            >
              <Input
                id="trusted-phone"
                type="tel"
                required
                placeholder="+1 404 555 0142"
                value={draft.trustedPhone}
                onChange={(e) => set({ trustedPhone: e.target.value })}
              />
            </Field>

            <Field id="alert-contact" label="Family alert contact" hint="Who should hear about a suspicious call?">
              <Input
                id="alert-contact"
                required
                placeholder="Maya (daughter) · +1 404 555 0199"
                value={draft.alertContact}
                onChange={(e) => set({ alertContact: e.target.value })}
              />
            </Field>

            <fieldset className="space-y-4">
              <legend className="text-lg font-bold">Challenge questions (2–3)</legend>
              <p className="flex gap-2 rounded-xl bg-caution-soft p-3 text-sm">
                <Lightbulb className="size-5 shrink-0 text-caution" />
                Pick private memories, not facts on social media (no pet names you&apos;ve posted, no mother&apos;s
                maiden name).
              </p>
              {draft.challenges.map((c, i) => (
                <div key={i} className="space-y-3 rounded-2xl bg-muted/60 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-base font-bold">Question {i + 1}</span>
                    {draft.challenges.length > 2 && (
                      <Button
                        type="button"
                        variant="ghost"
                        onClick={() => set({ challenges: draft.challenges.filter((_, j) => j !== i) })}
                      >
                        <Trash2 />
                        Remove
                      </Button>
                    )}
                  </div>
                  <Input
                    aria-label={`Question ${i + 1}`}
                    required
                    minLength={3}
                    value={c.question}
                    onChange={(e) => setChallenge(i, { question: e.target.value })}
                  />
                  <Input
                    aria-label={`Answer ${i + 1}`}
                    required={!c.id}
                    autoComplete="off"
                    placeholder={c.id ? "Answer (leave blank to keep)" : "Answer"}
                    value={c.answer}
                    onChange={(e) => setChallenge(i, { answer: e.target.value })}
                  />
                </div>
              ))}
              {draft.challenges.length < 3 && (
                <Button
                  type="button"
                  variant="outline"
                  size="lg"
                  onClick={() => set({ challenges: [...draft.challenges, { question: SUGGESTED[2], answer: "" }] })}
                >
                  <Plus />
                  Add a third question
                </Button>
              )}
            </fieldset>

            {error && <p className="text-base text-destructive">{error}</p>}
            {saved && <p className="text-base font-bold text-safe">Saved. Your family shield is up to date.</p>}
            <Button type="submit" size="xl" className="w-full" disabled={saving || loading}>
              {saving && <Loader2 className="animate-spin" />}
              {editing ? "Save changes" : "Turn on family shield"}
            </Button>
          </form>
        </Card>

        <div className="space-y-6 lg:sticky lg:top-6 lg:self-start">
          <FamilyShieldCard family={family} />
          <Card className="gap-2 p-5 text-base">
            <h2 className="text-xl font-semibold">The family plan</h2>
            <ol className="list-decimal space-y-1.5 pl-5">
              <li>Anyone asking for money urgently gets asked for the safe-word.</li>
              <li>If they can&apos;t answer, hang up and call the trusted number.</li>
              <li>Tell the family, even if it turns out fine.</li>
            </ol>
          </Card>
        </div>
      </div>
    </div>
  );
}
