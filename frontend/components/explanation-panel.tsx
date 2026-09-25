import { Info } from "lucide-react";
import type { AnalyzeResult } from "@/lib/api";
import type { Reading } from "@/hooks/use-shield";
import { bandFor, stabilityWord, type Thresholds } from "@/lib/signal";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function reasons(result: AnalyzeResult | null, readings: Reading[], t: Thresholds): string[] {
  if (!result || result.band === "idle") {
    return [
      "Verity listens to short samples of audio (about 2½ seconds each) for characteristics common in AI-generated voices.",
      "It looks at several samples together, so one odd moment doesn't raise an alarm.",
    ];
  }
  const recent = readings.slice(-result.samples);
  const elevated = recent.filter((r) => bandFor(r.score, t) !== "green").length;
  const steadiness = stabilityWord(result.stability, result.samples);
  const sampleWord = (n: number) => `${n} sample${n === 1 ? "" : "s"}`;

  if (result.band === "red") {
    return [
      `Synthetic characteristics remained elevated across ${elevated > 1 ? "several" : "recent"} audio samples (${elevated} of the last ${sampleWord(recent.length)}).`,
      `The readings were ${steadiness}, which makes this signal more meaningful.`,
      "Scammers often add urgency, secrecy (\"don't tell anyone\") and requests for money, gift cards or wire transfers.",
    ];
  }
  if (result.band === "amber") {
    return [
      `Some synthetic characteristics appeared (${elevated} of the last ${sampleWord(recent.length)}).`,
      "This can also happen with a bad connection, speakerphone echo or background noise.",
      "If the caller asks for money or secrecy, verify them before doing anything.",
    ];
  }
  return [
    `Recent samples showed few synthetic characteristics, and readings were ${steadiness}.`,
    "That's reassuring, but not proof: a good voice clone can still sound like someone you love.",
    "If anything feels off, ask for your safe-word.",
  ];
}

export function ExplanationPanel({
  result,
  readings,
  thresholds,
}: {
  result: AnalyzeResult | null;
  readings: Reading[];
  thresholds: Thresholds;
}) {
  const concerned = result?.band === "red" || result?.band === "amber";
  return (
    <Card className="gap-3 p-5">
      <CardHeader className="px-0">
        <CardTitle className="flex items-center gap-2 text-xl">
          <Info className="size-5 text-primary" />
          {concerned ? "Why are we concerned?" : "How this works"}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 px-0 text-base leading-relaxed">
        <ul className="list-disc space-y-2 pl-5">
          {reasons(result, readings, thresholds).map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
        <p className="rounded-xl bg-muted px-3 py-2 text-sm text-muted-foreground">
          This is a probabilistic signal, not proof. A low signal never means a caller is safe, and a high
          signal never means a voice is definitely fake. Verification is what protects you.
        </p>
      </CardContent>
    </Card>
  );
}
