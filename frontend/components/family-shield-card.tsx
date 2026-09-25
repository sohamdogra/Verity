import Link from "next/link";
import { CheckCircle2, Circle, ShieldCheck } from "lucide-react";
import type { Family } from "@/lib/api";
import { cn } from "@/lib/utils";
import { buttonVariants } from "@/components/ui/button";

function Item({ done, children }: { done: boolean; children: React.ReactNode }) {
  return (
    <li className="flex items-center gap-3 text-lg">
      {done ? <CheckCircle2 className="size-6 text-safe" /> : <Circle className="size-6 text-muted-foreground" />}
      <span className={done ? "" : "text-muted-foreground"}>{children}</span>
    </li>
  );
}

export function FamilyShieldCard({ family }: { family: Family | null }) {
  const protectedNow = !!family && family.safe_word_configured && family.challenges.length >= 2;
  return (
    <div
      className={cn(
        "rounded-3xl p-6 ring-1",
        protectedNow ? "bg-safe-soft ring-safe/30" : "bg-card ring-border",
      )}
    >
      <div className="flex items-center gap-3">
        <span
          className={cn(
            "grid size-12 place-items-center rounded-2xl",
            protectedNow ? "bg-safe text-white" : "bg-muted text-muted-foreground",
          )}
        >
          <ShieldCheck className="size-7" />
        </span>
        <div>
          <p className="text-sm tracking-wide text-muted-foreground uppercase">Family Shield</p>
          <p className="font-heading text-2xl font-semibold">{family ? family.name : "Not set up yet"}</p>
        </div>
      </div>
      <ul className="mt-5 space-y-2.5">
        <Item done={protectedNow}>Protected</Item>
        <Item done={!!family?.safe_word_configured}>Safe-word configured</Item>
        <Item done={(family?.challenges.length ?? 0) >= 2}>
          Challenges configured{family ? ` (${family.challenges.length})` : ""}
        </Item>
        <Item done={!!family?.trusted_phone}>
          Trusted callback configured
          {family?.trusted_phone ? <span className="font-bold"> · {family.trusted_phone}</span> : null}
        </Item>
        <Item done={!!family?.alert_contact}>
          Family alert contact
          {family?.alert_contact ? <span className="font-bold"> · {family.alert_contact}</span> : null}
        </Item>
      </ul>
      {!family && (
        <Link href="/setup" className={cn(buttonVariants({ size: "lg" }), "mt-5 w-full")}>
          Set up your family shield
        </Link>
      )}
    </div>
  );
}
