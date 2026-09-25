"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BadgeCheck, Clock, FileSearch, ShieldCheck, Users } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Live Shield", icon: ShieldCheck },
  { href: "/verify", label: "Verify caller", icon: BadgeCheck },
  { href: "/check", label: "Check a recording", icon: FileSearch },
  { href: "/history", label: "History", icon: Clock },
  { href: "/setup", label: "Family setup", icon: Users },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="border-b border-border/70 bg-card/70 backdrop-blur">
      <div className="mx-auto flex w-full max-w-5xl flex-wrap items-center justify-between gap-x-6 gap-y-3 px-4 py-4 sm:px-6">
        <Link href="/" className="flex items-center gap-3">
          <span className="grid size-11 place-items-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
            <ShieldCheck className="size-6" />
          </span>
          <span className="leading-tight">
            <span className="block font-heading text-2xl font-semibold">Verity</span>
            <span className="block text-sm text-muted-foreground">Is this real?</span>
          </span>
        </Link>
        <nav className="flex flex-wrap gap-1">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-2 rounded-xl px-3 py-2 text-base transition-colors",
                  active ? "bg-primary/10 font-bold text-primary" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                <Icon className="size-5" />
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
