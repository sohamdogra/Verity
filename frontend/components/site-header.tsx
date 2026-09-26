"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BadgeCheck, Clock, FileSearch, ShieldCheck, Users } from "lucide-react";
import { AccessibilityMenu } from "@/components/accessibility-menu";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Live Shield", short: "Shield", icon: ShieldCheck },
  { href: "/verify", label: "Verify caller", short: "Verify", icon: BadgeCheck },
  { href: "/check", label: "Check a recording", short: "Check", icon: FileSearch },
  { href: "/history", label: "History", short: "History", icon: Clock },
  { href: "/setup", label: "Family setup", short: "Family", icon: Users },
];

function isActive(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <>
      <header className="sticky top-0 z-40 border-b border-border/70 bg-card/85 backdrop-blur">
        <div className="mx-auto flex w-full max-w-5xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <Link href="/" className="flex items-center gap-3">
            <span className="grid size-11 place-items-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
              <ShieldCheck className="size-6" />
            </span>
            <span className="leading-tight">
              <span className="block font-heading text-2xl font-semibold">Verity</span>
              <span className="block text-sm whitespace-nowrap text-muted-foreground">Is this real?</span>
            </span>
          </Link>
          <nav className="desktop-nav hidden items-center gap-1 lg:flex" aria-label="Main">
            {NAV.map(({ href, label, short, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                aria-current={isActive(pathname, href) ? "page" : undefined}
                className={cn(
                  "flex items-center gap-2 rounded-xl px-3 py-2 text-base whitespace-nowrap transition-colors",
                  isActive(pathname, href)
                    ? "bg-primary/10 font-bold text-primary"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                <Icon className="size-5 shrink-0" />
                <span className="nav-short xl:hidden">{short}</span>
                <span className="nav-full hidden xl:inline">{label}</span>
              </Link>
            ))}
          </nav>
          <AccessibilityMenu />
        </div>
      </header>

      {/* Phones and tablets: a big, always-visible tab bar at the bottom. */}
      <nav
        aria-label="Main"
        className="tab-bar fixed inset-x-0 bottom-0 z-40 border-t border-border bg-card/95 pb-[env(safe-area-inset-bottom)] backdrop-blur lg:hidden"
      >
        <div className="mx-auto grid max-w-xl grid-cols-5">
          {NAV.map(({ href, short, icon: Icon }) => {
            const active = isActive(pathname, href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "flex flex-col items-center gap-0.5 py-2.5 text-xs font-bold transition-colors",
                  active ? "text-primary" : "text-muted-foreground",
                )}
              >
                <span className={cn("grid h-8 w-12 place-items-center rounded-full", active && "bg-primary/12")}>
                  <Icon className="size-6" />
                </span>
                {short}
              </Link>
            );
          })}
        </div>
      </nav>
    </>
  );
}
