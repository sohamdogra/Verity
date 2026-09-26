"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Contrast, PhoneCall, Type } from "lucide-react";
import { readPrefs, setHighContrast, setTextSize, type TextSize } from "@/lib/a11y";
import { cn } from "@/lib/utils";

const SIZES: { id: TextSize; label: string; sample: string }[] = [
  { id: "md", label: "Standard", sample: "text-base" },
  { id: "lg", label: "Large", sample: "text-lg" },
  { id: "xl", label: "Extra large", sample: "text-xl" },
];

export function AccessibilityMenu() {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState<TextSize>("md");
  const [contrast, setContrast] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const prefs = readPrefs();
    // eslint-disable-next-line react-hooks/set-state-in-effect -- sync with prefs applied before hydration
    setText(prefs.text);
    setContrast(prefs.highContrast);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label="Display settings"
        className="flex items-center gap-1.5 rounded-xl px-3 py-2 text-base text-muted-foreground ring-1 ring-border hover:bg-muted hover:text-foreground"
      >
        <Type className="size-5" />
        <span className="hidden sm:inline">Display</span>
      </button>
      {open && (
        <div className="absolute top-full right-0 z-50 mt-2 w-72 space-y-4 rounded-2xl bg-card p-4 shadow-xl ring-1 ring-border">
          <div>
            <p className="mb-2 text-sm font-bold">Text size</p>
            <div className="grid grid-cols-3 gap-2">
              {SIZES.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => {
                    setTextSize(s.id);
                    setText(s.id);
                  }}
                  aria-pressed={text === s.id}
                  className={cn(
                    "flex flex-col items-center gap-0.5 rounded-xl px-2 py-2 ring-1",
                    text === s.id ? "bg-primary text-primary-foreground ring-primary" : "ring-border hover:bg-muted",
                  )}
                >
                  <span className={cn("font-bold", s.sample)}>Aa</span>
                  <span className="text-xs">{s.label}</span>
                </button>
              ))}
            </div>
          </div>
          <button
            type="button"
            onClick={() => {
              setHighContrast(!contrast);
              setContrast(!contrast);
            }}
            aria-pressed={contrast}
            className="flex w-full items-center justify-between rounded-xl px-3 py-2.5 ring-1 ring-border hover:bg-muted"
          >
            <span className="flex items-center gap-2 text-base">
              <Contrast className="size-5" /> High contrast
            </span>
            <span className={cn("h-6 w-11 rounded-full p-0.5 transition-colors", contrast ? "bg-primary" : "bg-input")}>
              <span className={cn("block size-5 rounded-full bg-card transition-transform", contrast && "translate-x-5")} />
            </span>
          </button>
          <Link
            href="/call"
            onClick={() => setOpen(false)}
            className="flex items-center gap-2 rounded-xl bg-primary/10 px-3 py-2.5 text-base font-bold text-primary hover:bg-primary/15"
          >
            <PhoneCall className="size-5" /> Simple call mode
          </Link>
        </div>
      )}
    </div>
  );
}
