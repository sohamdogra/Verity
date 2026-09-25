"use client";

import { useEffect, useState } from "react";
import { api, type Health } from "@/lib/api";

/** Polls /health until detection is settled (ready / unavailable / disabled). */
export function useHealth() {
  const [health, setHealth] = useState<Health | null>(null);
  const [reachable, setReachable] = useState(true);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const h = await api.health();
        if (cancelled) return;
        setHealth(h);
        setReachable(true);
        if (h.detection_state === "loading") timer = setTimeout(poll, 2000);
      } catch {
        if (cancelled) return;
        setReachable(false);
        timer = setTimeout(poll, 4000);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  return { health, reachable };
}
