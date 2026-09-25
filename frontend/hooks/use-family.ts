"use client";

import { useCallback, useEffect, useState } from "react";
import { api, ApiError, type Family } from "@/lib/api";
import { getStoredFamilyId, setStoredFamilyId } from "@/lib/family-store";

/** Loads this browser's family (single-family mode), falling back to the latest one on the server. */
export function useFamily() {
  const [family, setFamily] = useState<Family | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const storedId = getStoredFamilyId();
      let found: Family | null = null;
      if (storedId != null) {
        found = await api.getFamily(storedId).catch((e) => {
          if (e instanceof ApiError && e.status === 404) return null;
          throw e;
        });
      }
      if (!found) {
        found = await api.currentFamily().catch((e) => {
          if (e instanceof ApiError && e.status === 404) return null;
          throw e;
        });
      }
      setStoredFamilyId(found?.id ?? null);
      setFamily(found);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load your family shield.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial fetch on mount
    void reload();
  }, [reload]);

  const save = useCallback((next: Family) => {
    setStoredFamilyId(next.id);
    setFamily(next);
  }, []);

  return { family, loading, error, reload, save };
}
