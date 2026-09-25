/** Single-family mode: remember which family this browser belongs to. */

const KEY = "verity.familyId";

export function getStoredFamilyId(): number | null {
  try {
    const value = window.localStorage.getItem(KEY);
    return value ? Number(value) || null : null;
  } catch {
    return null;
  }
}

export function setStoredFamilyId(id: number | null) {
  try {
    if (id == null) window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, String(id));
  } catch {
    // Storage can be blocked (private mode); the backend's /family/current is the fallback.
  }
}
