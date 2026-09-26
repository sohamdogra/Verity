/** Per-browser accessibility preferences (text size, contrast). Applied to <html> as data attributes. */

export type TextSize = "md" | "lg" | "xl";

const TEXT_KEY = "verity.text";
const CONTRAST_KEY = "verity.contrast";

/** Runs inline in <head> before first paint so the page never flashes at the wrong size. */
export const A11Y_BOOT_SCRIPT = `try{var d=document.documentElement,t=localStorage.getItem("${TEXT_KEY}");if(t&&t!=="md")d.dataset.text=t;if(localStorage.getItem("${CONTRAST_KEY}")==="high")d.dataset.contrast="high"}catch(e){}`;

export function readPrefs(): { text: TextSize; highContrast: boolean } {
  const d = document.documentElement;
  return { text: (d.dataset.text as TextSize) || "md", highContrast: d.dataset.contrast === "high" };
}

export function setTextSize(size: TextSize) {
  const d = document.documentElement;
  if (size === "md") delete d.dataset.text;
  else d.dataset.text = size;
  try {
    localStorage.setItem(TEXT_KEY, size);
  } catch {}
}

export function setHighContrast(on: boolean) {
  const d = document.documentElement;
  if (on) d.dataset.contrast = "high";
  else delete d.dataset.contrast;
  try {
    localStorage.setItem(CONTRAST_KEY, on ? "high" : "normal");
  } catch {}
}
