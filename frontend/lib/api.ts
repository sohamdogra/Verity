export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type Band = "idle" | "green" | "amber" | "red";

export interface AnalyzeResult {
  synthetic_likelihood: number | null;
  smoothed: number | null;
  band: Band;
  stability: number;
  samples: number;
  samples_seen: number;
  history: number[];
  silent: boolean;
  model: string | null;
}

export interface Health {
  status: string;
  model: string | null;
  device: string;
  detection_available: boolean;
  detection_state: "loading" | "ready" | "unavailable" | "disabled";
  detection_error: string | null;
  thresholds: { green_max: number; red_min: number };
  window_seconds: number;
  sample_rate: number;
  webhook_configured: boolean;
}

export interface Challenge {
  index: number;
  id: number;
  question: string;
}

export interface Family {
  id: number;
  name: string;
  trusted_phone: string;
  alert_contact: string;
  safe_word_configured: boolean;
  challenges: Challenge[];
  created_at: string;
  updated_at: string;
}

export interface FamilyInput {
  id?: number;
  name: string;
  safe_word?: string;
  trusted_phone: string;
  alert_contact: string;
  challenges: { id?: number; question: string; answer?: string }[];
}

export interface VerifyResult {
  passed: boolean;
  method: "challenge" | "safe_word";
  locked: boolean;
  attempts_remaining: number;
  locked_until: string | null;
  message: string;
}

export interface AlertResult {
  event_id: number;
  delivery: "simulated" | "webhook" | "webhook_failed";
  alert_contact: string | null;
  message: string;
  timestamp: string;
}

export interface VerityEvent {
  id: number;
  family_id: number | null;
  timestamp: string;
  event_type: string;
  synthetic_likelihood: number | null;
  band: string | null;
  verification_result: string | null;
  notes: string | null;
}

export interface ForensicTechnique {
  id: string;
  kind: "neural_detector" | "signal";
  name: string;
  score: number | null;
  finding: string;
  windows?: number[];
}

export interface ForensicReport {
  file: string;
  status: "ok" | "too_short" | "no_speech" | "unavailable";
  synthetic_likelihood: number | null; // 0-100
  prediction?: "synthetic" | "bonafide";
  band?: Exclude<Band, "idle">;
  manipulation_type?: string;
  manipulation_label?: string;
  type_confidence?: number | null;
  calibrated?: boolean;
  fusion?: string;
  summary: string;
  techniques?: ForensicTechnique[];
  transcript?: { text: string; cues: { category: string; phrase: string }[]; scam_language: number } | null;
  steps: string[];
  metadata: {
    duration_s: number;
    sample_rate: number;
    channels: number;
    container: string | null;
    codec: string | null;
    bit_rate: number | null;
    lossy: boolean;
  };
  elapsed_seconds?: number;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
  /** Backend unreachable (status 0) or detection unavailable (503). */
  get detectionUnavailable() {
    return this.status === 503;
  }
}

function messageFrom(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const first = detail[0] as { msg?: string; loc?: unknown[] } | undefined;
    if (first?.msg) {
      const field = first.loc?.[first.loc.length - 1];
      return typeof field === "string" ? `${field.replace(/_/g, " ")}: ${first.msg}` : first.msg;
    }
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    return String((detail as { message: unknown }).message);
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError("Can't reach the Verity server. Is the backend running on port 8000?", 0);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new ApiError(messageFrom(body?.detail, `Request failed (${res.status})`), res.status);
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),

  analyze: (wav: Blob, sessionId: string, familyId?: number | null) => {
    const form = new FormData();
    form.append("file", wav, "window.wav");
    form.append("session_id", sessionId);
    if (familyId != null) form.append("family_id", String(familyId));
    return request<AnalyzeResult>("/analyze", { method: "POST", body: form });
  },

  forensics: (file: Blob, filename: string, familyId?: number | null) => {
    const form = new FormData();
    form.append("file", file, filename);
    if (familyId != null) form.append("family_id", String(familyId));
    return request<ForensicReport>("/forensics", { method: "POST", body: form });
  },

  saveFamily: (input: FamilyInput) => request<Family>("/family", json(input)),
  getFamily: (id: number) => request<Family>(`/family/${id}`),
  currentFamily: () => request<Family>("/family/current"),

  verify: (body: { family_id: number; method: "challenge" | "safe_word"; challenge_index?: number; answer: string }) =>
    request<VerifyResult>("/verify", json(body)),

  alert: (body: {
    family_id?: number | null;
    reason?: string;
    synthetic_likelihood?: number | null;
    band?: string | null;
    notes?: string;
  }) => request<AlertResult>("/alert", json(body)),

  logEvent: (body: { family_id: number; event_type: "callback_started"; notes?: string }) =>
    request<VerityEvent>("/events", json(body)),

  events: (familyId: number) => request<VerityEvent[]>(`/events/${familyId}`),
};
