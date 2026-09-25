"""Speech-to-text (Whisper) + scam-language cues. This is about *what* is said, not whether
the voice is synthetic, so it is reported separately and never changes the likelihood."""

import re
import threading

import numpy as np

from detection.registry import device_name

SR = 16_000

SCAM_CUES: dict[str, list[str]] = {
    "urgency": [r"right now", r"immediately", r"\bhurry\b", r"\burgent", r"as soon as possible", r"\btoday\b", r"before it'?s too late"],
    "money": [r"\bmoney\b", r"\bwire\b", r"\btransfer", r"gift ?cards?", r"bitcoin|crypto", r"\bbail\b", r"\bcash\b",
              r"venmo|zelle|cash ?app|western union|paypal", r"\bpay\b|\bpayment", r"\$\s?\d|\d+ dollars", r"bank account"],
    "secrecy": [r"don'?t tell", r"keep (this|it) (between|secret|quiet)", r"\bsecret\b", r"don'?t call", r"nobody can know"],
    "distress": [r"\baccident\b", r"arrested|\bjail\b|police|lawyer|attorney", r"hospital", r"in trouble", r"kidnap", r"\bhelp me\b"],
    "authority": [r"\birs\b|social security|medicare", r"\bofficer\b|detective|federal|warrant", r"your account (has been|was|is)"],
}


def find_cues(text: str) -> list[dict]:
    lowered = text.lower()
    found = []
    for category, patterns in SCAM_CUES.items():
        for pattern in patterns:
            m = re.search(pattern, lowered)
            if m:
                found.append({"category": category, "phrase": text[m.start() : m.end()]})
                break  # one cue per category is enough
    return found


class Transcriber:
    def __init__(self, model_id: str = "openai/whisper-base.en", force_cpu: bool = False):
        self.model_id, self.force_cpu = model_id, force_cpu
        self._pipe = None
        self._lock = threading.Lock()

    def transcribe(self, y: np.ndarray) -> dict:
        with self._lock:
            if self._pipe is None:
                from transformers import pipeline

                self._pipe = pipeline("automatic-speech-recognition", model=self.model_id,
                                      device=0 if device_name(self.force_cpu) == "cuda" else -1)
            out = self._pipe({"raw": y.astype(np.float32), "sampling_rate": SR}, chunk_length_s=30)
        text = (out.get("text") or "").strip()
        cues = find_cues(text)
        return {
            "text": text,
            "cues": cues,
            "scam_language": round(min(1.0, len(cues) / 3), 2),  # 3+ categories = strong script match
            "model": self.model_id,
        }
