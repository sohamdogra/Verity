"""All tunable settings live here. Thresholds are served to the frontend via /health,
so this file is the single source of truth for the signal bands."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --- Signal bands (on the smoothed synthetic likelihood, 0.0 human .. 1.0 synthetic) ---
BAND_GREEN_MAX = float(os.getenv("BAND_GREEN_MAX", "0.40"))  # below this: low signal
BAND_RED_MIN = float(os.getenv("BAND_RED_MIN", "0.65"))  # above this: high concern

# --- Audio windowing ---
TARGET_SAMPLE_RATE = 16_000
WINDOW_SECONDS = 2.5
MIN_AUDIO_SECONDS = 0.5  # shorter clips are rejected
HISTORY_WINDOWS = 3  # rolling history used for smoothing
MIN_SAMPLES_FOR_RED = 2  # one odd window can reach amber, never red on its own
SILENCE_RMS = float(os.getenv("SILENCE_RMS", "0.004"))  # quieter windows are not scored
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# --- Verification rate limiting ---
MAX_VERIFY_ATTEMPTS = 5  # failed attempts before a temporary lock
VERIFY_LOCK_SECONDS = int(os.getenv("VERIFY_LOCK_SECONDS", "300"))
VERIFY_ATTEMPT_WINDOW_SECONDS = 15 * 60  # failures older than this are forgotten

# --- Detection ---
# Comma-separated HuggingFace model ids, tried in order. Empty = built-in fallback list.
DETECTOR_MODELS = [m.strip() for m in os.getenv("DETECTOR_MODELS", "").split(",") if m.strip()]
DETECTOR_DISABLED = _env_bool("DETECTOR_DISABLED")  # simulate "detection unavailable"
FORCE_CPU = _env_bool("FORCE_CPU")

# --- Forensics ("Check a recording" + HEARSAY challenge CLI) ---
FORENSICS_DETECTORS = [
    m.strip()
    for m in os.getenv(
        "FORENSICS_DETECTORS",
        "Bisher/wav2vec2_ASV_deepfake_audio_detection,MelodyMachine/Deepfake-audio-detection-V2,"
        "mo-thecreator/Deepfake-audio-detection,Hemgg/Deepfake-audio-detection,"
        "Om-Parab/distilhubert-finetuned-audio-deepfake-in-the-wild",
    ).split(",")
    if m.strip()
]
FORENSICS_EMBEDDING_MODEL = os.getenv("FORENSICS_EMBEDDING_MODEL", "microsoft/wavlm-base-plus")
# Extra SSL front-ends (comma-separated; see forensics/neural.py EXTRA_EMBEDDERS). Empty = none.
FORENSICS_EXTRA_EMBEDDINGS = [
    m.strip() for m in os.getenv("FORENSICS_EXTRA_EMBEDDINGS", "facebook/wav2vec2-xls-r-300m").split(",") if m.strip()
]
FORENSICS_ASR_MODEL = os.getenv("FORENSICS_ASR_MODEL", "openai/whisper-base.en")
FORENSICS_ASR_ENABLED = not _env_bool("FORENSICS_ASR_DISABLED")
# Live Shield engine: "hearsay" = trained live model (embeddings + signal features, falls back to
# "single" if the model file is missing); "single" = one pretrained detector (DETECTOR_MODELS).
LIVE_DETECTOR = os.getenv("LIVE_DETECTOR", "hearsay").strip().lower()
HEARSAY_LIVE_MODEL_PATH = Path(os.getenv("HEARSAY_LIVE_MODEL_PATH", str(BASE_DIR / "models" / "hearsay_live.joblib")))
LIVE_CONTEXT_WINDOWS = 3  # the live model scores the last ~7.5 s of audio, not one 2.5 s window
FORENSICS_WARMUP = not _env_bool("FORENSICS_WARMUP_DISABLED")  # preload at startup for a fast first check
FORENSICS_WINDOW_SECONDS = 4.0
FORENSICS_HOP_SECONDS = 3.0
FORENSICS_MAX_WINDOWS = 20
# Orchestration: when the fast model (embeddings + signal features) is already confident,
# skip the expensive 5-detector ensemble. Thresholds are on P(synthetic).
FORENSICS_CASCADE = not _env_bool("FORENSICS_CASCADE_DISABLED")
CASCADE_LOW, CASCADE_HIGH = 0.03, 0.97
HEARSAY_MODEL_PATH = Path(os.getenv("HEARSAY_MODEL_PATH", str(BASE_DIR / "models" / "hearsay.joblib")))

# --- Storage / integrations ---
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "verity.db")))
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").strip() or None
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]
