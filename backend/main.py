"""Verity API — verify first, detect second, protect always.

Run:  uvicorn main:app --port 8000
"""

import io
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import numpy as np

import alerts
import audio_io
import config
import db
from detection import DetectionUnavailable, Detector
from detection.models import resolve_models
from forensics import get_analyzer
from forensics.live import HearsayLiveDetector
from scoring import SessionStore
from security import hash_secret, normalize_secret, verify_secret

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("verity")

_single = Detector(
    resolve_models(config.DETECTOR_MODELS),
    sample_rate=config.TARGET_SAMPLE_RATE,
    force_cpu=config.FORCE_CPU,
    disabled=config.DETECTOR_DISABLED,
)
detector = (
    HearsayLiveDetector(config.HEARSAY_LIVE_MODEL_PATH, _single, disabled=config.DETECTOR_DISABLED)
    if config.LIVE_DETECTOR == "hearsay"
    else _single
)
sessions = SessionStore()


def _warm_forensics() -> None:
    """Load the forensic models and JIT-compile the DSP code so the first real check is fast."""
    import numpy as np
    import soundfile as sf

    t = np.arange(3 * 16_000) / 16_000
    tone = (0.1 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.01 * np.random.randn(t.size))
    buf = io.BytesIO()
    sf.write(buf, tone.astype("float32"), 16_000, format="WAV")
    try:
        started = time.perf_counter()
        get_analyzer().analyze(buf.getvalue(), "warmup.wav", transcribe=config.FORENSICS_ASR_ENABLED)
        log.info("Forensic analyzer warm (%.0fs)", time.perf_counter() - started)
    except Exception as exc:
        log.warning("Forensic warm-up failed (the page will still try on demand): %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    detector.load_in_background()  # server is usable (verification) while the model loads
    if config.FORENSICS_WARMUP and not config.DETECTOR_DISABLED:
        threading.Thread(target=_warm_forensics, name="forensics-warmup", daemon=True).start()
    yield


app = FastAPI(title="Verity API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- schemas ------------------------------------------------------------------------


class AnalyzeResponse(BaseModel):
    synthetic_likelihood: float | None  # this request's raw score; None if only silence
    smoothed: float | None  # rolling average of the last few windows
    band: Literal["idle", "green", "amber", "red"]
    stability: float  # 0..1, how consistent recent windows are
    samples: int  # windows currently in the rolling history
    samples_seen: int  # windows scored this session
    history: list[float]
    silent: bool
    model: str | None


class ChallengeIn(BaseModel):
    id: int | None = None  # existing challenge id; lets you keep its answer on update
    question: str = Field(min_length=3, max_length=200)
    answer: str | None = Field(default=None, max_length=200)


class FamilyIn(BaseModel):
    id: int | None = None
    name: str = Field(min_length=1, max_length=80)
    safe_word: str | None = Field(default=None, max_length=100)
    trusted_phone: str = Field(min_length=3, max_length=40)
    alert_contact: str = Field(min_length=2, max_length=120)
    challenges: list[ChallengeIn] = Field(min_length=2, max_length=3)

    @field_validator("name", "trusted_phone", "alert_contact")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class ChallengeOut(BaseModel):
    index: int
    id: int
    question: str


class FamilyOut(BaseModel):
    id: int
    name: str
    trusted_phone: str
    alert_contact: str
    safe_word_configured: bool
    challenges: list[ChallengeOut]
    created_at: str
    updated_at: str


class VerifyIn(BaseModel):
    family_id: int
    method: Literal["challenge", "safe_word"] = "challenge"
    challenge_index: int = 0
    answer: str = Field(max_length=200)


class VerifyOut(BaseModel):
    passed: bool
    method: str
    locked: bool
    attempts_remaining: int
    locked_until: str | None
    message: str


class AlertIn(BaseModel):
    family_id: int | None = None
    reason: str = Field(default="manual", max_length=60)
    synthetic_likelihood: float | None = Field(default=None, ge=0, le=1)
    band: str | None = None
    notes: str | None = Field(default=None, max_length=500)


class AlertOut(BaseModel):
    event_id: int
    delivery: Literal["simulated", "webhook", "webhook_failed"]
    alert_contact: str | None
    message: str
    timestamp: str


class EventIn(BaseModel):
    family_id: int
    event_type: Literal["callback_started"]
    notes: str | None = Field(default=None, max_length=500)


class EventOut(BaseModel):
    id: int
    family_id: int | None
    timestamp: str
    event_type: str
    synthetic_likelihood: float | None
    band: str | None
    verification_result: str | None
    notes: str | None


def _family_out(conn, row) -> FamilyOut:
    challenges = db.get_challenges(conn, row["id"])
    return FamilyOut(
        id=row["id"],
        name=row["name"],
        trusted_phone=row["trusted_phone"],
        alert_contact=row["alert_contact"],
        safe_word_configured=bool(row["safe_word_hash"]),
        challenges=[ChallengeOut(index=i, id=c["id"], question=c["question"]) for i, c in enumerate(challenges)],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _require_family(conn, family_id: int):
    row = db.get_family(conn, family_id)
    if row is None:
        raise HTTPException(404, "Family not found. Set up your family shield first.")
    return row


# ---- health -------------------------------------------------------------------------


@app.get("/health")
def health():
    status = detector.status
    return {
        "status": "ok",
        "model": status.model_id,
        "device": status.device,
        "detection_available": detector.available,
        "detection_state": status.state,  # loading | ready | unavailable | disabled
        "detection_error": status.error,
        "thresholds": {"green_max": config.BAND_GREEN_MAX, "red_min": config.BAND_RED_MIN},
        "window_seconds": config.WINDOW_SECONDS,
        "sample_rate": config.TARGET_SAMPLE_RATE,
        "webhook_configured": bool(config.ALERT_WEBHOOK_URL),
        "forensics": {
            "detectors": config.FORENSICS_DETECTORS,
            "trained_model": getattr(get_analyzer().bundle, "version", None),
            "transcription": config.FORENSICS_ASR_ENABLED,
        },
    }


# ---- detection ----------------------------------------------------------------------


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    session_id: str = Form(..., min_length=1, max_length=100),
    family_id: int | None = Form(None),
):
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file is too large (25 MB max).")
    try:
        audio = audio_io.load_mono(data, config.TARGET_SAMPLE_RATE)
    except audio_io.AudioDecodeError as exc:
        raise HTTPException(400, str(exc))
    if audio.size < config.MIN_AUDIO_SECONDS * config.TARGET_SAMPLE_RATE:
        raise HTTPException(400, "Audio is too short to analyze.")

    if not detector.available:
        raise HTTPException(
            503,
            {
                "message": "Detection unavailable. Verification remains active.",
                "detection_state": detector.status.state,
            },
        )

    windows = audio_io.split_windows(
        audio, config.TARGET_SAMPLE_RATE, config.WINDOW_SECONDS, config.MIN_AUDIO_SECONDS
    )
    voiced = [w for w in windows if audio_io.rms(w) >= config.SILENCE_RMS]

    state = sessions.get(session_id)
    previous_band = state.band
    scores: list[float] = []
    try:
        for window in voiced:
            clip = window
            if getattr(detector, "wants_context", False):
                state.audio.append(window)
                clip = np.concatenate(list(state.audio))
            score = await run_in_threadpool(detector.analyze, clip)
            scores.append(score)
            state = sessions.push(session_id, score)
    except DetectionUnavailable as exc:
        raise HTTPException(503, {"message": "Detection unavailable. Verification remains active.", "detail": str(exc)})
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(503, {"message": "Detection hit an error. Verification remains active.", "detail": str(exc)})

    raw = round(sum(scores) / len(scores), 3) if scores else None
    smoothed = round(state.smoothed, 3) if state.smoothed is not None else None

    if state.band == "red" and previous_band != "red":
        with db.connect() as conn:
            fid = family_id if family_id is not None and db.get_family(conn, family_id) else None
            db.log_event(
                conn, fid, "suspicious_voice",
                synthetic_likelihood=smoothed, band="red",
                notes="Synthetic characteristics stayed elevated across recent audio samples.",
            )

    return AnalyzeResponse(
        synthetic_likelihood=raw,
        smoothed=smoothed,
        band=state.band,
        stability=state.stability,
        samples=len(state.history),
        samples_seen=state.samples_seen,
        history=[round(s, 3) for s in state.history],
        silent=not scores,
        model=detector.status.model_id,
    )


# ---- forensics ("Check a recording" / HEARSAY) ----------------------------------------


@app.post("/forensics")
async def forensics(
    file: UploadFile = File(...),
    family_id: int | None = Form(None),
    transcribe: bool = Form(True),
):
    """Any audio format -> multi-technique forensic report with a 0-100 synthetic likelihood."""
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file is too large (25 MB max).")
    if config.DETECTOR_DISABLED:
        raise HTTPException(503, {"message": "Detection unavailable. Verification remains active."})
    try:
        report = await run_in_threadpool(get_analyzer().analyze, data, file.filename or "", transcribe=transcribe)
    except audio_io.AudioDecodeError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.exception("Forensic analysis failed")
        raise HTTPException(503, {"message": "Forensic analysis hit an error. Verification remains active.", "detail": str(exc)})

    if report.get("status") == "ok" and family_id is not None:
        with db.connect() as conn:
            if db.get_family(conn, family_id):
                db.log_event(
                    conn, family_id, "recording_checked",
                    synthetic_likelihood=report["synthetic_likelihood"] / 100, band=report["band"],
                    notes=f"{file.filename or 'Recording'}: {report['manipulation_label']}.",
                )
    return report


# ---- family -------------------------------------------------------------------------


@app.post("/family", response_model=FamilyOut)
def upsert_family(body: FamilyIn):
    now = db.now_iso()
    with db.connect() as conn:
        existing = db.get_family(conn, body.id) if body.id is not None else None
        if body.id is not None and existing is None:
            raise HTTPException(404, "Family not found.")

        safe_word = (body.safe_word or "").strip()
        if safe_word and not normalize_secret(safe_word):
            raise HTTPException(422, "Safe-word must contain letters or numbers.")
        if existing is None and not safe_word:
            raise HTTPException(422, "A safe-word is required.")

        old_challenges = {c["id"]: c for c in db.get_challenges(conn, existing["id"])} if existing else {}
        new_challenges = []
        for item in body.challenges:
            answer = (item.answer or "").strip()
            if answer:
                if not normalize_secret(answer):
                    raise HTTPException(422, "Answers must contain letters or numbers.")
                new_challenges.append((item.question.strip(), hash_secret(answer)))
            elif item.id in old_challenges:
                new_challenges.append((item.question.strip(), old_challenges[item.id]["answer_hash"]))
            else:
                raise HTTPException(422, f'Please add an answer for "{item.question}".')

        if existing is None:
            cur = conn.execute(
                """INSERT INTO families (name, safe_word_hash, trusted_phone, alert_contact, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (body.name, hash_secret(safe_word), body.trusted_phone, body.alert_contact, now, now),
            )
            family_id = cur.lastrowid
        else:
            family_id = existing["id"]
            conn.execute(
                """UPDATE families SET name = ?, safe_word_hash = ?, trusted_phone = ?,
                   alert_contact = ?, updated_at = ? WHERE id = ?""",
                (
                    body.name,
                    hash_secret(safe_word) if safe_word else existing["safe_word_hash"],
                    body.trusted_phone,
                    body.alert_contact,
                    now,
                    family_id,
                ),
            )
            conn.execute("DELETE FROM challenges WHERE family_id = ?", (family_id,))

        conn.executemany(
            "INSERT INTO challenges (family_id, question, answer_hash) VALUES (?, ?, ?)",
            [(family_id, q, h) for q, h in new_challenges],
        )
        db.log_event(conn, family_id, "family_updated" if existing else "family_created")
        return _family_out(conn, db.get_family(conn, family_id))


@app.get("/family/current", response_model=FamilyOut)
def current_family():
    """Single-family mode: the most recently updated family."""
    with db.connect() as conn:
        row = db.latest_family(conn)
        if row is None:
            raise HTTPException(404, "No family set up yet.")
        return _family_out(conn, row)


@app.get("/family/{family_id}", response_model=FamilyOut)
def get_family(family_id: int):
    with db.connect() as conn:
        return _family_out(conn, _require_family(conn, family_id))


# ---- verification -------------------------------------------------------------------


@app.post("/verify", response_model=VerifyOut)
def verify(body: VerifyIn):
    now = time.time()
    with db.connect() as conn:
        family = _require_family(conn, body.family_id)

        failures = db.recent_failures(conn, family["id"], since=now - config.VERIFY_ATTEMPT_WINDOW_SECONDS)
        if len(failures) >= config.MAX_VERIFY_ATTEMPTS and now - failures[-1] < config.VERIFY_LOCK_SECONDS:
            locked_until = datetime.fromtimestamp(failures[-1] + config.VERIFY_LOCK_SECONDS, timezone.utc)
            return VerifyOut(
                passed=False, method=body.method, locked=True, attempts_remaining=0,
                locked_until=locked_until.isoformat(timespec="seconds"),
                message="Too many attempts. Hang up and call them back using your trusted number.",
            )

        if body.method == "safe_word":
            passed = verify_secret(body.answer, family["safe_word_hash"])
        else:
            challenges = db.get_challenges(conn, family["id"])
            if not 0 <= body.challenge_index < len(challenges):
                raise HTTPException(400, "That challenge question doesn't exist.")
            passed = verify_secret(body.answer, challenges[body.challenge_index]["answer_hash"])

        db.record_attempt(conn, family["id"], passed, now)
        label = "safe-word" if body.method == "safe_word" else "challenge question"

        if passed:
            db.log_event(conn, family["id"], "verification_passed", verification_result="passed",
                         notes=f"Caller answered the {label} correctly.")
            return VerifyOut(
                passed=True, method=body.method, locked=False,
                attempts_remaining=config.MAX_VERIFY_ATTEMPTS, locked_until=None,
                message="Verification succeeded.",
            )

        failures_now = len(failures) + 1
        remaining = max(0, config.MAX_VERIFY_ATTEMPTS - failures_now)
        locked = remaining == 0
        db.log_event(conn, family["id"], "verification_locked" if locked else "verification_failed",
                     verification_result="failed", notes=f"Caller could not answer the {label}.")
        locked_until = (
            datetime.fromtimestamp(now + config.VERIFY_LOCK_SECONDS, timezone.utc).isoformat(timespec="seconds")
            if locked else None
        )
        return VerifyOut(
            passed=False, method=body.method, locked=locked, attempts_remaining=remaining,
            locked_until=locked_until,
            message=(
                "Too many attempts. Hang up and call them back using your trusted number."
                if locked
                else "We couldn't verify this caller. Call them back using your trusted number."
            ),
        )


# ---- alerts & events ----------------------------------------------------------------


def _alert_note(family, delivery: str) -> str:
    who = family["alert_contact"] if family else "your family"
    return {
        "simulated": f"Alert for {who} recorded (demo mode, no message sent).",
        "webhook": f"Alert sent to {who}.",
        "webhook_failed": f"Alert for {who} recorded, but the message channel couldn't be reached.",
    }[delivery]


@app.post("/alert", response_model=AlertOut)
def alert(body: AlertIn):
    with db.connect() as conn:
        family = db.get_family(conn, body.family_id) if body.family_id is not None else None
        family_name = family["name"] if family else "your family"
        message = (
            f"Verity alert for {family_name}: a possibly suspicious call is happening right now. "
            f"Please check in using a number you trust."
        )
        delivery = alerts.send_alert(
            message,
            {
                "family": family_name,
                "alert_contact": family["alert_contact"] if family else None,
                "reason": body.reason,
                "band": body.band,
            },
        )
        event = db.log_event(
            conn, family["id"] if family else None, "alert_sent",
            synthetic_likelihood=body.synthetic_likelihood, band=body.band,
            notes=body.notes or _alert_note(family, delivery),
        )
        return AlertOut(
            event_id=event["id"],
            delivery=delivery,
            alert_contact=family["alert_contact"] if family else None,
            message=message,
            timestamp=event["timestamp"],
        )


@app.post("/events", response_model=EventOut)
def add_event(body: EventIn):
    with db.connect() as conn:
        _require_family(conn, body.family_id)
        return EventOut(**dict(db.log_event(conn, body.family_id, body.event_type, notes=body.notes)))


@app.get("/events/{family_id}", response_model=list[EventOut])
def events(family_id: int, limit: int = 100):
    with db.connect() as conn:
        _require_family(conn, family_id)
        return [EventOut(**dict(r)) for r in db.list_events(conn, family_id, min(max(limit, 1), 500))]
