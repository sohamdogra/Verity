"""HEARSAY-style forensic orchestrator: decode anything, decide which techniques to run,
run them, fuse the evidence, and explain the result in plain language."""

import logging
import threading
import time
from pathlib import Path

import numpy as np

import config
from audio_io import rms

from . import dsp
from .decode import DecodedAudio, decode_any
from .fusion import default_fusion, heuristic_type, load_bundle, type_label
from .neural import DetectorEnsemble, Embedder, windows
from .transcript import Transcriber

log = logging.getLogger("verity.forensics")


def _band(p: float) -> str:
    if p > config.BAND_RED_MIN:
        return "red"
    if p >= config.BAND_GREEN_MAX:
        return "amber"
    return "green"


def _summary(p: float, kind: str, n_detectors: int) -> str:
    if p > config.BAND_RED_MIN:
        lead = "Strong synthetic characteristics"
    elif p >= config.BAND_GREEN_MAX:
        lead = "Some synthetic characteristics"
    else:
        lead = "Few synthetic characteristics"
    tail = f" Likely manipulation: {type_label(kind).lower()}." if kind != "none" else ""
    return f"{lead} across {n_detectors} detectors and signal checks.{tail} This is a probabilistic signal, not proof."


class ForensicAnalyzer:
    def __init__(self):
        self.ensemble = DetectorEnsemble(config.FORENSICS_DETECTORS, force_cpu=config.FORCE_CPU)
        self.embedder = Embedder(config.FORENSICS_EMBEDDING_MODEL, force_cpu=config.FORCE_CPU)
        self.transcriber = Transcriber(config.FORENSICS_ASR_MODEL, force_cpu=config.FORCE_CPU)
        self.model_path = Path(config.HEARSAY_MODEL_PATH)
        self._bundle = None
        self._bundle_lock = threading.Lock()

    @property
    def bundle(self):
        with self._bundle_lock:
            self._bundle = load_bundle(self.model_path, self._bundle)
            return self._bundle

    # ---- feature extraction (shared by the API and the training CLI) ------------------

    def extract(self, decoded: DecodedAudio, *, embeddings: bool, detectors: bool = True) -> tuple[dict, list[dict], list[str]]:
        steps: list[str] = []
        y = decoded.audio
        clips = windows(y, config.FORENSICS_WINDOW_SECONDS, config.FORENSICS_HOP_SECONDS)
        voiced = [c for c in clips if rms(c) >= config.SILENCE_RMS] or clips
        if len(voiced) > config.FORENSICS_MAX_WINDOWS:  # long files: sample evenly
            idx = np.linspace(0, len(voiced) - 1, config.FORENSICS_MAX_WINDOWS).round().astype(int)
            voiced = [voiced[i] for i in idx]
            steps.append(f"Long recording: sampled {len(voiced)} segments evenly.")

        t = time.perf_counter()
        feats, techniques = dsp.analyze(decoded)
        steps.append(f"Ran 5 signal-processing checks ({time.perf_counter() - t:.1f}s).")

        if detectors:
            t = time.perf_counter()
            det_feats, det_techniques = self.ensemble.run(voiced)
            ok = sum(1 for d in det_techniques if d["score"] is not None)
            steps.append(f"Ran {ok} neural detectors over {len(voiced)} segment(s) ({time.perf_counter() - t:.1f}s).")
            feats.update(det_feats)
            techniques = det_techniques + techniques

        if embeddings:
            t = time.perf_counter()
            try:
                feats.update(self.embedder.embed(y))
                steps.append(f"Extracted {config.FORENSICS_EMBEDDING_MODEL} speech embeddings ({time.perf_counter() - t:.1f}s).")
            except Exception as exc:
                log.warning("Embedding failed: %s", exc)
                steps.append("Speech embeddings unavailable; the trained model used the remaining evidence.")
        return feats, techniques, steps

    # ---- full report ------------------------------------------------------------------

    def analyze(self, data: bytes, filename: str = "", *, transcribe: bool = True) -> dict:
        started = time.perf_counter()
        decoded = decode_any(data, filename)
        channel_note = f", {decoded.channels} channels mixed to mono" if decoded.channels > 1 else ""
        steps = [
            f"Decoded {filename or 'audio'} with {decoded.decoder} ({decoded.codec or decoded.container}, "
            f"{decoded.native_rate / 1000:g} kHz, {decoded.duration:.1f}s{channel_note})."
        ]
        metadata = {
            "duration_s": round(decoded.duration, 2), "sample_rate": decoded.native_rate,
            "channels": decoded.channels, "container": decoded.container, "codec": decoded.codec,
            "bit_rate": decoded.bit_rate, "lossy": decoded.lossy,
        }
        base = {"file": filename, "metadata": metadata}

        if decoded.duration < config.MIN_AUDIO_SECONDS:
            return {**base, "status": "too_short", "synthetic_likelihood": None, "steps": steps,
                    "summary": "The recording is too short to analyze."}
        if rms(decoded.audio) < config.SILENCE_RMS:
            return {**base, "status": "no_speech", "synthetic_likelihood": None, "steps": steps,
                    "summary": "We couldn't find any speech in this recording."}

        bundle = self.bundle
        feats, techniques, more = self.extract(decoded, embeddings=bool(bundle and bundle.uses_embeddings))
        steps += more

        type_conf = None
        if bundle is not None:
            p, kind, type_conf, missing = bundle.predict(feats)
            fusion = f"trained model ({bundle.version})"
            threshold = bundle.threshold
            if kind is None or p < threshold:
                kind, type_conf = heuristic_type(p, techniques, threshold), None
            if missing:
                steps.append(f"{missing} trained features were unavailable and filled with defaults.")
        else:
            p = default_fusion(techniques)
            fusion, threshold = "default weighted ensemble (uncalibrated)", 0.5
            if p is None:
                return {**base, "status": "unavailable", "synthetic_likelihood": None, "techniques": techniques,
                        "steps": steps, "summary": "Detection unavailable. Verification remains active."}
            kind = heuristic_type(p, techniques, threshold)
        steps.append(f"Fused evidence with the {fusion}.")

        transcript = None
        if transcribe and config.FORENSICS_ASR_ENABLED:
            t = time.perf_counter()
            try:
                transcript = self.transcriber.transcribe(decoded.audio)
                steps.append(f"Transcribed speech and checked it for scam-script language ({time.perf_counter() - t:.1f}s).")
            except Exception as exc:
                log.warning("Transcription failed: %s", exc)
                steps.append("Transcription unavailable.")

        n_detectors = sum(1 for t in techniques if t["kind"] == "neural_detector" and t["score"] is not None)
        return {
            **base,
            "status": "ok",
            "synthetic_likelihood": round(p * 100, 1),
            "prediction": "synthetic" if p >= threshold else "bonafide",
            "band": _band(p),
            "manipulation_type": kind,
            "manipulation_label": type_label(kind),
            "type_confidence": None if type_conf is None else round(type_conf, 2),
            "calibrated": bundle is not None,
            "fusion": fusion,
            "summary": _summary(p, kind, n_detectors),
            "techniques": techniques,
            "transcript": transcript,
            "steps": steps,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }


_analyzer: ForensicAnalyzer | None = None
_lock = threading.Lock()


def get_analyzer() -> ForensicAnalyzer:
    global _analyzer
    with _lock:
        if _analyzer is None:
            _analyzer = ForensicAnalyzer()
        return _analyzer
