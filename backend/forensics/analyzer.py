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
from . import metadata as container_forensics
from .decode import DecodedAudio, decode_any
from .fusion import default_fusion, heuristic_type, load_bundle, type_label
from .neural import EXTRA_EMBEDDERS, DetectorEnsemble, Embedder, windows
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


SR = 16_000


def speech_segments(y: np.ndarray) -> list[dict]:
    """Where someone is speaking (10 ms frames, energy relative to the clip), merged into regions."""
    db = dsp._frame_db(y)
    threshold = max(np.percentile(db, 95) - 30, np.percentile(db, 10) + 8)
    voiced = db > threshold
    segments, start, gap = [], None, 0
    for i, v in enumerate(voiced):
        if v:
            start = i if start is None else start
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 25:  # a pause longer than 250 ms ends the region
                segments.append((start, i - gap))
                start, gap = None, 0
    if start is not None:
        segments.append((start, len(voiced) - 1 - gap))
    return [{"start": round(s * 0.01, 2), "end": round((e + 1) * 0.01, 2)} for s, e in segments if e - s >= 20]


class ForensicAnalyzer:
    def __init__(self):
        self.ensemble = DetectorEnsemble(config.FORENSICS_DETECTORS, force_cpu=config.FORCE_CPU)
        self.embedder = Embedder(config.FORENSICS_EMBEDDING_MODEL, force_cpu=config.FORCE_CPU)
        self.extra_embedders = [
            Embedder(mid, EXTRA_EMBEDDERS[mid][0], force_cpu=config.FORCE_CPU, prefix=EXTRA_EMBEDDERS[mid][1], drift=False)
            for mid in config.FORENSICS_EXTRA_EMBEDDINGS if mid in EXTRA_EMBEDDERS
        ]
        self.transcriber = Transcriber(config.FORENSICS_ASR_MODEL, force_cpu=config.FORCE_CPU)
        self.model_path = Path(config.HEARSAY_MODEL_PATH)
        self.fast_path = Path(config.HEARSAY_LIVE_MODEL_PATH)
        self._bundle = self._fast = None
        self._bundle_lock = threading.Lock()

    def extra_features(self, y: np.ndarray, prefixes: set[str] | None = None) -> dict:
        """Features from the extra SSL front-ends (optionally only those whose prefix is wanted)."""
        feats: dict = {}
        for emb in self.extra_embedders:
            if prefixes is None or emb.prefix in prefixes:
                feats.update(emb.embed(y))
        return feats

    @property
    def bundle(self):
        with self._bundle_lock:
            self._bundle = load_bundle(self.model_path, self._bundle)
            return self._bundle

    @property
    def fast_bundle(self):
        """The fast model (no detector ensemble), used as stage 1 of the cascade."""
        with self._bundle_lock:
            self._fast = load_bundle(self.fast_path, self._fast)
            return self._fast

    # ---- feature extraction (shared by the API and the training CLI) ------------------

    @staticmethod
    def voiced_windows(y: np.ndarray, steps: list[str] | None = None) -> list[np.ndarray]:
        clips = windows(y, config.FORENSICS_WINDOW_SECONDS, config.FORENSICS_HOP_SECONDS)
        voiced = [c for c in clips if rms(c) >= config.SILENCE_RMS] or clips
        if len(voiced) > config.FORENSICS_MAX_WINDOWS:  # long files: sample evenly
            idx = np.linspace(0, len(voiced) - 1, config.FORENSICS_MAX_WINDOWS).round().astype(int)
            voiced = [voiced[i] for i in idx]
            if steps is not None:
                steps.append(f"Long recording: sampled {len(voiced)} segments evenly.")
        return voiced

    def run_detectors(self, voiced: list[np.ndarray], steps: list[str]) -> tuple[dict, list[dict]]:
        t = time.perf_counter()
        det_feats, det_techniques = self.ensemble.run(voiced)
        ok = sum(1 for d in det_techniques if d["score"] is not None)
        steps.append(f"Ran {ok} neural detectors over {len(voiced)} segment(s) ({time.perf_counter() - t:.1f}s).")
        return det_feats, det_techniques

    def extract(self, decoded: DecodedAudio, *, embeddings: bool, detectors: bool = True) -> tuple[dict, list[dict], list[str]]:
        steps: list[str] = []
        y = decoded.audio
        voiced = self.voiced_windows(y, steps)

        t = time.perf_counter()
        feats, techniques = dsp.analyze(decoded)
        steps.append(f"Ran {len(techniques)} signal-processing checks ({time.perf_counter() - t:.1f}s).")

        if detectors:
            det_feats, det_techniques = self.run_detectors(voiced, steps)
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

    def timeline(self, decoded: DecodedAudio, techniques: list[dict]) -> tuple[list[dict], str | None]:
        """Synthetic-likelihood over time, for the evidence view."""
        y, duration = decoded.audio, decoded.audio.size / SR
        fast = self.fast_bundle
        if fast is not None and not fast.data.get("uses_detectors", True) and not fast.extra_prefixes:
            win = 2.0
            hop = max(1.0, (duration - win) / 23) if duration > win else win
            starts = np.arange(0.0, max(duration - win, 0.0) + 1e-6, hop)
            out = []
            for s in starts:
                seg = y[int(s * SR): int((s + win) * SR)]
                item = {"start": round(float(s), 2), "end": round(min(float(s) + win, duration), 2), "score": None}
                if seg.size >= 0.75 * SR and rms(seg) >= config.SILENCE_RMS:
                    d = DecodedAudio(audio=seg, native=seg, native_rate=SR, channels=1, duration=seg.size / SR)
                    feats, _, _ = self.extract(d, embeddings=fast.uses_embeddings, detectors=False)
                    item["score"] = round(float(fast.predict(feats)[0]), 3)
                out.append(item)
            return out, "trained fast model on 2-second windows"
        detectors = [t for t in techniques if t["kind"] == "neural_detector" and t.get("windows")]
        if detectors:
            scores = np.mean([t["windows"] for t in detectors if len(t["windows"]) == len(detectors[0]["windows"])], axis=0)
            win, hop = config.FORENSICS_WINDOW_SECONDS, config.FORENSICS_HOP_SECONDS
            return [{"start": round(i * hop, 2), "end": round(min(i * hop + win, duration), 2), "score": round(float(v), 3)}
                    for i, v in enumerate(scores)], "mean of the pretrained detectors on 4-second windows"
        return [], None

    # ---- full report ------------------------------------------------------------------

    def analyze(self, data: bytes, filename: str = "", *, transcribe: bool = True, path: Path | None = None) -> dict:
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
        fast = self.fast_bundle
        confident_fast = None
        if (config.FORENSICS_CASCADE and bundle is not None and fast is not None
                and not fast.data.get("uses_detectors", True) and bundle.data.get("uses_detectors", True)):
            # Stage 1: cheap evidence only (signal forensics + WavLM embeddings).
            feats, techniques, more = self.extract(decoded, embeddings=True, detectors=False)
            steps += more
            p_fast = fast.predict(feats)[0]
            if p_fast <= config.CASCADE_LOW or p_fast >= config.CASCADE_HIGH:
                confident_fast = p_fast
                steps.append(f"Stage 1 was already confident ({p_fast:.2f}), so the 5-detector ensemble was skipped.")
            else:
                steps.append(f"Stage 1 was uncertain ({p_fast:.2f}); escalating to the full detector ensemble.")
                det_feats, det_techniques = self.run_detectors(self.voiced_windows(decoded.audio), steps)
                feats.update(det_feats)
                techniques = det_techniques + techniques
        else:
            feats, techniques, more = self.extract(decoded, embeddings=bool(bundle and bundle.uses_embeddings))
            steps += more
            wanted = bundle.extra_prefixes if bundle is not None else set()
            if wanted:
                t = time.perf_counter()
                feats.update(self.extra_features(decoded.audio, wanted))
                steps.append(f"Extracted extra SSL embeddings ({', '.join(sorted(wanted))}, {time.perf_counter() - t:.1f}s).")
        meta_technique, container = container_forensics.inspect(data, filename, decoded.container, decoded.codec, path)
        techniques.append(meta_technique)
        steps.append("Inspected container structure, encoder tags" + (" and file timestamps." if path else "."))

        type_conf = None
        if confident_fast is not None:
            p, threshold = confident_fast, fast.threshold
            fusion = f"fast model, confident at stage 1 ({fast.version})"
            kind = heuristic_type(p, techniques, threshold)
        elif bundle is not None:
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

        t = time.perf_counter()
        try:
            timeline, timeline_source = self.timeline(decoded, techniques)
            steps.append(f"Scored the clip over time for the evidence view ({time.perf_counter() - t:.1f}s).")
        except Exception as exc:
            log.warning("Timeline failed: %s", exc)
            timeline, timeline_source = [], None

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
            "container": container,
            "timeline": timeline,
            "timeline_source": timeline_source,
            "speech_segments": speech_segments(decoded.audio),
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
