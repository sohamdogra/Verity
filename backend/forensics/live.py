"""Live Shield backed by the trained HEARSAY 'live' model (WavLM embeddings + signal features).

Drop-in for detection.Detector (same load/status/available/analyze). If the live model file
is missing or broken it falls back to the single-model detector, so the app never loses
detection because of a missing training step."""

import logging
import threading
from pathlib import Path

import numpy as np

from detection.detector import DetectionUnavailable, Detector, DetectorStatus

from .decode import DecodedAudio
from .fusion import load_bundle

log = logging.getLogger("verity.live")
SR = 16_000


class HearsayLiveDetector:
    def __init__(self, model_path: Path, fallback: Detector, disabled: bool = False):
        self.model_path = model_path
        self.fallback = fallback
        self._bundle = None
        self._using_fallback = False
        self._status = DetectorStatus("disabled" if disabled else "loading", None, "cpu")
        self._lock = threading.Lock()

    def load_in_background(self) -> None:
        if self._status.state == "disabled":
            log.warning("Detection disabled via DETECTOR_DISABLED; verification stays active.")
            return
        threading.Thread(target=self.load, name="live-detector-load", daemon=True).start()

    def load(self) -> None:
        bundle = load_bundle(self.model_path, None)
        if bundle is None:
            log.warning("No live HEARSAY model at %s; falling back to the single-model detector. "
                        "Train one with: python scripts/hearsay.py train --feature-set live --out %s",
                        self.model_path, self.model_path)
            self._use_fallback()
            return
        try:
            from detection.registry import device_name

            self._bundle = bundle
            self._status = DetectorStatus("ready", f"hearsay-live ({bundle.version})", device_name())
            self.analyze((0.05 * np.random.default_rng(0).standard_normal(SR * 3)).astype(np.float32))  # warm-up
            log.info("Live detection ready: trained HEARSAY model %s", bundle.version)
        except Exception as exc:
            log.warning("Live HEARSAY model failed (%s); falling back to single-model detector.", exc)
            self._bundle = None
            self._use_fallback()

    def _use_fallback(self) -> None:
        self._using_fallback = True
        self.fallback.load()

    @property
    def wants_context(self) -> bool:
        """Score recent context (several windows) rather than one window."""
        return not self._using_fallback

    @property
    def status(self) -> DetectorStatus:
        return self.fallback.status if self._using_fallback else self._status

    @property
    def available(self) -> bool:
        return self.fallback.available if self._using_fallback else (self._status.state == "ready" and self._bundle is not None)

    def analyze(self, audio: np.ndarray) -> float:
        if self._using_fallback:
            return self.fallback.analyze(audio)
        bundle = self._bundle
        if bundle is None:
            raise DetectionUnavailable(self._status.error or f"Detector is {self._status.state}")
        from .analyzer import get_analyzer

        decoded = DecodedAudio(audio=audio, native=audio, native_rate=SR, channels=1, duration=audio.size / SR)
        with self._lock:
            feats, _, _ = get_analyzer().extract(decoded, embeddings=bundle.uses_embeddings,
                                                 detectors=bundle.data.get("uses_detectors", True))
        p, _, _, _ = bundle.predict(feats)
        return p
