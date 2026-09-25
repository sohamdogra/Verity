"""Model-agnostic detector. The app only ever calls `detector.analyze(audio)`."""

import logging
import threading
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .labels import synthetic_likelihood, validate_labels
from .models import ModelSpec

log = logging.getLogger("verity.detection")

DetectorState = Literal["loading", "ready", "unavailable", "disabled"]


class DetectionUnavailable(RuntimeError):
    pass


@dataclass
class DetectorStatus:
    state: DetectorState
    model_id: str | None
    device: str
    error: str | None = None


class Detector:
    def __init__(
        self,
        specs: list[ModelSpec],
        sample_rate: int = 16_000,
        force_cpu: bool = False,
        disabled: bool = False,
    ):
        self._specs = specs
        self._sample_rate = sample_rate
        self._force_cpu = force_cpu
        self._pipe = None
        self._spec: ModelSpec | None = None
        self._infer_lock = threading.Lock()
        self._status = DetectorStatus(
            state="disabled" if disabled else "loading", model_id=None, device="cpu"
        )

    # ---- lifecycle -------------------------------------------------------------

    def load_in_background(self) -> None:
        if self._status.state == "disabled":
            log.warning("Detection disabled via DETECTOR_DISABLED; verification stays active.")
            return
        threading.Thread(target=self.load, name="detector-load", daemon=True).start()

    def load(self) -> None:
        try:
            from .registry import device_name as resolve_device, load_pipeline
        except Exception as exc:  # torch/transformers missing or broken
            self._fail(f"ML libraries unavailable: {exc}")
            return

        try:
            device_name = resolve_device(self._force_cpu)
        except Exception as exc:
            self._fail(f"ML libraries unavailable: {exc}")
            return
        self._status.device = device_name
        errors = []
        for spec in self._specs:
            try:
                log.info("Loading detection model %s on %s ...", spec.model_id, device_name)
                pipe = load_pipeline(spec.model_id, self._force_cpu)
                validate_labels(pipe.model.config.id2label.values(), spec.label_overrides)
                self._pipe, self._spec = pipe, spec
                self._status = DetectorStatus("ready", spec.model_id, device_name)
                self.analyze(np.zeros(self._sample_rate, dtype=np.float32))  # warm-up
                log.info("Detection ready: %s (%s)", spec.model_id, device_name)
                return
            except Exception as exc:
                self._pipe = self._spec = None
                errors.append(f"{spec.model_id}: {exc}")
                log.warning("Could not load %s: %s", spec.model_id, exc)
        self._fail("; ".join(errors) or "No models configured")

    def _fail(self, error: str) -> None:
        self._status = DetectorStatus("unavailable", None, self._status.device, error)
        log.error("Detection unavailable (verification remains active): %s", error)

    # ---- public API ------------------------------------------------------------

    @property
    def status(self) -> DetectorStatus:
        return self._status

    @property
    def available(self) -> bool:
        return self._status.state == "ready" and self._pipe is not None

    def analyze(self, audio: np.ndarray) -> float:
        """Mono float32 audio at the detector's sample rate -> synthetic likelihood (0..1)."""
        if not self.available or self._spec is None:
            raise DetectionUnavailable(self._status.error or f"Detector is {self._status.state}")
        num_labels = len(self._pipe.model.config.id2label)
        with self._infer_lock:
            predictions = self._pipe(
                {"raw": audio.astype(np.float32), "sampling_rate": self._sample_rate},
                top_k=num_labels,
            )
        return synthetic_likelihood(predictions, self._spec.label_overrides)
