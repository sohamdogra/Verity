"""Turn per-technique evidence into one synthetic likelihood (0-100) and a manipulation type.

* With a trained bundle (scripts/hearsay.py train) we use its classifier(s).
* Without one we use a transparent weighted average: neural detectors dominate, signal
  cues nudge. That default is uncalibrated and says so (`calibrated: false`)."""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("verity.forensics")

TYPE_LABELS = {
    "none": "No manipulation indicated",
    "synthetic_speech": "Fully synthetic speech (text-to-speech or voice clone)",
    "partial_synthetic": "Partially synthetic or spliced (real and generated audio mixed)",
}

# The trained type model predicts the generator it was trained on ("xtts_v2", "devset_freevc").
# Those names are meaningful in HEARSAY.md but meaningless to a worried family, so the app
# shows the attack family instead. Unknown names fall back to a tidied version of themselves.
GENERATOR_FAMILIES = {
    "voice clone": ("xtts", "your_tts", "openvoice", "elevenlabs", "playht", "unit_speech", "clone"),
    "voice conversion": ("freevc", "conversion", "_vc"),
    "spliced": ("splice", "partial"),
    "text-to-speech": ("tts", "diffgan", "grad", "pro_diff", "wavegrad", "sapi", "mms"),
}
FAMILY_LABELS = {
    "voice clone": "Voice clone (a real person's voice, synthesized)",
    "voice conversion": "Voice conversion (one speaker converted into another)",
    "spliced": "Partially synthetic (generated speech spliced into real audio)",
    "text-to-speech": "Fully synthetic speech (text-to-speech)",
}

# Relative trust in each detector for the untrained default (tuned on our dev clips; see HEARSAY.md).
DEFAULT_DETECTOR_WEIGHTS = {
    "Bisher/wav2vec2_ASV_deepfake_audio_detection": 3.0,
    "MelodyMachine/Deepfake-audio-detection-V2": 1.0,
    "mo-thecreator/Deepfake-audio-detection": 1.0,
    "Hemgg/Deepfake-audio-detection": 1.0,
    "Om-Parab/distilhubert-finetuned-audio-deepfake-in-the-wild": 1.0,
}
SIGNAL_WEIGHT = 0.15


def type_family(key: str) -> str | None:
    """Map a trained generator name to the attack family a person would recognise."""
    name = key.lower()
    for family, markers in GENERATOR_FAMILIES.items():
        if any(m in name for m in markers):
            return family
    return None


def type_label(key: str) -> str:
    if key in TYPE_LABELS:
        return TYPE_LABELS[key]
    family = type_family(key)
    if family:
        return FAMILY_LABELS[family]
    return key.replace("_", " ").strip().capitalize()


class TrainedBundle:
    def __init__(self, path: Path):
        import joblib

        self.path = path
        self.mtime = path.stat().st_mtime
        data = joblib.load(path)
        self.data = data
        self.feature_names: list[str] = data["feature_names"]
        self.binary = data["binary"]
        self.threshold: float = data.get("threshold", 0.5)
        self.type_model = data.get("type_model")
        self.uses_embeddings: bool = data.get("uses_embeddings", False)
        self.detectors: list[str] = data.get("detectors", [])
        self.version: str = data.get("version", path.stem)
        from .neural import EXTRA_EMBEDDERS

        known = {prefix for _, prefix in EXTRA_EMBEDDERS.values()}
        # Extra SSL front-ends this model was trained with (feature names like "xlsr_mean_0").
        self.extra_prefixes: set[str] = {n.split("_", 1)[0] for n in self.feature_names} & known

    def vector(self, feats: dict[str, float]) -> tuple[np.ndarray, int]:
        missing = sum(1 for n in self.feature_names if n not in feats)
        x = np.array([[feats.get(n, 0.0) for n in self.feature_names]], dtype=np.float64)
        return np.nan_to_num(x), missing

    def predict(self, feats: dict[str, float]) -> tuple[float, str | None, float | None, int]:
        x, missing = self.vector(feats)
        p = float(self.binary.predict_proba(x)[0, 1])
        kind = conf = None
        if self.type_model is not None:
            probs = self.type_model.predict_proba(x)[0]
            i = int(np.argmax(probs))
            kind, conf = str(self.type_model.classes_[i]), float(probs[i])
        return p, kind, conf, missing


def load_bundle(path: Path, current: "TrainedBundle | None") -> "TrainedBundle | None":
    if not path.exists():
        return None
    if current is not None and current.path == path and current.mtime == path.stat().st_mtime:
        return current
    try:
        bundle = TrainedBundle(path)
        log.info("Loaded trained HEARSAY model %s (%d features)", path.name, len(bundle.feature_names))
        return bundle
    except Exception as exc:
        log.warning("Could not load %s, using default fusion: %s", path, exc)
        return None


def default_fusion(techniques: list[dict]) -> float | None:
    num = den = 0.0
    for t in techniques:
        if t["kind"] == "neural_detector" and t["score"] is not None:
            w = DEFAULT_DETECTOR_WEIGHTS.get(t["id"].split(":", 1)[1], 1.0)
            num, den = num + w * t["score"], den + w
    neural = num / den if den else None
    signal_scores = [t["score"] for t in techniques if t["kind"] in ("signal", "metadata") and t["score"] is not None]
    signal = float(np.mean(signal_scores)) if signal_scores else None
    if neural is None:
        return signal
    if signal is None:
        return neural
    return (1 - SIGNAL_WEIGHT) * neural + SIGNAL_WEIGHT * signal


def heuristic_type(likelihood: float, techniques: list[dict], threshold: float = 0.5) -> str:
    if likelihood < threshold:
        return "none"
    detector = max(
        (t for t in techniques if t["kind"] == "neural_detector" and t.get("windows")),
        key=lambda t: DEFAULT_DETECTOR_WEIGHTS.get(t["id"].split(":", 1)[1], 1.0),
        default=None,
    )
    splice = next((t for t in techniques if t["id"] == "signal:continuity"), None)
    if detector and len(detector["windows"]) >= 3:
        share = float(np.mean(np.array(detector["windows"]) > 0.5))
        if 0.15 < share < 0.75:
            return "partial_synthetic"
    if splice and (splice["score"] or 0) >= 0.3:
        return "partial_synthetic"
    return "synthetic_speech"
