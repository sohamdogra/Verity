"""Normalize arbitrary classifier labels into one synthetic_likelihood in [0, 1].

Matching is case-insensitive and never depends on label order or index, because
models disagree: some use fake/real, others real/fake, bonafide/spoof, etc."""

import re
from typing import Iterable, Literal, Mapping

LabelClass = Literal["synthetic", "human"]

SYNTHETIC_TERMS = {
    "fake", "spoof", "spoofed", "synthetic", "synth", "ai", "aigenerated", "generated",
    "deepfake", "clone", "cloned", "tts", "vc", "artificial",
}
HUMAN_TERMS = {
    "real", "bonafide", "bona", "genuine", "human", "authentic", "natural", "live", "original",
}
NEGATIONS = {"not", "non", "no"}


class LabelMappingError(ValueError):
    pass


def _split_camel(label: str) -> str:
    """"AIVoice" -> "AI Voice", "HumanVoice" -> "Human Voice"."""
    label = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", label)
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", label)


def _decide(is_synthetic: bool, is_human: bool) -> LabelClass | None:
    if is_synthetic == is_human:  # neither, or ambiguous like "not fake"
        return None
    return "synthetic" if is_synthetic else "human"


def classify_label(label: str, overrides: Mapping[str, str] | None = None) -> LabelClass | None:
    raw = label.strip().lower()
    if overrides:
        lowered = {k.strip().lower(): v for k, v in overrides.items()}
        if raw in lowered:
            return lowered[raw]  # type: ignore[return-value]

    compact = re.sub(r"[^a-z0-9]", "", raw)
    tokens = {t for t in re.split(r"[^a-z0-9]+", _split_camel(label.strip()).lower()) if t}
    if tokens & NEGATIONS:  # "not fake", "non-human": too risky to guess; use label_overrides
        return None
    decided = _decide(
        compact in SYNTHETIC_TERMS or bool(tokens & SYNTHETIC_TERMS),
        compact in HUMAN_TERMS or bool(tokens & HUMAN_TERMS),
    )
    if decided:
        return decided
    # Last resort: substrings like "fakeaudio" / "realspeech" (long terms only, to avoid noise).
    return _decide(
        any(term in compact for term in SYNTHETIC_TERMS if len(term) >= 4),
        any(term in compact for term in HUMAN_TERMS if len(term) >= 4),
    )


def validate_labels(labels: Iterable[str], overrides: Mapping[str, str] | None = None) -> None:
    """Raise if we cannot interpret a model's labels, so the loader can try the next model."""
    classes = {classify_label(label, overrides) for label in labels}
    if not classes & {"synthetic", "human"}:
        raise LabelMappingError(f"Unrecognized labels: {list(labels)}")


def synthetic_likelihood(
    predictions: Iterable[Mapping[str, float]], overrides: Mapping[str, str] | None = None
) -> float:
    """predictions: [{"label": str, "score": float}, ...] from an audio-classification pipeline."""
    synthetic = human = 0.0
    saw_synthetic = saw_human = False
    for pred in predictions:
        cls = classify_label(str(pred["label"]), overrides)
        if cls == "synthetic":
            synthetic += float(pred["score"])
            saw_synthetic = True
        elif cls == "human":
            human += float(pred["score"])
            saw_human = True

    if saw_synthetic and saw_human:
        total = synthetic + human
        value = synthetic / total if total > 0 else 0.5
    elif saw_synthetic:
        value = synthetic
    elif saw_human:
        value = 1.0 - human
    else:
        raise LabelMappingError("No prediction labels could be mapped to synthetic/human")
    return min(1.0, max(0.0, value))
