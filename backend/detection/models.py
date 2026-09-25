"""Registry of detection models. Swap or add models here without touching the app."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    # Explicit raw-label -> "synthetic" | "human" mapping, for models whose labels are
    # opaque (e.g. LABEL_0 / LABEL_1). Most models are handled by keyword matching.
    label_overrides: dict[str, str] = field(default_factory=dict)


DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec("MelodyMachine/Deepfake-audio-detection-V2"),  # labels: fake / real
    ModelSpec("mo-thecreator/Deepfake-audio-detection"),  # labels: fake / real
    ModelSpec("Bisher/wav2vec2_ASV_deepfake_audio_detection"),  # Wav2Vec2, ASVspoof-trained
]


def resolve_models(model_ids: list[str]) -> list[ModelSpec]:
    """Use env-configured model ids when given, otherwise the built-in fallback order."""
    if not model_ids:
        return list(DEFAULT_MODELS)
    known = {spec.model_id: spec for spec in DEFAULT_MODELS}
    return [known.get(model_id, ModelSpec(model_id)) for model_id in model_ids]
