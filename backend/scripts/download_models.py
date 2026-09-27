"""Pre-download the HuggingFace models this deployment actually needs (Docker / offline use).

By default it inspects the trained bundle at HEARSAY_MODEL_PATH and fetches only the
front-ends that bundle refers to, which keeps the image small: the pretrained detector
ensemble alone is ~1.8 GB and the current model does not use it.

  python scripts/download_models.py              # only what the shipped model needs
  python scripts/download_models.py --all        # everything the code can use
  python scripts/download_models.py --with-asr   # also the Whisper model for transcripts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from forensics.neural import EXTRA_EMBEDDERS  # noqa: E402


def needed() -> tuple[list[str], list[str]]:
    """(detector model ids, embedding model ids) required by the bundled model."""
    if "--all" in sys.argv:
        return config.FORENSICS_DETECTORS, [config.FORENSICS_EMBEDDING_MODEL, *config.FORENSICS_EXTRA_EMBEDDINGS]

    path = Path(config.HEARSAY_MODEL_PATH)
    if not path.exists():
        print(f"No trained model at {path}; fetching everything.")
        return config.FORENSICS_DETECTORS, [config.FORENSICS_EMBEDDING_MODEL, *config.FORENSICS_EXTRA_EMBEDDINGS]

    from forensics.fusion import load_bundle

    bundle = load_bundle(path, None)
    if bundle is None:
        print(f"Could not read {path}; fetching everything.")
        return config.FORENSICS_DETECTORS, [config.FORENSICS_EMBEDDING_MODEL, *config.FORENSICS_EXTRA_EMBEDDINGS]

    detectors = config.FORENSICS_DETECTORS if bundle.data.get("uses_detectors", True) else []
    embeddings = [config.FORENSICS_EMBEDDING_MODEL] if bundle.uses_embeddings else []
    by_prefix = {prefix: model_id for model_id, (_, prefix) in EXTRA_EMBEDDERS.items()}
    embeddings += [by_prefix[p] for p in sorted(bundle.extra_prefixes) if p in by_prefix]
    print(f"Model {bundle.version} needs {len(detectors)} detector(s) and {len(embeddings)} embedding model(s).")
    return detectors, embeddings


def main() -> None:
    from transformers import AutoFeatureExtractor, AutoModel, pipeline

    detectors, embeddings = needed()
    for model_id in detectors:
        print(f"detector  {model_id}", flush=True)
        pipeline("audio-classification", model=model_id, device=-1)
    for model_id in embeddings:
        print(f"embedding {model_id}", flush=True)
        AutoFeatureExtractor.from_pretrained(model_id)
        AutoModel.from_pretrained(model_id)
    if "--with-asr" in sys.argv:
        print(f"asr       {config.FORENSICS_ASR_MODEL}", flush=True)
        pipeline("automatic-speech-recognition", model=config.FORENSICS_ASR_MODEL, device=-1)
    print("All required models cached.")


if __name__ == "__main__":
    main()
