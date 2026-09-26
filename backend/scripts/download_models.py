"""Pre-download every HuggingFace model the HEARSAY pipeline uses (for Docker / offline runs)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def main() -> None:
    from transformers import AutoFeatureExtractor, AutoModel, pipeline

    for model_id in config.FORENSICS_DETECTORS:
        print(f"detector  {model_id}", flush=True)
        pipeline("audio-classification", model=model_id, device=-1)
    print(f"embedding {config.FORENSICS_EMBEDDING_MODEL}", flush=True)
    AutoFeatureExtractor.from_pretrained(config.FORENSICS_EMBEDDING_MODEL)
    AutoModel.from_pretrained(config.FORENSICS_EMBEDDING_MODEL)
    for model_id in config.FORENSICS_EXTRA_EMBEDDINGS:
        print(f"embedding {model_id}", flush=True)
        AutoFeatureExtractor.from_pretrained(model_id)
        AutoModel.from_pretrained(model_id)
    if "--with-asr" in sys.argv:
        print(f"asr       {config.FORENSICS_ASR_MODEL}", flush=True)
        pipeline("automatic-speech-recognition", model=config.FORENSICS_ASR_MODEL, device=-1)
    print("All models cached.")


if __name__ == "__main__":
    main()
