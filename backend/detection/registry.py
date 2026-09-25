"""Process-wide cache of HuggingFace audio-classification pipelines, shared by the live
detector and the forensic ensemble so each model is loaded into memory only once."""

import threading

_cache: dict[str, object] = {}
_lock = threading.Lock()


def device_name(force_cpu: bool = False) -> str:
    import torch

    return "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"


def load_pipeline(model_id: str, force_cpu: bool = False):
    with _lock:
        if model_id not in _cache:
            from transformers import pipeline

            device = 0 if device_name(force_cpu) == "cuda" else -1
            _cache[model_id] = pipeline("audio-classification", model=model_id, device=device)
        return _cache[model_id]
