"""Neural techniques: an ensemble of pretrained deepfake detectors, and self-supervised
speech embeddings (WavLM) that a trained classifier can learn from."""

import logging
import threading

import numpy as np

from detection.labels import synthetic_likelihood, validate_labels
from detection.registry import device_name, load_pipeline

log = logging.getLogger("verity.forensics")

SR = 16_000


def windows(y: np.ndarray, seconds: float, hop_seconds: float, min_seconds: float = 1.0) -> list[np.ndarray]:
    size, hop = int(seconds * SR), int(hop_seconds * SR)
    if y.size <= size:
        return [y]
    out = [y[i : i + size] for i in range(0, y.size - size + hop, hop)]
    return [w for w in out if w.size >= min_seconds * SR]


class DetectorEnsemble:
    def __init__(self, model_ids: list[str], force_cpu: bool = False):
        self.model_ids = model_ids
        self.force_cpu = force_cpu
        self._lock = threading.Lock()

    def _pipe(self, model_id: str):
        pipe = load_pipeline(model_id, self.force_cpu)
        validate_labels(pipe.model.config.id2label.values())
        return pipe

    def run(self, clips: list[np.ndarray]) -> tuple[dict[str, float], list[dict]]:
        feats: dict[str, float] = {}
        techniques = []
        for model_id in self.model_ids:
            key = model_id.split("/")[-1]
            try:
                pipe = self._pipe(model_id)
                top_k = len(pipe.model.config.id2label)
                with self._lock:
                    scores = [
                        synthetic_likelihood(pipe({"raw": c, "sampling_rate": SR}, top_k=top_k)) for c in clips
                    ]
            except Exception as exc:  # one broken model must not sink the analysis
                log.warning("Detector %s failed: %s", model_id, exc)
                techniques.append({"id": f"detector:{model_id}", "kind": "neural_detector", "name": key,
                                   "score": None, "finding": f"Unavailable ({type(exc).__name__}).", "windows": []})
                continue
            s = np.array(scores)
            feats.update({
                f"det_{key}_mean": float(s.mean()), f"det_{key}_max": float(s.max()),
                f"det_{key}_min": float(s.min()), f"det_{key}_std": float(s.std()),
                f"det_{key}_frac": float((s > 0.5).mean()),
            })
            share = (s > 0.5).mean()
            if share >= 0.66:
                finding = f"Flagged synthetic characteristics in {int(round(share * len(s)))} of {len(s)} segments."
            elif share > 0:
                finding = f"Mixed: {int(round(share * len(s)))} of {len(s)} segments looked synthetic."
            else:
                finding = "No segments looked synthetic to this model."
            techniques.append({"id": f"detector:{model_id}", "kind": "neural_detector", "name": key,
                               "score": round(float(s.mean()), 3), "finding": finding,
                               "windows": [round(float(v), 3) for v in s]})
        return feats, techniques


class Embedder:
    """Mean+std pooled hidden states from middle layers of a self-supervised speech model.
    Middle layers of wav2vec2/WavLM carry the most spoofing-relevant information."""

    def __init__(self, model_id: str = "microsoft/wavlm-base-plus", layers: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
                 force_cpu: bool = False, prefix: str = "emb", drift: bool = True):
        self.model_id, self.layers, self.force_cpu = model_id, layers, force_cpu
        self.prefix, self.drift = prefix, drift
        self._model = self._fe = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from transformers import AutoFeatureExtractor, AutoModel

            self._fe = AutoFeatureExtractor.from_pretrained(self.model_id)
            model = AutoModel.from_pretrained(self.model_id)
            self._device = device_name(self.force_cpu)
            self._model = model.to(self._device).eval()

    def embed(self, y: np.ndarray, max_seconds: float = 20.0) -> dict[str, float]:
        import torch

        with self._lock:
            self._load()
            y = y[: int(max_seconds * SR)]
            inputs = self._fe(y, sampling_rate=SR, return_tensors="pt").to(self._device)
            with torch.no_grad():
                hidden = self._model(**inputs, output_hidden_states=True).hidden_states
            h = torch.stack([hidden[i][0] for i in self.layers]).mean(dim=0)  # (time, dim)
            mean, std = h.mean(dim=0).cpu().numpy(), h.std(dim=0).cpu().numpy()
            # Voice consistency: compare the embedding of each third of the clip with the others.
            thirds = [c.mean(dim=0) for c in torch.tensor_split(h, 3, dim=0) if c.shape[0] > 0]
            sims = [float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                    for i, a in enumerate(thirds) for b in thirds[i + 1 :]]
        feats = {f"{self.prefix}_mean_{i}": float(v) for i, v in enumerate(mean)}
        feats.update({f"{self.prefix}_std_{i}": float(v) for i, v in enumerate(std)})
        if self.drift:
            feats["spk_drift_max"] = 1.0 - min(sims) if sims else 0.0
            feats["spk_drift_mean"] = 1.0 - float(np.mean(sims)) if sims else 0.0
        return feats


# Additional self-supervised front-ends: model id -> (layers to pool, feature prefix).
# XLS-R 300M is the front-end behind many top ASVspoof systems; early-middle layers work best.
EXTRA_EMBEDDERS = {
    "facebook/wav2vec2-xls-r-300m": ((5, 6, 7, 8, 9, 10), "xlsr"),
    # WavLM-large: 24 layers; the middle block carries the most spoofing cues.
    "microsoft/wavlm-large": ((8, 9, 10, 11, 12, 13), "wlml"),
}
