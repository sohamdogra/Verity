"""Helpers for training/evaluating the HEARSAY fusion model (used by scripts/hearsay.py)."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from detection.labels import classify_label

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm", ".mp4", ".wma", ".amr", ".3gp"}
FEATURE_VERSION = 1

FILE_COLS = ("filename", "file", "file_name", "path", "filepath", "audio", "audio_path", "clip", "name", "id", "utt", "utterance")
LABEL_COLS = ("label", "labels", "class", "target", "is_fake", "is_synthetic", "fake", "spoof", "y", "ground_truth", "gt", "key")
TYPE_COLS = ("manipulation_type", "manipulation", "type", "attack", "attack_type", "method", "system", "system_id", "generator")


# ---- labels ---------------------------------------------------------------------------


def _pick(fields: list[str], wanted: tuple[str, ...], explicit: str | None, what: str, required: bool) -> str | None:
    if explicit:
        if explicit not in fields:
            raise SystemExit(f"Column '{explicit}' not in labels file (columns: {fields})")
        return explicit
    lowered = {f.lower().strip(): f for f in fields}
    for w in wanted:
        if w in lowered:
            return lowered[w]
    if required:
        raise SystemExit(f"Couldn't find the {what} column in {fields}. Pass it explicitly (see --help).")
    return None


def parse_label(value: str, positive: str | None = None) -> int:
    v = value.strip()
    if positive is not None:
        return int(v.lower() == positive.lower())
    try:
        return int(float(v) >= 0.5)  # numeric: 1 = synthetic
    except ValueError:
        pass
    cls = classify_label(v)
    if cls is None:
        raise ValueError(v)
    return int(cls == "synthetic")


def read_labels(path: Path, file_col=None, label_col=None, type_col=None, positive=None) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ") if sample else csv.excel
        rows = list(csv.DictReader(f, dialect=dialect))
    if not rows:
        raise SystemExit(f"{path} is empty")
    fields = list(rows[0].keys())
    fc = _pick(fields, FILE_COLS, file_col, "filename", True)
    lc = _pick(fields, LABEL_COLS, label_col, "label", True)
    tc = _pick(fields, TYPE_COLS, type_col, "manipulation type", False)
    out, bad = [], set()
    for r in rows:
        try:
            y = parse_label(r[lc], positive)
        except ValueError:
            bad.add(r[lc])
            continue
        kind = (r.get(tc) or "").strip() if tc else ""
        out.append({"file": r[fc].strip(), "y": y, "type": kind or ("none" if y == 0 else None)})
    if bad:
        raise SystemExit(f"Unrecognized label values {sorted(bad)[:10]}; pass --positive-label <value meaning synthetic>.")
    return out


def resolve(data_dir: Path, name: str) -> Path | None:
    p = data_dir / name
    if p.exists():
        return p
    for ext in AUDIO_EXTS:
        if (q := p.with_suffix(ext)).exists() or (q := data_dir / f"{name}{ext}").exists():
            return q
    return None


def list_audio(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


# ---- feature cache --------------------------------------------------------------------


class FeatureCache:
    def __init__(self, root: Path, signature: dict):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.tag = hashlib.sha1(json.dumps({**signature, "v": FEATURE_VERSION}, sort_keys=True).encode()).hexdigest()[:10]

    def _path(self, data: bytes) -> Path:
        return self.root / f"{hashlib.sha1(data).hexdigest()}_{self.tag}.json"

    def get(self, data: bytes) -> dict | None:
        p = self._path(data)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, data: bytes, feats: dict) -> None:
        self._path(data).write_text(json.dumps(feats))


# ---- metrics --------------------------------------------------------------------------


def eer(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold where it occurs."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    out = {"n": int(len(y)), "accuracy": float(accuracy_score(y, p >= threshold)),
           "balanced_accuracy": float(balanced_accuracy_score(y, p >= threshold))}
    if len(set(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["eer"], out["eer_threshold"] = eer(y, p)
    return out


# ---- models ---------------------------------------------------------------------------


class SubsetBlend:
    """Average of classifiers that each see a subset of the columns (picklable, sklearn-like)."""

    def __init__(self, members: list[tuple[list[int], object]]):
        self.members = members
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        for cols, model in self.members:
            model.fit(X[:, cols], y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X[:, cols]) for cols, m in self.members], axis=0)
