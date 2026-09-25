"""Forensics tests that need no model download."""

import io
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forensics import dsp  # noqa: E402
from forensics.decode import decode_any  # noqa: E402
from forensics.fusion import default_fusion, heuristic_type  # noqa: E402
from forensics.training import metrics, parse_label, read_labels  # noqa: E402

SR = 16_000


def _speechlike(seconds: float, noise: float = 0.003, gap_zeros: bool = False) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SR)) / SR
    y = 0.2 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 2 * t) > 0) + noise * rng.standard_normal(t.size)
    if gap_zeros:
        y[SR : SR + SR // 2] = 0.0  # half a second of perfect digital silence
    return y.astype(np.float32)


def _wav(y: np.ndarray, rate: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_decode_wav_resamples_and_keeps_native():
    y = _speechlike(2.0)
    d = decode_any(_wav(np.repeat(y, 3)[: 3 * y.size], 48_000), "x.wav")
    assert d.native_rate == 48_000 and abs(d.audio.size - 2 * SR) < 50


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_decode_m4a_via_ffmpeg(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(src, _speechlike(2.0), SR)
    out = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "aac", str(out)], check=True)
    d = decode_any(out.read_bytes(), "a.m4a")
    assert d.decoder == "ffmpeg" and d.lossy and 1.8 < d.duration < 2.3


def test_digital_silence_is_flagged_only_when_present():
    clean = decode_any(_wav(_speechlike(4.0)), "a.wav")
    gappy = decode_any(_wav(_speechlike(4.0, gap_zeros=True)), "b.wav")
    score = lambda d: next(t for t in dsp.analyze(d)[1] if t["id"] == "signal:digital_silence")["score"]  # noqa: E731
    assert score(gappy) > 0.4 > score(clean)


def test_default_fusion_weights_detectors_over_signals():
    techniques = [
        {"id": "detector:Bisher/wav2vec2_ASV_deepfake_audio_detection", "kind": "neural_detector", "score": 0.9},
        {"id": "detector:MelodyMachine/Deepfake-audio-detection-V2", "kind": "neural_detector", "score": 0.1},
        {"id": "signal:bandwidth", "kind": "signal", "score": 0.1},
    ]
    p = default_fusion(techniques)
    assert 0.6 < p < 0.8
    assert default_fusion([]) is None


def test_heuristic_type_detects_partial():
    mixed = [{"id": "detector:Bisher/wav2vec2_ASV_deepfake_audio_detection", "kind": "neural_detector",
              "score": 0.5, "windows": [0.1, 0.9, 0.95, 0.1, 0.05]}]
    full = [{**mixed[0], "windows": [0.9, 0.95, 0.99]}]
    assert heuristic_type(0.7, mixed) == "partial_synthetic"
    assert heuristic_type(0.9, full) == "synthetic_speech"
    assert heuristic_type(0.2, full) == "none"


@pytest.mark.parametrize("value,expected", [("spoof", 1), ("bonafide", 0), ("1", 1), ("0", 0), ("FAKE", 1), ("real", 0)])
def test_parse_label(value, expected):
    assert parse_label(value) == expected


def test_read_labels_autodetects_columns(tmp_path):
    f = tmp_path / "labels.csv"
    f.write_text("File_Name,Class,Attack\na.wav,spoof,tts\nb.wav,bonafide,\n", encoding="utf-8")
    rows = read_labels(f)
    assert rows == [{"file": "a.wav", "y": 1, "type": "tts"}, {"file": "b.wav", "y": 0, "type": "none"}]


def test_metrics_perfect_separation():
    m = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc"] == 1.0 and m["eer"] == 0.0 and m["accuracy"] == 1.0
