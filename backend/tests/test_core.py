"""Fast tests that need no model download:  .venv\\Scripts\\python -m pytest tests -q"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.labels import LabelMappingError, classify_label, synthetic_likelihood  # noqa: E402
from security import hash_secret, normalize_secret, verify_secret  # noqa: E402


@pytest.mark.parametrize(
    "label,expected",
    [
        ("fake", "synthetic"), ("FAKE", "synthetic"), ("Spoof", "synthetic"), ("spoofed", "synthetic"),
        ("synthetic", "synthetic"), ("AI-Generated", "synthetic"), ("AIVoice", "synthetic"),
        ("deepfake", "synthetic"), ("fake_audio", "synthetic"),
        ("real", "human"), ("Bonafide", "human"), ("bona-fide", "human"), ("HumanVoice", "human"),
        ("genuine", "human"), ("REAL", "human"),
        ("LABEL_0", None), ("not fake", None),
    ],
)
def test_classify_label(label, expected):
    assert classify_label(label) == expected


def test_likelihood_is_order_independent():
    a = [{"label": "fake", "score": 0.8}, {"label": "real", "score": 0.2}]
    b = [{"label": "Real", "score": 0.2}, {"label": "FAKE", "score": 0.8}]
    assert synthetic_likelihood(a) == pytest.approx(0.8)
    assert synthetic_likelihood(b) == pytest.approx(0.8)


def test_likelihood_single_sided_and_overrides():
    assert synthetic_likelihood([{"label": "bonafide", "score": 0.9}]) == pytest.approx(0.1)
    preds = [{"label": "LABEL_0", "score": 0.3}, {"label": "LABEL_1", "score": 0.7}]
    assert synthetic_likelihood(preds, {"LABEL_1": "synthetic", "LABEL_0": "human"}) == pytest.approx(0.7)
    with pytest.raises(LabelMappingError):
        synthetic_likelihood(preds)


def test_secret_normalization_and_hashing():
    assert normalize_secret("  Pizza   Place! ") == "pizza place"
    stored = hash_secret("Blue Pelican")
    assert verify_secret("blue pelican", stored)
    assert verify_secret("  BLUE PELICAN. ", stored)
    assert not verify_secret("blue penguin", stored)
    assert hash_secret("x") != hash_secret("x")  # salted


def test_single_window_never_reaches_red():
    from scoring import SessionStore

    store = SessionStore()
    assert store.push("s", 0.99).band == "amber"
    assert store.push("s", 0.98).band == "red"
    assert store.push("g", 0.1).band == "green"


def test_alert_payload_formats():
    from alerts import _payload

    assert _payload("https://discord.com/api/webhooks/1/abc", "hi", {"x": 1}) == {"content": "hi"}
    assert _payload("https://hooks.slack.com/services/T/B/C", "hi", {"x": 1}) == {"text": "hi"}
    assert _payload("https://example.com/hook", "hi", {"x": 1}) == {"message": "hi", "x": 1}
