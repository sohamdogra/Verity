"""Make training audio look like the HEARSAY test set, then augment it.

The official test clips were all normalized the same way: 16 kHz mono PCM16, 3-5 s long,
peak-normalized to full scale, with a low-pass near 7.45 kHz. The training data is not
(LJ real = 16 kHz, 4.7-9 s, peak ~0.5; DiffSSD fakes = 22.05 kHz, longer, quieter). A model
trained on raw files learns those format differences instead of synthesis artifacts, and
the test set has none of them. So every training clip, real or fake, goes through the same
`to_test_domain` pipeline, and augmentations are applied to both classes equally.
"""

import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_io import resample

SR = 16_000
TEST_LOWPASS_HZ = 7_450
CROP_SECONDS = (3.0, 5.0)


def lowpass(y: np.ndarray, cutoff_hz: float = TEST_LOWPASS_HZ) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(10, cutoff_hz, btype="low", fs=SR, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def crop(y: np.ndarray, rng: random.Random, seconds: tuple[float, float] = CROP_SECONDS) -> np.ndarray:
    """Random 3-5 s segment that starts on speech (the test clips rarely begin with silence)."""
    length = int(rng.uniform(*seconds) * SR)
    if y.size <= length:
        return y
    voiced = np.where(np.abs(y) > 0.02 * np.abs(y).max())[0]
    lo = max(0, int(voiced[0]) - int(0.05 * SR)) if voiced.size else 0
    hi = max(lo, min(y.size - length, (int(voiced[-1]) if voiced.size else y.size) - length // 2))
    start = rng.randint(lo, hi) if hi > lo else lo
    return y[start : start + length]


def peak_normalize(y: np.ndarray, peak: float = 0.999) -> np.ndarray:
    m = float(np.abs(y).max())
    return (y * (peak / m)).astype(np.float32) if m > 0 else y


def to_test_domain(y: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    y = resample(y.astype(np.float32), sr, SR)
    return peak_normalize(lowpass(crop(y, rng)))


# ---- augmentations (applied to real and fake alike) ------------------------------------


def _add_noise(y: np.ndarray, rng: random.Random) -> np.ndarray:
    snr_db = rng.uniform(8, 35)
    n = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(y.size).astype(np.float32)
    if rng.random() < 0.5:  # pink-ish noise: integrate white noise a little
        n = np.cumsum(n) * 0.02
        n -= np.convolve(n, np.ones(400) / 400, mode="same")
    signal_power = np.mean(y**2) + 1e-12
    n *= np.sqrt(signal_power / (10 ** (snr_db / 10)) / (np.mean(n**2) + 1e-12))
    return y + n


def _telephone(y: np.ndarray, rng: random.Random) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    y8 = resample(y, SR, 8_000)
    sos = butter(6, [300, 3400], btype="band", fs=8_000, output="sos")
    return resample(sosfiltfilt(sos, y8).astype(np.float32), 8_000, SR)


def _reverb(y: np.ndarray, rng: random.Random) -> np.ndarray:
    rt60 = rng.uniform(0.15, 0.6)
    t = np.arange(int(rt60 * SR)) / SR
    rir = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(t.size) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    wet = np.convolve(y, rir / np.abs(rir).sum() * 4)[: y.size]
    return (1 - 0.5) * y + 0.5 * wet.astype(np.float32)


def _codec(y: np.ndarray, rng: random.Random) -> np.ndarray:
    codec, args, ext = rng.choice([
        ("mp3", ["-c:a", "libmp3lame", "-b:a", rng.choice(["32k", "48k", "64k", "96k"])], ".mp3"),
        ("aac", ["-c:a", "aac", "-b:a", rng.choice(["32k", "64k"])], ".m4a"),
        ("opus", ["-c:a", "libopus", "-b:a", rng.choice(["12k", "16k", "24k"])], ".ogg"),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        src, enc = Path(tmp) / "in.wav", Path(tmp) / f"enc{ext}"
        sf.write(src, y, SR, subtype="PCM_16")
        r1 = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *args, str(enc)], capture_output=True)
        if r1.returncode != 0:
            return y
        out = Path(tmp) / "out.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(enc), "-ar", str(SR), "-ac", "1", str(out)],
                       capture_output=True, check=True)
        dec, _ = sf.read(out, dtype="float32")
    return dec[: y.size] if dec.size >= y.size else np.pad(dec, (0, y.size - dec.size))


AUGMENTATIONS = {"noise": _add_noise, "telephone": _telephone, "reverb": _reverb, "codec": _codec}


def augment(y: np.ndarray, rng: random.Random, p: float = 0.5) -> tuple[np.ndarray, list[str]]:
    """With probability p apply 1-2 random perturbations, then re-apply the test-domain finish."""
    if rng.random() >= p:
        return y, []
    names = rng.sample(list(AUGMENTATIONS), k=rng.choice([1, 1, 2]))
    for name in names:
        y = AUGMENTATIONS[name](y, rng)
    return peak_normalize(lowpass(y)), names
