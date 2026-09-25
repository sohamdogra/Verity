"""Decode uploaded audio into mono float32 at 16 kHz and cut it into analysis windows."""

import io

import numpy as np
import soundfile as sf


class AudioDecodeError(ValueError):
    pass


def decode(data: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioDecodeError("Could not read audio. Please send a WAV file.") from exc
    return audio.mean(axis=1), int(sample_rate)


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate or audio.size == 0:
        return audio
    try:
        import torch
        import torchaudio.functional as F

        return F.resample(torch.from_numpy(audio), from_rate, to_rate).numpy()
    except Exception:
        # Fallback without torchaudio: linear interpolation is good enough for a signal.
        duration = audio.size / from_rate
        n_out = int(round(duration * to_rate))
        x_old = np.linspace(0.0, duration, num=audio.size, endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def load_mono(data: bytes, target_rate: int) -> np.ndarray:
    audio, sample_rate = decode(data)
    return resample(audio, sample_rate, target_rate).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def split_windows(
    audio: np.ndarray, sample_rate: int, window_seconds: float, min_seconds: float
) -> list[np.ndarray]:
    """Short audio (<= ~1.5 windows) stays whole; longer audio is cut into windows.
    A trailing piece shorter than min_seconds is dropped."""
    window = int(window_seconds * sample_rate)
    if audio.size <= int(window * 1.5):
        return [audio]
    pieces = [audio[i : i + window] for i in range(0, audio.size, window)]
    return [p for p in pieces if p.size >= int(min_seconds * sample_rate)]
