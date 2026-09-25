"""Decode any audio file (WAV, FLAC, OGG, MP3, M4A/AAC, OPUS, WEBM, ...) to mono float32.

soundfile handles WAV/FLAC/OGG/MP3 directly; everything else goes through ffmpeg.
We keep the native-rate signal too, because some forensic cues (e.g. a band-limit far
below Nyquist) are only visible before resampling to 16 kHz."""

import io
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from audio_io import AudioDecodeError, resample

TARGET_RATE = 16_000
MAX_NATIVE_SECONDS = 60  # native-rate copy is only for spectral analysis


@dataclass
class DecodedAudio:
    audio: np.ndarray  # mono float32 @ 16 kHz
    native: np.ndarray  # mono float32 @ native_rate (first MAX_NATIVE_SECONDS)
    native_rate: int
    channels: int
    duration: float
    container: str | None = None
    codec: str | None = None
    bit_rate: int | None = None
    decoder: str = "soundfile"
    notes: list[str] = field(default_factory=list)

    @property
    def lossy(self) -> bool:
        text = f"{self.container or ''} {self.codec or ''}".lower()
        return any(k in text for k in ("mp3", "mpeg", "aac", "m4a", "mp4", "opus", "vorbis", "ogg", "webm", "amr", "wma"))


def ffmpeg_path() -> str | None:
    return os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg")


def _ffprobe(path: str) -> dict:
    ffmpeg = ffmpeg_path()
    probe = shutil.which("ffprobe") or (ffmpeg and os.path.join(os.path.dirname(ffmpeg), "ffprobe"))
    if not probe:
        return {}
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
            capture_output=True, timeout=30, check=True,
        ).stdout
        return json.loads(out or b"{}")
    except Exception:
        return {}


def _decode_ffmpeg(data: bytes, suffix: str) -> DecodedAudio:
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise AudioDecodeError("This format needs ffmpeg. Install it (or set FFMPEG_PATH), or send a WAV file.")
    # Containers like M4A keep their index at the end, so decode from a real (seekable) file.
    with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        info = _ffprobe(path)
        stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
        rate = int(stream.get("sample_rate") or 44_100)
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(rate), "pipe:1"],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0 or not result.stdout:
            raise AudioDecodeError("Could not read audio. The file may be damaged or not an audio file.")
        native = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32)
        fmt = info.get("format", {})
        return DecodedAudio(
            audio=resample(native, rate, TARGET_RATE).astype(np.float32),
            native=native[: rate * MAX_NATIVE_SECONDS],
            native_rate=rate,
            channels=int(stream.get("channels") or 1),
            duration=native.size / rate,
            container=fmt.get("format_name"),
            codec=stream.get("codec_name"),
            bit_rate=int(stream.get("bit_rate") or fmt.get("bit_rate") or 0) or None,
            decoder="ffmpeg",
        )
    finally:
        os.unlink(path)


def decode_any(data: bytes, filename: str = "") -> DecodedAudio:
    if not data:
        raise AudioDecodeError("The file is empty.")
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in {".m4a", ".mp4", ".aac", ".webm", ".opus", ".amr", ".wma", ".3gp"}:
        try:
            info = sf.info(io.BytesIO(data))
            audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            return DecodedAudio(
                audio=resample(mono, rate, TARGET_RATE).astype(np.float32),
                native=mono[: rate * MAX_NATIVE_SECONDS],
                native_rate=int(rate),
                channels=audio.shape[1],
                duration=mono.size / rate,
                container=info.format,
                codec=info.subtype,
                decoder="soundfile",
            )
        except Exception:
            pass  # fall through to ffmpeg
    return _decode_ffmpeg(data, suffix)
