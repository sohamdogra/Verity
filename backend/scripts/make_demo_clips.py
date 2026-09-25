"""Create STAND-IN demo clips until you record real ones (see README "Demo clips").

  real.wav  — genuine human speech from LibriSpeech (CC BY 4.0), via a tiny HF test dataset
  clone.wav — synthetic speech from Windows text-to-speech (SAPI)

Replace both files in frontend/public/demo-clips/ with your own recording and an
ElevenLabs clone of it for the real demo. No code changes needed.

Usage (from backend/):  .venv\\Scripts\\python scripts\\make_demo_clips.py
Needs: pyarrow (pip install pyarrow) for the LibriSpeech parquet; Windows for SAPI TTS.
"""

import io
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_io import resample  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "frontend" / "public" / "demo-clips"
RATE = 16_000
TARGET_SECONDS = 12.0
LIBRISPEECH_PARQUET = (
    "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/"
    "resolve/main/clean/validation-00000-of-00001.parquet"
)
TTS_TEXT = (
    "Mom, it's me. I'm in trouble and I really need your help right now. "
    "I got into an accident and they won't let me go until I pay. "
    "Please don't tell Dad. Can you send the money today? I'm scared."
)


def _normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.8).astype(np.float32)


def make_real() -> np.ndarray:
    import pyarrow.parquet as pq

    print("Downloading LibriSpeech sample utterances ...")
    data = urllib.request.urlopen(LIBRISPEECH_PARQUET, timeout=60).read()
    table = pq.read_table(io.BytesIO(data)).to_pylist()
    # Stitch consecutive utterances from one speaker into ~12 seconds.
    speaker = table[0]["speaker_id"]
    pieces, total = [], 0.0
    for row in table:
        if row["speaker_id"] != speaker:
            continue
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
        audio = resample(audio.mean(axis=1), sr, RATE)
        pieces += [audio, np.zeros(int(0.25 * RATE), dtype=np.float32)]
        total += audio.size / RATE
        if total >= TARGET_SECONDS:
            break
    return _normalize(np.concatenate(pieces)[: int(TARGET_SECONDS * RATE)])


def make_clone() -> np.ndarray:
    if sys.platform != "win32":
        raise SystemExit("clone.wav stand-in uses Windows SAPI; on other OSes export any TTS clip as WAV.")
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "tts.wav"
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { $s.SelectVoice('Microsoft Zira Desktop') } catch {}; "
            "$s.Rate = 0; "
            f"$s.SetOutputToWaveFile('{wav_path}'); "
            f"$s.Speak(\"{TTS_TEXT}\"); $s.Dispose()"
        )
        print("Synthesizing speech with Windows TTS ...")
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    return _normalize(resample(audio.mean(axis=1), sr, RATE))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, make in (("real", make_real), ("clone", make_clone)):
        audio = make()
        path = OUT_DIR / f"{name}.wav"
        sf.write(path, audio, RATE, subtype="PCM_16")
        print(f"Wrote {path} ({audio.size / RATE:.1f}s)")


if __name__ == "__main__":
    main()
