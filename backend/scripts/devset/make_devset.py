"""Build a labeled, HEARSAY-style development set so the forensic pipeline can be trained and
evaluated before the official NSA data arrives. It also produces the Live Shield demo clips.

Classes (manipulation_type):
  none               real LibriSpeech dev-clean speech (40 speakers, CC BY 4.0)
  tts_legacy         Windows SAPI text-to-speech (2 voices, varied rate)
  tts_neural         MMS-TTS (VITS) neural text-to-speech
  voice_clone        XTTS-v2 zero-shot clones of real dev-clean speakers
  voice_conversion   FreeVC: one real speaker converted into another's voice
  partial_synthetic  real speech with a cloned phrase spliced into the middle

Anti-leakage choices: every file is resampled to 16 kHz and loudness-normalized, 25% of
every class is re-encoded to MP3/M4A, filenames are anonymized, synthetic speech reads
LibriSpeech transcripts (same vocabulary as the real class), and train/test speakers are
disjoint.

Usage (from backend/):
  .venv\\Scripts\\python scripts\\devset\\make_devset.py --librispeech <path>\\LibriSpeech\\dev-clean ^
      --out devset --tts-python <path to Coqui env python>   (omit to skip clones/conversion)
Download dev-clean: https://www.openslr.org/resources/12/dev-clean.tar.gz (337 MB)
"""

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from audio_io import resample  # noqa: E402

SR = 16_000
DEMO_DIR = BACKEND.parent / "frontend" / "public" / "demo-clips"
SCAM_PHRASES = [
    "so please send the money today",
    "and don't tell anyone about this call",
    "I need you to wire it right now",
    "can you buy the gift cards for me",
    "they said I can't leave until I pay",
]
DEMO_SCRIPT = (
    "Mom, it's me. I'm in trouble and I really need your help right now. "
    "I got into an accident, and they won't let me go until I pay. Please don't tell Dad. Can you send the money today?"
)


def load_librispeech(root: Path) -> dict[str, list[dict]]:
    speakers: dict[str, list[dict]] = {}
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, text = line.split(" ", 1)
            path = trans.parent / f"{uid}.flac"
            speakers.setdefault(uid.split("-")[0], []).append(
                {"path": path, "text": text, "dur": sf.info(path).duration}
            )
    return speakers


def sentence(text: str) -> str:
    text = text.lower().strip()
    return text[0].upper() + text[1:] + "."


def read_mono(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), rate, SR).astype(np.float32)


def normalize(y: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    level = np.sqrt(np.mean(y**2)) + 1e-9
    y = y * (10 ** (target_db / 20) / level)
    peak = np.abs(y).max()
    return (y * (0.95 / peak) if peak > 0.95 else y).astype(np.float32)


def splice(real: np.ndarray, phrase: np.ndarray, rng: random.Random) -> np.ndarray:
    """Insert `phrase` at the quietest point near the middle of `real`, with 20 ms crossfades."""
    frame = SR // 50
    lo, hi = int(real.size * 0.35), int(real.size * 0.65)
    energies = [np.mean(real[i : i + frame] ** 2) for i in range(lo, hi - frame, frame)]
    at = lo + int(np.argmin(energies)) * frame if energies else real.size // 2
    speech_rms = np.sqrt(np.mean(real[np.abs(real) > 0.02] ** 2)) if np.any(np.abs(real) > 0.02) else 0.1
    phrase = phrase * (speech_rms / (np.sqrt(np.mean(phrase[np.abs(phrase) > 0.02] ** 2)) + 1e-9))
    fade = SR // 50
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    phrase = phrase.copy()
    phrase[:fade] *= ramp
    phrase[-fade:] *= ramp[::-1]
    return np.concatenate([real[:at], phrase, real[at:]]).astype(np.float32)


def run_sapi(jobs: list[dict]) -> None:
    if not jobs:
        return
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "sapi.json"
        manifest.write_text(json.dumps(jobs), encoding="utf-8")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            f"$jobs = Get-Content -Raw -Encoding UTF8 '{manifest}' | ConvertFrom-Json; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "foreach ($j in $jobs) { $s.SelectVoice($j.voice); $s.Rate = $j.rate; "
            "$s.SetOutputToWaveFile($j.out); $s.Speak($j.text) }; $s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)


def run_mms(jobs: list[dict]) -> None:
    import torch
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
    for i, job in enumerate(jobs):
        torch.manual_seed(1000 + i)
        model.speaking_rate = job["rate"]
        with torch.no_grad():
            wav = model(**tok(job["text"], return_tensors="pt")).waveform[0].numpy()
        sf.write(job["out"], wav, model.config.sampling_rate)


def encode(src: Path, dst_stem: Path, rng: random.Random) -> Path:
    """Write as WAV, or (25% of the time) MP3/M4A to mimic real-world uploads."""
    y = normalize(read_mono(src))
    roll = rng.random()
    if roll >= 0.25:
        out = dst_stem.with_suffix(".wav")
        sf.write(out, y, SR, subtype="PCM_16")
        return out
    ext, args = (".mp3", ["-c:a", "libmp3lame", "-b:a", rng.choice(["64k", "96k", "128k"])]) if roll < 0.15 \
        else (".m4a", ["-c:a", "aac", "-b:a", "96k"])
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, SR, subtype="PCM_16")
    out = dst_stem.with_suffix(ext)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp.name, *args, str(out)], check=True)
    Path(tmp.name).unlink()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--librispeech", type=Path, required=True, help="Path to LibriSpeech/dev-clean")
    ap.add_argument("--out", type=Path, default=BACKEND / "devset")
    ap.add_argument("--tts-python", help="Python of a Coqui TTS env (enables voice_clone/conversion/partial)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-demo-clips", action="store_true", help="Don't overwrite frontend demo clips")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    speakers = load_librispeech(args.librispeech)
    ids = sorted(speakers)
    rng.shuffle(ids)
    test_speakers = set(ids[:12])
    split_of = {s: ("test" if s in test_speakers else "train") for s in ids}
    all_texts = [u["text"] for s in ids for u in speakers[s] if 40 < len(u["text"]) < 200]

    work = args.out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []  # {src, label, type, split, provenance}
    sapi_jobs, mms_jobs, coqui_jobs, partial_jobs = [], [], [], []

    for s in ids:
        utts = [u for u in speakers[s] if 3 <= u["dur"] <= 12]
        rng.shuffle(utts)
        ref = next((u for u in utts if u["dur"] >= 6), utts[0])
        for u in [u for u in utts if u is not ref][:4]:
            items.append({"src": u["path"], "label": "bonafide", "type": "none", "split": split_of[s], "prov": u["path"].name})
        if args.tts_python:
            out = work / f"clone_{s}.wav"
            coqui_jobs.append({"kind": "xtts", "text": sentence(rng.choice(all_texts)), "speaker_wav": str(ref["path"]), "out": str(out)})
            items.append({"src": out, "label": "spoof", "type": "voice_clone", "split": split_of[s], "prov": f"xtts<-{ref['path'].name}"})

    if args.tts_python:
        pool = [(s, u) for s in ids for u in speakers[s] if 3 <= u["dur"] <= 10]
        for i in range(30):
            a, src = rng.choice(pool)
            b = rng.choice([x for x in ids if x != a and split_of[x] == split_of[a]])
            target = max(speakers[b], key=lambda u: min(u["dur"], 10))
            out = work / f"vc_{i:02d}.wav"
            coqui_jobs.append({"kind": "freevc", "source_wav": str(src["path"]), "target_wav": str(target["path"]), "out": str(out)})
            items.append({"src": out, "label": "spoof", "type": "voice_conversion", "split": split_of[a], "prov": f"freevc {a}->{b}"})
        for i, s in enumerate(rng.sample(ids, 30)):
            real = rng.choice([u for u in speakers[s] if 4 <= u["dur"] <= 10])
            phrase_out = work / f"phrase_{i:02d}.wav"
            coqui_jobs.append({"kind": "xtts", "text": rng.choice(SCAM_PHRASES), "speaker_wav": str(real["path"]), "out": str(phrase_out)})
            partial_jobs.append({"real": real["path"], "phrase": phrase_out, "out": work / f"partial_{i:02d}.wav"})
            items.append({"src": work / f"partial_{i:02d}.wav", "label": "spoof", "type": "partial_synthetic", "split": split_of[s], "prov": f"splice {real['path'].name}"})

    for i in range(40):
        split = "test" if i % 10 < 3 else "train"
        out = work / f"sapi_{i:02d}.wav"
        sapi_jobs.append({"voice": ["Microsoft David Desktop", "Microsoft Zira Desktop"][i % 2], "rate": rng.randint(-2, 2),
                          "text": sentence(rng.choice(all_texts)), "out": str(out)})
        items.append({"src": out, "label": "spoof", "type": "tts_legacy", "split": split, "prov": "sapi"})
        out = work / f"mms_{i:02d}.wav"
        mms_jobs.append({"text": sentence(rng.choice(all_texts)), "rate": rng.uniform(0.9, 1.15), "out": str(out)})
        items.append({"src": out, "label": "spoof", "type": "tts_neural", "split": split, "prov": "mms-tts-eng"})

    demo_speaker = sorted(test_speakers)[0]
    demo_real = sorted([u for u in speakers[demo_speaker] if 3 <= u["dur"] <= 8], key=lambda u: u["path"].name)[:3]
    demo_ref = max(speakers[demo_speaker], key=lambda u: min(u["dur"], 12))
    if args.tts_python and not args.no_demo_clips:
        coqui_jobs.append({"kind": "xtts", "text": DEMO_SCRIPT, "speaker_wav": str(demo_ref["path"]), "out": str(work / "demo_clone.wav")})

    print(f"Speakers: {len(ids)} ({len(test_speakers)} test). Generating {len(sapi_jobs)} SAPI, {len(mms_jobs)} MMS, "
          f"{len(coqui_jobs)} Coqui jobs ...", flush=True)
    run_sapi([j for j in sapi_jobs if not Path(j["out"]).exists()])
    run_mms([j for j in mms_jobs if not Path(j["out"]).exists()])
    if coqui_jobs:
        manifest = work / "coqui_jobs.json"
        manifest.write_text(json.dumps(coqui_jobs), encoding="utf-8")
        subprocess.run([args.tts_python, str(Path(__file__).with_name("clone_voices.py")), str(manifest)], check=True)
    for job in partial_jobs:
        if job["phrase"].exists():
            sf.write(job["out"], splice(read_mono(job["real"]), read_mono(job["phrase"]), rng), SR)

    # Anonymize, encode, split.
    rng.shuffle(items)
    rows = {"train": [], "test": []}
    provenance = []
    for split in rows:
        (args.out / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for item in items:
        if not Path(item["src"]).exists():
            print(f"  skipping missing {item['src']}")
            continue
        n += 1
        out = encode(Path(item["src"]), args.out / item["split"] / f"clip_{n:04d}", rng)
        rows[item["split"]].append({"filename": out.name, "label": item["label"], "manipulation_type": item["type"]})
        provenance.append({"filename": out.name, "split": item["split"], "type": item["type"], "source": item["prov"]})

    def write(path: Path, data: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    write(args.out / "train" / "labels.csv", rows["train"])
    write(args.out / "test_labels.csv", rows["test"])  # kept OUTSIDE test/ so prediction is blind
    write(args.out / "provenance.csv", provenance)
    for split, data in rows.items():
        counts: dict[str, int] = {}
        for r in data:
            counts[r["manipulation_type"]] = counts.get(r["manipulation_type"], 0) + 1
        print(f"{split}: {len(data)} files  {counts}")

    if not args.no_demo_clips:
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        real = np.concatenate([read_mono(u["path"]) for u in demo_real])
        sf.write(DEMO_DIR / "real.wav", normalize(real), SR, subtype="PCM_16")
        if (work / "demo_clone.wav").exists():
            sf.write(DEMO_DIR / "clone.wav", normalize(read_mono(work / "demo_clone.wav")), SR, subtype="PCM_16")
        print(f"Demo clips: real = LibriSpeech speaker {demo_speaker} (held-out test speaker); "
              f"clone = XTTS-v2 clone of the same speaker reading a scam script -> {DEMO_DIR}")


if __name__ == "__main__":
    main()
