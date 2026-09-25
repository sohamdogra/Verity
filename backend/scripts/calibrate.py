"""Score WAV files window-by-window exactly like the app does, to tune the bands in config.py
and to compare detection models on YOUR clips before a demo.

Usage (from backend/):
  .venv\\Scripts\\python scripts\\calibrate.py                          # built-in demo clips
  .venv\\Scripts\\python scripts\\calibrate.py my_real.wav my_clone.wav
  .venv\\Scripts\\python scripts\\calibrate.py --models MelodyMachine/Deepfake-audio-detection-V2,Bisher/wav2vec2_ASV_deepfake_audio_detection
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audio_io  # noqa: E402
import config  # noqa: E402
from detection import Detector  # noqa: E402
from detection.models import resolve_models  # noqa: E402
from scoring import SessionStore  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parents[2] / "frontend" / "public" / "demo-clips"


def score_file(detector: Detector, path: Path) -> list[float]:
    audio = audio_io.load_mono(path.read_bytes(), config.TARGET_SAMPLE_RATE)
    window = int(config.WINDOW_SECONDS * config.TARGET_SAMPLE_RATE)
    sessions = SessionStore()
    scores = []
    print(f"  == {path.name} ({audio.size / config.TARGET_SAMPLE_RATE:.1f}s)")
    for i in range(0, audio.size, window):
        chunk = audio[i : i + window]
        if chunk.size < config.MIN_AUDIO_SECONDS * config.TARGET_SAMPLE_RATE:
            continue
        if audio_io.rms(chunk) < config.SILENCE_RMS:
            print(f"     window {i // window + 1:>2}: (silence, skipped)")
            continue
        score = detector.analyze(chunk)
        scores.append(score)
        state = sessions.push("calibrate", score)
        print(f"     window {i // window + 1:>2}: raw {score:.3f}  smoothed {state.smoothed:.3f}  "
              f"band {state.band:<5}  stability {state.stability:.2f}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--models", help="Comma-separated model ids to compare (default: configured model)")
    args = parser.parse_args()

    files = args.files or sorted(DEMO_DIR.glob("*.wav"))
    if not files:
        raise SystemExit(f"No WAV files given and none found in {DEMO_DIR}")
    model_lists = [[m.strip()] for m in args.models.split(",")] if args.models else [config.DETECTOR_MODELS]

    print(f"Bands: green < {config.BAND_GREEN_MAX} <= amber <= {config.BAND_RED_MIN} < red\n")
    summary = []
    for model_ids in model_lists:
        detector = Detector(resolve_models(model_ids), sample_rate=config.TARGET_SAMPLE_RATE,
                            force_cpu=config.FORCE_CPU)
        detector.load()
        if not detector.available:
            print(f"!! {model_ids}: unavailable — {detector.status.error}\n")
            continue
        print(f"Model: {detector.status.model_id} on {detector.status.device}")
        for path in files:
            scores = score_file(detector, path)
            if scores:
                summary.append((detector.status.model_id, path.name, sum(scores) / len(scores)))
        print()

    print("Summary (mean raw score per file — you want real clips low, clones high):")
    for model_id, name, mean in summary:
        print(f"  {mean:.2f}  {name:<24} {model_id}")


if __name__ == "__main__":
    main()
