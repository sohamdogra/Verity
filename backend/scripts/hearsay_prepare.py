"""Build the HEARSAY training set: sample, harmonize to the test-set format, augment.

Sources (paths default to the repo's data/ folder):
  real   data/hearsay/real                      NSA LJRealResampled (one speaker; cropped 3x)
  real   data/extra/LibriSpeech/dev-clean       40 speakers (bona fide diversity)
  real   data/extra/LibriSpeech/dev-other       33 speakers, harder/noisier (if present)
  spoof  data/hearsay/spoof/DiffSSD/generated_speech/<generator>/...   10 generators
  spoof  backend/devset/_work                   our XTTS/FreeVC/MMS/SAPI/splice fakes (if present)

Every clip goes through forensics.harmonize.to_test_domain (16 kHz, 3-5 s crop, 7.45 kHz
low-pass, peak-normalized) and then, with p=0.5, the same random perturbations regardless
of label. Output: data/hearsay/prepared/train/*.wav + labels.csv with a `group` column
(generator for fakes, speaker/source file for real) so CV can hold out whole generators.

Usage (from backend/):  .venv\\Scripts\\python scripts\\hearsay_prepare.py
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
DATA = BACKEND.parent / "data"
AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO) if root.exists() else []


def plan(args) -> list[dict]:
    rng = random.Random(args.seed)
    items: list[dict] = []

    # --- bona fide ---
    lj = _files(DATA / "hearsay" / "real")
    for crop_i in range(args.lj_crops):
        for f in lj:
            items.append({"src": f, "label": "bonafide", "type": "none", "group": f"lj:{f.stem}", "source": "nsa_lj"})
    for split, per_spk in (("dev-clean", args.libri_per_speaker), ("dev-other", args.libri_other_per_speaker)):
        by_spk = defaultdict(list)
        for f in _files(DATA / "extra" / "LibriSpeech" / split):
            by_spk[f.name.split("-")[0]].append(f)
        for spk, files in sorted(by_spk.items()):
            for f in rng.sample(files, min(per_spk, len(files))):
                items.append({"src": f, "label": "bonafide", "type": "none", "group": f"libri:{spk}", "source": f"librispeech_{split}"})

    # --- spoof: DiffSSD, balanced per generator and spread over speakers ---
    diff = DATA / "hearsay" / "spoof" / "DiffSSD" / "generated_speech"
    for gen in sorted(p for p in diff.iterdir() if p.is_dir()) if diff.exists() else []:
        by_dir = defaultdict(list)
        for f in _files(gen):
            by_dir[f.parent].append(f)
        dirs = sorted(by_dir)
        per_dir = max(1, args.per_generator // len(dirs))
        for d in dirs:
            for f in rng.sample(by_dir[d], min(per_dir, len(by_dir[d]))):
                items.append({"src": f, "label": "spoof", "type": gen.name, "group": f"gen:{gen.name}", "source": "nsa_diffssd"})

    # --- spoof: our dev-set generators (extra diversity: VC, splices, legacy/neural TTS) ---
    work = BACKEND / "devset" / "_work"
    kinds = {"clone": "devset_xtts_clone", "vc": "devset_freevc", "mms": "devset_mms_tts",
             "sapi": "devset_sapi_tts", "partial": "devset_partial_splice"}
    for f in _files(work):
        kind = kinds.get(f.stem.split("_")[0])
        if kind and ".partial" not in f.name:
            items.append({"src": f, "label": "spoof", "type": kind, "group": f"gen:{kind}", "source": "devset"})

    rng.shuffle(items)
    for i, item in enumerate(items):
        item["seed"] = args.seed * 1_000_003 + i
        item["out"] = f"{args.prefix}_{i:05d}.wav"
        # Weight this batch towards attack types NSA lists but our sources lack.
        # Chosen without looking at the label, so it cannot become a shortcut.
        roll = rng.random()
        if roll < args.replay_share:
            item["force"] = "replay"
        elif roll < args.replay_share + args.scene_share:
            item["force"] = "scene"
    return items


def _process(job: tuple[dict, str]) -> dict | None:
    item, out_dir = job
    import numpy as np
    import soundfile as sf

    from forensics.harmonize import augment, to_test_domain

    rng = random.Random(item["seed"])
    try:
        y, sr = sf.read(item["src"], dtype="float32", always_2d=True)
        y = to_test_domain(y.mean(axis=1), sr, rng)
        if y.size < 16_000 * 1.5 or np.abs(y).max() < 1e-3:
            return None
        forced = item.get("force")
        y, augs = augment(y, rng, p=0.5, force=forced)
        sf.write(Path(out_dir) / item["out"], y, 16_000, subtype="PCM_16")
        return {"filename": item["out"], "label": item["label"], "manipulation_type": item["type"],
                "group": item["group"], "source": item["source"], "augment": "+".join(augs) or "none",
                "origin": os.path.relpath(item["src"], DATA.parent)}
    except Exception as exc:
        print(f"  ! {item['src']}: {exc}", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "hearsay" / "prepared" / "train")
    ap.add_argument("--per-generator", type=int, default=200)
    ap.add_argument("--lj-crops", type=int, default=3, help="Random crops per LJ file (only 242 real NSA files)")
    ap.add_argument("--libri-per-speaker", type=int, default=22)
    ap.add_argument("--libri-other-per-speaker", type=int, default=15)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replay-share", type=float, default=0.0,
                    help="Fraction of clips played through a simulated speaker->room->mic chain")
    ap.add_argument("--scene-share", type=float, default=0.0,
                    help="Fraction of clips given a fabricated acoustic background")
    ap.add_argument("--prefix", default="train", help="Filename prefix, so extra batches don't collide")
    ap.add_argument("--append", action="store_true", help="Add to an existing labels.csv instead of replacing it")
    args = ap.parse_args()

    items = plan(args)
    args.out.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    for it in items:
        counts[(it["label"], it["type"] if it["label"] == "spoof" else it["source"])] += 1
    print(f"Preparing {len(items)} clips with {args.workers} workers:")
    for (label, kind), n in sorted(counts.items()):
        print(f"   {label:<9} {kind:<26} {n}")

    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(_process, [(it, str(args.out)) for it in items], chunksize=16), 1):
            if row:
                rows.append(row)
            if i % 500 == 0:
                print(f"  {i}/{len(items)}", flush=True)
    labels = args.out / "labels.csv"
    if args.append and labels.exists():
        existing = list(csv.DictReader(labels.open(encoding="utf-8")))
        keep = {r["filename"] for r in rows}
        rows = [r for r in existing if r["filename"] not in keep] + rows
        print(f"Appending to {len(existing)} existing rows -> {len(rows)} total")
    with labels.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} clips + labels.csv to {args.out}")


if __name__ == "__main__":
    main()
