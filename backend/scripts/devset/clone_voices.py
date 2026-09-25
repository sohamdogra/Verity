"""Voice cloning (XTTS-v2) and voice conversion (FreeVC) jobs for the HEARSAY dev set.

Runs in a SEPARATE Python env with Coqui TTS (it needs older torch/transformers than the app):
    python -m venv .venv-tts
    .venv-tts\\Scripts\\pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
    .venv-tts\\Scripts\\pip install coqui-tts "transformers>=4.57,<5"
make_devset.py calls it as:  <tts-python> clone_voices.py jobs.json

XTTS-v2 is released under the Coqui Public Model License (non-commercial). Only clone voices
you have the right to use; LibriSpeech speakers are public-domain audiobook readers.
"""

import json
import os
import sys
import time

os.environ.setdefault("COQUI_TOS_AGREED", "1")


def main(manifest: str) -> None:
    jobs = json.load(open(manifest, encoding="utf-8"))
    from TTS.api import TTS

    xtts = freevc = None
    started = time.time()
    for i, job in enumerate(jobs, 1):
        if os.path.exists(job["out"]):
            continue
        tmp = job["out"] + ".partial.wav"  # write-then-rename so an interrupted run never leaves a bad file
        try:
            if job["kind"] == "xtts":
                xtts = xtts or TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                xtts.tts_to_file(text=job["text"], speaker_wav=job["speaker_wav"], language="en", file_path=tmp)
            elif job["kind"] == "freevc":
                freevc = freevc or TTS("voice_conversion_models/multilingual/vctk/freevc24")
                freevc.voice_conversion_to_file(source_wav=job["source_wav"], target_wav=job["target_wav"],
                                                file_path=tmp)
            os.replace(tmp, job["out"])
        except Exception as exc:  # keep going; make_devset skips missing outputs
            print(f"[{i}/{len(jobs)}] FAILED {job['out']}: {exc}", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] {job['kind']} -> {os.path.basename(job['out'])} ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
