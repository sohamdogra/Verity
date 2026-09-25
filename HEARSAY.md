# HEARSAY — The Audio Authentication Challenge (NSA)

> Build an AI system that autonomously analyzes an audio file, outputs a synthetic likelihood (0–100%) and optionally the type of manipulation, and submit predictions for the held-out test set as a CSV.

Verity's forensic engine does this. It powers **Check a recording** in the app (`/check`, `POST /forensics`), and `backend/scripts/hearsay.py` trains on the provided labeled set and writes the submission CSV.

## How it works

For each file, the orchestrator (`backend/forensics/analyzer.py`) does the following:

1. **Decodes anything.** WAV/FLAC/OGG/MP3 are read with libsndfile. M4A/AAC/OPUS/WEBM/AMR/WMA go through ffmpeg. It keeps both a 16 kHz mono copy (for the models) and the native-rate signal (for spectral forensics), and records codec, bit rate, sample rate and channels.
2. **Decides what to run.** Too-short or silent input stops early with a clear status. It cuts 4 s windows with a 3 s hop and skips silent windows. Long files are sampled evenly (at most 20 windows). Embeddings are computed only when the trained model uses them, and transcription only for the interactive report.
3. **Runs independent techniques:**

   | Family | Technique | What it catches |
   |---|---|---|
   | Neural detectors | 5 pretrained wav2vec2/HuBERT spoof classifiers (ASVspoof, in-the-wild, generic deepfake sets), scored per window | Vocoder and acoustic-model artifacts; each model generalizes to different generators |
   | Self-supervised embeddings | WavLM-base-plus, middle layers 3–8, mean+std pooled (1,536 dims) | The strongest general-purpose spoofing representation when you have labeled data to learn from |
   | Signal forensics | Digital-silence ratio and longest zero run | Generated/edited audio has perfect silence; microphones never do |
   | | Effective bandwidth vs Nyquist (Welch PSD, native rate) | Audio generated at 16–24 kHz and upsampled; codec low-pass |
   | | Noise floor, dynamic range, per-second floor spread | Unnaturally clean backgrounds; mixed sources |
   | | Pitch (YIN): F0 spread, jitter, voicing | Over-smooth prosody (used as learned features) |
   | | Spectral centroid/bandwidth/rolloff/flatness/flux, ZCR, 20 MFCCs (+Δ) | Generic timbre statistics for the classifier |
   | | Segment-to-segment envelope jumps | Splices / inserted segments |
   | Content (report only) | Whisper transcript + scam-script cues (urgency, money, secrecy, emergency, authority) | *What* is said, reported separately; never changes the likelihood |

4. **Fuses the evidence.**
   - With a trained bundle (`backend/models/hearsay.joblib`), a classifier chosen by cross-validation gives the probability, and a second classifier trained on synthetic files only gives the manipulation type.
   - Without one, a transparent weighted average is used (neural detectors 85%, signal cues 15%), plus heuristic typing: mixed per-window verdicts or a splice cue → `partial_synthetic`, otherwise `synthetic_speech`. It is marked `calibrated: false`.
5. **Explains itself.** Every technique returns a plain-language finding, and the report lists the steps it took and why.

## Using it for the challenge

```powershell
cd backend
.\.venv\Scripts\Activate.ps1

# Train on the provided labeled set. Columns are auto-detected (filename/file/path + label/class/target
# + optional type/attack/method); labels like spoof/bonafide, fake/real, 1/0 all work.
python scripts\hearsay.py train --data D:\hearsay\train --labels D:\hearsay\train_labels.csv

# If files share speakers/sources, keep them in the same CV fold for honest numbers:
python scripts\hearsay.py train --data ... --labels ... --group-col speaker_id

# Predict the held-out test set -> submission CSV
python scripts\hearsay.py predict --data D:\hearsay\test --out submission.csv
```

`submission.csv` has the columns `filename, synthetic_likelihood (0-100), prediction (synthetic|bonafide), manipulation_type`. If the organizers want different column names or a 0–1 score, use `--id-col` and `--scale 1`, or rename the columns. The CSV is plain.

- **Speed:** on this laptop's CPU, about 1.5–2.5 s per file the first time. Features are cached in `backend/.hearsay_cache/` keyed by file content, so re-training with different models is instant.
- **Output:** `train` prints cross-validated AUC / EER / balanced accuracy for each candidate model, a per-class accuracy table and the manipulation-type recall. It saves `models/hearsay.joblib` plus `models/hearsay_metrics.json`. The running app picks up a new model automatically on the next request.

Candidate models compared in CV:
- `detectors_only (baseline)`: logistic regression on the 5 detectors' per-window statistics
- `logreg_all_features`: standardized logistic regression on everything, including embeddings
- `gboost_signal+detectors`: gradient boosting on detectors + signal features
- `blend(logreg_all, gboost)`: average of the two

## Dev set (until the official data arrives)

`backend/scripts/devset/make_devset.py` builds a labeled practice set with the same shape as the challenge (a train folder + labels CSV, and a blind test folder):

| Type | Source |
|---|---|
| `none` (bonafide) | LibriSpeech dev-clean, 40 speakers |
| `tts_legacy` | Windows SAPI voices, varied rate |
| `tts_neural` | MMS-TTS (VITS) |
| `voice_clone` | XTTS-v2 zero-shot clones of the dev-clean speakers |
| `voice_conversion` | FreeVC, one real speaker converted into another |
| `partial_synthetic` | Real speech with a cloned scam phrase spliced in |

The set is built to stop the classifier from cheating:
- Every file is resampled to 16 kHz and loudness-normalized.
- 25% of every class is re-encoded to MP3/M4A.
- Filenames are anonymized.
- Synthetic speech reads LibriSpeech transcripts, so fakes and real speech use the same vocabulary.
- Train and test speakers are disjoint.

Build and score it:

```powershell
cd backend
# needs LibriSpeech dev-clean + a Coqui TTS env for clones (see the script's docstring)
python scripts\devset\make_devset.py --librispeech <path>\LibriSpeech\dev-clean --out devset --tts-python <coqui python>
python scripts\hearsay.py train    --data devset\train --labels devset\train\labels.csv
python scripts\hearsay.py predict  --data devset\test  --out devset\submission.csv
python scripts\hearsay.py evaluate --data devset\test  --labels devset\test_labels.csv

# Fast model for Live Shield (WavLM embeddings + signal features, no detector ensemble)
python scripts\hearsay.py train --data devset\train --labels devset\train\labels.csv --feature-set live --out models\hearsay_live.joblib
```

The current build has 237 training files and 103 blind test files from 12 held-out speakers. Metrics are written to `backend/models/hearsay_metrics.json`.

**What we already know:** on a real speaker and an XTTS-v2 clone of that same speaker, none of the 5 pretrained detectors reliably flags the clone. Their mean scores on the clone range from 0.00 to 0.48, and one scores the *real* voice 0.91. Each off-the-shelf model catches some generators and misses others. That is why the final decision comes from a model trained on labeled data, and why Verity leads with verification.

## Honest limits

- The dev set is small and built from a handful of generators. **Its numbers show the pipeline works; they don't predict the official score.** Retrain on the official training data; that is what the classifier is for.
- Pretrained detectors disagree wildly on unseen generators (see the README's model table). That is why the final score comes from a model fit on labeled data rather than any single detector.
- The manipulation-type labels in the official data may use a different taxonomy. `train` learns whatever values are in the type column.
