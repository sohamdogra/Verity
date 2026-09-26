# HEARSAY — The Audio Authentication Challenge (NSA)

> Build an AI system that autonomously analyzes an audio file and outputs a synthetic probability (0.0–1.0, 1.0 = synthetic), using multiple forensic techniques, with explainable results. Submit a TSV for the held-out test set.

Verity's forensic engine does this. It is the same code that powers **Check a recording** in the app (`/check`, `POST /forensics`). `backend/scripts/hearsay.py` trains it on the NSA data and writes the submission.

## Quick start

```powershell
# Docker (what judges run): scores every file in the mounted folder
docker build -t verity-hearsay backend
docker run --rm -v "C:\path\to\HackGTHearsayTesting:/data:ro" -v "${PWD}:/out" verity-hearsay
# -> ./predictions.tsv   (header: filename<TAB>cm-score)

# Local
cd backend
.\.venv\Scripts\python scripts\hearsay.py predict --data ..\data\hearsay\test --out teamName_predictions.tsv
```

The output matches NSA's template exactly: a tab-separated file with the header `filename	cm-score`. Each row has the file name including its extension, and `cm-score` = P(synthetic) from 0.0 to 1.0.

> ⚠️ **Score direction.** The ASVspoof5 scorer (`calculate_metrics.py`) treats **higher = bona fide**. NSA's template defines `cm-score` as the probability of **synthetic**. We follow NSA's written spec. If their scorer is run unmodified, the scores need `1 - cm-score` first.

## The data, and the one thing that matters most

| Set | Contents |
|---|---|
| NSA training, bona fide | `LJRealResampled`: **242** clips, one speaker (LJ Speech), 16 kHz, 4.7–9 s |
| NSA training, spoof | `DiffSSD`: **70,000** clips from **10 generators**, including DiffGAN-TTS, Grad-TTS, ProDiff and WaveGrad2 (single-speaker), and OpenVoice v2, XTTS-v2, YourTTS, UnitSpeech, PlayHT and ElevenLabs (zero-shot cloning, 10 speakers each). Mostly 22.05 kHz, 6–11 s |
| NSA test (unlabeled) | **1,671** clips |
| Extra bona fide (public) | LibriSpeech dev-clean (40 speakers) + dev-other (33 speakers), to break the "one speaker = real" shortcut |
| Extra spoof (ours) | XTTS clones, FreeVC voice conversion, MMS-TTS, SAPI and spliced partial fakes from `scripts/devset/` |

**The test set was normalized; the training set was not.** We measured it:

| | Test | NSA real | NSA fake (e.g. OpenVoice) |
|---|---|---|---|
| Sample rate | 16 kHz | 16 kHz | 22.05 kHz |
| Duration | 3.1–5.0 s | 4.7–9 s | 6–11 s |
| Peak level | 1.0 (peak-normalized) | ~0.54 | ~0.47 |
| Effective bandwidth | ~7.45 kHz, every file | 8.0 kHz | ~7.7 kHz |
| Encoder tag | `Lavf58.29.100`, all 1,671 files | `Lavf58.29.100`, all 242 | none (ElevenLabs: MP3, no ID3), every generator sampled |

A model trained on the raw files learns "22 kHz / long / quiet → fake", and the test set contains none of those differences. So `forensics/harmonize.py` runs **every** training clip, real or fake, through the test pipeline:
1. resample to 16 kHz;
2. crop a random 3–5 s segment;
3. apply a 7.45 kHz low-pass;
4. peak-normalize.

Then, with p = 0.5 and **regardless of label**, it applies 1–2 perturbations: noise at 8–35 dB SNR, telephone band (300–3400 Hz via 8 kHz), MP3/AAC/Opus re-encoding, or reverb. That matches "real audio clips might be augmented with noise and other perturbations". After this step the real and fake training clips match each other and the test set on every format statistic above. Features that describe the *file* rather than the *voice* (duration, sample rate, bandwidth cutoff, lossy flag) are excluded from the model.

## Techniques (rubric: forensic diversity)

| Rubric category | What Verity computes | Where |
|---|---|---|
| Deep-learning anti-spoofing | 5 pretrained spoof classifiers (wav2vec2 ASVspoof, generic deepfake and in-the-wild models), per 4 s window: mean/max/min/std/fraction flagged | `forensics/neural.py` |
| SSL front-end | WavLM-base-plus, layers 3–8, mean+std pooled (1,536 dims) | `forensics/neural.py` |
| Speaker-embedding consistency | Cosine drift between the WavLM embeddings of each third of the clip ("does the voice drift?") | `forensics/neural.py` |
| Spectral / frequency domain | Welch bandwidth vs Nyquist; spectral centroid/bandwidth/rolloff/flatness/flux; ZCR; 20 MFCC means/stds/Δ-stds | `forensics/dsp.py` |
| Prosody & phonetics | YIN pitch: F0 median, spread in semitones, frame jitter, voicing ratio; pause structure via digital silence and noise floor | `forensics/dsp.py` |
| Acoustic environment / ENF | 50/60 Hz mains hum (+ harmonics) relative to the 35–135 Hz band; noise floor, dynamic range, per-second floor spread | `forensics/dsp.py` |
| Compression forensics | High-band "spectral holes" typical of low-bitrate codecs; per-frame 99% rolloff variability; codec/bit rate from the container | `forensics/dsp.py`, `decode.py` |
| Splice / discontinuity | Mel-envelope jumps between half-second segments; DC-offset steps; phase breaks (instantaneous-frequency spikes) | `forensics/dsp.py` |
| Container & metadata | RIFF chunk walk, LIST/INFO tags (encoder), header-vs-payload size check, synthesis-tool signatures, MAC timestamps (modified/accessed/created, created-after-modified, future dates) | `forensics/metadata.py` (report only, see below) |
| Content (app only) | Whisper transcript + scam-script cues | `forensics/transcript.py` |

**Why metadata is report-only:** in the training data it separates the classes *perfectly*. Every real clip has the `Lavf58.29.100` encoder tag and no fake has one. In the test set it's constant: all 1,671 files have the same tag. A classifier fed container fields would score 100% in training and learn nothing usable ("which tool wrote this file"). We still inspect and report metadata for every file, and a synthesis-tool signature or a broken header is flagged in the report.

## Orchestration (rubric: agentic bonus)

The analyzer decides what to run instead of brute-forcing everything:
- **By container and codec.** libsndfile decodes WAV/FLAC/OGG/MP3; M4A/AAC/OPUS/WEBM/MP4 go through ffmpeg. Lossy input changes how the bandwidth and compression findings are worded, and the bandwidth-vs-Nyquist check only runs when the native rate is ≥ 32 kHz.
- **By content.** Too-short or silent input stops early with a clear status. Silent windows are skipped. Long files are sampled evenly (at most 20 windows).
- **By confidence (cascade).** Stage 1 runs the cheap evidence (signal forensics + WavLM) through the fast model. If it is already confident (P ≤ 0.03 or ≥ 0.97), the 5-detector ensemble is skipped and the report says so. Otherwise it escalates to the full model.
- **By purpose.** Transcription runs only for the interactive report, since it never affects the score.

## Model and validation

- **Candidates:** logistic regression on detector features only (baseline), on WavLM only, and on everything; gradient boosting on signal + detector features; and a blend.
- **Selection metric: the challenge's own.** ASVspoof5 minDCF with **π_spoof = 0.3, C_miss = 1, C_fa = 4**, as NSA specified. `forensics/training.py:asvspoof_min_dcf` re-implements `compute_det_curve` / `compute_eer` / `compute_mindcf` and is verified to match the official `calculate_modules.py` to 6 decimals, ties included.
- **Validation that predicts the test.** 5-fold `StratifiedGroupKFold`, where the group is the **generator** for fakes and the speaker or source file for real clips. Every fold scores generators the model has never seen, which is the situation NSA's test set creates.

## Results (leave-generator-out CV on 4,279 harmonized training clips)

Every number below comes from folds where the model **never saw the generator it is scored on**. Metric: minDCF with π_spoof = 0.3, C_fa = 4 (lower is better; 1.0 = no better than always guessing one class).

### Candidate models (v1: WavLM + signal + detector features)

| Model | minDCF | EER | AUC |
|---|---|---|---|
| **logreg, all features (selected)** | **0.253** | **9.8%** | **0.966** |
| logreg, all except the 5 detectors | 0.254 | 9.8% | 0.966 |
| logreg, WavLM embeddings only | 0.264 | 9.8% | 0.964 |
| blend(logreg, gradient boosting) | 0.298 | 11.4% | 0.957 |
| gradient boosting, signal + detectors | 0.524 | 20.2% | 0.879 |
| 5 pretrained detectors only (baseline) | 0.944 | 43.5% | 0.590 |

### What worked and what had no effect

Each technique family on its own, in the same grouped CV:

| Technique family | minDCF | EER | AUC | Verdict |
|---|---|---|---|---|
| Self-supervised embeddings (WavLM) | 0.268 | 10.0% | 0.963 | **Carries the system** |
| Spectral statistics + MFCC | 0.619 | 24.6% | 0.828 | Useful, complementary |
| ENF mains hum | 0.882 | 43.2% | 0.609 | Weak signal |
| Speaker-embedding drift | 0.987 | 42.4% | 0.606 | Weak alone (3–5 s clips give little room to drift) |
| Splice / seams (envelope, DC, phase) | 1.000 | 43.8% | 0.582 | No effect on whole-clip fakes; kept for partial fakes |
| Prosody (pitch, jitter, voicing) | 0.997 | 44.9% | 0.552 | No effect |
| Noise floor & dynamics | 1.000 | 48.8% | 0.533 | No effect after augmentation (by design: noise is label-blind) |
| Digital silence | 1.000 | 48.5% | 0.523 | No effect after harmonization (the test set has no silence gaps) |
| Compression / transcoding traces | 0.986 | 48.4% | 0.505 | No effect (both classes are re-encoded in augmentation) |
| **5 pretrained deepfake detectors** | 0.946 | 43.6% | 0.590 | **Barely generalize to unseen generators** |
| Container metadata | — | — | — | Perfect but spurious in training (all real = Lavf tag, no fake has it); constant in test → report-only |

Two lessons. **Off-the-shelf detectors fail on new generators.** They were trained on ASVspoof-era systems and score near chance on DiffSSD's diffusion and zero-shot models. **Hand-crafted cues are fragile once the obvious shortcuts are removed.** Several of them looked strong on raw files, but only because of format differences that we deliberately neutralized. What transfers is a strong self-supervised representation plus a simple, well-regularized classifier.

### Per generator (held out)

| Held-out class | Correct | | Held-out class | Correct |
|---|---|---|---|---|
| YourTTS | 100% | | PlayHT | 91% |
| XTTS-v2 | 99% | | Real speech (all sources) | 90% |
| DiffGAN-TTS | 99% | | ElevenLabs | 86% |
| ProDiff | 98% | | WaveGrad2 | 81% |
| UnitSpeech | 98% | | Grad-TTS | 69% |
| SAPI TTS (ours) | 97% | | MMS-TTS (ours) | 68% |
| OpenVoice v2 | 96% | | Spliced partial fakes (ours) | 63% |
| | | | FreeVC voice conversion (ours) | 57% |

The hardest held-out cases are voice conversion and partial splices, the attack types NSA lists that DiffSSD doesn't contain. The manipulation-type classifier identifies the generator of a detected fake with 96.8% CV accuracy.

## Reproduce

```powershell
cd backend
# 1. training clips: sample, harmonize, augment -> data/hearsay/prepared/train (+ labels.csv with a group column)
.\.venv\Scripts\python scripts\hearsay_prepare.py
# 2. features (cached; shards can run in parallel terminals)
.\.venv\Scripts\python scripts\hearsay.py extract --data ..\data\hearsay\prepared\train --shard 0 --shards 3
.\.venv\Scripts\python scripts\hearsay.py extract --data ..\data\hearsay\test --shard 0 --shards 3
# 3. train (full model) and the fast stage-1 / Live Shield model
.\.venv\Scripts\python scripts\hearsay.py train --data ..\data\hearsay\prepared\train --labels ..\data\hearsay\prepared\train\labels.csv --group-col group
.\.venv\Scripts\python scripts\hearsay.py train --data ..\data\hearsay\prepared\train --labels ..\data\hearsay\prepared\train\labels.csv --group-col group --feature-set live --out models\hearsay_live.joblib --skip-family-report
# 4. submission
.\.venv\Scripts\python scripts\hearsay.py predict --data ..\data\hearsay\test --out teamName_predictions.tsv
```

Expected layout (the `data/` folder is git-ignored; NSA data is never committed):

```
data/hearsay/real/resampled/LJ*.wav             (LJRealResampled)
data/hearsay/spoof/DiffSSD/generated_speech/…   (DiffSSD)
data/hearsay/test/HackGTHearsayTesting/HGT*.wav (test set)
data/extra/LibriSpeech/dev-clean, dev-other     (openslr.org/12)
```

## Honest limits

- There is only one real speaker in the NSA training data. We add public LibriSpeech speakers, but the test set's bona fide conditions (smartphone, telephony, field) may still differ from anything we trained on.
- Leave-generator-out CV estimates performance on *unseen* generators, but the test set may also contain attacks absent from training: replay, scene manipulation, laundering chains.
- Scores are probabilities from a classifier trained with balanced classes. minDCF depends only on the ranking; actDCF/CLLR in the ASVspoof5 package expect log-likelihood ratios and are not meaningful for these scores.
