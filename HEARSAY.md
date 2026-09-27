# HEARSAY — The Audio Authentication Challenge (NSA)

> Build an AI system that autonomously analyzes an audio file and outputs a synthetic probability (0.0–1.0, 1.0 = synthetic), using multiple forensic techniques, with explainable results. Submit a TSV for the held-out test set.

Verity's forensic engine does this. It is the same code that powers **Check a recording** in the app (`/check`, `POST /forensics`). `backend/scripts/hearsay.py` trains it on the NSA data and writes the submission.

**In one paragraph:** we found that the single biggest risk in this challenge was not detection but *leakage* — the provided real and synthetic clips differ in sample rate, length, loudness and encoder tags, so a model can score near-perfectly in training by learning the file format and then fail completely on NSA's normalised test set. So we first rebuild the training set to match the test set exactly, and augment both classes identically. On top of that we stack two frozen self-supervised speech models (WavLM and XLS-R), a set of classical signal-forensic features, and container/metadata analysis, and fuse them with a regularised linear model selected by NSA's own minDCF. Cross-validated against generators it has never seen, the current model reaches **minDCF 0.177 / EER 6.6%**; against the generator families we believe the test set uses, **minDCF 0.028 / EER 1.0%**.

## Quick start

### Docker — what judges run

```bash
# macOS / Linux
docker build -t verity-hearsay backend
docker run --rm -v "/path/to/HackGTHearsayTesting:/data:ro" -v "$PWD:/out" verity-hearsay
```
```powershell
# Windows PowerShell
docker build -t verity-hearsay backend
docker run --rm -v "C:\path\to\HackGTHearsayTesting:/data:ro" -v "${PWD}:/out" verity-hearsay
```

This writes `predictions.tsv` (header `filename<TAB>cm-score`) into the folder you mounted at `/out`.

The build takes about 20 minutes and produces an ~8 GB image, because the trained model and both speech front-ends are baked in. In exchange the container needs **no network at run time** — we verified it with `--network none`, and it reproduced our host predictions to within 0.000007 on all 1,671 test clips. Scoring the full test set takes roughly an hour on 8 CPU cores; pass `--workers 8` to use more parallelism if the machine has it.

### Running it directly

```powershell

# Local
cd backend
.\.venv\Scripts\python scripts\hearsay.py predict --data ..\data\hearsay\test --out teamName_predictions.tsv

# Pass NSA's score-key file to emit rows in exactly its order, and warn on any missing file
.\.venv\Scripts\python scripts\hearsay.py predict --data ..\data\hearsay\test ^
    --template HearsayScoreKey.tsv --out teamName_predictions.tsv
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
| SSL front-end #1 | WavLM-base-plus, layers 3–8, mean+std pooled (1,536 dims) | `forensics/neural.py` |
| SSL front-end #2 | XLS-R 300M, layers 5–10, mean+std pooled (2,048 dims) | `forensics/neural.py` |
| Deep-learning anti-spoofing | 5 pretrained spoof classifiers (wav2vec2 ASVspoof, generic deepfake and in-the-wild models), per 4 s window: mean/max/min/std/fraction flagged. **Shown to the analyst but excluded from the v3 model** — see results | `forensics/neural.py` |
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
- **By confidence (cascade).** Stage 1 runs the cheap evidence — signal forensics plus the WavLM front-end — through a fast model. If that is already decisive (P ≤ 0.03 or ≥ 0.97) the expensive stage is skipped and the report says so; otherwise the full model runs. On our demo clips this short-circuits most decisions, and the saving is real: the skipped stage costs roughly four seconds per clip.
- **By model.** Feature extraction asks the loaded model which front-ends it actually needs, so a model trained without XLS-R never pays to compute it.
- **By purpose.** Transcription runs only for the interactive report, since it never affects the score.

## Model and validation

**Selection metric: the challenge's own.** ASVspoof5 minDCF with **π_spoof = 0.3, C_miss = 1, C_fa = 4**, the values NSA specified rather than the package defaults. `forensics/training.py:asvspoof_min_dcf` re-implements `compute_det_curve` / `compute_eer` / `compute_mindcf` and is verified to match the official `calculate_modules.py` to six decimals, ties included. Note the score direction: NSA's template defines `cm-score` as P(synthetic), while the ASVspoof scorer treats higher as bona fide, so our implementation negates before scoring.

**Two validation protocols.** Both use 5-fold `StratifiedGroupKFold`; only the grouping differs.

| Protocol | Group | What it answers |
|---|---|---|
| **Unseen generators** | generator for fakes, speaker for real | How well do we handle an attack we have never seen? Deliberately pessimistic. |
| **Seen generators** | source recording | How well do we do when the test set uses the same generators as training, but never the same audio? |

We report both. We believe the second is closer to NSA's test set, because our predictions flag 25% of the test files at the model's own threshold against a stated 30% spoof rate — close enough to suggest the same attack families. But the first is the honest measure of generalisation, so it drives model selection.

## Results

Three models, each a strict superset of the last. All numbers are cross-validated with **unseen generators** unless stated.

| | v1 | v2 | **v3 (current)** |
|---|---|---|---|
| Training clips | 4,279 | 4,279 | **10,076** |
| Front-ends | WavLM-base | + XLS-R 300M | + XLS-R 300M |
| **minDCF** | 0.2533 | 0.1912 | **0.1772** |
| EER | 9.77% | 7.78% | **6.63%** |
| AUC | 0.9661 | 0.9813 | **0.9848** |
| minDCF, *seen* generators | 0.1011 | 0.0361 | **0.0283** |
| Manipulation-type accuracy | 96.8% | 98.4% | **99.3%** |

Two things drove the gains. **Adding a second self-supervised front-end mattered most** (v1→v2 cut minDCF by 25%): XLS-R alone beats WavLM alone, and the two together beat either. **Doubling the training data helped less but consistently** (v2→v3), and it particularly improved the hardest category, partial splices, from 0.73 to 0.87 recall.

### Candidate models considered (v3 feature set)

| Model | minDCF | EER | AUC |
|---|---|---|---|
| **Logistic regression, all features (selected)** | **0.1772** | **6.63%** | **0.9848** |
| Blend of logistic regression + gradient boosting | 0.2168 | 8.03% | 0.9749 |
| Logistic regression, embeddings only | 0.2378 | 9.14% | 0.9713 |

A well-regularised linear model on top of strong frozen representations beat everything else we tried, including gradient boosting and an SVM. With ~3,700 features and ~10,000 clips, the linear model generalises better across generators.

### What worked and what had no effect

This is the part the challenge explicitly asks about, so here is every technique family scored **on its own**, same grouped CV:

| Technique family | minDCF | AUC | Verdict |
|---|---|---|---|
| Self-supervised embeddings (XLS-R 300M) | 0.226 | 0.975 | **Strongest single technique** |
| Self-supervised embeddings (WavLM-base) | 0.268 | 0.963 | **Strong, and complementary to XLS-R** |
| Spectral statistics + MFCC | 0.619 | 0.828 | Genuinely useful, adds to the embeddings |
| ENF mains hum (50/60 Hz) | 0.882 | 0.609 | Weak but above chance |
| Speaker-embedding drift | 0.987 | 0.606 | Weak alone — 3–5 s clips give a voice little room to drift |
| Splice / seams (envelope, DC, phase) | 1.000 | 0.582 | No effect on whole-clip fakes; retained because it is the only thing that can localise a partial fake |
| Prosody (pitch, jitter, voicing) | 0.997 | 0.552 | No effect |
| Noise floor & dynamics | 1.000 | 0.533 | No effect |
| Digital silence | 1.000 | 0.523 | No effect |
| Compression / transcoding traces | 0.986 | 0.505 | No effect |
| **5 pretrained deepfake detectors** | **0.946** | **0.590** | **Near chance — dropped from v3** |
| Container metadata | — | — | Perfect but spurious (see below) — report only |

**The two findings we did not expect:**

1. **Off-the-shelf deepfake detectors were almost useless here.** Five published models, all scoring close to chance on DiffSSD's diffusion and zero-shot systems. They were trained on ASVspoof-era vocoders, and modern generators simply do not leave the same traces. They were also ~80% of our compute, so dropping them made v3 five times cheaper to train. We kept them in the *application* because their per-window opinions are useful evidence to show a user, but they contribute nothing to the score.

2. **Most hand-crafted forensic cues collapsed once we removed the shortcuts.** Digital silence, noise floor and compression traces all looked strong on the raw files — and all of that was format leakage, not synthesis artefacts. After harmonisation and label-blind augmentation they sit at chance. We think this is the single most important methodological point in our submission: *a cue that only works before you control for format was never detecting synthesis at all.*

**Why metadata is report-only.** In the training data it separates the classes perfectly: every real clip carries the `Lavf58.29.100` encoder tag and no fake does. In the test set it is constant — all 1,671 files share that tag. A classifier fed container fields would score 100% in training and learn nothing transferable. We still parse and report it (encoder tags, RIFF structure, header-vs-payload consistency, MAC timestamps, synthesis-tool signatures), and a broken header or a tool signature is surfaced to the analyst, but it never reaches the model.

### Per generator (held out, v3)

| Held-out class | Correct | | Held-out class | Correct |
|---|---|---|---|---|
| DiffGAN-TTS | 100% | | ElevenLabs | 92% |
| ProDiff | 100% | | WaveGrad2 | 93% |
| UnitSpeech | 100% | | Real speech (all sources) | 93% |
| OpenVoice v2 | 100% | | Grad-TTS | 56% |
| XTTS-v2 | 100% | | Partial splices (ours) | 58% |
| YourTTS | 100% | | FreeVC voice conversion (ours) | 67% |
| PlayHT | 98% | | | |

The three weak cases are exactly the ones our training data covers thinnest: Grad-TTS, and the two attack types DiffSSD does not contain at all — voice conversion and partial splices, both of which we had to generate ourselves.

## Reproduce

```powershell
cd backend
# 1. Build the training set: sample, harmonise to the test format, augment label-blind.
#    Run twice with different --prefix/--seed to grow it (the second call appends).
.\.venv\Scripts\python scripts\hearsay_prepare.py
.\.venv\Scripts\python scripts\hearsay_prepare.py --prefix batch2 --append --seed 21 --per-generator 250

# 2. Extract features. Cached by file content, so this is only paid once.
#    Shards run in parallel terminals; --no-detectors skips the ensemble we found useless.
.\.venv\Scripts\python scripts\hearsay.py extract --data ..\data\hearsay\prepared\train --shard 0 --shards 4 --no-detectors --extra-embeddings
.\.venv\Scripts\python scripts\hearsay.py extract --data ..\data\hearsay\test --shard 0 --shards 4 --no-detectors --extra-embeddings

# 3. Train. --group-col group gives the leave-generator-out protocol.
.\.venv\Scripts\python scripts\hearsay.py train --data ..\data\hearsay\prepared\train ^
    --labels ..\data\hearsay\prepared\train\labels.csv --group-col group --no-detectors

# 4. Submission TSV, in the exact file order of NSA's score key.
.\.venv\Scripts\python scripts\hearsay.py predict --data ..\data\hearsay\test ^
    --template HearsayScoreKey.tsv --out verity_predictions.tsv
```

Expected layout (`data/` is git-ignored — NSA's audio is never committed):

```
data/hearsay/real/resampled/LJ*.wav             LJRealResampled
data/hearsay/spoof/DiffSSD/generated_speech/...  DiffSSD, 10 generators
data/hearsay/test/HackGTHearsayTesting/HGT*.wav  the 1,671 test clips
data/extra/LibriSpeech/dev-clean, dev-other      openslr.org/12
```

**Runtime on the laptop we developed on** (8-core CPU, no GPU): feature extraction runs at roughly one clip per second per worker, so the full 10,076-clip training set took about 90 minutes across four parallel workers. Training is about 20 minutes; scoring the 1,671 test clips from cache is under a minute.

## Honest limits

- **Our real speech is the narrow side of the data.** One LJ Speech speaker plus 73 LibriSpeech audiobook readers. NSA's brief mentions studio, smartphone, telephony and field recordings; a bona fide clip unlike anything we trained on is our most likely failure mode, and it costs 0.7 per unit rate in their cost function.
- **Three attack types in the brief are absent from our training data**: replay attacks, scene manipulation, and metadata-spoofed containers. We generate our own voice conversion and partial splices to partially cover the gap, and those are measurably our weakest categories (67% and 58%).
- **The seen-generator number is an estimate, not a score.** It assumes the test set draws on the same attack families as training. The evidence for that is circumstantial: our flag rate lines up with the stated 30% spoof prevalence.
- **minDCF depends only on ranking**, so our scores are usable for the challenge metric but are not calibrated probabilities; actDCF and CLLR from the ASVspoof package would not be meaningful for them.
- **No GPU.** Everything here is frozen pretrained front-ends plus a linear classifier. Fine-tuning the front-end is the obvious next step and is where the remaining headroom almost certainly is, but it was not reachable on a CPU-only laptop in the time available.
