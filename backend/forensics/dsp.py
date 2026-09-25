"""Classic signal-processing forensics. Each check yields
  * numeric features (for the trained fusion model), and
  * a plain-language finding with an optional weak suspicion score (0..1) used by the
    default, untrained fusion. Scores here are deliberately conservative: these cues
    have innocent explanations too (editing, noise suppression, codecs)."""

import numpy as np

from .decode import DecodedAudio

SR = 16_000
FRAME = 400  # 25 ms
HOP = 160  # 10 ms
EPS = 1e-10


def _frame_db(y: np.ndarray) -> np.ndarray:
    if y.size < FRAME:
        y = np.pad(y, (0, FRAME - y.size))
    n = 1 + (y.size - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(y[idx] ** 2, axis=1))
    return 20 * np.log10(rms + EPS)


def _technique(tid: str, name: str, finding: str, score: float | None = None) -> dict:
    return {"id": tid, "kind": "signal", "name": name, "score": None if score is None else round(float(score), 3), "finding": finding}


def _silence(d: DecodedAudio, feats: dict) -> dict:
    x = d.native
    exact = float(np.mean(np.abs(x) < 1.5e-5)) if x.size else 0.0
    # Longest run of digital zeros, in seconds.
    zero = np.abs(x) < 1.5e-5
    longest = 0
    if zero.any():
        edges = np.diff(np.concatenate(([0], zero.astype(np.int8), [0])))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        longest = int((ends - starts).max())
    longest_s = longest / d.native_rate
    feats.update(digital_silence_ratio=exact, digital_silence_longest_s=longest_s)
    if exact > 0.02 and longest_s > 0.05:
        return _technique(
            "signal:digital_silence", "Digital silence",
            f"{exact:.0%} of the audio is perfect digital silence (gaps up to {longest_s:.1f}s). Microphones always "
            "pick up some room noise; perfect silence is common in generated or heavily edited audio.",
            min(1.0, 0.35 + exact * 3),
        )
    return _technique("signal:digital_silence", "Digital silence",
                      "Pauses contain natural background noise, as expected from a real microphone.", 0.1)


def _bandwidth(d: DecodedAudio, feats: dict) -> dict:
    from scipy.signal import welch

    nyquist = d.native_rate / 2
    x = d.native
    if x.size < 4096 or d.native_rate < 16_000:
        feats.update(bandwidth_cutoff_hz=nyquist, bandwidth_ratio=1.0)
        return _technique("signal:bandwidth", "Frequency range", "Too little audio at a high enough sample rate to check.")
    freqs, psd = welch(x, fs=d.native_rate, nperseg=4096)
    db = 10 * np.log10(psd + EPS)
    ref = db[(freqs > 100) & (freqs < 4000)].max()
    above = np.where(db > ref - 65)[0]
    cutoff = float(freqs[above[-1]]) if above.size else nyquist
    ratio = cutoff / nyquist
    hf = float(psd[freqs > 4000].sum() / (psd.sum() + EPS))
    feats.update(bandwidth_cutoff_hz=cutoff, bandwidth_ratio=ratio, hf_energy_ratio=hf)
    if d.native_rate >= 32_000 and ratio < 0.7:
        cause = "compressed" if d.lossy else "generated at a lower sample rate and upsampled"
        return _technique(
            "signal:bandwidth", "Frequency range",
            f"Sound stops sharply at about {cutoff / 1000:.1f} kHz although the file can hold up to {nyquist / 1000:.0f} kHz. "
            f"That usually means the audio was {cause}, which is common for synthetic voices but also for phone audio.",
            0.35 if d.lossy else 0.55,
        )
    return _technique("signal:bandwidth", "Frequency range",
                      f"Uses the file's full frequency range (up to ~{cutoff / 1000:.1f} kHz).", 0.15)


def _noise_floor(y: np.ndarray, feats: dict) -> dict:
    db = _frame_db(y)
    floor, level = float(np.percentile(db, 5)), float(np.percentile(db, 95))
    # Background consistency: noise floor per second (spliced audio mixes different rooms).
    per_sec = [float(np.percentile(db[i : i + 100], 10)) for i in range(0, max(1, db.size - 50), 100)]
    spread = float(np.std(per_sec)) if len(per_sec) > 1 else 0.0
    feats.update(noise_floor_db=floor, speech_level_db=level, dynamic_range_db=level - floor,
                 frame_db_std=float(np.std(db)), noise_floor_spread_db=spread)
    if level - floor > 70 and floor < -85:
        return _technique("signal:noise_floor", "Background noise",
                          "The background is unnaturally clean between words, with no room tone. Can indicate generated audio "
                          "or aggressive noise removal.", 0.45)
    return _technique("signal:noise_floor", "Background noise",
                      f"Background noise level looks natural ({level - floor:.0f} dB between speech and pauses).", 0.15)


def _prosody(y: np.ndarray, feats: dict) -> dict:
    import librosa

    f0 = librosa.yin(y, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    db = _frame_db(y)[: f0.size]
    f0 = f0[: db.size]
    voiced = db > (np.percentile(db, 95) - 25)
    v = f0[voiced]
    if v.size < 20:
        feats.update(f0_median=0.0, f0_std_semitones=0.0, jitter=0.0, voiced_ratio=float(voiced.mean()))
        return _technique("signal:prosody", "Pitch & rhythm", "Not enough voiced speech to measure pitch.")
    semis = 12 * np.log2(v / np.median(v))
    consecutive = voiced[1:] & voiced[:-1]  # frame pairs that are both voiced
    step = np.abs(np.diff(f0)) / f0[1:]
    jitter = float(np.median(step[consecutive])) if consecutive.any() else 0.0
    feats.update(f0_median=float(np.median(v)), f0_std_semitones=float(np.std(semis)), jitter=jitter,
                 voiced_ratio=float(voiced.mean()))
    return _technique("signal:prosody", "Pitch & rhythm",
                      f"Pitch varies by about {np.std(semis):.1f} semitones around {np.median(v):.0f} Hz "
                      "(measured for the trained model; not used as a cue on its own).")


def _spectral(y: np.ndarray, feats: dict) -> None:
    import librosa

    S = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP)) + EPS
    for name, values in {
        "centroid": librosa.feature.spectral_centroid(S=S, sr=SR)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=SR)[0],
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=SR)[0],
        "flatness": librosa.feature.spectral_flatness(S=S)[0],
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=HOP)[0],
        "flux": librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=SR),
    }.items():
        feats[f"spec_{name}_mean"] = float(np.mean(values))
        feats[f"spec_{name}_std"] = float(np.std(values))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S**2, sr=SR, n_mels=40)), n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    for i in range(20):
        feats[f"mfcc{i}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc{i}_std"] = float(mfcc[i].std())
        feats[f"mfcc{i}_dstd"] = float(delta[i].std())


def _continuity(y: np.ndarray, feats: dict) -> dict:
    """Splice cue: abrupt jumps in spectral envelope between adjacent half-second segments."""
    import librosa

    seg = SR // 2
    if y.size < seg * 4:
        feats.update(continuity_max_jump=0.0)
        return _technique("signal:continuity", "Splice check", "Too short to check for edits.")
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=512, hop_length=HOP, n_mels=40))
    per = seg // HOP
    env = np.stack([mel[:, i : i + per].mean(axis=1) for i in range(0, mel.shape[1] - per + 1, per)])
    jumps = np.linalg.norm(np.diff(env, axis=0), axis=1)
    rel = float(jumps.max() / (np.median(jumps) + EPS))
    feats.update(continuity_max_jump=rel)
    if rel > 3.5:
        at = (int(np.argmax(jumps)) + 1) * 0.5
        return _technique("signal:continuity", "Splice check",
                          f"The sound character changes abruptly around {at:.1f}s, which can indicate an edit or an "
                          "inserted segment.", min(0.6, 0.2 + (rel - 3.5) * 0.1))
    return _technique("signal:continuity", "Splice check", "No abrupt changes in sound character between segments.", 0.1)


def analyze(d: DecodedAudio) -> tuple[dict[str, float], list[dict]]:
    feats: dict[str, float] = {
        "duration_s": d.duration, "native_rate": float(d.native_rate), "lossy": float(d.lossy),
    }
    y = d.audio
    techniques = [_silence(d, feats), _bandwidth(d, feats), _noise_floor(y, feats), _prosody(y, feats), _continuity(y, feats)]
    _spectral(y, feats)
    return feats, techniques
