"""HEARSAY: The Audio Authentication Challenge — extract, train, predict (submission TSV), evaluate.

  # 0. (optional, parallel) pre-compute features: run N shards in separate terminals
  .venv\\Scripts\\python scripts\\hearsay.py extract --data <dir> --shard 0 --shards 3

  # 1. Train the fusion model on the labeled training set (any audio formats)
  .venv\\Scripts\\python scripts\\hearsay.py train --data path\\to\\train --labels path\\to\\labels.csv --group-col group

  # 2. Predict the held-out test set -> submission TSV (filename <tab> cm-score, 0.0-1.0, 1.0 = synthetic)
  .venv\\Scripts\\python scripts\\hearsay.py predict --data path\\to\\test --out teamName_predictions.tsv

  # 3. (Optional) score predictions against labels you have
  .venv\\Scripts\\python scripts\\hearsay.py evaluate --data path\\to\\test --labels test_labels.csv

Features are cached in backend/.hearsay_cache, so re-training after the first run is fast.
Column names are auto-detected (filename/file/path, label/class/target, type/attack/...);
override with --file-col / --label-col / --type-col / --positive-label.
"""

import argparse
import copy
import csv
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

import config  # noqa: E402
from forensics.analyzer import get_analyzer  # noqa: E402
from forensics.decode import decode_any  # noqa: E402
from forensics.fusion import heuristic_type, load_bundle  # noqa: E402
from forensics.training import FeatureCache, SubsetBlend, list_audio, metrics, read_labels, resolve  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
CACHE_DIR = BACKEND / ".hearsay_cache"
# Properties of the FILE, not the voice. The HEARSAY test set is uniform on all of them
# (16 kHz, 3-5 s, same low-pass), so a model must never learn from them.
FORMAT_FEATURES = {"duration_s", "native_rate", "lossy", "bandwidth_cutoff_hz", "bandwidth_ratio"}
FEATURE_SETS = {
    "all": lambda n: n not in FORMAT_FEATURES,
    "no-embeddings": lambda n: n not in FORMAT_FEATURES and not n.startswith("emb_"),
    # Fast enough for Live Shield: no detector ensemble, just embeddings + signal features.
    # The live model must stay fast: base WavLM + signal features only.
    "live": lambda n: n not in FORMAT_FEATURES and not n.startswith(("det_", "xlsr_", "wlml_")),
    "no-extra-embeddings": lambda n: n not in FORMAT_FEATURES and not n.startswith(("xlsr_", "wlml_")),
    "no-detectors": lambda n: n not in FORMAT_FEATURES and not n.startswith("det_"),
}
# Technique families, for the "what worked / what had no effect" report.
FAMILIES = {
    "Deep-learning anti-spoofing detectors (5 models)": ("det_",),
    "Self-supervised embeddings (WavLM)": ("emb_mean_", "emb_std_"),
    "Self-supervised embeddings (XLS-R 300M)": ("xlsr_",),
    "Self-supervised embeddings (WavLM-large)": ("wlml_",),
    "Speaker-embedding drift": ("spk_drift",),
    "Spectral statistics + MFCC": ("spec_", "mfcc"),
    "Prosody (pitch, jitter, voicing)": ("f0_", "jitter", "voiced_ratio"),
    "Noise floor & dynamics": ("noise_floor", "speech_level", "dynamic_range", "frame_db_std"),
    "Digital silence": ("digital_silence",),
    "Splice / seams (envelope, DC, phase)": ("continuity", "dc_offset", "phase_jump"),
    "ENF mains hum": ("enf_",),
    "Compression / transcoding traces": ("hf_hole", "rolloff99", "hf_energy"),
}


def _signature(embeddings: bool, detectors: bool = True) -> dict:
    return {"detectors": config.FORENSICS_DETECTORS if detectors else [], "embedding": config.FORENSICS_EMBEDDING_MODEL if embeddings else None,
            "window": config.FORENSICS_WINDOW_SECONDS, "hop": config.FORENSICS_HOP_SECONDS}


def featurize(paths: list[Path], embeddings: bool = True, threads: int | None = None,
              extra: bool | set[str] = False, base: bool = True, detectors: bool = True) -> list[dict | None]:
    """Base features (+ optional extra SSL embeddings), each cached separately by content hash.

    detectors=False skips the 5-detector ensemble (about 80% of the compute) and reuses any
    full cached result by dropping its det_* features."""
    if threads:
        import torch

        torch.set_num_threads(threads)
    analyzer = get_analyzer()
    cache = FeatureCache(CACHE_DIR, _signature(embeddings, detectors))
    full_cache = None if detectors else FeatureCache(CACHE_DIR, _signature(embeddings, True))
    extra_embedders = [e for e in analyzer.extra_embedders
                       if extra is True or (isinstance(extra, set) and e.prefix in extra)]
    extra_caches = [(e, FeatureCache(CACHE_DIR, {"embedding": e.model_id, "layers": list(e.layers)}))
                    for e in extra_embedders]
    out: list[dict | None] = []
    started = time.time()
    fresh = 0
    for i, path in enumerate(paths, 1):
        data = path.read_bytes()
        feats = cache.get(data) if base else {}
        if feats is None and full_cache is not None:
            full = full_cache.get(data)
            if full is not None:
                feats = {k: v for k, v in full.items() if not k.startswith("det_")}
        decoded = None
        if feats is None:
            try:
                decoded = decode_any(data, path.name)
                feats, _, _ = analyzer.extract(decoded, embeddings=embeddings, detectors=detectors)
                cache.put(data, feats)
                fresh += 1
            except Exception as exc:
                print(f"  ! {path.name}: {exc}")
                feats = None
        if feats is not None:
            for emb, ecache in extra_caches:
                extra_feats = ecache.get(data)
                if extra_feats is None:
                    decoded = decoded or decode_any(data, path.name)
                    extra_feats = emb.embed(decoded.audio)
                    ecache.put(data, extra_feats)
                    fresh += 1
                feats = {**feats, **extra_feats}
        out.append(feats)
        if i % 10 == 0 or i == len(paths):
            rate = (time.time() - started) / max(fresh, 1)
            eta = rate * sum(1 for _ in paths[i:]) if fresh else 0
            print(f"  features {i}/{len(paths)}  ({fresh} computed, ~{eta / 60:.1f} min left)", flush=True)
    return out


def _matrix(feats: list[dict], names: list[str]) -> np.ndarray:
    return np.nan_to_num(np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=np.float64))


def _models(names: list[str]) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    every = list(range(len(names)))
    no_emb = [i for i, n in enumerate(names) if not n.startswith("emb_")]
    detectors = [i for i, n in enumerate(names) if n.startswith("det_")]
    no_det = [i for i, n in enumerate(names) if not n.startswith("det_")]
    # C=0.02 was best in leave-generator-out CV on the NSA training data (see HEARSAY.md).
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=5000, class_weight="balanced"))  # noqa: E731
    hgb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,  # noqa: E731
                                                 l2_regularization=1.0, class_weight="balanced", random_state=0)
    embeddings = [i for i, n in enumerate(names) if n.startswith("emb_")]
    candidates = {
        "detectors_only (baseline)": lambda: SubsetBlend([(detectors, lr())]),
        "logreg_embeddings_only": lambda: SubsetBlend([(embeddings, lr())]),
        "logreg_all_features": lambda: SubsetBlend([(every, lr())]),
        # The pretrained detectors barely generalize to unseen generators and add noise.
        "logreg_no_detectors": lambda: SubsetBlend([(no_det, lr())]),
        "gboost_signal+detectors": lambda: SubsetBlend([(no_emb, hgb())]),
        "blend(logreg_all, gboost)": lambda: SubsetBlend([(every, lr()), (no_emb, hgb())]),
    }
    return {k: v for k, v in candidates.items()
            if (detectors or "detectors" not in k) and (embeddings or "embeddings" not in k)}


def _family_report(names, X, y, groups, folds) -> list[dict]:
    """Train a small model on each technique family alone: which techniques carry signal?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for family, prefixes in FAMILIES.items():
        cols = [i for i, n in enumerate(names) if n.startswith(prefixes)]
        if not cols:
            continue
        factory = lambda cols=cols: SubsetBlend([(cols, make_pipeline(  # noqa: E731
            StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced")))])
        m = metrics(y, _oof(factory, X, y, groups, folds))
        rows.append({"family": family, "features": len(cols), **{k: m.get(k) for k in ("min_dcf", "eer", "auc")}})
    return sorted(rows, key=lambda r: r["min_dcf"])


def _oof(factory, X, y, groups, folds: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=0) if groups is not None \
        else StratifiedKFold(folds, shuffle=True, random_state=0)
    p = np.zeros(len(y))
    for tr, te in splitter.split(X, y, groups):
        p[te] = factory().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def cmd_train(args) -> None:
    import joblib

    rows = read_labels(args.labels, args.file_col, args.label_col, args.type_col, args.positive_label)
    if args.limit:
        rows = rows[: args.limit]
    paths = [resolve(args.data, r["file"]) for r in rows]
    missing = [r["file"] for r, p in zip(rows, paths) if p is None]
    if missing:
        print(f"Warning: {len(missing)} labeled files not found (e.g. {missing[:3]})")
    rows, paths = zip(*[(r, p) for r, p in zip(rows, paths) if p is not None])
    print(f"Training on {len(rows)} files ({sum(r['y'] for r in rows)} synthetic).")
    feats = featurize(list(paths), embeddings=not args.no_embeddings, threads=args.threads,
                      extra=not args.no_extra_embeddings, detectors=not args.no_detectors)
    keep = [i for i, f in enumerate(feats) if f is not None]
    rows, feats = [rows[i] for i in keep], [feats[i] for i in keep]
    names = sorted(k for k in {k for f in feats for k in f} if FEATURE_SETS[args.feature_set](k))
    X, y = _matrix(feats, names), np.array([r["y"] for r in rows])
    print(f"Feature set '{args.feature_set}': {len(names)} features.")
    groups = None
    if args.group_col:
        lookup = {row[args.file_col or "filename"]: row[args.group_col] for row in csv.DictReader(args.labels.open(encoding="utf-8-sig"))}
        groups = np.array([lookup.get(r["file"], r["file"]) for r in rows])

    folds = max(2, min(args.folds, int(np.bincount(y).min())))
    print(f"\n{folds}-fold cross-validation ({'grouped' if groups is not None else 'stratified'}):")
    results = {}
    for name, factory in _models(names).items():
        p = _oof(factory, X, y, groups, folds)
        results[name] = (metrics(y, p), p, factory)
        m = results[name][0]
        print(f"  {name:<28} minDCF {m.get('min_dcf', float('nan')):.4f}  EER {m.get('eer', float('nan')):.4f}  "
              f"AUC {m.get('auc', float('nan')):.4f}")
    # The challenge metric: minDCF (Pspoof 0.3, Cfa 4) — lower is better.
    best = min((k for k in results if "baseline" not in k),
               key=lambda k: (results[k][0].get("min_dcf", 9), results[k][0].get("eer", 1)))
    best_metrics, best_oof, factory = results[best]
    threshold = float(np.clip(best_metrics.get("eer_threshold", 0.5), 0.05, 0.95))
    model = factory().fit(X, y)
    print(f"\nSelected: {best} (threshold {threshold:.3f})")

    # Manipulation-type model, trained on synthetic files only.
    type_model, type_metrics = None, None
    fake = [i for i, r in enumerate(rows) if r["y"] == 1 and r["type"]]
    if fake:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        kinds = np.array([rows[i]["type"] for i in fake])
        classes, counts = np.unique(kinds, return_counts=True)
        if len(classes) >= 2 and counts.min() >= 3:
            Xf = X[fake]
            tm = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
            cv = StratifiedKFold(min(5, int(counts.min())), shuffle=True, random_state=0)
            pred = cross_val_predict(copy.deepcopy(tm), Xf, kinds, cv=cv)
            type_metrics = {"accuracy": float((pred == kinds).mean()),
                            "per_type_recall": {c: float((pred[kinds == c] == c).mean()) for c in classes}}
            type_model = tm.fit(Xf, kinds)
            print(f"Manipulation-type model: {len(classes)} types, CV accuracy {type_metrics['accuracy']:.3f}")
            for c, rec in type_metrics["per_type_recall"].items():
                print(f"    {c:<22} recall {rec:.2f}")

    per_type = {}
    for kind in sorted({r["type"] or "unknown" for r in rows}):
        idx = [i for i, r in enumerate(rows) if (r["type"] or "unknown") == kind]
        target = rows[idx[0]]["y"]
        per_type[kind] = float(np.mean((best_oof[idx] >= threshold) == target))
    print("Cross-validated accuracy per class (held-out generators / speakers):")
    for kind, acc in per_type.items():
        print(f"    {kind:<26} {acc:.2f}")

    families = []
    if not args.skip_family_report:
        print("\nWhat each technique family achieves on its own (same grouped CV):")
        families = _family_report(names, X, y, groups, folds)
        for r in families:
            print(f"    {r['family']:<48} minDCF {r['min_dcf']:.4f}  EER {r['eer']:.4f}  "
                  f"AUC {r['auc']:.4f}  ({r['features']} features)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("hearsay-%Y%m%d-%H%M")
    joblib.dump({
        "version": version, "trained_at": datetime.now(timezone.utc).isoformat(), "feature_names": names,
        "binary": model, "model_name": best, "threshold": threshold, "type_model": type_model,
        "uses_embeddings": any(n.startswith("emb_") for n in names),
        "uses_detectors": any(n.startswith("det_") for n in names), "feature_set": args.feature_set,
        "detectors": config.FORENSICS_DETECTORS,
        "n_train": len(rows),
    }, args.out)
    report = {"version": version, "selected": best, "threshold": threshold, "cv": {k: v[0] for k, v in results.items()},
              "per_class_accuracy": per_type, "type_model": type_metrics, "n_train": len(rows),
              "technique_families": families, "cv_grouping": args.group_col or "stratified"}
    args.out.with_name(args.out.stem + "_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved {args.out} (+ metrics JSON)")


def predict_paths(paths: list[Path], model_path: Path, threads: int | None = None) -> list[dict]:
    bundle = load_bundle(model_path, None)
    out = []
    if bundle is None:
        print(f"No trained model at {model_path}; using the default (uncalibrated) ensemble.")
        analyzer = get_analyzer()
        for i, path in enumerate(paths, 1):
            try:
                r = analyzer.analyze(path.read_bytes(), path.name, transcribe=False)
                p = (r["synthetic_likelihood"] or 0) / 100
                out.append({"path": path, "p": p, "type": r.get("manipulation_type") or "none", "threshold": 0.5})
            except Exception as exc:
                print(f"  ! {path.name}: {exc}")
                out.append({"path": path, "p": 0.5, "type": "unknown", "threshold": 0.5})
            if i % 10 == 0:
                print(f"  {i}/{len(paths)}", flush=True)
        return out
    feats = featurize(paths, embeddings=bundle.uses_embeddings, threads=threads, extra=bundle.extra_prefixes,
                      detectors=bundle.data.get("uses_detectors", True))
    for path, f in zip(paths, feats):
        if f is None:
            out.append({"path": path, "p": 0.5, "type": "unknown", "threshold": bundle.threshold})
            continue
        p, kind, _, _ = bundle.predict(f)
        if p < bundle.threshold:
            kind = "none"
        elif kind is None:
            kind = "synthetic_speech"
        out.append({"path": path, "p": p, "type": kind, "threshold": bundle.threshold})
    return out


def cmd_predict(args) -> None:
    paths = list_audio(args.data)[: args.limit or None]
    print(f"Predicting {len(paths)} files ...")
    results = predict_paths(paths, args.model, threads=args.threads)
    if args.format == "tsv":
        # Official HEARSAY format: header "filename<TAB>cm-score", score = P(synthetic) in [0, 1].
        scores = {r["path"].name: min(1.0, max(0.0, r["p"])) for r in results}
        names = sorted(scores)
        if args.template:
            # Follow NSA's score-key template exactly: same files, same order.
            lines = args.template.read_text(encoding="utf-8-sig").splitlines()[1:]
            names = [n for n in (line.split("\t")[0].strip() for line in lines) if n]
            missing = [n for n in names if n not in scores]
            extra = sorted(set(scores) - set(names))
            if missing:
                print(f"WARNING: {len(missing)} template files had no audio/score (written as 0.5), e.g. {missing[:3]}")
            if extra:
                print(f"Note: {len(extra)} scored files are not in the template and were left out, e.g. {extra[:3]}")
        with args.out.open("w", newline="", encoding="utf-8") as f:
            f.write("filename\tcm-score\n")
            for name in names:
                f.write(f"{name}\t{scores.get(name, 0.5):.6f}\n")
    else:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([args.id_col, "synthetic_likelihood", "prediction", "manipulation_type"])
            for r in results:
                name = r["path"].relative_to(args.data).as_posix()
                w.writerow([name, round(r["p"] * 100, 2), "synthetic" if r["p"] >= r["threshold"] else "bonafide",
                            r["type"]])
    print(f"Wrote {len(results)} predictions to {args.out}")


def cmd_extract(args) -> None:
    paths = list_audio(args.data)
    mine = paths[args.shard :: args.shards]
    print(f"Shard {args.shard + 1}/{args.shards}: extracting features for {len(mine)} of {len(paths)} files ...")
    featurize(mine, embeddings=not args.no_embeddings, threads=args.threads,
              extra=args.extra_embeddings or args.only_extra, base=not args.only_extra,
              detectors=not args.no_detectors)


def cmd_evaluate(args) -> None:
    rows = read_labels(args.labels, args.file_col, args.label_col, args.type_col, args.positive_label)
    pairs = [(r, resolve(args.data, r["file"])) for r in rows]
    pairs = [(r, p) for r, p in pairs if p is not None][: args.limit or None]
    results = predict_paths([p for _, p in pairs], args.model)
    y = np.array([r["y"] for r, _ in pairs])
    p = np.array([x["p"] for x in results])
    thr = results[0]["threshold"] if results else 0.5
    m = metrics(y, p, thr)
    print(f"\nHeld-out: n={m['n']}  minDCF {m.get('min_dcf', float('nan')):.4f}  AUC {m.get('auc', float('nan')):.3f}  EER {m.get('eer', float('nan')):.3f}  "
          f"accuracy {m['accuracy']:.3f}  balanced {m['balanced_accuracy']:.3f}  (threshold {thr:.3f})")
    kinds = sorted({r["type"] or "unknown" for r, _ in pairs})
    for kind in kinds:
        idx = [i for i, (r, _) in enumerate(pairs) if (r["type"] or "unknown") == kind]
        acc = np.mean((p[idx] >= thr) == y[idx])
        print(f"    {kind:<22} n={len(idx):<4} correct {acc:.2f}")
    fake_idx = [i for i, (r, _) in enumerate(pairs) if r["y"] == 1 and r["type"] and p[i] >= thr]
    if fake_idx:
        type_acc = np.mean([results[i]["type"] == pairs[i][0]["type"] for i in fake_idx])
        print(f"Manipulation type accuracy (on detected fakes): {type_acc:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def label_args(p):
        p.add_argument("--labels", type=Path, required=True)
        p.add_argument("--file-col")
        p.add_argument("--label-col")
        p.add_argument("--type-col")
        p.add_argument("--positive-label", help="Label value meaning synthetic, if not auto-detectable")

    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    label_args(t)
    t.add_argument("--group-col", help="Column to keep together in CV folds (e.g. speaker) for honest estimates")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--no-embeddings", action="store_true", help="Skip WavLM embeddings (faster, usually less accurate)")
    t.add_argument("--feature-set", choices=list(FEATURE_SETS), default="all",
                   help="'live' trains the fast Live Shield model (no detector ensemble)")
    t.add_argument("--out", type=Path, default=config.HEARSAY_MODEL_PATH)
    t.add_argument("--limit", type=int)
    t.add_argument("--threads", type=int, help="Torch CPU threads")
    t.add_argument("--skip-family-report", action="store_true")
    t.add_argument("--no-extra-embeddings", action="store_true", help="Don't compute/use the extra SSL front-ends")
    t.add_argument("--no-detectors", action="store_true", help="Skip the pretrained detector ensemble (5x faster)")
    t.set_defaults(fn=cmd_train)

    x = sub.add_parser("extract", help="Pre-compute cached features (run shards in parallel)")
    x.add_argument("--data", type=Path, required=True)
    x.add_argument("--shard", type=int, default=0)
    x.add_argument("--shards", type=int, default=1)
    x.add_argument("--threads", type=int)
    x.add_argument("--no-embeddings", action="store_true")
    x.add_argument("--extra-embeddings", action="store_true", help="Also compute the extra SSL front-ends (XLS-R)")
    x.add_argument("--only-extra", action="store_true", help="Compute only the extra SSL front-ends")
    x.add_argument("--no-detectors", action="store_true", help="Skip the pretrained detector ensemble (5x faster)")
    x.set_defaults(fn=cmd_extract)

    p = sub.add_parser("predict")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("predictions.tsv"))
    p.add_argument("--format", choices=["tsv", "csv"], default="tsv", help="tsv = official HEARSAY submission format")
    p.add_argument("--template", type=Path, help="NSA score-key TSV: output follows its file list and order")
    p.add_argument("--threads", type=int)
    p.add_argument("--model", type=Path, default=config.HEARSAY_MODEL_PATH)
    p.add_argument("--id-col", default="filename", help="Header for the file column")
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_predict)

    e = sub.add_parser("evaluate")
    e.add_argument("--data", type=Path, required=True)
    label_args(e)
    e.add_argument("--model", type=Path, default=config.HEARSAY_MODEL_PATH)
    e.add_argument("--limit", type=int)
    e.set_defaults(fn=cmd_evaluate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
