"""HEARSAY: The Audio Authentication Challenge — train, predict (submission CSV), evaluate.

  # 1. Train the fusion model on the labeled training set (any audio formats)
  .venv\\Scripts\\python scripts\\hearsay.py train --data path\\to\\train --labels path\\to\\train_labels.csv

  # 2. Predict the held-out test set -> submission CSV
  .venv\\Scripts\\python scripts\\hearsay.py predict --data path\\to\\test --out predictions.csv

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
FEATURE_SETS = {
    "all": lambda n: True,
    "no-embeddings": lambda n: not n.startswith("emb_"),
    # Fast enough for Live Shield: no detector ensemble, just embeddings + signal features.
    "live": lambda n: not n.startswith("det_"),
}


def _signature(embeddings: bool) -> dict:
    return {"detectors": config.FORENSICS_DETECTORS, "embedding": config.FORENSICS_EMBEDDING_MODEL if embeddings else None,
            "window": config.FORENSICS_WINDOW_SECONDS, "hop": config.FORENSICS_HOP_SECONDS}


def featurize(paths: list[Path], embeddings: bool = True) -> list[dict | None]:
    analyzer = get_analyzer()
    cache = FeatureCache(CACHE_DIR, _signature(embeddings))
    out: list[dict | None] = []
    started = time.time()
    fresh = 0
    for i, path in enumerate(paths, 1):
        data = path.read_bytes()
        feats = cache.get(data)
        if feats is None:
            try:
                feats, _, _ = analyzer.extract(decode_any(data, path.name), embeddings=embeddings)
                cache.put(data, feats)
                fresh += 1
            except Exception as exc:
                print(f"  ! {path.name}: {exc}")
                feats = None
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
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))  # noqa: E731
    hgb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,  # noqa: E731
                                                 l2_regularization=1.0, class_weight="balanced", random_state=0)
    embeddings = [i for i, n in enumerate(names) if n.startswith("emb_")]
    candidates = {
        "detectors_only (baseline)": lambda: SubsetBlend([(detectors, lr())]),
        "logreg_embeddings_only": lambda: SubsetBlend([(embeddings, lr())]),
        "logreg_all_features": lambda: SubsetBlend([(every, lr())]),
        "gboost_signal+detectors": lambda: SubsetBlend([(no_emb, hgb())]),
        "blend(logreg_all, gboost)": lambda: SubsetBlend([(every, lr()), (no_emb, hgb())]),
    }
    return {k: v for k, v in candidates.items()
            if (detectors or "detectors" not in k) and (embeddings or "embeddings" not in k)}


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
    feats = featurize(list(paths), embeddings=not args.no_embeddings)
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
        print(f"  {name:<28} AUC {m.get('auc', float('nan')):.3f}  EER {m.get('eer', float('nan')):.3f}  "
              f"balanced acc {m['balanced_accuracy']:.3f}")
    best = max((k for k in results if "baseline" not in k), key=lambda k: (results[k][0].get("auc", 0), -results[k][0].get("eer", 1)))
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
    print("Cross-validated accuracy per class:")
    for kind, acc in per_type.items():
        print(f"    {kind:<22} {acc:.2f}")

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
              "per_class_accuracy": per_type, "type_model": type_metrics, "n_train": len(rows)}
    args.out.with_name(args.out.stem + "_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved {args.out} (+ metrics JSON)")


def predict_paths(paths: list[Path], model_path: Path) -> list[dict]:
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
    feats = featurize(paths, embeddings=bundle.uses_embeddings)
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
    results = predict_paths(paths, args.model)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([args.id_col, "synthetic_likelihood", "prediction", "manipulation_type"])
        for r in results:
            name = r["path"].relative_to(args.data).as_posix()
            score = round(r["p"] * 100, 2) if args.scale == 100 else round(r["p"], 4)
            w.writerow([name, score, "synthetic" if r["p"] >= r["threshold"] else "bonafide", r["type"]])
    print(f"Wrote {args.out}")


def cmd_evaluate(args) -> None:
    rows = read_labels(args.labels, args.file_col, args.label_col, args.type_col, args.positive_label)
    pairs = [(r, resolve(args.data, r["file"])) for r in rows]
    pairs = [(r, p) for r, p in pairs if p is not None][: args.limit or None]
    results = predict_paths([p for _, p in pairs], args.model)
    y = np.array([r["y"] for r, _ in pairs])
    p = np.array([x["p"] for x in results])
    thr = results[0]["threshold"] if results else 0.5
    m = metrics(y, p, thr)
    print(f"\nHeld-out: n={m['n']}  AUC {m.get('auc', float('nan')):.3f}  EER {m.get('eer', float('nan')):.3f}  "
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
    t.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("predictions.csv"))
    p.add_argument("--model", type=Path, default=config.HEARSAY_MODEL_PATH)
    p.add_argument("--id-col", default="filename", help="Header for the file column")
    p.add_argument("--scale", type=int, choices=[1, 100], default=100, help="100 = percent (default), 1 = probability")
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
