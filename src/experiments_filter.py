"""Eksperimen opsional - pengaruh filter citra pada model terbaik.

Tidak mengubah hasil utama. Jalankan setelah src.train_classic.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config as C
from src.features import extract_all
from src.preprocessing import FILTER_METHODS, preprocess_gray
from src.train_classic import compute_metrics, ordered_proba


def build_features_from_raw(splits, filter_method):
    rows = []
    for _, r in splits.iterrows():
        img = cv2.imread(r["path"])
        if img is None:
            continue
        gray = preprocess_gray(img, filter_method=filter_method)
        names, vec = extract_all(gray)
        rows.append([r["split"], r["label"], Path(r["path"]).stem + ".png", *vec])
    return pd.DataFrame(rows, columns=["split", "label", "file", *names])


def run_filter_experiment(verbose=True):
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(C.MODELS_DIR / "best_model.joblib")
    splits = pd.read_csv(C.PROCESSED_DIR / "splits.csv")
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    rows = []
    for method in FILTER_METHODS:
        if verbose:
            print(f"[filter] {method} ...", flush=True)
        feats = build_features_from_raw(splits, method)
        cols = bundle["columns"]
        tr = feats[feats["split"] == "train"]
        te = feats[feats["split"] == "test"]

        model = clone(bundle["model"])
        model.set_params(**bundle["best_params"])
        model.fit(tr[cols], tr["label"])

        cv_scores = cross_val_score(
            model, tr[cols], tr["label"], cv=cv,
            scoring="f1_macro", n_jobs=C.N_JOBS)
        pred = model.predict(te[cols])
        proba = ordered_proba(model, te[cols])
        metrics = compute_metrics(te["label"], pred, proba)

        rows.append({
            "filter": method,
            "model": bundle["name"],
            "fitur": bundle["feature_set"],
            "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_macro_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(accuracy_score(te["label"], pred), 4),
            "test_f1_macro": round(f1_score(
                te["label"], pred, labels=C.CLASSES,
                average="macro", zero_division=0), 4),
            "test_recall_aus": metrics["recall_aus"],
        })

    out = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False)
    out.to_csv(C.TAB_DIR / "eksperimen_filter.csv", index=False)
    if verbose:
        print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run_filter_experiment()
