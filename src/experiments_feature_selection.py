"""Eksperimen opsional - pengaruh seleksi fitur pada model terbaik.

Tidak mengubah hasil utama. Jalankan setelah src.train_classic.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from src.train_classic import compute_metrics, ordered_proba


def make_model(bundle, selector=None):
    if selector is None:
        return clone(bundle["model"]).set_params(**bundle["best_params"])

    clf = clone(bundle["model"].named_steps["clf"])
    params = {k.split("__", 1)[1]: v for k, v in bundle["best_params"].items()
              if k.startswith("clf__")}
    clf.set_params(**params)
    return Pipeline([("scaler", StandardScaler()), ("selector", selector), ("clf", clf)])


def run_feature_selection_experiment(verbose=True):
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(C.MODELS_DIR / "best_model.joblib")
    df = pd.read_csv(C.PROCESSED_DIR / "features.csv")
    cols = bundle["columns"]
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    candidates = [("none", None)]
    candidates += [
        ("variance_1e-4", VarianceThreshold(threshold=1e-4)),
        ("selectk_50", SelectKBest(f_classif, k=min(50, len(cols)))),
        ("selectk_100", SelectKBest(f_classif, k=min(100, len(cols)))),
        ("selectk_200", SelectKBest(f_classif, k=min(200, len(cols)))),
    ]
    n_comp = min(tr[cols].shape[0], len(cols))
    for k in (20, 50, 100):
        if k < n_comp:
            candidates.append((f"pca_{k}", PCA(n_components=k, random_state=C.RANDOM_STATE)))

    rows = []
    for name, selector in candidates:
        if verbose:
            print(f"[seleksi] {name} ...", flush=True)
        model = make_model(bundle, selector)
        model.fit(tr[cols], tr["label"])

        cv_scores = cross_val_score(
            model, tr[cols], tr["label"], cv=cv,
            scoring="f1_macro", n_jobs=C.N_JOBS)
        pred = model.predict(te[cols])
        proba = ordered_proba(model, te[cols])
        metrics = compute_metrics(te["label"], pred, proba)

        if selector is None:
            n_out = len(cols)
        else:
            fitted = model.named_steps["selector"]
            n_out = int(getattr(fitted, "n_components_", None)
                        or getattr(fitted, "get_support")().sum())
        rows.append({
            "metode": name,
            "model": bundle["name"],
            "fitur": bundle["feature_set"],
            "n_features_awal": len(cols),
            "n_features_hasil": n_out,
            "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_macro_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(accuracy_score(te["label"], pred), 4),
            "test_f1_macro": round(f1_score(
                te["label"], pred, labels=C.CLASSES,
                average="macro", zero_division=0), 4),
            "test_recall_aus": metrics["recall_aus"],
        })

    out = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False)
    out.to_csv(C.TAB_DIR / "eksperimen_seleksi_fitur.csv", index=False)
    if verbose:
        print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run_feature_selection_experiment()
