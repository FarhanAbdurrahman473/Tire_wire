"""TAHAP 4 KDD - Data Mining: 5 model klasik + GridSearchCV + studi ablasi fitur."""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config as C
from src.features import FEATURE_SETS, select_columns

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*probability.*")

MODELS = {
    "KNN": (KNeighborsClassifier(metric="euclidean"),
            {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]}),
    "SVM": (SVC(probability=True, random_state=C.RANDOM_STATE),
            {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001],
             "clf__kernel": ["rbf"]}),
    "RandomForest": (
        RandomForestClassifier(random_state=C.RANDOM_STATE, class_weight="balanced"),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 20]}),
    "ExtraTrees": (
        ExtraTreesClassifier(random_state=C.RANDOM_STATE, class_weight="balanced"),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 20]}),
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=C.RANDOM_STATE),
        {"clf__C": [0.1, 1, 10]}),
}


def ordered_proba(model, X):
    """Probabilitas dengan urutan kolom sesuai C.CLASSES."""
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    idx = [classes.index(c) for c in C.CLASSES]
    return proba[:, idx]


def compute_metrics(y_true, y_pred, proba=None):
    labels = C.CLASSES
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=labels,
                                           average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels,
                                     average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels,
                             average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels,
                                average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    for c in labels:
        out[f"precision_{c}"] = precision_score(
            y_true, y_pred, labels=[c], average="macro", zero_division=0)
        out[f"recall_{c}"] = recall_score(
            y_true, y_pred, labels=[c], average="macro", zero_division=0)
        out[f"f1_{c}"] = f1_score(
            y_true, y_pred, labels=[c], average="macro", zero_division=0)

    if proba is not None:
        try:
            labels_sorted = sorted(labels)
            order = [labels.index(c) for c in labels_sorted]
            out["roc_auc_ovr_macro"] = roc_auc_score(
                y_true, proba[:, order], multi_class="ovr",
                average="macro", labels=labels_sorted)
        except ValueError:
            out["roc_auc_ovr_macro"] = np.nan
        try:
            out["log_loss"] = log_loss(
                y_true, proba[:, order], labels=labels_sorted)
        except ValueError:
            out["log_loss"] = np.nan
    else:
        out["roc_auc_ovr_macro"] = np.nan
        out["log_loss"] = np.nan
    return {k: (round(float(v), 4) if np.isfinite(v) else np.nan) for k, v in out.items()}


def _feature_importance_df(model, columns, X_test, y_test, feature_set, model_name):
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        kind = "builtin"
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        values = np.abs(coef).mean(axis=0)
        kind = "coef_abs_mean"
    else:
        try:
            r = permutation_importance(
                model, X_test, y_test, scoring="f1_macro",
                n_repeats=3, random_state=C.RANDOM_STATE, n_jobs=C.N_JOBS)
            values = r.importances_mean
            kind = "permutation_test"
        except Exception as exc:  # pragma: no cover - proteksi tambahan
            print("feature importance dilewati:", exc)
            return None
    df = pd.DataFrame({"feature": columns, "importance": values})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["feature_set"] = feature_set
    df["model"] = model_name
    df["importance_type"] = kind
    return df


def run_training(verbose=True):
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(C.PROCESSED_DIR / "features.csv")
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    if tr.empty or te.empty:
        raise RuntimeError("features.csv tidak memuat split train/test yang lengkap.")

    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    rep_cv = RepeatedStratifiedKFold(
        n_splits=C.CV_FOLDS, n_repeats=C.CV_REPEATS, random_state=C.RANDOM_STATE)

    results = []
    best = None
    best_columns = None
    best_X_test = None

    for fs_name, prefixes in FEATURE_SETS.items():
        cols = select_columns(df, prefixes)
        X_tr, y_tr = tr[cols], tr["label"]
        X_te, y_te = te[cols], te["label"]

        for m_name, (est, grid) in MODELS.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(est))])
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=C.N_JOBS)
            gs.fit(X_tr, y_tr)

            model = gs.best_estimator_
            pred = model.predict(X_te)
            proba = ordered_proba(model, X_te)
            metrics = compute_metrics(y_te, pred, proba)

            rep_scores = pd.Series(cross_val_score(
                gs.best_estimator_, X_tr, y_tr, cv=rep_cv,
                scoring="f1_macro", n_jobs=C.N_JOBS))

            row = {
                "fitur": fs_name,
                "model": m_name,
                "n_features": len(cols),
                "cv_grid_f1_macro": round(float(gs.best_score_), 4),
                "cv_repeat_f1_macro_mean": round(float(rep_scores.mean()), 4),
                "cv_repeat_f1_macro_std": round(float(rep_scores.std()), 4),
                **{f"test_{k}": v for k, v in metrics.items()},
                "best_params": str(gs.best_params_),
            }
            results.append(row)
            if verbose:
                print(f"[{m_name:18s} | {fs_name:18s}] "
                      f"CV F1={row['cv_repeat_f1_macro_mean']:.4f} "
                      f"test F1={row['test_f1_macro']:.4f} "
                      f"recall aus={row['test_recall_aus']:.4f}", flush=True)

            if best is None or gs.best_score_ > best["cv_grid_f1_macro"]:
                best = {
                    "cv_grid_f1_macro": float(gs.best_score_),
                    "model": model,
                    "columns": cols,
                    "feature_set": fs_name,
                    "name": m_name,
                    "best_params": gs.best_params_,
                    "X_test": X_te,
                }
                best_columns = cols
                best_X_test = X_te

    res = pd.DataFrame(results).sort_values(
        ["cv_repeat_f1_macro_mean", "cv_grid_f1_macro"], ascending=False)
    res.to_csv(C.TAB_DIR / "perbandingan_model.csv", index=False)
    res.to_csv(C.TAB_DIR / "cv_results.csv", index=False)

    bundle = {
        "model": best["model"],
        "columns": best_columns,
        "feature_set": best["feature_set"],
        "name": best["name"],
        "classes": C.CLASSES,
        "best_params": best["best_params"],
        "filter_method": C.FILTER_METHOD,
    }
    joblib.dump(bundle, C.MODELS_DIR / "best_model.joblib")

    info = {
        "model": best["name"],
        "feature_set": best["feature_set"],
        "n_features": len(best_columns),
        "cv_grid_f1_macro": round(best["cv_grid_f1_macro"], 4),
        "best_params": {k: str(v) for k, v in best["best_params"].items()},
        "filter_method": C.FILTER_METHOD,
        "random_state": C.RANDOM_STATE,
    }
    with open(C.MODELS_DIR / "best_model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    imp = _feature_importance_df(
        best["model"], best_columns, best_X_test,
        te["label"], best["feature_set"], best["name"])
    if imp is not None:
        imp.to_csv(C.TAB_DIR / "feature_importance.csv", index=False)
        imp.head(20).to_csv(C.TAB_DIR / "feature_importance_top20.csv", index=False)

    if verbose:
        print("\nModel terbaik (berdasar CV):", best["name"], "+", best["feature_set"])
        print("Detail:", info)
    return res, bundle


if __name__ == "__main__":
    run_training()
