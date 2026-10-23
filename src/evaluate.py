"""TAHAP 5 KDD - Evaluasi & interpretasi pada data uji."""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    balanced_accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

import config as C
from src.train_classic import ordered_proba

COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _plot_confusion(y_true, y_pred, name, out_path, normalize=None, title_suffix=""):
    cm = confusion_matrix(y_true, y_pred, labels=C.CLASSES, normalize=normalize)
    disp = ConfusionMatrixDisplay(cm, display_labels=C.CLASSES)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp.plot(cmap="Blues", ax=ax, values_format=".2f" if normalize else "d")
    ax.set_title(f"Confusion Matrix - {name}{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def _plot_roc_pr(y_true, proba, name):
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(C.CLASSES):
        y_bin = (np.asarray(y_true) == c).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
        auc = roc_auc_score(y_bin, proba[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC={auc:.3f})", color=COLORS[i])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (OvR) - {name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(C.CLASSES):
        y_bin = (np.asarray(y_true) == c).astype(int)
        if y_bin.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin, proba[:, i])
        ap = average_precision_score(y_bin, proba[:, i])
        ax.plot(rec, prec, label=f"{c} (AP={ap:.3f})", color=COLORS[i])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {name}")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)


def _plot_comparison():
    path = C.TAB_DIR / "perbandingan_model.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    pivot = df.pivot(index="model", columns="fitur",
                     values="cv_repeat_f1_macro_mean")
    ax = pivot.plot(kind="bar", figsize=(11, 5), rot=20)
    ax.set_ylabel("CV F1-macro (rerata)")
    ax.set_title("Perbandingan model x skenario fitur")
    ax.legend(title="Skenario fitur", fontsize=8)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "perbandingan_model.png", dpi=150)
    plt.close()


def _plot_learning_curve(model, X, y, name):
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring="f1_macro",
        train_sizes=np.linspace(0.4, 1.0, 4), n_jobs=C.N_JOBS,
        error_score=np.nan)
    tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    va_mean, va_std = val_scores.mean(axis=1), val_scores.std(axis=1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, tr_mean, "o-", label="Train")
    plt.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15)
    plt.plot(sizes, va_mean, "o-", label="Validasi silang")
    plt.fill_between(sizes, va_mean - va_std, va_mean + va_std, alpha=0.15)
    plt.xlabel("Jumlah sampel latih")
    plt.ylabel("F1-macro")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "learning_curve.png", dpi=150)
    plt.close()


def run_evaluation(verbose=True):
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(C.MODELS_DIR / "best_model.joblib")
    model, cols = bundle["model"], bundle["columns"]
    name = bundle["name"]
    feature_set = bundle["feature_set"]

    df = pd.read_csv(C.PROCESSED_DIR / "features.csv")
    te = df[df["split"] == "test"].reset_index(drop=True)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    X_te, y_te = te[cols], te["label"]
    pred = model.predict(X_te)
    proba = ordered_proba(model, X_te)

    report = classification_report(
        y_te, pred, labels=C.CLASSES, target_names=C.CLASSES,
        digits=3, zero_division=0)
    (C.TAB_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    if verbose:
        print(report)

    metrics = {
        "model": name,
        "feature_set": feature_set,
        "n_features": len(cols),
        "n_test": int(len(te)),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_te, pred), 4),
        "precision_macro": round(precision_score(
            y_te, pred, labels=C.CLASSES, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(
            y_te, pred, labels=C.CLASSES, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(
            y_te, pred, labels=C.CLASSES, average="macro", zero_division=0), 4),
        "cohen_kappa": round(cohen_kappa_score(y_te, pred), 4),
    }
    for c in C.CLASSES:
        metrics[f"recall_{c}"] = round(recall_score(
            y_te, pred, labels=[c], average="macro", zero_division=0), 4)
    if proba is not None:
        try:
            labels_sorted = sorted(C.CLASSES)
            order = [C.CLASSES.index(c) for c in labels_sorted]
            metrics["roc_auc_ovr_macro"] = round(roc_auc_score(
                y_te, proba[:, order], multi_class="ovr", average="macro",
                labels=labels_sorted), 4)
        except ValueError:
            metrics["roc_auc_ovr_macro"] = None
        try:
            metrics["log_loss"] = round(
                log_loss(y_te, proba[:, order], labels=labels_sorted), 4)
        except ValueError:
            metrics["log_loss"] = None
    with open(C.TAB_DIR / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    cm = _plot_confusion(y_te, pred, name, C.FIG_DIR / "confusion_matrix.png")
    _plot_confusion(y_te, pred, name, C.FIG_DIR / "confusion_matrix_normalized.png",
                    normalize="true", title_suffix=" (ternormalisasi)")
    pd.DataFrame(cm, index=C.CLASSES, columns=C.CLASSES).to_csv(
        C.TAB_DIR / "confusion_matrix.csv")

    if proba is not None:
        _plot_roc_pr(y_te, proba, name)

    out = te[["file", "label"]].copy()
    out["prediksi"] = pred
    if proba is not None:
        for i, c in enumerate(C.CLASSES):
            out[f"proba_{c}"] = np.round(proba[:, i], 4)
    out.to_csv(C.TAB_DIR / "test_predictions.csv", index=False)

    salah = pred != y_te.to_numpy()
    err = out.loc[salah, ["file", "label", "prediksi"]].copy()
    if proba is not None:
        err["proba_prediksi"] = np.round(proba[salah].max(axis=1), 4)
    err.to_csv(C.TAB_DIR / "salah_prediksi.csv", index=False)

    _plot_comparison()
    _plot_learning_curve(model, tr[cols], tr["label"],
                         f"{name} ({feature_set})")

    if verbose:
        print(f"{len(err)} citra salah prediksi -> reports/tables/salah_prediksi.csv")
        print("Metrik test:", metrics)
    return metrics, cm, out


if __name__ == "__main__":
    run_evaluation()
