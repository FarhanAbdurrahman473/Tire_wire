"""TAHAP 6 KDD - Penerapan: prediksi 1 foto ban baru.

Pakai: python -m src.predict path/foto.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import pandas as pd

import config as C
from src.features import extract_all
from src.preprocessing import preprocess_gray

PESAN = {
    "baik": "Ban masih layak pakai.",
    "sedang": "Ban mulai aus - pantau dan rencanakan penggantian.",
    "aus": "Ban sudah aus/gundul - SEGERA GANTI demi keselamatan.",
}


def predict_image(path, bundle=None):
    bundle = bundle or joblib.load(C.MODELS_DIR / "best_model.joblib")
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"File gambar tidak terbaca: {path}")

    gray = preprocess_gray(img, filter_method=bundle.get("filter_method", "gaussian"))
    names, vec = extract_all(gray)
    X = pd.DataFrame([vec], columns=names)[bundle["columns"]]
    label = bundle["model"].predict(X)[0]

    result = {"label": label, "pesan": PESAN.get(label, ""), "proba": {}}
    if hasattr(bundle["model"], "predict_proba"):
        proba = bundle["model"].predict_proba(X)[0]
        classes = list(bundle["model"].classes_)
        result["proba"] = {c: float(proba[classes.index(c)]) for c in C.CLASSES}
    return result


def main():
    ap = argparse.ArgumentParser(description="Prediksi tingkat keausan ban dari satu foto.")
    ap.add_argument("image", help="path ke foto ban")
    args = ap.parse_args()

    if not Path(args.image).exists():
        raise SystemExit(f"File tidak ditemukan: {args.image}")

    r = predict_image(args.image)
    print("Prediksi:", r["label"].upper(), "-", r["pesan"])
    for c, p in r["proba"].items():
        print(f"  {c:7s}: {p:.1%}")


if __name__ == "__main__":
    main()
