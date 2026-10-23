"""TAHAP 2 KDD - Preprocessing: crop-resize, grayscale, CLAHE, filter."""
from pathlib import Path

import cv2
import pandas as pd

import config as C

FILTER_METHODS = ("gaussian", "median", "bilateral", "unsharp")


def center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def preprocess_rgb(img_bgr):
    """Crop tengah (persegi) lalu resize. Dipakai juga untuk jalur CNN."""
    return cv2.resize(center_square(img_bgr), (C.IMG_SIZE, C.IMG_SIZE),
                      interpolation=cv2.INTER_AREA)


def apply_filter(gray, method="gaussian"):
    """Filter peredam noise. Default gaussian sama seperti panduan."""
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (3, 3), 0)
    if method == "median":
        return cv2.medianBlur(gray, 3)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, 5, 50, 50)
    if method == "unsharp":
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    raise ValueError(f"Filter tidak dikenal: {method}. Pilih salah satu {FILTER_METHODS}")


def preprocess_gray(img_bgr, filter_method="gaussian"):
    """Pipeline untuk machine learning klasik."""
    sq = preprocess_rgb(img_bgr)
    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return apply_filter(gray, filter_method)


def run_preprocessing(verbose=True):
    df = pd.read_csv(C.PROCESSED_DIR / "splits.csv")
    gagal = 0
    for _, r in df.iterrows():
        img = cv2.imread(r["path"])
        if img is None:
            gagal += 1
            if verbose:
                print("gagal baca:", r["path"])
            continue
        name = Path(r["path"]).stem + ".png"
        for kind, out in (("gray", preprocess_gray(img)),
                          ("rgb", preprocess_rgb(img))):
            d = C.INTERIM_DIR / kind / r["split"] / r["label"]
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / name), out)

    if verbose:
        print("Preprocessing selesai ->", C.INTERIM_DIR)
        print("foto gagal dibaca:", gagal)
        for kind in ("gray", "rgb"):
            n = len(list((C.INTERIM_DIR / kind).rglob("*.png")))
            print(f"  {kind}: {n} file  (splits.csv: {len(df)} baris)")
    return df


if __name__ == "__main__":
    run_preprocessing()
