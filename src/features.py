"""TAHAP 3 KDD - Transformation: ekstraksi fitur (LBP, arah garis, GLCM, tepi)."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

import config as C

LBP256_COLS = [f"lbp256_{i}" for i in range(256)]
LBPMS_COLS = [f"lbpms_{i}" for i in range(sum(p + 2 for p, _ in C.LBP_MS))]
DIR_COLS = ["dir_edge", "dir_vert", "dir_diagr", "dir_horiz", "dir_diagl"]
GLCM_COLS = ["glcm_contrast", "glcm_homog", "glcm_energy", "glcm_corr"]
ALL_FEATURE_COLS = LBP256_COLS + LBPMS_COLS + DIR_COLS + GLCM_COLS


def lbp256(gray):
    """LBP dasar 3x3 (P=8,R=1), histogram 256-bin -> seperti jurnal (Pers. 1 & 2)."""
    lbp = local_binary_pattern(gray, 8, 1, method="default")
    h, _ = np.histogram(lbp, bins=256, range=(0, 256))
    return h / h.sum()


def lbp_multiscale(gray):
    """LBP uniform melingkar (P,R) berbeda -> menangkap alur skala besar."""
    feats = []
    for P, R in C.LBP_MS:
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2))
        feats.append(h / h.sum())
    return np.concatenate(feats)


def direction_stats(gray):
    """Analogi kode arah jurnal (2=vertikal, 3=diagonal kanan, 4=horizontal, 5=diagonal kiri).
    K(n) = jumlah piksel tepi berarah n / jumlah piksel total (Pers. 3)."""
    edges = cv2.Canny(gray, 60, 150) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180)[edges]
    n = gray.size
    vert = np.sum((ang < 22.5) | (ang >= 157.5))
    diag_r = np.sum((ang >= 22.5) & (ang < 67.5))
    horiz = np.sum((ang >= 67.5) & (ang < 112.5))
    diag_l = np.sum((ang >= 112.5) & (ang < 157.5))
    return np.array([edges.sum() / n, vert / n, diag_r / n, horiz / n, diag_l / n])


def glcm_stats(gray):
    g = (gray // 4).astype(np.uint8)
    m = graycomatrix(g, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=64, symmetric=True, normed=True)
    return np.array([graycoprops(m, p).mean()
                     for p in ("contrast", "homogeneity", "energy", "correlation")])


def extract_all(gray):
    """Satu vektor fitur; nama kolom berawalan grup (lbp256_, lbpms_, dir_, glcm_)."""
    a = lbp256(gray)
    b = lbp_multiscale(gray)
    c = direction_stats(gray)
    d = glcm_stats(gray)
    names = (LBP256_COLS + LBPMS_COLS + DIR_COLS + GLCM_COLS)
    return names, np.concatenate([a, b, c, d])


FEATURE_SETS = {
    "A_LBP256 (jurnal)": ["lbp256_"],
    "B_LBP multi-skala": ["lbpms_"],
    "C_LBP-MS + arah": ["lbpms_", "dir_"],
    "D_Lengkap": ["lbpms_", "dir_", "glcm_"],
}


def select_columns(df, prefixes):
    return [c for c in df.columns if any(str(c).startswith(p) for p in prefixes)]


def build_feature_frame(interim_dir=None, splits=None, filter_method="gaussian"):
    """Bangun features.csv dari citra hasil preprocessing.

    interim_dir : folder data/interim (default C.INTERIM_DIR)
    splits      : DataFrame splits.csv (default dibaca dari processed)
    """
    interim_dir = Path(interim_dir) if interim_dir is not None else C.INTERIM_DIR
    if splits is None:
        splits = pd.read_csv(C.PROCESSED_DIR / "splits.csv")

    rows = []
    for i, (_, r) in enumerate(splits.iterrows()):
        name = Path(r["path"]).stem + ".png"
        gray = cv2.imread(str(interim_dir / "gray" / r["split"] / r["label"] / name), 0)
        if gray is None:
            print("gagal baca:", r["path"])
            continue
        names, vec = extract_all(gray)
        rows.append([r["split"], r["label"], name, *vec])
        if (i + 1) % 100 == 0:
            print(f"... {i + 1}/{len(splits)} foto")

    return pd.DataFrame(rows, columns=["split", "label", "file", *names])


def run_feature_extraction():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = build_feature_frame()
    out.to_csv(C.PROCESSED_DIR / "features.csv", index=False)
    print("features.csv:", out.shape)
    return out


if __name__ == "__main__":
    run_feature_extraction()
