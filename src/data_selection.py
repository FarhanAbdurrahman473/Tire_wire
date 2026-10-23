"""TAHAP 1 KDD - Data Selection: audit dataset + pembagian train/test."""
from pathlib import Path

import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import config as C

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit(raw_dir=None):
    """Audit setiap foto: terbaca, duplikat MD5, dan skor blur."""
    raw_dir = Path(raw_dir) if raw_dir is not None else C.RAW_DIR
    rows = []
    for label in C.CLASSES:
        folder = raw_dir / label
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() not in EXT:
                continue
            img = cv2.imread(str(p))
            if img is None:
                rows.append(dict(path=str(p), label=label, valid=False, reason="tidak terbaca"))
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            md5 = __import__("hashlib").md5(p.read_bytes()).hexdigest()
            rows.append(dict(path=str(p), label=label, width=w, height=h,
                             blur_score=round(float(blur), 2), md5=md5,
                             valid=True, reason=""))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    dup = df["valid"] & df.duplicated("md5", keep="first")
    df.loc[dup, "reason"] = "duplikat"
    df.loc[dup, "valid"] = False

    blur = df["valid"] & (df["blur_score"] < C.BLUR_THRESHOLD)
    df.loc[blur, "reason"] = "buram"
    df.loc[blur, "valid"] = False
    return df


def split(df):
    """Bagi data valid menjadi train/test (stratified, opsional group by id ban)."""
    v = df[df["valid"]].copy()
    if v.empty:
        return v

    if C.GROUP_BY_TIRE_ID:
        v["group"] = v["path"].map(lambda s: Path(s).name.split("_")[0])
        gss = GroupShuffleSplit(1, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
        _, te = next(gss.split(v, v["label"], v["group"]))
        v["split"] = "train"
        v.iloc[te, v.columns.get_loc("split")] = "test"
    else:
        _, te = train_test_split(
            v.index, test_size=C.TEST_SIZE, stratify=v["label"],
            random_state=C.RANDOM_STATE)
        v["split"] = "train"
        v.loc[te, "split"] = "test"
    return v


def run_audit_and_split(verbose=True):
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = audit()
    df.to_csv(C.PROCESSED_DIR / "audit.csv", index=False)

    sp = split(df)
    sp[["path", "label", "split"]].to_csv(C.PROCESSED_DIR / "splits.csv", index=False)

    if verbose:
        print("Ringkasan audit:\n", df.groupby(["label", "valid"]).size().unstack(fill_value=0))
        print("Alasan dibuang:\n", df.loc[~df["valid"], "reason"].value_counts())
        print("\nDistribusi split:\n",
              sp.groupby(["split", "label"]).size().unstack(fill_value=0))

    if not sp.empty:
        ax = sp.groupby(["label", "split"]).size().unstack().loc[C.CLASSES].plot(kind="bar")
        ax.set_title("Distribusi data per kelas")
        ax.set_ylabel("jumlah citra")
        ax.set_xticklabels(C.CLASSES, rotation=0)
        plt.tight_layout()
        plt.savefig(C.FIG_DIR / "distribusi_kelas.png", dpi=150)
        plt.close()

    return df, sp


if __name__ == "__main__":
    run_audit_and_split()
