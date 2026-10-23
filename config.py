"""Konfigurasi terpusat. Ubah parameter di sini, bukan di tiap skrip/notebook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"            # foto asli (salinan), 1 sub-folder per kelas
INTERIM_DIR = ROOT / "data" / "interim"    # hasil preprocessing
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"

CLASSES = ["baik", "sedang", "aus"]        # urutan = label 0,1,2
IMG_SIZE = 256                             # sisi gambar setelah resize (px)
RANDOM_STATE = 42
TEST_SIZE = 0.20

BLUR_THRESHOLD = 50.0                      # variance of Laplacian; di bawah ini = buram
GROUP_BY_TIRE_ID = False                   # True jika nama file "ID-BAN_nomor.jpg"
                                           # (cegah foto ban yang sama masuk train & test)

# Parameter LBP (P titik sampel, R radius) -> multi-skala
LBP_MS = [(8, 1), (16, 2), (24, 3)]

# Filter citra setelah CLAHE: "gaussian", "median", "bilateral", "unsharp"
FILTER_METHOD = "gaussian"

# Validasi silang
CV_FOLDS = 5
CV_REPEATS = 3
N_JOBS = -1
