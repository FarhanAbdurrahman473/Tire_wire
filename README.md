# Klasifikasi Tingkat Keausan Ban Motor

Klasifikasi citra tapak ban menjadi 3 kelas: **baik**, **sedang**, **aus**.
Jalur utama memakai machine learning klasik dengan fitur tekstur LBP, arah tepi, dan GLCM; model yang dibandingkan: KNN, SVM, Random Forest, Extra Trees, dan Logistic Regression.

## Struktur

```text
Ban_Motor/
|-- config.py                 # parameter terpusat
|-- requirements.txt
|-- requirements-lock.txt
|-- data/
|   |-- raw/{baik,sedang,aus} # foto asli
|   |-- interim/              # [otomatis] hasil preprocessing
|   `-- processed/            # [otomatis] audit.csv, splits.csv, features.csv
|-- src/
|   |-- data_selection.py     # Tahap 1: audit + split
|   |-- preprocessing.py      # Tahap 2: crop, resize, grayscale, CLAHE, filter
|   |-- features.py           # Tahap 3: LBP, arah, GLCM
|   |-- train_classic.py      # Tahap 4: 5 model x 4 skenario fitur
|   |-- evaluate.py           # Tahap 5: metrik, confusion matrix, ROC/PR
|   |-- predict.py            # Tahap 6: prediksi 1 foto
|   |-- experiments_filter.py           # opsional: perbandingan filter citra
|   `-- experiments_feature_selection.py# opsional: seleksi fitur / PCA
|-- models/                   # [otomatis] best_model.joblib
|-- reports/
|   |-- figures/
|   `-- tables/
`-- notebooks/                # 01-06 dokumentasi eksplorasi
```

## Menjalankan

Dari folder induk proyek:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_selection
python -m src.preprocessing
python -m src.features
python -m src.train_classic
python -m src.evaluate
python -m src.predict "data/raw/baik/ban_bagus_1.jpg"
```

Eksperimen opsional (setelah training):

```powershell
python -m src.experiments_filter
python -m src.experiments_feature_selection
```

## Hasil singkat

Model terbaik: **SVM RBF + D_Lengkap (63 fitur)**. CV F1-macro berulang 0.7474 ± 0.0365; akurasi test 0.7358; F1-macro test 0.7091; recall kelas aus 0.9048. Ringkasan lengkap: `reports/RINGKASAN_HASIL.md`.

## Keluaran utama

| File | Isi |
|---|---|
| `data/processed/audit.csv` | hasil audit tiap foto (blur, duplikat, valid) |
| `data/processed/splits.csv` | path, label, split train/test |
| `data/processed/features.csv` | 319 fitur per foto |
| `models/best_model.joblib` | model terbaik + daftar kolom + best params |
| `reports/RINGKASAN_HASIL.md` | ringkasan hasil, metrik, confusion matrix, eksperimen |
| `reports/tables/perbandingan_model.csv` | 20 kombinasi model x skenario |
| `reports/tables/classification_report.txt` | precision, recall, F1 per kelas |
| `reports/tables/metrics_test.json` | ringkasan metrik test |
| `reports/tables/confusion_matrix.csv` | confusion matrix (jumlah) |
| `reports/tables/test_predictions.csv` | prediksi + probabilitas tiap foto test |
| `reports/tables/salah_prediksi.csv` | daftar foto yang salah diprediksi |
| `reports/tables/feature_importance*.csv` | kontribusi fitur model terbaik |
| `reports/figures/*.png` | distribusi kelas, confusion matrix, ROC, PR, learning curve |

## Catatan metodologi

- Split 80/20 stratified dengan `RANDOM_STATE = 42`; data uji tidak dipakai saat tuning.
- Model dipilih berdasarkan **CV F1-macro**, bukan skor test.
- Kelas tidak seimbang ditangani dengan `class_weight='balanced'` pada SVM, Random Forest, Extra Trees, dan Logistic Regression.
- Angka performa dilaporkan apa adanya; set test hanya 106 foto sehingga hasil mudah berubah.
- Filter default pada preprocessing adalah Gaussian blur. Eksperimen median/bilateral/unsharp mask tersedia di `src/experiments_filter.py`.
- Hasil eksperimen filter dan seleksi fitur disimpan terpisah dan tidak mengubah hasil utama.
