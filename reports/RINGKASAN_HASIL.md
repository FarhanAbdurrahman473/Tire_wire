# Ringkasan Hasil

Dokumen ini merangkum hasil yang dihasilkan skrip. Angka dilaporkan apa adanya; tidak ada penyesuaian untuk menaikkan akurasi.

## Dataset

| Kelas | Foto mentah | Valid setelah audit | Latih | Uji |
|---|---:|---:|---:|---:|
| baik | 215 | 189 | 151 | 38 |
| sedang | 132 | 128 | 102 | 26 |
| aus | 229 | 212 | 170 | 42 |
| **Total** | **576** | **529** | **423** | **106** |

- 47 foto dibuang karena blur score di bawah `BLUR_THRESHOLD = 50.0` (baik 26, aus 17, sedang 4).
- Tidak ditemukan duplikat MD5 atau foto yang tidak terbaca.
- Split 80/20 stratified, `RANDOM_STATE = 42`.

## Model terbaik

Dipilih berdasarkan **CV F1-macro**, bukan skor test:

| Item | Nilai |
|---|---|
| Model | **SVM (RBF)** |
| Skenario fitur | **D_Lengkap** (LBP multiskala + arah + GLCM) |
| Jumlah fitur | 63 |
| Hyperparameter | `C=10`, `gamma=scale` |
| CV F1-macro (5-fold) | 0.7687 |
| CV F1-macro berulang (5x3) | **0.7474 ± 0.0365** |
| Test accuracy | 0.7358 |
| Test balanced accuracy | 0.7091 |
| Test F1-macro | 0.7091 |
| Test precision-macro | 0.7176 |
| Test recall-macro | 0.7091 |
| Cohen's kappa | 0.5948 |
| ROC-AUC OvR macro | 0.8531 |
| Log loss | 0.6853 |
| Recall kelas **aus** | **0.9048** |

### Metrik per kelas (data uji)

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| baik | 0.839 | 0.684 | 0.754 | 38 |
| sedang | 0.538 | 0.538 | 0.538 | 26 |
| aus | 0.776 | 0.905 | 0.835 | 42 |

### Confusion matrix (baris = label asli, kolom = prediksi)

| Asli \ Prediksi | baik | sedang | aus |
|---|---:|---:|---:|
| baik | 26 | 8 | 4 |
| sedang | 5 | 14 | 7 |
| aus | 0 | 4 | 38 |

Tidak ada ban **aus** yang diprediksi **baik**. Kesalahan terbanyak adalah kelas **sedang** yang tertukar dengan **baik/aus**, sesuai dugaan karena batas kelasnya kontinu.

## Perbandingan model (10 teratas dari 20 kombinasi)

| Skenario fitur | Model | CV F1 (5-fold) | CV F1 berulang | Test acc | Test F1-macro | Recall aus |
|---|---|---:|---:|---:|---:|---:|
| D_Lengkap | SVM | 0.7687 | 0.7474 ± 0.0365 | 0.7358 | 0.7091 | 0.9048 |
| C_LBP-MS + arah | SVM | 0.7560 | 0.7452 ± 0.0306 | 0.7358 | 0.7072 | 0.9286 |
| B_LBP multi-skala | SVM | 0.7487 | 0.7371 ± 0.0538 | 0.7453 | 0.7150 | 0.9524 |
| D_Lengkap | Logistic Regression | 0.7350 | 0.7324 ± 0.0339 | 0.7642 | 0.7482 | 0.8810 |
| B_LBP multi-skala | Logistic Regression | 0.7224 | 0.7208 ± 0.0472 | 0.7453 | 0.7292 | 0.8810 |
| C_LBP-MS + arah | Logistic Regression | 0.7211 | 0.7198 ± 0.0312 | 0.7642 | 0.7500 | 0.8810 |
| C_LBP-MS + arah | Random Forest | 0.7266 | 0.7148 ± 0.0483 | 0.7170 | 0.6918 | 0.8810 |
| D_Lengkap | Random Forest | 0.7238 | 0.7119 ± 0.0456 | 0.6981 | 0.6771 | 0.8571 |
| D_Lengkap | Extra Trees | 0.7230 | 0.7116 ± 0.0410 | 0.7170 | 0.6775 | 0.9286 |
| B_LBP multi-skala | Random Forest | 0.7240 | 0.7092 ± 0.0392 | 0.7264 | 0.7053 | 0.8571 |

Hasil lengkap ada di `reports/tables/perbandingan_model.csv`.

## Eksperimen opsional

### Filter citra (model terbaik, fitur D)

| Filter | CV F1 (5-fold) | Test accuracy | Test F1-macro | Recall aus |
|---|---:|---:|---:|---:|
| gaussian (default) | 0.7687 | 0.7358 | 0.7091 | 0.9048 |
| bilateral | 0.7591 | 0.8019 | 0.7725 | 0.9286 |
| median | 0.7574 | 0.7453 | 0.7094 | 0.9048 |
| unsharp | 0.6971 | 0.7642 | 0.7322 | 0.9048 |

Catatan: filter bilateral memberi skor test lebih tinggi, tetapi skor CV-nya lebih rendah dari Gaussian. Karena pemilihan model memakai CV, hasil utama tetap memakai Gaussian.

### Seleksi fitur (model terbaik, fitur D = 63 fitur)

| Metode | Jumlah fitur | CV F1 | Test F1-macro | Recall aus |
|---|---:|---:|---:|---:|
| none | 63 | 0.7687 | 0.7091 | 0.9048 |
| variance threshold | 63 | 0.7687 | 0.7091 | 0.9048 |
| SelectKBest-100/200 | 63 | 0.7687 | 0.7091 | 0.9048 |
| PCA-50 | 50 | 0.7663 | 0.7091 | 0.9048 |
| PCA-20 | 20 | 0.7579 | 0.7399 | 0.9048 |
| SelectKBest-50 | 50 | 0.7430 | 0.6839 | 0.8810 |

## Keterbatasan

- Data uji hanya 106 foto, sehingga satu angka akurasi mudah berubah.
- Label berasal dari struktur folder dataset, bukan pengukuran kedalaman alur per ban.
- Kelas sedang paling sulit diprediksi (F1 0.538), wajar karena batas kelas kontinu.
- Feature importance model terbaik dihitung dengan permutation importance pada data uji dan hanya untuk interpretasi, bukan untuk pemilihan model.
- Sistem ini alat bantu skrining, bukan pengganti pemeriksaan teknisi.

## Artefak

- Model: `models/best_model.joblib`, `models/best_model_info.json`
- Tabel: `reports/tables/`
- Gambar: `reports/figures/`
- Notebook: `notebooks/04_training.ipynb`, `05_evaluasi.ipynb`, `06_prediksi.ipynb`
