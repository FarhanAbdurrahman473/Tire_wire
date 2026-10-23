        # Panduan Pengerjaan Penelitian: Klasifikasi Tingkat Keausan Ban Kendaraan Berbasis Citra (Tire Wear Classification)

Struktur folder, daftar file, dan tahapan **Knowledge Discovery in Databases (KDD)** dengan Machine Learning klasik (KNN, SVM, Random Forest) dan opsi CNN.

Metode ekstraksi fitur mengacu pada jurnal: A. Sitompul, *Implementasi Pengenalan Jenis Pola Tapak Ban (Tread) Menggunakan Metode Local Binary Patterns*, Jurnal Sistem Komputer dan Informatika (JSON), Vol. 1 No. 3, Mei 2020, hal. 189-194 (DOI 10.30865/json.v1i3.2094).

> **Status verifikasi kode**
> Seluruh kode jalur klasik (Lampiran A) telah dijalankan end-to-end pada data sintetis: audit, preprocessing, ekstraksi fitur, tuning KNN/SVM/RF, evaluasi, dan prediksi berjalan tanpa error. Angka performa pada data sintetis **tidak** berlaku untuk foto ban Anda.
> Kode CNN (`train_cnn.py`) **belum dijalankan** karena TensorFlow tidak tersedia di lingkungan penyusunan; perlakukan sebagai template.

## Daftar Isi

1. [Gambaran Umum Penelitian](#1-gambaran-umum-penelitian)
2. [Analisis Jurnal Acuan dan Cara Mengadaptasinya](#2-analisis-jurnal-acuan-dan-cara-mengadaptasinya)
3. [Indikator dan Kelas Tingkat Keausan](#3-indikator-dan-kelas-tingkat-keausan)
4. [Struktur Folder dan Daftar File](#4-struktur-folder-dan-daftar-file)
5. [Persiapan Lingkungan dan Pengambilan Foto](#5-persiapan-lingkungan-dan-pengambilan-foto)
6. [Proses KDD Langkah demi Langkah](#6-proses-kdd-langkah-demi-langkah)
7. [Template Hasil dan Kerangka Laporan](#7-template-hasil-dan-kerangka-laporan)
8. [Pemecahan Masalah Umum](#8-pemecahan-masalah-umum)
9. [Lampiran A. Kode Lengkap](#lampiran-a-kode-lengkap)

---

## 1. Gambaran Umum Penelitian

Penelitian ini membangun model yang menerima **foto tapak ban** dan mengeluarkan **tingkat keausan** (mis. baik / sedang / aus) sehingga pengendara motor tahu kapan ban harus diganti. Panduan ini menyusun pekerjaan Anda menjadi urutan yang bisa langsung dieksekusi: dari struktur folder, daftar file, sampai enam tahap proses KDD.

### 1.1 Ruang lingkup dan keputusan desain yang disarankan

- **Objek**: ban sepeda motor (foto tapak dari atas, tegak lurus).
- **Kelas** (usulan): *baik*, *sedang*, *aus*. Boleh disederhanakan menjadi 2 kelas (layak / tidak layak) bila data tiap kelas terlalu sedikit.
- **Jalur utama**: machine learning klasik (LBP + KNN/SVM/Random Forest). Alasannya: dataset hanya beberapa ratus foto, sehingga jalur klasik lebih stabil dan mudah dijelaskan di laporan.
- **Jalur opsional**: CNN transfer learning (MobileNetV2) sebagai pembanding.
- **Metrik utama**: F1-macro dan **recall kelas "aus"**. Kesalahan paling berbahaya adalah ban aus yang diprediksi "baik".

### 1.2 Peta tahap KDD ke file dan keluaran

| Tahap | Kegiatan inti | File yang dijalankan | Keluaran utama |
|---|---|---|---|
| 0. Pemahaman domain | Definisi kelas, protokol foto, pelabelan | (manual) + metadata | Aturan kelas, dataset terlabel |
| 1. Selection | Audit, buang duplikat/buram/rusak, split | `src/data_selection.py` | `splits.csv`, `audit.csv`, grafik distribusi |
| 2. Preprocessing | Crop, resize, grayscale, CLAHE, denoise | `src/preprocessing.py` | `data/interim/gray`, `data/interim/rgb` |
| 3. Transformation | Ekstraksi fitur LBP, arah, GLCM | `src/features.py` | `data/processed/features.csv` |
| 4. Data Mining | Tuning dan pelatihan KNN, SVM, RF (opsi CNN) | `src/train_classic.py` (`src/train_cnn.py`) | `models/best_model.joblib`, `perbandingan_model.csv` |
| 5. Evaluation | Metrik, confusion matrix, analisis kesalahan | `src/evaluate.py` | `classification_report.txt`, `confusion_matrix.png` |
| 6. Knowledge/Deployment | Prediksi foto baru, laporan | `src/predict.py` | Label + pesan rekomendasi + probabilitas |

---

## 2. Analisis Jurnal Acuan dan Cara Mengadaptasinya

Jurnal Sitompul (2020) menggunakan *Local Binary Patterns* (LBP) untuk mengenali **jenis pola tapak** (Rib, Lug, Rib-Lug, Block). Topik Anda adalah **tingkat keausan**. Karena itu yang dipinjam dari jurnal adalah **teknik ekstraksi cirinya**, bukan tujuan klasifikasinya.

| Elemen di jurnal | Isi (ringkas) | Penerapan di penelitian ini |
|---|---|---|
| Operator LBP 3x3, Pers. (1)-(2) | 8 piksel tetangga dibandingkan dengan piksel tengah; hasil ≥ 0 menjadi 1, selain itu 0; 8 bit dikonversi ke desimal. | Fitur `lbp256_*` (P=8, R=1) sebagai baseline yang setia pada jurnal. |
| Histogram 256-bin sebagai *texture descriptor* | Kode LBP dihitung per piksel, lalu dirangkum menjadi histogram 256 bin. | Histogram dinormalisasi (jumlah = 1) agar tidak bergantung ukuran citra. |
| Circular neighborhood (P,R) + interpolasi bilinear | Jurnal menyebut 3x3 tidak bisa menangkap ciri berskala besar; operator dikembangkan ke radius/titik sampel lain. | Fitur `lbpms_*`: LBP uniform multi-skala (8,1), (16,2), (24,3) untuk menangkap lebar alur. |
| Kode arah garis 2/3/4/5 dan K(n), Pers. (3) | Arah garis (vertikal, diagonal kanan, horizontal, diagonal kiri) dikodekan; K(n) = jumlah piksel berkode n / ukuran matriks. | Fitur `dir_*`: proporsi piksel tepi per arah (Sobel + Canny) dibagi jumlah piksel citra. |
| Praproses: binerisasi, thinning, normalisasi | Disebut sebelum ekstraksi ciri. | Grayscale + CLAHE (normalisasi kontras) + Gaussian blur. Binerisasi/thinning opsional untuk eksperimen tambahan. |
| Euclidean Distance | Kemiripan dua vektor fitur dihitung dari jarak Euclid. | KNN memakai `metric='euclidean'`; fitur di-standarisasi agar jarak adil. |
| Akurasi 91,82% (acuan [2] di jurnal) | LBP + Nearest Neighbor untuk deteksi kerusakan permukaan ban. | Hanya sebagai pembanding kasar di pembahasan; tugasnya berbeda sehingga bukan target resmi. |

> **Dua catatan penting tentang jurnal ini**
>
> **1. Jurnal tidak mendefinisikan tingkat keausan.** Indikator keausan (kelas dan ambangnya) harus Anda tetapkan sendiri; usulannya ada di Bab 3 dan harus disetujui pembimbing.
>
> **2. Perhitungan pada Subbab 3.1 jurnal (Pers. 4, faktor skala β, dan nilai V<sub>j</sub>) tidak konsisten secara matematis.** Contoh: β ditulis 0,7 × (11)^(1/400) = 25,45, padahal 11^(1/400) hanya sekitar 1,006 sehingga hasilnya sekitar 0,70. Tabel pembobotan juga tidak bisa direproduksi dari teksnya. Karena itu panduan ini hanya memakai bagian jurnal yang jelas dan baku (Pers. 1-3 dan Euclidean Distance). Sebaiknya Anda tidak menyalin prosedur V<sub>j</sub> tersebut ke laporan sebagai dasar perhitungan.

---

## 3. Indikator dan Kelas Tingkat Keausan

Ini bagian yang paling menentukan mutu penelitian: model hanya sebaik label yang diberikan. Label sebaiknya berasal dari **pengukuran fisik**, bukan sekadar tebakan visual.

### 3.1 Definisi kelas (usulan awal)

| Kelas | Kedalaman alur (contoh ambang) | Ciri visual | Rekomendasi |
|---|---|---|---|
| baik | lebih dari 2 mm | Alur dalam, bayangan alur tegas, blok tapak jelas, TWI belum terlihat | Layak pakai |
| sedang | 1 sampai 2 mm | Alur dangkal, tepi blok mulai membulat, TWI mendekati permukaan | Pantau, rencanakan ganti |
| aus | di bawah 1 mm | Permukaan hampir rata/gundul, alur nyaris hilang, TWI sejajar permukaan, kadang retak | Segera ganti |

> **Wajib diverifikasi.** Angka ambang di atas hanyalah **contoh awal**. Tetapkan ambang final dengan merujuk pada buku manual/rekomendasi produsen ban, regulasi yang berlaku, dan persetujuan pembimbing. Cantumkan sumber ambang itu di Bab Metodologi laporan.

### 3.2 Cara memberi label

1. Ukur kedalaman alur dengan *tread depth gauge* (atau jangka sorong bertangkai) pada 3 titik alur utama di tiap ban, catat rata-ratanya dalam mm.
2. Foto tapak ban **sesudah** diukur; simpan hasil ukur di `data/raw/metadata.csv` (kolom: `file, id_ban, kedalaman_mm, kelas`).
3. Tentukan kelas dari `kedalaman_mm` menurut ambang Bab 3.1, lalu pindahkan foto ke folder kelasnya.
4. Jika tidak ada alat ukur: minta **dua orang** melabeli terpisah, hitung kesepakatan (Cohen's kappa), dan selesaikan perbedaan lewat diskusi. Tulis di laporan bahwa label berbasis visual.

### 3.3 Pemetaan indikator visual ke fitur citra

| Indikator keausan | Dampak pada citra | Fitur yang menangkapnya |
|---|---|---|
| Kedalaman alur | Kontras tinggi antara blok tapak dan alur (bayangan) | `glcm_contrast`, `dir_edge`, bin LBP tepi/sudut |
| Kejelasan tepi blok dan alur | Banyak tepi tajam | `dir_edge`, `lbp256`, `lbpms` |
| Arah pola alur | Distribusi arah tepi (vertikal, diagonal, horizontal) | `dir_vert`, `dir_diagr`, `dir_horiz`, `dir_diagl` (K(n) jurnal) |
| Permukaan halus/gundul | Area datar luas, tekstur homogen | bin LBP seragam, `glcm_homog`, `glcm_energy` |
| Tekstur mikro karet (retak, aus tidak rata) | Variasi tekstur halus | LBP radius kecil (8,1), `lbp256` |
| Tread Wear Indicator (TWI) | Tonjolan kecil di dasar alur | Tidak ditangkap langsung; dipakai sebagai acuan pelabelan |

---

## 4. Struktur Folder dan Daftar File

```text
tire-wear-classification/
|-- README.md                  # ringkasan proyek + cara menjalankan (Anda tulis)
|-- requirements.txt           # daftar library
|-- config.py                  # SEMUA parameter terpusat (path, kelas, ukuran, LBP)
|-- data/
|   |-- raw/                   # foto ASLI - jangan diubah/ditimpa
|   |   |-- baik/
|   |   |-- sedang/
|   |   |-- aus/
|   |   `-- metadata.csv       # id_ban, kedalaman_mm, kelas (opsional tapi disarankan)
|   |-- interim/               # [otomatis] hasil preprocessing
|   |   |-- gray/{train,test}/{baik,sedang,aus}/
|   |   `-- rgb/{train,test}/{baik,sedang,aus}/      # untuk jalur CNN
|   `-- processed/             # [otomatis]
|       |-- audit.csv          # hasil audit tiap foto
|       |-- splits.csv         # path, label, split (train/test)
|       `-- features.csv       # matriks fitur
|-- src/
|   |-- __init__.py
|   |-- data_selection.py      # TAHAP 1
|   |-- preprocessing.py       # TAHAP 2
|   |-- features.py            # TAHAP 3
|   |-- train_classic.py       # TAHAP 4 (KNN, SVM, Random Forest)
|   |-- train_cnn.py           # TAHAP 4 opsional (CNN)
|   |-- evaluate.py            # TAHAP 5
|   `-- predict.py             # TAHAP 6
|-- models/                    # [otomatis] best_model.joblib, cnn_mobilenetv2.keras
|-- reports/
|   |-- figures/               # [otomatis] distribusi_kelas.png, confusion_matrix.png
|   `-- tables/                # [otomatis] perbandingan_model.csv, classification_report.txt,
|                              #            salah_prediksi.csv
`-- notebooks/                 # eksplorasi bebas (EDA, visualisasi LBP) - opsional
```

| File | Dibuat oleh | Fungsi |
|---|---|---|
| `config.py` | Anda (salin dari Lampiran A) | Satu-satunya tempat mengubah path, daftar kelas, ukuran gambar, parameter LBP, rasio split. |
| `src/data_selection.py` | Anda | Membaca foto per kelas, mengaudit (terbaca? duplikat? buram?), membagi train/test bertingkat. |
| `src/preprocessing.py` | Anda | Crop tengah persegi, resize 256x256, grayscale, CLAHE, Gaussian blur. Fungsi `preprocess_gray` dipakai ulang oleh `predict.py`. |
| `src/features.py` | Anda | Empat grup fitur (lbp256, lbpms, dir, glcm) dan definisi skenario ablasi. |
| `src/train_classic.py` | Anda | GridSearchCV 5-fold untuk 3 model × 4 skenario fitur; menyimpan model terbaik. |
| `src/evaluate.py` | Anda | Classification report, confusion matrix, daftar foto yang salah prediksi. |
| `src/predict.py` | Anda | Prediksi satu foto baru lewat command line. |
| `src/train_cnn.py` | Anda (opsional) | Transfer learning MobileNetV2 sebagai pembanding. |
| `data/processed/*.csv`, `models/*`, `reports/*` | Otomatis | Dihasilkan skrip; jangan diedit manual. |

### 4.1 Aturan penamaan dan kebiasaan baik

- Skrip dijalankan **dari folder induk** dengan `python -m src.nama_skrip` agar `import config` berfungsi.
- Nama foto unik, mis. `B01_001.jpg` (B01 = id ban). Bila satu ban difoto berkali-kali, set `GROUP_BY_TIRE_ID = True` di `config.py` agar foto ban yang sama tidak terbagi ke train dan test (kebocoran data).
- Foto asli di `data/raw` tidak pernah ditimpa; semua hasil olahan ditulis ke folder lain.
- Pakai Git (`.gitignore` untuk `data/` dan `models/`) agar perubahan kode terlacak.
- Semua keacakan memakai `RANDOM_STATE = 42` agar hasil bisa diulang.

---

## 5. Persiapan Lingkungan dan Pengambilan Foto

### 5.1 Instalasi

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip freeze > requirements-lock.txt # kunci versi untuk reproduktibilitas
```

### 5.2 Protokol pengambilan foto (jika menambah data)

- Jarak sekitar 15-25 cm, kamera **tegak lurus** permukaan tapak, tapak memenuhi bingkai.
- Cahaya merata (teduh/siang hari). Hindari flash langsung dan pantulan silau.
- Fokus tajam; buang foto goyang (akan ditandai "buram" oleh audit).
- **Variasikan latar dan pencahayaan di setiap kelas.** Bila semua foto ban aus diambil di tempat A dan ban baru di tempat B, model akan belajar membedakan *tempat*, bukan keausan.
- Gunakan beberapa merek/jenis ban dan beberapa motor bila memungkinkan agar model lebih general.
- Sertakan 3-5 foto per ban (tengah dan tepi tapak), tetapi catat `id_ban` untuk mencegah kebocoran data.

### 5.3 Target jumlah data

Untuk beberapa ratus foto, usahakan minimal sekitar 50-60 foto per kelas dan jumlah antarkelas relatif seimbang. Bila kelas tidak seimbang, tambahkan `class_weight='balanced'` pada `SVC` dan `RandomForestClassifier` di `train_classic.py` dan laporkan F1-macro, bukan akurasi saja.

---

## 6. Proses KDD Langkah demi Langkah

Setiap tahap di bawah ini berisi tujuan, langkah kerja, perintah, keluaran, dan hal yang harus diperiksa sebelum lanjut. Kode lengkap tiap file ada di Lampiran A.

### Tahap 0 - Pemahaman Domain dan Persiapan Data

| | |
|---|---|
| **Tujuan** | Menetapkan masalah, kelas, metrik sukses, dan label yang valid. |
| **Masukan** | Foto ban, alat ukur kedalaman alur, jurnal acuan. |
| **Keluaran** | Dataset terlabel di `data/raw/{baik,sedang,aus}`, aturan kelas tertulis. |
| **File** | (manual) |

1. Tulis rumusan masalah dan tujuan: *mengklasifikasikan keausan ban dari citra tapak menjadi 3 kelas*.
2. Tetapkan kelas dan ambang (Bab 3), minta persetujuan pembimbing.
3. Pindahkan foto ke tiga folder kelas di `data/raw/`; isi `metadata.csv`.
4. Tetapkan metrik sukses sejak awal: F1-macro dan recall kelas "aus".

**Periksa sebelum lanjut:** apakah tiap kelas punya foto dari beberapa ban dan latar berbeda? Apakah dasar pelabelan (ukur/visual) tercatat?

### Tahap 1 - Data Selection

| | |
|---|---|
| **Tujuan** | Memilih data yang layak dipakai dan memisahkan data uji sejak awal. |
| **Masukan** | `data/raw/*` |
| **Keluaran** | `audit.csv`, `splits.csv`, `distribusi_kelas.png` |
| **File** | `src/data_selection.py` |

1. Kode membaca semua foto, mencatat ukuran, *blur score* (variance of Laplacian) dan hash MD5.
2. Foto yang tidak terbaca, duplikat identik (MD5 sama), atau buram (blur score di bawah `BLUR_THRESHOLD`) ditandai `valid=False`.
3. Data valid dibagi **80% train / 20% test** secara bertingkat (stratified) per kelas. Set test dikunci dan **tidak boleh dipakai untuk tuning**.
4. Buka `audit.csv`, lihat foto yang dibuang; bila banyak foto bagus terbuang, turunkan `BLUR_THRESHOLD`.

```bash
python -m src.data_selection
```

> **Kesalahan umum.** Membagi data setelah augmentasi atau setelah ekstraksi fitur dari foto ban yang sama membuat akurasi tampak sangat tinggi tetapi menyesatkan (data leakage).

### Tahap 2 - Preprocessing

| | |
|---|---|
| **Tujuan** | Menyeragamkan citra agar fitur tekstur dapat dibandingkan secara adil. |
| **Masukan** | `splits.csv` + foto asli |
| **Keluaran** | `data/interim/gray/...`, `data/interim/rgb/...` |
| **File** | `src/preprocessing.py` |

1. **Crop tengah persegi** lalu **resize 256x256** (INTER_AREA) agar semua foto berukuran sama.
2. **Grayscale**: LBP bekerja pada intensitas keabuan (sesuai jurnal).
3. **CLAHE** (clipLimit 2.0, tile 8x8): meratakan kontras lokal sehingga perbedaan cahaya antarfoto tidak terbaca sebagai perbedaan keausan. Ini padanan modern dari "normalisasi" pada jurnal.
4. **Gaussian blur 3x3** ringan untuk meredam noise sensor.
5. Salinan RGB hasil resize disimpan terpisah untuk jalur CNN.

```bash
python -m src.preprocessing
```

**Periksa sebelum lanjut:** buka 10-20 gambar di `data/interim/gray` secara acak. Alur masih terlihat jelas? Bila CLAHE membuat noise berlebihan, kecilkan `clipLimit`. Lakukan pemeriksaan ini sebelum menjalankan tahap 3.

**Augmentasi**: untuk jalur klasik tidak wajib. Bila dipakai (flip, rotasi kecil, perubahan kecerahan), lakukan **hanya pada data train dan setelah split**. Pada jalur CNN augmentasi sudah ada di dalam model (lapisan `RandomFlip` dst.).

### Tahap 3 - Transformation (Ekstraksi Fitur)

| | |
|---|---|
| **Tujuan** | Mengubah tiap citra menjadi vektor fitur numerik yang membedakan tingkat keausan. |
| **Masukan** | `data/interim/gray/*` |
| **Keluaran** | `data/processed/features.csv` (satu baris per foto) |
| **File** | `src/features.py` |

| Grup fitur | Jumlah | Cara kerja | Dasar dari jurnal |
|---|---|---|---|
| `lbp256_*` | 256 | LBP default P=8, R=1; histogram 256-bin ternormalisasi | Pers. (1)-(2), 256-bin descriptor |
| `lbpms_*` | 10+18+26 = 54 | LBP uniform melingkar (8,1), (16,2), (24,3); histogram tiap skala digabung | Circular neighborhood (P,R) |
| `dir_*` | 5 | Proporsi piksel tepi total dan per arah (vertikal, diagonal kanan, horizontal, diagonal kiri) | Kode arah 2/3/4/5 dan K(n), Pers. (3) |
| `glcm_*` | 4 | Contrast, homogeneity, energy, correlation dari GLCM (jarak 1 dan 3, 4 sudut) | Tambahan (analisis tekstur) |

- **Normalisasi fitur** (StandardScaler) dilakukan <u>di dalam pipeline</u> pada tahap 4, sehingga statistiknya hanya dihitung dari data latih per fold. Jangan menormalisasi seluruh dataset sebelum split.
- **Studi ablasi**: `features.py` mendefinisikan 4 skenario (A: LBP256 jurnal; B: LBP multi-skala; C: B + arah; D: lengkap). Hasilnya menjawab pertanyaan penelitian: *apakah pengembangan dari jurnal memberi peningkatan?*
- Opsional: seleksi fitur/PCA, atau eksperimen binerisasi + thinning seperti disebut jurnal.

```bash
python -m src.features
```

**Periksa sebelum lanjut:** jalankan `pd.read_csv('data/processed/features.csv').describe()`. Tidak boleh ada NaN; tiap grup fitur harus bervariasi antarkelas (bandingkan rata-rata per kelas).

### Tahap 4 - Data Mining (Pemodelan)

| | |
|---|---|
| **Tujuan** | Melatih dan menyetel model klasifikasi, lalu memilih yang terbaik secara adil. |
| **Masukan** | `features.csv` (split = train) |
| **Keluaran** | `models/best_model.joblib`, `reports/tables/perbandingan_model.csv` |
| **File** | `src/train_classic.py` (opsional `src/train_cnn.py`) |

| Model | Pipeline | Hiperparameter yang dituning | Catatan |
|---|---|---|---|
| KNN | StandardScaler + KNN (Euclidean) | n_neighbors 3,5,7,9; weights uniform/distance | Paling dekat dengan jurnal (Euclidean Distance). |
| SVM | StandardScaler + SVC (RBF) | C 1,10,100; gamma scale,0.01,0.001 | Biasanya kuat untuk fitur histogram LBP. |
| Random Forest | StandardScaler + RF | n_estimators 200,400; max_depth None,10,20 | Memberi feature importance untuk interpretasi. |

- **GridSearchCV 5-fold** bertingkat pada data train, skor **F1-macro**.
- Model terbaik dipilih berdasarkan **skor validasi silang**, <u>bukan skor test</u>, supaya set test tetap murni untuk evaluasi akhir.
- Semua kombinasi (3 model × 4 skenario fitur) dicatat di `perbandingan_model.csv` untuk tabel laporan.
- **Jalur CNN (opsional)**: MobileNetV2 pra-latih (ImageNet), bagian dasar dibekukan, kepala klasifikasi baru, augmentasi + early stopping. Instal TensorFlow terlebih dahulu, lalu `python -m src.train_cnn`. Bandingkan hasilnya pada set test yang sama.

```bash
python -m src.train_classic
```

Waktu tuning pada beberapa ratus foto umumnya beberapa menit di laptop biasa; Random Forest yang paling lama.

### Tahap 5 - Evaluation / Interpretation

| | |
|---|---|
| **Tujuan** | Mengukur kinerja jujur pada data uji dan memahami kesalahan model. |
| **Masukan** | `best_model.joblib` + `features.csv` (split = test) |
| **Keluaran** | `classification_report.txt`, `confusion_matrix.png`, `salah_prediksi.csv` |
| **File** | `src/evaluate.py` |

- **Metrik**: accuracy, precision, recall, F1 per kelas dan macro. Utamakan **recall kelas "aus"**.
- **Confusion matrix**: perhatikan galat "aus diprediksi baik" (paling berbahaya) dan galat berdekatan (sedang tertukar baik/aus) yang wajar karena batas kelas memang kontinu.
- **Analisis kesalahan**: buka foto di `salah_prediksi.csv`. Apakah buram, silau, sudut miring, atau labelnya meragukan? Tuliskan temuan ini di pembahasan.
- **Tabel ablasi**: bandingkan skenario A-D dan KNN/SVM/RF (template di Bab 7).
- **Kestabilan**: set test hanya puluhan foto, jadi satu angka akurasi mudah berubah. Laporkan juga rerata ± simpangan baku dari validasi silang berulang (`RepeatedStratifiedKFold`).
- **Pembanding**: acuan 91,82% pada jurnal berasal dari tugas yang berbeda; sebut sebagai konteks, bukan sebagai klaim "lebih baik/lebih buruk".
- **Uji ketahanan** (disarankan): uji model dengan beberapa foto baru yang diambil pada kondisi cahaya berbeda.

```bash
python -m src.evaluate
```

### Tahap 6 - Penerapan dan Konsolidasi Pengetahuan

| | |
|---|---|
| **Tujuan** | Memakai model untuk foto baru dan merumuskan temuan penelitian. |
| **Masukan** | Foto ban baru, `best_model.joblib` |
| **Keluaran** | Label keausan, pesan rekomendasi, probabilitas per kelas |
| **File** | `src/predict.py` |

```bash
python -m src.predict contoh_foto.jpg
```

Contoh keluaran:

```text
Prediksi: AUS - Ban sudah aus/gundul - SEGERA GANTI demi keselamatan.
  aus    : 96.9%
  baik   : 1.8%
  sedang : 1.3%
```

- `predict.py` memakai fungsi preprocessing dan ekstraksi fitur **yang sama persis** dengan saat pelatihan; inilah alasan `preprocess_gray` dan `extract_all` dipisah sebagai fungsi.
- Pengembangan lanjutan (opsional): antarmuka web sederhana (Streamlit/Gradio) atau aplikasi ponsel.
- Cantumkan di laporan bahwa sistem adalah **alat bantu skrining**, bukan pengganti pemeriksaan langsung oleh teknisi.

---

## 7. Template Hasil dan Kerangka Laporan

### 7.1 Template tabel hasil (isi dari `perbandingan_model.csv`)

| Skenario fitur | Model | CV F1-macro (rerata ± sd) | Test accuracy | Test F1-macro | Recall "aus" |
|---|---|---|---|---|---|
| A. LBP256 (jurnal) | KNN / SVM / RF | | | | |
| B. LBP multi-skala | KNN / SVM / RF | | | | |
| C. LBP-MS + arah | KNN / SVM / RF | | | | |
| D. Lengkap | KNN / SVM / RF | | | | |
| CNN MobileNetV2 (opsional) | - | | | | |

### 7.2 Pemetaan ke kerangka laporan

| Bab laporan | Isi | Diambil dari |
|---|---|---|
| I. Pendahuluan | Latar belakang keselamatan, rumusan masalah, tujuan, batasan | Tahap 0 |
| II. Tinjauan pustaka | Ban dan tapak, pengolahan citra, LBP, GLCM, KNN/SVM/RF, KDD, jurnal Sitompul (2020) | Bab 2 |
| III. Metodologi | Pengumpulan data, definisi kelas dan ambang, tahap KDD 1-6, parameter | Bab 3-6 |
| IV. Hasil dan pembahasan | Distribusi data, tabel ablasi, confusion matrix, analisis kesalahan, keterbatasan | Tahap 5 |
| V. Penutup | Kesimpulan menjawab rumusan masalah, saran | Tahap 5-6 |

### 7.3 Perkiraan jadwal kerja

| Minggu | Kegiatan |
|---|---|
| 1 | Tahap 0: finalisasi kelas, ambang, pelabelan; setup folder dan lingkungan |
| 2 | Tahap 1-2: audit, split, preprocessing, pemeriksaan visual |
| 3 | Tahap 3: ekstraksi fitur dan eksplorasi (grafik rata-rata fitur per kelas) |
| 4 | Tahap 4: tuning KNN/SVM/RF (dan CNN bila ada waktu) |
| 5 | Tahap 5-6: evaluasi, analisis kesalahan, `predict.py` |
| 6 | Penulisan laporan, revisi, persiapan presentasi |

---

## 8. Pemecahan Masalah Umum

| Gejala | Kemungkinan penyebab | Tindakan |
|---|---|---|
| Akurasi test jauh di bawah skor CV | Data test kecil; atau foto ban yang sama tersebar di train dan CV | Aktifkan `GROUP_BY_TIRE_ID`; laporkan CV berulang; tambah data |
| Akurasi mendekati 100% (mencurigakan) | Model belajar latar/pencahayaan, bukan keausan; atau kebocoran data | Variasikan latar tiap kelas; periksa duplikat dekat; cek split per ban |
| Kelas "sedang" sering tertukar | Batas kelas kabur, label noisy | Tambah data di sekitar ambang; pertimbangkan 2 kelas atau regresi kedalaman (mm) |
| Banyak foto dibuang audit "buram" | `BLUR_THRESHOLD` terlalu tinggi untuk kamera Anda | Lihat `audit.csv`, turunkan ambang setelah memeriksa foto |
| Overfitting (CV tinggi, test rendah) | Fitur terlalu banyak untuk data sedikit | Pakai skenario B/C, PCA, atau regularisasi lebih kuat |
| `ModuleNotFoundError: config` | Skrip dijalankan dari dalam folder `src/` | Jalankan dari folder induk dengan `python -m src.nama` |
| Data sangat sedikit per kelas | Variasi tidak cukup | Tambah foto; untuk CNN gunakan transfer learning + augmentasi |

### Daftar periksa akhir

- [ ] Ambang kelas dan sumbernya tertulis dan disetujui pembimbing.
- [ ] Set test tidak pernah dipakai saat tuning; model dipilih dari skor CV.
- [ ] Tidak ada foto ban yang sama di train dan test (atau dilaporkan sebagai keterbatasan).
- [ ] Tabel ablasi A-D terisi, confusion matrix dan analisis kesalahan ada di laporan.
- [ ] Versi library dikunci (`requirements-lock.txt`) dan `RANDOM_STATE` dicatat.
- [ ] Keterbatasan dituliskan: ukuran data, label berbasis ukur/visual, ragam kamera dan cahaya.

---

## Lampiran A. Kode Lengkap

Salin tiap blok ke file dengan nama yang tertera (perhatikan lokasi folder). Kode ini sama dengan yang telah diuji pada data sintetis (jalur klasik).

### `requirements.txt`

```text
numpy
pandas
opencv-python-headless
scikit-image
scikit-learn
matplotlib
joblib
# opsional, hanya untuk jalur CNN:
# tensorflow
```

### `config.py`

```python
"""Konfigurasi terpusat. Ubah parameter di sini, bukan di tiap skrip."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"            # foto asli, 1 sub-folder per kelas
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
```

### `src/__init__.py`

```python
# file kosong: cukup buat src/__init__.py tanpa isi
```

### `src/data_selection.py`

```python
"""TAHAP 1 KDD - Data Selection: audit dataset + pembagian train/test."""
import hashlib
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import config as C

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit():
    rows = []
    for label in C.CLASSES:
        for p in sorted((C.RAW_DIR / label).glob("*")):
            if p.suffix.lower() not in EXT:
                continue
            img = cv2.imread(str(p))
            if img is None:
                rows.append(dict(path=str(p), label=label, valid=False, reason="tidak terbaca"))
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            md5 = hashlib.md5(p.read_bytes()).hexdigest()
            rows.append(dict(path=str(p), label=label, width=w, height=h,
                             blur_score=round(blur, 2), md5=md5, valid=True, reason=""))
    df = pd.DataFrame(rows)
    dup = df["valid"] & df.duplicated("md5", keep="first")
    df.loc[dup, "reason"] = "duplikat"
    df.loc[dup, "valid"] = False
    blur = df["valid"] & (df["blur_score"] < C.BLUR_THRESHOLD)
    df.loc[blur, "reason"] = "buram"
    df.loc[blur, "valid"] = False
    return df


def split(df):
    v = df[df["valid"]].copy()
    if C.GROUP_BY_TIRE_ID:
        v["group"] = v["path"].map(lambda s: s.split("/")[-1].split("_")[0])
        gss = GroupShuffleSplit(1, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
        _, te = next(gss.split(v, v["label"], v["group"]))
        v["split"] = "train"
        v.iloc[te, v.columns.get_loc("split")] = "test"
    else:
        _, te = train_test_split(v.index, test_size=C.TEST_SIZE,
                                 stratify=v["label"], random_state=C.RANDOM_STATE)
        v["split"] = "train"
        v.loc[te, "split"] = "test"
    return v


if __name__ == "__main__":
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = audit()
    df.to_csv(C.PROCESSED_DIR / "audit.csv", index=False)
    print("Ringkasan audit:\n", df.groupby(["label", "valid"]).size().unstack(fill_value=0))
    print("Alasan dibuang:\n", df.loc[~df["valid"], "reason"].value_counts())

    sp = split(df)
    sp[["path", "label", "split"]].to_csv(C.PROCESSED_DIR / "splits.csv", index=False)
    print("\nDistribusi split:\n", sp.groupby(["split", "label"]).size().unstack(fill_value=0))

    ax = sp.groupby(["label", "split"]).size().unstack().loc[C.CLASSES].plot(kind="bar")
    ax.set_title("Distribusi data per kelas"); ax.set_ylabel("jumlah citra")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "distribusi_kelas.png", dpi=150)
```

### `src/preprocessing.py`

```python
"""TAHAP 2 KDD - Preprocessing: crop-resize, grayscale, CLAHE, denoise."""
import cv2
import pandas as pd
import config as C


def center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def preprocess_rgb(img_bgr):
    """Crop tengah (persegi) lalu resize. Dipakai juga untuk jalur CNN."""
    return cv2.resize(center_square(img_bgr), (C.IMG_SIZE, C.IMG_SIZE),
                      interpolation=cv2.INTER_AREA)


def preprocess_gray(img_bgr):
    """Pipeline untuk machine learning klasik."""
    sq = preprocess_rgb(img_bgr)
    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # normalisasi kontras/cahaya
    gray = clahe.apply(gray)
    return cv2.GaussianBlur(gray, (3, 3), 0)                      # redam noise


if __name__ == "__main__":
    df = pd.read_csv(C.PROCESSED_DIR / "splits.csv")
    for _, r in df.iterrows():
        img = cv2.imread(r["path"])
        name = r["path"].split("/")[-1].rsplit(".", 1)[0] + ".png"
        for kind, out in (("gray", preprocess_gray(img)), ("rgb", preprocess_rgb(img))):
            d = C.INTERIM_DIR / kind / r["split"] / r["label"]
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / name), out)
    print("Preprocessing selesai ->", C.INTERIM_DIR)
```

### `src/features.py`

```python
"""TAHAP 3 KDD - Transformation: ekstraksi fitur (LBP, arah garis, GLCM, tepi)."""
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import config as C


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
    ang = (np.degrees(np.arctan2(gy, gx)) % 180)[edges]      # arah gradien 0..180
    n = gray.size
    vert = np.sum((ang < 22.5) | (ang >= 157.5))             # gradien horizontal -> garis vertikal
    diag_r = np.sum((ang >= 22.5) & (ang < 67.5))            # garis "/"
    horiz = np.sum((ang >= 67.5) & (ang < 112.5))            # garis horizontal
    diag_l = np.sum((ang >= 112.5) & (ang < 157.5))          # garis "\"
    return np.array([edges.sum() / n, vert / n, diag_r / n, horiz / n, diag_l / n])


def glcm_stats(gray):
    g = (gray // 4).astype(np.uint8)                         # 64 level -> GLCM ringan
    m = graycomatrix(g, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=64, symmetric=True, normed=True)
    return np.array([graycoprops(m, p).mean()
                     for p in ("contrast", "homogeneity", "energy", "correlation")])


def extract_all(gray):
    """Satu vektor fitur; nama kolom berawalan grup (lbp256_, lbpms_, dir_, glcm_)."""
    a, b, c, d = lbp256(gray), lbp_multiscale(gray), direction_stats(gray), glcm_stats(gray)
    names = ([f"lbp256_{i}" for i in range(len(a))] + [f"lbpms_{i}" for i in range(len(b))]
             + ["dir_edge", "dir_vert", "dir_diagr", "dir_horiz", "dir_diagl"]
             + ["glcm_contrast", "glcm_homog", "glcm_energy", "glcm_corr"])
    return names, np.concatenate([a, b, c, d])


# Skenario fitur untuk studi ablasi (seberapa besar tiap grup fitur berkontribusi)
FEATURE_SETS = {
    "A_LBP256 (jurnal)": ["lbp256_"],
    "B_LBP multi-skala": ["lbpms_"],
    "C_LBP-MS + arah": ["lbpms_", "dir_"],
    "D_Lengkap": ["lbpms_", "dir_", "glcm_"],
}

if __name__ == "__main__":
    df = pd.read_csv(C.PROCESSED_DIR / "splits.csv")
    rows = []
    for _, r in df.iterrows():
        name = r["path"].split("/")[-1].rsplit(".", 1)[0] + ".png"
        gray = cv2.imread(str(C.INTERIM_DIR / "gray" / r["split"] / r["label"] / name), 0)
        names, vec = extract_all(gray)
        rows.append([r["split"], r["label"], name, *vec])
    out = pd.DataFrame(rows, columns=["split", "label", "file", *names])
    out.to_csv(C.PROCESSED_DIR / "features.csv", index=False)
    print("features.csv:", out.shape)
```

### `src/train_classic.py`

```python
"""TAHAP 4 KDD - Data Mining: KNN, SVM, Random Forest + GridSearchCV."""
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import config as C
from src.features import FEATURE_SETS

MODELS = {
    "KNN": (KNeighborsClassifier(metric="euclidean"),           # Euclidean Distance seperti jurnal
            {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]}),
    "SVM": (SVC(probability=True, random_state=C.RANDOM_STATE),
            {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001], "clf__kernel": ["rbf"]}),
    "RandomForest": (RandomForestClassifier(random_state=C.RANDOM_STATE),
                     {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 20]}),
}


def select_cols(df, prefixes):
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


if __name__ == "__main__":
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.PROCESSED_DIR / "features.csv")
    tr, te = df[df.split == "train"], df[df.split == "test"]
    cv = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)

    results, best = [], None
    for fs_name, prefixes in FEATURE_SETS.items():
        cols = select_cols(df, prefixes)
        for m_name, (est, grid) in MODELS.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
            gs.fit(tr[cols], tr["label"])
            pred = gs.predict(te[cols])
            row = dict(fitur=fs_name, model=m_name, cv_f1_macro=round(float(gs.best_score_), 4),
                       test_acc=round(accuracy_score(te["label"], pred), 4),
                       test_f1_macro=round(f1_score(te["label"], pred, average="macro"), 4),
                       best_params=str(gs.best_params_))
            results.append(row)
            print(row)
            if best is None or gs.best_score_ > best[0]:         # pilih berdasar CV, BUKAN test
                best = (gs.best_score_, gs.best_estimator_, cols, fs_name, m_name)

    res = pd.DataFrame(results).sort_values("cv_f1_macro", ascending=False)
    res.to_csv(C.TAB_DIR / "perbandingan_model.csv", index=False)
    _, model, cols, fs_name, m_name = best
    joblib.dump(dict(model=model, columns=cols, feature_set=fs_name, name=m_name,
                     classes=C.CLASSES), C.MODELS_DIR / "best_model.joblib")
    print(f"\nModel terbaik (berdasar CV): {m_name} + {fs_name}")
```

### `src/evaluate.py`

```python
"""TAHAP 5 KDD - Evaluasi & interpretasi pada data uji."""
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import config as C

if __name__ == "__main__":
    bundle = joblib.load(C.MODELS_DIR / "best_model.joblib")
    df = pd.read_csv(C.PROCESSED_DIR / "features.csv")
    te = df[df.split == "test"].reset_index(drop=True)
    pred = bundle["model"].predict(te[bundle["columns"]])

    rep = classification_report(te["label"], pred, labels=C.CLASSES, digits=3)
    (C.TAB_DIR / "classification_report.txt").write_text(rep)
    print(rep)

    cm = confusion_matrix(te["label"], pred, labels=C.CLASSES)
    ConfusionMatrixDisplay(cm, display_labels=C.CLASSES).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {bundle['name']} ({bundle['feature_set']})")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "confusion_matrix.png", dpi=150)

    salah = te["label"].values != pred
    err = te.loc[salah, ["file", "label"]].assign(prediksi=pred[salah])
    err.to_csv(C.TAB_DIR / "salah_prediksi.csv", index=False)   # bahan analisis kesalahan
    print(f"{len(err)} citra salah prediksi -> reports/tables/salah_prediksi.csv")
```

### `src/predict.py`

```python
"""TAHAP 6 KDD - Penerapan: prediksi 1 foto ban baru.
Pakai: python -m src.predict path/foto.jpg"""
import sys
import cv2
import joblib
import pandas as pd
import config as C
from src.preprocessing import preprocess_gray
from src.features import extract_all

PESAN = {"baik": "Ban masih layak pakai.",
         "sedang": "Ban mulai aus - pantau dan rencanakan penggantian.",
         "aus": "Ban sudah aus/gundul - SEGERA GANTI demi keselamatan."}

if __name__ == "__main__":
    bundle = joblib.load(C.MODELS_DIR / "best_model.joblib")
    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit("File gambar tidak terbaca.")
    names, vec = extract_all(preprocess_gray(img))
    X = pd.DataFrame([vec], columns=names)[bundle["columns"]]
    label = bundle["model"].predict(X)[0]
    print("Prediksi:", label.upper(), "-", PESAN[label])
    if hasattr(bundle["model"], "predict_proba"):
        for c, p in zip(bundle["model"].classes_, bundle["model"].predict_proba(X)[0]):
            print(f"  {c:7s}: {p:.1%}")
```

### `src/train_cnn.py` (OPSIONAL - belum diuji, butuh TensorFlow)

```python
"""TAHAP 4 (jalur opsional) - CNN transfer learning MobileNetV2 (butuh tensorflow)."""
import tensorflow as tf
import config as C

TR = C.INTERIM_DIR / "rgb" / "train"
TE = C.INTERIM_DIR / "rgb" / "test"
SIZE = (C.IMG_SIZE, C.IMG_SIZE)

train = tf.keras.utils.image_dataset_from_directory(
    TR, validation_split=0.2, subset="training", seed=C.RANDOM_STATE,
    image_size=SIZE, batch_size=16, class_names=C.CLASSES, label_mode="categorical")
val = tf.keras.utils.image_dataset_from_directory(
    TR, validation_split=0.2, subset="validation", seed=C.RANDOM_STATE,
    image_size=SIZE, batch_size=16, class_names=C.CLASSES, label_mode="categorical")
test = tf.keras.utils.image_dataset_from_directory(
    TE, image_size=SIZE, batch_size=16, class_names=C.CLASSES,
    label_mode="categorical", shuffle=False)

augment = tf.keras.Sequential([                  # augmentasi HANYA pada data latih
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.15)])

base = tf.keras.applications.MobileNetV2(input_shape=SIZE + (3,), include_top=False,
                                         weights="imagenet")
base.trainable = False
inp = tf.keras.Input(SIZE + (3,))
x = augment(inp)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(len(C.CLASSES), activation="softmax")(x)
model = tf.keras.Model(inp, out)
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
model.fit(train, validation_data=val, epochs=30, callbacks=cb)
print("Evaluasi data uji:", model.evaluate(test))
C.MODELS_DIR.mkdir(exist_ok=True)
model.save(C.MODELS_DIR / "cnn_mobilenetv2.keras")
```

---

## Referensi

1. A. Sitompul, "Implementasi Pengenalan Jenis Pola Tapak Ban (Tread) Menggunakan Metode Local Binary Patterns," *Jurnal Sistem Komputer dan Informatika (JSON)*, vol. 1, no. 3, hal. 189-194, Mei 2020. DOI: 10.30865/json.v1i3.2094.
2. T. Ojala, M. Pietikainen, T. Maenpaa, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE Trans. PAMI*, vol. 24, no. 7, 2002 (sumber asli LBP, disarankan dibaca untuk Bab II).
3. U. Fayyad, G. Piatetsky-Shapiro, P. Smyth, "From Data Mining to Knowledge Discovery in Databases," *AI Magazine*, vol. 17, no. 3, 1996 (rujukan proses KDD).
