# SentimenID API

Proyek UAS Machine Learning end-to-end untuk mengklasifikasikan teks Bahasa Indonesia ke dalam tiga kelas sentimen: `negative`, `neutral`, dan `positive`.

## Identitas

- **Nama:** Panji Arya Soma
- **NIM:** 1003240008
- **Repository:** `UAS-ML-1003240008`
- **Kasus:** Case C — Klasifikasi Sentimen Teks Bahasa Indonesia

## Berkas Pengumpulan

- [Laporan PDF](Laporan_UAS_ML_SentimenID_Panji_Arya_Soma.pdf)
- [Link Video Demonstrasi](LINK_VIDEO.md)


## Masalah Bisnis

Media, marketplace, dan layanan digital dapat menerima banyak ulasan atau komentar berbahasa Indonesia. Pemeriksaan manual membutuhkan waktu, sulit diskalakan, dan dapat menghasilkan keputusan yang tidak konsisten.

SentimenID API menyediakan baseline machine learning untuk membantu proses triage awal terhadap teks masuk. Sistem ini tidak dimaksudkan sebagai pengganti keputusan manusia atau sebagai mesin yang memahami konteks bahasa secara sempurna.

## Ringkasan Solusi

Alur proyek mencakup:

1. pengunduhan dan validasi dataset publik;
2. audit kualitas data dan EDA;
3. normalisasi teks Bahasa Indonesia;
4. representasi TF-IDF unigram dan bigram;
5. perbandingan tiga model melalui stratified 5-fold cross-validation;
6. evaluasi satu kali pada test set terkunci;
7. penyimpanan pipeline utuh;
8. serving melalui FastAPI;
9. mechanical test, behavioral test, dan privacy-safe logging test.

Model final adalah **Calibrated Linear SVC**, yaitu `LinearSVC` yang dikalibrasi agar dapat menghasilkan probabilitas kelas.

## Dataset

Dataset yang digunakan adalah **IndoNLU SmSA** (`smsa_doc-sentiment-prosa`) dari IndoNLP. Dataset berisi teks Bahasa Indonesia dengan tiga label:

- `negative`
- `neutral`
- `positive`

Sumber canonical:

- https://github.com/IndoNLP/indonlu/tree/master/dataset/smsa_doc-sentiment-prosa

File `train_preprocess.tsv` dan `valid_preprocess.tsv` diunduh oleh `src/load_data.py`, diverifikasi, lalu digabungkan. Test resmi IndoNLU tidak digunakan karena labelnya disamarkan.

### Profil Data

- Data awal berlabel: **12.260 baris**
- Exact duplicate yang dihapus: **81 baris**
- Data setelah kurasi: **12.179 baris**
- Train set: **9.743 baris**
- Test set terkunci: **2.436 baris**
- Missing value standar: **tidak ditemukan**
- Konflik label: **tidak ditemukan**
- Baris dengan placeholder token: **45**
- Baris dengan pemanjangan huruf: **103**

### Lisensi dan Atribusi

- Lisensi benchmark dataset IndoNLU: **MIT**, berdasarkan dataset card resmi.
- Lisensi repository/code upstream IndoNLU: **Apache License 2.0**, berdasarkan file `LICENSE` pada repository IndoNLU.
- Detail atribusi tersedia pada [`DATASET_ATTRIBUTION.md`](DATASET_ATTRIBUTION.md).
- Lisensi kode repository ini mengikuti file [`LICENSE`](LICENSE).

Sitasi dataset:

> Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim, Z. Y., Soleman, S., Mahendra, R., Fung, P., Bahar, S., dan Purwarianti, A. (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding*.

## Hasil EDA dan Preprocessing

Temuan utama:

- kelas `positive` merupakan kelas terbesar;
- kelas `neutral` merupakan kelas terkecil;
- exact duplicate dihapus sebelum split;
- negasi formal dipertahankan;
- variasi negasi informal dinormalisasi secara terbatas;
- placeholder, whitespace, lowercase, dan pemanjangan huruf ditangani tanpa pembersihan agresif.

Seluruh preprocessing yang dipelajari berada di dalam `sklearn.Pipeline` untuk mencegah perbedaan antara training dan inferensi.

Dokumentasi rinci:

- [`reports/eda_findings.md`](reports/eda_findings.md)
- [`reports/preprocessing_findings.md`](reports/preprocessing_findings.md)

## Pemilihan Model

Tiga kandidat dibandingkan menggunakan stratified 5-fold cross-validation pada train set:

| Model | Mean F1-macro CV | Std. deviasi |
|---|---:|---:|
| Calibrated Linear SVC | 0.850229 | 0.009328 |
| Logistic Regression | 0.847877 | 0.007200 |
| Multinomial Naive Bayes | 0.754436 | 0.014869 |

Calibrated Linear SVC dipilih karena memperoleh F1-macro CV terbaik, tetap dapat memberikan probabilitas kelas, dan dapat disimpan sebagai pipeline utuh.

Dokumentasi rinci tersedia di [`reports/training_findings.md`](reports/training_findings.md).

## Evaluasi Final

Test set hanya digunakan setelah model dan konfigurasi dikunci.

| Metrik | Nilai |
|---|---:|
| Accuracy | 0.889163 |
| F1-macro | 0.855483 |
| F1-weighted | 0.888721 |
| Log loss | 0.312220 |
| Prediksi benar | 2.166 |
| Prediksi salah | 270 |

Performa per kelas:

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| negative | 0.859230 | 0.851316 | 0.855254 | 760 |
| neutral | 0.805785 | 0.767717 | 0.786290 | 254 |
| positive | 0.918806 | 0.931083 | 0.924904 | 1.422 |

Kelas `neutral` menjadi kelas tersulit. Kesalahan juga ditemukan pada negasi komposisional, teks panjang dengan sentimen campuran, teks lintas-domain, dan label ambigu.

Dokumentasi rinci tersedia di [`reports/evaluation_findings.md`](reports/evaluation_findings.md).

## Environment yang Diuji

Runtime utama:

- Python `3.12.7`
- NumPy `2.5.1`
- pandas `3.0.5`
- scikit-learn `1.9.0`
- Matplotlib `3.11.1`
- joblib `1.5.3`
- pytest `9.1.1`
- FastAPI `0.140.0`
- Uvicorn `0.51.0`
- Pydantic `2.13.4`
- httpx2 `2.7.0`

Versi yang digunakan untuk training dan serving harus konsisten karena artefak `joblib` dapat sensitif terhadap perbedaan versi library.

## Struktur Repository

```text
UAS-ML-1003240008/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── data/
├── models/
│   └── metadata.json
├── notebooks/
├── reports/
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── load_data.py
│   ├── train.py
│   └── transformers.py
├── tests/
│   ├── test_api.py
│   ├── test_behavior.py
│   ├── test_eda.py
│   ├── test_evaluate.py
│   ├── test_load_data.py
│   ├── test_logging.py
│   ├── test_train.py
│   └── test_transformers.py
├── .gitignore
├── DATASET_ATTRIBUTION.md
├── LICENSE
├── PRD.md
├── README.md
├── requirements-api.txt
└── requirements.txt
```

### Folder yang Diabaikan Git

Folder `data/` dan artefak hasil training pada `models/` diabaikan agar repository tidak menyimpan file hasil unduhan atau artefak biner yang dapat dibuat ulang. Penguji dapat memproduksi ulang isinya secara deterministik dengan menjalankan `python -m src.load_data` dan `python -m src.train`; metadata ringan dan laporan tetap dikomit sebagai bukti konfigurasi serta hasil eksperimen.

Virtual environment, Python cache, pytest cache, file environment, dan konfigurasi editor juga diabaikan karena bersifat lokal dan tidak diperlukan untuk reproduksi proyek.

## Menjalankan Proyek dari Nol

Instruksi berikut ditulis untuk Windows PowerShell.

### 1. Clone repository

```powershell
git clone <URL_REPOSITORY>
cd UAS-ML-1003240008
```

Ganti `<URL_REPOSITORY>` dengan URL repository GitHub proyek.

### 2. Buat virtual environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Apabila PowerShell menolak aktivasi script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 3. Instal dependency

`requirements-api.txt` memanggil `requirements.txt`, sehingga satu perintah berikut memasang dependency training, testing, dan API:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-api.txt
```

### 4. Unduh dan audit dataset

```powershell
python -m src.load_data
```

Perintah ini mengunduh data resmi, memverifikasi sumber, melakukan audit awal, menghapus exact duplicate secara deterministik, dan membuat split train/test.

### 5. Jalankan EDA

```powershell
python -m src.eda
```

Grafik dan temuan EDA disimpan di `reports/`.

### 6. Latih model

```powershell
python -m src.train
```

Perintah ini membandingkan tiga kandidat melalui cross-validation, memilih model final, lalu membuat:

```text
models/model.joblib
models/metadata.json
```

### 7. Jalankan evaluasi final

```powershell
python -m src.evaluate
```

Perintah ini menghasilkan classification report, confusion matrix, prediction table, dan error analysis di `reports/`.

Test set tidak boleh digunakan untuk mengubah model setelah evaluasi final dilakukan.

### 8. Jalankan automated test

```powershell
python -m pytest tests/ -q
```

Hasil environment terakhir:

```text
40 passed, 33 warnings
```

Warning yang terlihat berasal dari kompatibilitas internal `joblib` dengan NumPy 2.5 dan tidak menyebabkan test gagal.

### 9. Jalankan API

```powershell
python -m uvicorn app.main:app --reload
```

Dokumentasi interaktif:

- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- Health check: http://127.0.0.1:8000/health

## Endpoint API

| Method | Endpoint | Fungsi |
|---|---|---|
| `GET` | `/` | informasi layanan |
| `GET` | `/health` | status model dan versi |
| `POST` | `/predict-teks` | prediksi sentimen |

### Request Valid — HTTP 200

Gunakan `curl.exe`, bukan alias `curl` PowerShell:

```powershell
curl.exe -i -X POST "http://127.0.0.1:8000/predict-teks" `
  -H "Content-Type: application/json" `
  -d '{"text":"aplikasinya tidak bagus dan sering error","language":"id"}'
```

Bentuk respons:

```json
{
  "prediction": "negative",
  "confidence": 0.0,
  "probabilities": {
    "negative": 0.0,
    "neutral": 0.0,
    "positive": 0.0
  },
  "model_version": "<versi-model>"
}
```

Nilai probabilitas pada contoh hanya menunjukkan struktur respons. Nilai aktual berasal dari model yang dijalankan.

### Request Invalid — HTTP 422

Contoh request tanpa field `text`:

```powershell
curl.exe -i -X POST "http://127.0.0.1:8000/predict-teks" `
  -H "Content-Type: application/json" `
  -d '{"language":"id"}'
```

Contoh request dengan bahasa yang tidak didukung:

```powershell
curl.exe -i -X POST "http://127.0.0.1:8000/predict-teks" `
  -H "Content-Type: application/json" `
  -d '{"text":"contoh teks","language":"en"}'
```

Keduanya harus mengembalikan:

```text
HTTP/1.1 422 Unprocessable Entity
```

## Pengujian

Test suite mencakup:

- pemuatan dan validasi dataset;
- audit kualitas data dan EDA;
- normalisasi teks;
- training dan pemilihan model;
- evaluasi final;
- kontrak endpoint FastAPI;
- behavioral test untuk negasi, keluhan, dan pujian;
- privacy-safe prediction logging.

Logging prediksi mencatat:

- `timestamp_utc`
- `latency_ms`
- `text_length`
- `language`
- `prediction`
- `confidence`
- `model_version`

Teks mentah pengguna tidak disimpan pada log.

Detail pengujian tersedia di [`reports/testing_findings.md`](reports/testing_findings.md).

## Keterbatasan

- Model hanya ditujukan untuk teks Bahasa Indonesia.
- Model dapat gagal pada sarkasme, ironi, code-switching, dan konteks yang sangat panjang.
- Kosakata baru atau domain di luar data training dapat menurunkan kualitas prediksi.
- Kelas `neutral` memiliki performa paling rendah.
- Confidence bukan jaminan bahwa prediksi benar.
- Sistem belum diuji untuk traffic tinggi atau concurrent requests.
- API ini merupakan baseline akademik, bukan layanan produksi skala besar.

## Privasi dan Penggunaan

Dataset bersifat publik dan proyek tidak mengumpulkan data pribadi baru. API tidak menyimpan isi teks mentah pengguna dalam prediction log.

Hasil klasifikasi sebaiknya digunakan untuk triage awal dan tetap ditinjau manusia ketika keputusan memiliki dampak penting.
