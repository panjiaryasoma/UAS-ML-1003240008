# Training Findings — SentimenID API

**Proyek:** UAS Machine Learning End-to-End  
**Kasus:** Case C — Klasifikasi Sentimen Teks Bahasa Indonesia  
**Dataset:** IndoNLU SmSA  
**Status:** Training dan model selection selesai; test set belum dievaluasi  
**Model terpilih:** `calibrated_linear_svc`  
**Versi model:** `sentimenid-20260725T181613Z`  
**Waktu pembuatan:** `2026-07-25T18:16:13+00:00`  

---

## 1. Tujuan

Tahap ini bertujuan membandingkan minimal tiga algoritma klasifikasi teks menggunakan train set saja, memilih konfigurasi terbaik berdasarkan cross-validation, lalu menyimpan pipeline final beserta metadata reproduksi.

Test set tidak digunakan untuk pemilihan model, tuning, maupun interpretasi hasil pada tahap ini. Status test set tetap:

```text
locked_until_evaluate.py
```

---

## 2. Data yang Digunakan

Setelah exact duplicate dihapus, jumlah data modeling adalah:

| Partisi | Jumlah |
|---|---:|
| Data modeling | 12.179 |
| Train set | 9.743 |
| Test set terkunci | 2.436 |

Split dilakukan dengan konfigurasi:

```text
test_size = 0.2
stratified = True
random_state = 42
```

Hanya train set yang digunakan pada training dan cross-validation.

---

## 3. Pipeline Kandidat

Seluruh kandidat memakai alur:

```text
IndonesianTextNormalizer
→ TfidfVectorizer
→ classifier
```

Untuk Calibrated Linear SVC, pipeline lengkap ditempatkan di dalam `CalibratedClassifierCV` agar preprocessing dan vectorizer tetap dipelajari secara benar pada fold kalibrasi.

Classifier yang dibandingkan:

1. Multinomial Naive Bayes;
2. Logistic Regression;
3. Calibrated Linear SVC.

Seluruh artefak model menyimpan preprocessing lengkap, bukan hanya classifier.

---

## 4. Kontrak Cross-Validation

Model selection menggunakan:

```text
scoring = f1_macro
cv = StratifiedKFold
n_splits = 5
shuffle = True
random_state = 42
calibration_cv = 3
```

Metrik utama adalah **F1-macro** karena distribusi kelas tidak seimbang dan kelas `neutral` jauh lebih sedikit dibandingkan `positive`.

---

## 5. Hasil Cross-Validation

| Peringkat | Model | Mean F1-macro | Std F1-macro | Mean fit time | Total search time |
|---:|---|---:|---:|---:|---:|
| 1 | `calibrated_linear_svc` | 0.850229 | 0.009328 | 4.531 detik | 10.634 detik |
| 2 | `logistic_regression` | 0.847877 | 0.007200 | 2.518 detik | 6.799 detik |
| 3 | `multinomial_nb` | 0.754436 | 0.014869 | 1.148 detik | 7.974 detik |

Model dengan rata-rata F1-macro tertinggi adalah **Calibrated Linear SVC** sebesar **0.850229**.

Logistic Regression berada sangat dekat dengan skor **0.847877**. Selisih keduanya hanya **0.002352**, atau sekitar **0.235 poin persentase**.

Karena kriteria pemilihan telah dikunci berdasarkan mean F1-macro tertinggi, Calibrated Linear SVC dipilih. Namun, hasil ini harus ditulis sebagai kemenangan tipis, bukan perbedaan performa yang besar.

---

## 6. Parameter Terbaik

### 6.1 Calibrated Linear SVC

```text
C = 0.5
ngram_range = (1, 2)
class_weight = balanced
calibration_method = sigmoid
calibration_cv = 3
random_state = 42
```

Konfigurasi ini menjadi pipeline final yang disimpan ke `models/model.joblib`.

### 6.2 Logistic Regression

```text
C = 2.0
ngram_range = (1, 2)
class_weight = balanced
max_iter = 2000
random_state = 42
```

### 6.3 Multinomial Naive Bayes

```text
alpha = 0.5
ngram_range = (1, 1)
```

Dua model teratas memilih kombinasi unigram dan bigram, sedangkan Multinomial Naive Bayes memilih unigram saja.

---

## 7. Alasan Pemilihan Model

Calibrated Linear SVC dipilih karena:

1. memiliki mean F1-macro tertinggi;
2. mendukung probabilitas melalui kalibrasi sigmoid;
3. memakai pipeline utuh dengan normalizer dan TF-IDF;
4. memakai unigram dan bigram;
5. telah di-refit pada seluruh train set setelah konfigurasi terbaik ditemukan.

Standard deviation Calibrated Linear SVC sedikit lebih tinggi daripada Logistic Regression. Artinya, Logistic Regression sedikit lebih stabil antar-fold, tetapi skor rata-ratanya tetap lebih rendah.

Keputusan final tetap mengikuti aturan yang ditetapkan sebelumnya: mean F1-macro tertinggi diprioritaskan, lalu standard deviation digunakan sebagai tie-breaker.

---

## 8. Evaluasi Prakiraan Sebelum Training

### 8.1 Bigram negasi akan membantu

Prakiraan ini **mendapat dukungan awal**, tetapi belum terbukti secara kausal.

Dua model terbaik, Calibrated Linear SVC dan Logistic Regression, sama-sama memilih:

```text
ngram_range = (1, 2)
```

Temuan tersebut konsisten dengan dugaan bahwa konteks bigram membantu membedakan frasa seperti `tidak bagus` dan `tidak mengecewakan`. Namun, bukti final tetap memerlukan error analysis setelah evaluasi test set.

### 8.2 Kelas neutral kemungkinan paling sulit

Prakiraan ini **belum dapat diputuskan** pada tahap training.

Skor cross-validation yang tersimpan adalah F1-macro agregat, bukan precision, recall, dan F1 per kelas. Kesulitan kelas `neutral` baru boleh disimpulkan setelah classification report dan confusion matrix pada test set tersedia.

### 8.3 Normalisasi informal terbatas akan membantu

Prakiraan ini **belum terbukti sebagai peningkatan performa**.

Normalizer telah terbukti bekerja secara teknis dan menjadi bagian dari semua pipeline kandidat. Namun, training saat ini tidak membandingkan pipeline dengan normalizer melawan pipeline tanpa normalizer. Karena itu, manfaat kausal normalisasi tidak boleh diklaim dari hasil ini saja.

---

## 9. Artefak Training

Artefak yang dihasilkan:

```text
models/
├── model.joblib
└── metadata.json

reports/
└── model_selection_results.csv
```

### 9.1 `model.joblib`

Berisi model final beserta preprocessing lengkap:

```text
CalibratedClassifierCV
└── Pipeline
    ├── IndonesianTextNormalizer
    ├── TfidfVectorizer
    └── LinearSVC
```

### 9.2 `metadata.json`

Mencatat:

- versi model;
- waktu training;
- dataset;
- versi runtime;
- konfigurasi split;
- status test set;
- hasil seluruh kandidat;
- parameter terbaik;
- urutan kelas;
- informasi pipeline.

Urutan kelas model:

```text
negative, neutral, positive
```

### 9.3 `model_selection_results.csv`

Menyimpan ringkasan hasil terbaik setiap kandidat agar model selection dapat diaudit tanpa menjalankan training ulang.

---

## 10. Runtime Reproduksi

Training dijalankan menggunakan:

```text
Python = 3.12.7
scikit-learn = 1.9.0
pandas = 3.0.5
```

Artefak joblib sebaiknya dimuat menggunakan versi dependency yang sama atau kompatibel. Perbedaan versi scikit-learn dapat memunculkan `InconsistentVersionWarning` atau masalah kompatibilitas serialisasi.

---

## 11. Hal yang Belum Boleh Diklaim

Pada checkpoint ini, belum boleh diklaim bahwa:

- model memiliki F1-macro final sebesar 0.850229 pada data baru;
- kelas `neutral` pasti paling sulit;
- normalisasi pasti meningkatkan performa;
- bigram pasti memperbaiki semua kasus negasi;
- model siap produksi tanpa evaluasi test set.

Skor yang tersedia masih merupakan hasil cross-validation pada train set.

---

## 12. Langkah Berikutnya

Tahap selanjutnya adalah evaluasi final satu kali pada test set terkunci melalui:

```text
src/evaluate.py
```

Evaluasi harus menghasilkan:

- F1-macro final;
- accuracy sebagai metrik sekunder;
- classification report;
- confusion matrix PNG;
- contoh prediksi benar dan salah;
- error analysis;
- pemeriksaan ulang tiga prakiraan sebelum training.

Setelah evaluasi dilakukan, test set tidak boleh digunakan kembali untuk mengubah model atau parameter tanpa membuat split evaluasi baru.

---

## 13. Status Checkpoint

- Tiga kandidat model: selesai.
- Stratified 5-fold CV: selesai.
- F1-macro sebagai metrik utama: selesai.
- Unigram dan bigram diuji: selesai.
- Model final dipilih: selesai.
- Pipeline lengkap disimpan: selesai.
- Metadata reproduksi disimpan: selesai.
- Test set masih terkunci: ya.
- Tahap berikutnya: evaluasi final satu kali.
