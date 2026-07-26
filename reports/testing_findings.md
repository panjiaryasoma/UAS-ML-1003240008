# Testing Findings

## 1. Tujuan Pengujian

Pengujian dilakukan untuk memastikan bahwa seluruh komponen proyek klasifikasi sentimen berjalan sesuai kontrak yang telah ditetapkan, mulai dari pemuatan data, exploratory data analysis (EDA), preprocessing, training, evaluasi model, layanan FastAPI, behavioral testing, hingga logging prediksi yang aman terhadap privasi.

Fokus utama pengujian adalah:

- memastikan setiap modul dapat dijalankan secara konsisten;
- mencegah kebocoran data antara train set dan test set;
- memastikan pipeline preprocessing yang sama digunakan saat training dan inferensi;
- memastikan endpoint API memberikan respons yang benar;
- memastikan perilaku model sesuai arah sentimen yang diharapkan;
- memastikan log operasional tidak menyimpan teks mentah pengguna;
- memastikan seluruh test dapat dijalankan secara otomatis menggunakan `pytest`.

---

## 2. Struktur Test Suite

Test suite disimpan pada direktori `tests/` dan mencakup beberapa kelompok pengujian berikut.

| File pengujian | Ruang lingkup |
|---|---|
| `tests/test_load_data.py` | pembacaan dataset dan validasi label |
| `tests/test_eda.py` | audit kualitas data, pembersihan duplikat, split, dan artefak EDA |
| `tests/test_transformers.py` | normalisasi teks, bentuk input, negasi, serialisasi, dan kompatibilitas pipeline |
| `tests/test_train.py` | kandidat model, cross-validation, pemilihan model, dan artefak training |
| `tests/test_evaluate.py` | evaluasi final, metrik, probabilitas, contoh prediksi, dan artefak evaluasi |
| `tests/test_api.py` | endpoint FastAPI, validasi request, model loading, dan response schema |
| `tests/test_behavior.py` | perilaku probabilitas model terhadap negasi, keluhan, dan pujian |
| `tests/test_logging.py` | logging metadata prediksi tanpa menyimpan raw text |

---

## 3. Mechanical API Testing

Mechanical testing digunakan untuk memeriksa apakah API memenuhi kontrak teknisnya, terlepas dari apakah prediksi tertentu dianggap baik secara semantik.

### 3.1 Endpoint sistem

Endpoint berikut telah diuji:

- `GET /`
- `GET /health`
- `POST /predict-teks`

Hasil yang diverifikasi:

- `GET /` mengembalikan informasi layanan;
- `GET /health` mengembalikan status `200`;
- `GET /health` menyatakan bahwa model telah dimuat;
- `GET /health` mengembalikan `model_version`;
- `POST /predict-teks` mengembalikan label prediksi;
- respons prediksi memuat `confidence`;
- respons prediksi memuat probabilitas setiap kelas;
- respons prediksi memuat versi model.

### 3.2 Validasi request

Request yang tidak memenuhi schema menghasilkan status `422 Unprocessable Entity`.

Kasus invalid yang diuji mencakup:

- teks kosong;
- field `text` hilang;
- field `language` hilang;
- kode bahasa tidak didukung;
- tipe data field tidak sesuai;
- payload kosong;
- bentuk payload tidak sesuai schema.

Validasi request dilakukan oleh Pydantic sebelum data diteruskan ke model. Dengan demikian, model tidak menerima input yang kosong atau tidak valid.

### 3.3 Model loading

Model dimuat satu kali melalui mekanisme lifespan FastAPI. Pengujian memverifikasi bahwa:

- model tersedia selama aplikasi berjalan;
- metadata model ikut dimuat;
- urutan kelas tersedia;
- request prediksi tidak melakukan load model ulang;
- endpoint menggunakan pipeline yang telah disimpan.

### 3.4 Pipeline preprocessing

API tidak melakukan preprocessing manual secara terpisah. Teks asli diteruskan langsung ke pipeline yang telah menyimpan:

1. `IndonesianTextNormalizer`;
2. TF-IDF vectorizer;
3. classifier terpilih.

Desain ini mencegah perbedaan preprocessing antara tahap training dan tahap inferensi.

---

## 4. Behavioral Testing

Mechanical test hanya membuktikan bahwa program berjalan. Behavioral test digunakan untuk memeriksa apakah arah perubahan probabilitas model masih masuk akal.

### 4.1 Pengujian negasi

Test memeriksa bahwa penambahan negasi pada kalimat menggeser probabilitas ke arah kelas `negative`.

Contoh hubungan yang diuji secara konseptual:

- pernyataan positif tanpa negasi;
- pernyataan serupa setelah ditambahkan negasi.

Assertion tidak mewajibkan label tertentu secara kaku. Test membandingkan arah perubahan probabilitas agar tidak terlalu rapuh terhadap retraining model.

### 4.2 Keluhan dibandingkan pujian

Test membandingkan:

- teks keluhan;
- teks pujian.

Hasil yang diwajibkan adalah probabilitas kelas `negative` pada keluhan lebih tinggi daripada probabilitas kelas `negative` pada pujian.

Pendekatan ini lebih informatif daripada hanya memeriksa satu label hasil prediksi, karena menunjukkan hubungan semantik antara dua input.

### 4.3 Hasil behavioral testing

Dua behavioral test berhasil dijalankan:

- `test_negation_moves_probability_toward_negative`;
- `test_complaint_has_higher_negative_probability_than_praise`.

Hasil:

- passed: 2;
- failed: 0.

---

## 5. Privacy-Safe Prediction Logging

Logging ditambahkan untuk kebutuhan observabilitas dan audit operasional tanpa menyimpan isi teks mentah pengguna.

### 5.1 Metadata yang dicatat

Setiap prediksi berhasil mencatat:

- `timestamp_utc`;
- `latency_ms`;
- `text_length`;
- `language`;
- `prediction`;
- `confidence`;
- `model_version`.

Format log prediksi mengikuti pola berikut:

```text
prediction_completed timestamp_utc=<timestamp> latency_ms=<value> text_length=<value> language=<value> prediction=<label> confidence=<value> model_version=<version>
```

### 5.2 Informasi yang tidak dicatat

Teks asli pengguna tidak dimasukkan ke dalam log.

Hal ini diverifikasi melalui test yang:

1. mengirim teks khusus ke endpoint prediksi;
2. menangkap log menggunakan `caplog`;
3. memastikan metadata prediksi tersedia;
4. memastikan teks mentah tidak muncul di log.

Dengan pendekatan ini, log masih berguna untuk debugging dan monitoring tanpa menciptakan salinan tersembunyi dari data pengguna.

---

## 6. Hasil Pengujian Keseluruhan

Perintah yang digunakan:

```powershell
python -m pytest tests/ -q
```

Hasil akhir:

| Metrik | Nilai |
|---|---:|
| Total test | 40 |
| Passed | 40 |
| Failed | 0 |
| Tingkat kelulusan | 100% |
| Warnings | 33 |
| Durasi | sekitar 4,10 detik |

Output akhir:

```text
40 passed, 33 warnings in 4.10s
```

Seluruh test yang dikumpulkan oleh `pytest` berhasil dijalankan tanpa kegagalan.

---

## 7. Warning yang Ditemukan

Warning yang muncul adalah `DeprecationWarning` dari proses internal `joblib` ketika menangani array NumPy.

Pesan utama warning:

```text
Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
```

Warning tersebut muncul ketika artefak model dibaca dan digunakan selama pengujian.

### 7.1 Dampak warning

Warning tidak menyebabkan:

- kegagalan load model;
- perubahan schema API;
- kegagalan prediksi;
- probabilitas tidak valid;
- test gagal.

Seluruh 40 test tetap lulus.

### 7.2 Interpretasi

Warning berasal dari kompatibilitas implementasi internal dependency dengan NumPy 2.5, bukan dari logika aplikasi yang ditulis pada proyek ini.

Karena proyek telah menggunakan environment pin dan seluruh test berhasil, warning dicatat sebagai isu kompatibilitas dependency yang perlu dipantau, bukan sebagai defect fungsional.

---

## 8. Batasan Pengujian

Walaupun seluruh test lulus, hasil tersebut tidak berarti model bebas dari kesalahan.

Batasan yang masih berlaku meliputi:

- behavioral test hanya mencakup beberapa hubungan dasar;
- sarkasme belum dapat diuji secara menyeluruh;
- bahasa campuran dan slang baru mungkin tidak terwakili;
- teks sangat pendek dapat memiliki probabilitas yang tidak stabil;
- label sentimen tertentu dapat bersifat ambigu;
- test belum mengukur beban tinggi atau concurrent requests;
- test belum menjadi bukti bahwa model cocok untuk seluruh domain teks Bahasa Indonesia.

Dengan demikian, kelulusan test menunjukkan implementasi konsisten terhadap kontrak proyek, bukan jaminan kesempurnaan model.

---

## 9. Kesimpulan

Test suite proyek telah mencakup alur utama machine learning dan API, yaitu:

- validasi dan audit data;
- preprocessing;
- training dan cross-validation;
- evaluasi final;
- penyimpanan artefak;
- inferensi melalui FastAPI;
- validasi input;
- behavioral testing;
- privacy-safe logging.

Hasil akhir menunjukkan:

- 40 test berhasil;
- tidak ada test gagal;
- endpoint valid mengembalikan `200`;
- request invalid mengembalikan `422`;
- model dimuat satu kali;
- pipeline training digunakan kembali oleh API;
- perilaku dasar sentimen memenuhi assertion;
- log operasional tidak menyimpan teks mentah pengguna.

Berdasarkan hasil tersebut, implementasi telah memenuhi kontrak pengujian yang ditetapkan dan siap dilanjutkan ke tahap dokumentasi README, penyusunan laporan PDF, video demonstrasi, serta clone test pada environment baru.
