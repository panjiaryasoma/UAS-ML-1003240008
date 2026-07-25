# API Findings

## 1. Tujuan

Dokumen ini mencatat desain, pengujian, dan hasil aktual REST API untuk klasifikasi sentimen teks Bahasa Indonesia. API dibangun menggunakan FastAPI dan memuat pipeline machine learning yang telah disimpan pada `models/model.joblib`.

API berfungsi sebagai lapisan inferensi. Seluruh preprocessing tetap dilakukan oleh pipeline tersimpan sehingga tidak ada duplikasi logika preprocessing di endpoint.

## 2. Ruang Lingkup

Pengujian API mencakup:

- pemuatan model saat aplikasi dimulai;
- pemeriksaan kondisi layanan;
- prediksi sentimen melalui request JSON;
- pengembalian probabilitas setiap kelas;
- validasi teks kosong;
- validasi kode bahasa;
- konsistensi versi model;
- pengujian otomatis dengan `pytest`;
- pengujian manual melalui Swagger UI.

Pengujian tidak mencakup deployment ke server publik, autentikasi pengguna, rate limiting, logging terpusat, dan pengujian beban.

## 3. Struktur Implementasi

Komponen utama API adalah:

```text
app/
├── __init__.py
├── main.py
└── schemas.py

tests/
└── test_api.py

models/
├── model.joblib
└── metadata.json
```

Peran setiap komponen:

- `app/main.py` mendefinisikan aplikasi FastAPI, lifecycle, serta endpoint.
- `app/schemas.py` mendefinisikan request dan response menggunakan Pydantic.
- `models/model.joblib` menyimpan seluruh pipeline inferensi.
- `models/metadata.json` menyimpan informasi versi dan metadata model.
- `tests/test_api.py` menguji kontrak dan perilaku API.

## 4. Lingkungan Pengujian

Pengujian dilakukan pada lingkungan berikut:

| Komponen | Versi |
|---|---:|
| Python | 3.12.7 |
| FastAPI | 0.140.0 |
| Pydantic | 2.13.4 |
| Uvicorn | 0.51.0 |
| pytest | 9.1.1 |

Server pengembangan dijalankan dengan perintah:

```powershell
python -m uvicorn app.main:app --reload
```

Swagger UI tersedia pada:

```text
http://127.0.0.1:8000/docs
```

## 5. Endpoint API

### 5.1 `GET /`

Endpoint ini menyediakan informasi dasar mengenai layanan API.

Hasil pengujian otomatis:

```text
test_root_returns_service_information PASSED
```

### 5.2 `GET /health`

Endpoint ini memeriksa apakah aplikasi aktif dan model berhasil dimuat.

Respons aktual:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "sentimenid-20260725T181613Z"
}
```

Status HTTP:

```text
200 OK
```

Hasil ini membuktikan bahwa model tersedia sebelum endpoint prediksi digunakan.

### 5.3 `POST /predict-teks`

Endpoint ini menerima teks Bahasa Indonesia dan menghasilkan prediksi sentimen.

Contoh request:

```json
{
  "text": "aplikasinya tidak bagus dan sering error",
  "language": "id"
}
```

Struktur respons:

```json
{
  "prediction": "negative",
  "confidence": 0.6397032143194422,
  "probabilities": {
    "negative": 0.6397032143194422,
    "neutral": 0.007033869713914838,
    "positive": 0.3532629159666428
  },
  "model_version": "sentimenid-20260725T181613Z"
}
```

Respons menyediakan:

- label prediksi;
- confidence prediksi terpilih;
- probabilitas seluruh kelas;
- versi model yang digunakan.

## 6. Hasil Pengujian Otomatis

Pengujian khusus API menghasilkan:

```text
12 passed in 0.68s
```

Seluruh test suite proyek menghasilkan:

```text
37 passed, 6 warnings in 3.74s
```

Daftar kontrak API yang berhasil diuji:

```text
test_root_returns_service_information
test_health_reports_loaded_model_and_version
test_predict_returns_prediction_confidence_and_probabilities
test_api_passes_original_text_directly_to_saved_pipeline
test_model_is_loaded_only_once_during_lifespan
test_invalid_requests_return_422[payload0]
test_invalid_requests_return_422[payload1]
test_invalid_requests_return_422[payload2]
test_invalid_requests_return_422[payload3]
test_invalid_requests_return_422[payload4]
test_invalid_requests_return_422[payload5]
test_invalid_requests_return_422[payload6]
```

Temuan utama dari pengujian otomatis:

1. Model berhasil dimuat pada lifecycle aplikasi.
2. Model tidak dimuat ulang pada setiap request.
3. Endpoint prediksi mengembalikan label, confidence, probabilitas, dan versi model.
4. Teks asli diteruskan langsung ke pipeline tersimpan.
5. Request yang tidak memenuhi schema ditolak dengan status `422`.
6. API tidak melakukan preprocessing manual di luar pipeline.

## 7. Hasil Smoke Test Swagger UI

### 7.1 Pemeriksaan layanan

Request:

```http
GET /health
```

Hasil:

```text
200 OK
status = ok
model_loaded = true
```

Interpretasi:

API aktif dan artefak model berhasil dibaca.

### 7.2 Contoh prediksi negatif

Request:

```json
{
  "text": "aplikasinya tidak bagus dan sering error",
  "language": "id"
}
```

Hasil:

| Kelas | Probabilitas |
|---|---:|
| negative | 0.639703 |
| neutral | 0.007034 |
| positive | 0.353263 |

Prediksi:

```text
negative
```

Confidence:

```text
0.639703
```

Teks mengandung negasi dan keluhan langsung. Prediksi negatif konsisten dengan makna kalimat.

Namun, probabilitas positif masih mencapai sekitar `0.353`. Hal ini menunjukkan bahwa keputusan benar tetapi tidak sepenuhnya tegas. Kemungkinan penyebabnya adalah token positif seperti `bagus` tetap memberikan sinyal sebelum konteks negasi `tidak bagus` diselesaikan oleh fitur bigram.

### 7.3 Contoh prediksi positif

Request:

```json
{
  "text": "makanannya enak dan pelayanannya sangat baik",
  "language": "id"
}
```

Hasil:

| Kelas | Probabilitas |
|---|---:|
| negative | 0.011089 |
| neutral | 0.000856 |
| positive | 0.988055 |

Prediksi:

```text
positive
```

Confidence:

```text
0.988055
```

Model sangat yakin karena kalimat memuat beberapa indikator positif yang eksplisit, yaitu `enak`, `pelayanan`, `sangat`, dan `baik`.

### 7.4 Contoh prediksi netral

Request:

```json
{
  "text": "jadwal kegiatan dilaksanakan pada hari senin",
  "language": "id"
}
```

Hasil:

| Kelas | Probabilitas |
|---|---:|
| negative | 0.323925 |
| neutral | 0.640929 |
| positive | 0.035146 |

Prediksi:

```text
neutral
```

Confidence:

```text
0.640929
```

Kalimat bersifat informatif dan tidak memuat penilaian eksplisit. Prediksi netral benar, tetapi probabilitas negatif relatif tinggi. Hasil ini sejalan dengan temuan evaluasi bahwa kelas netral merupakan kelas yang paling sulit dibedakan.

### 7.5 Validasi teks kosong

Request:

```json
{
  "text": "",
  "language": "id"
}
```

Hasil:

```text
422 Unprocessable Entity
```

Pesan validasi:

```text
String should have at least 1 character
```

Interpretasi:

Schema berhasil mencegah teks kosong diteruskan ke pipeline.

### 7.6 Validasi bahasa

Request:

```json
{
  "text": "bagus sekali",
  "language": "en"
}
```

Hasil:

```text
422 Unprocessable Entity
```

Pesan validasi:

```text
Input should be 'id'
```

Interpretasi:

API secara eksplisit membatasi penggunaan pada teks Bahasa Indonesia. Pembatasan ini sesuai dengan data pelatihan dan mencegah klaim dukungan bahasa yang tidak diuji.

## 8. Konsistensi Pipeline

Endpoint tidak melakukan tahapan berikut secara manual:

- lowercase;
- normalisasi whitespace;
- normalisasi slang;
- TF-IDF;
- pemilihan unigram atau bigram;
- klasifikasi.

Seluruh proses tersebut telah menjadi bagian dari pipeline yang disimpan dalam `model.joblib`.

Keputusan ini penting karena:

1. preprocessing saat training dan inferensi tetap identik;
2. risiko training-serving skew berkurang;
3. API tetap sederhana;
4. artefak model dapat diaudit sebagai satu unit;
5. perubahan normalizer tidak perlu disalin ke endpoint.

Test berikut secara khusus memverifikasi kontrak tersebut:

```text
test_api_passes_original_text_directly_to_saved_pipeline PASSED
```

## 9. Validasi Request dan Response

### 9.1 Request

Request valid minimal terdiri atas:

```json
{
  "text": "contoh teks",
  "language": "id"
}
```

Aturan utama:

- `text` wajib ada;
- `text` harus berupa string;
- `text` tidak boleh kosong;
- panjang teks dibatasi oleh schema;
- `language` wajib bernilai `id`;
- field yang tidak sesuai schema ditolak.

### 9.2 Response

Respons prediksi terdiri atas:

```json
{
  "prediction": "positive",
  "confidence": 0.95,
  "probabilities": {
    "negative": 0.02,
    "neutral": 0.03,
    "positive": 0.95
  },
  "model_version": "sentimenid-..."
}
```

Nilai probabilitas dapat digunakan untuk:

- mengetahui keyakinan model;
- membedakan prediksi kuat dan ambigu;
- melakukan audit;
- menerapkan threshold pada aplikasi lain;
- menandai prediksi berisiko untuk pemeriksaan manual.

## 10. Temuan

### 10.1 Temuan yang berhasil

- API dapat dijalankan melalui Uvicorn.
- Swagger UI berhasil dibuat otomatis.
- Endpoint sistem dan prediksi dapat diakses.
- Model dimuat dengan sukses.
- Model dimuat satu kali pada lifecycle aplikasi.
- Prediksi tiga kelas berhasil dihasilkan.
- Probabilitas seluruh kelas tersedia.
- Versi model disertakan dalam respons.
- Input tidak valid ditolak sebelum inferensi.
- Seluruh test API lulus.
- Seluruh test suite proyek lulus.

### 10.2 Temuan yang perlu diperhatikan

- Prediksi negatif pada frasa negasi masih memiliki probabilitas positif yang cukup besar.
- Prediksi netral masih memiliki probabilitas negatif yang relatif tinggi.
- Confidence tidak selalu menunjukkan bahwa teks benar-benar mudah dipahami secara linguistik.
- API belum menggunakan threshold khusus untuk menandai prediksi ambigu.
- API belum menyediakan batch prediction.
- API belum menyediakan autentikasi atau pembatasan request.
- API belum diuji pada beban tinggi.
- API belum diuji pada server publik.
- API hanya mendukung Bahasa Indonesia.

## 11. Hubungan dengan Evaluasi Model

Perilaku API konsisten dengan evaluasi final model:

- kelas positive merupakan kelas yang paling mudah diprediksi;
- kelas neutral merupakan kelas yang paling sulit;
- negasi dapat diproses, tetapi beberapa kasus tetap ambigu;
- probabilitas perlu dibaca bersama label, bukan dianggap sebagai kebenaran absolut.

API tidak memperbaiki kelemahan model. API hanya menyediakan cara terstruktur untuk menggunakan model yang telah dievaluasi.

## 12. Risiko dan Keterbatasan

API masih memiliki beberapa keterbatasan:

1. Belum ada autentikasi.
2. Belum ada rate limiting.
3. Belum ada monitoring produksi.
4. Belum ada log request terstruktur.
5. Belum ada penyimpanan histori prediksi.
6. Belum ada deteksi bahasa otomatis.
7. Belum ada dukungan multi-bahasa.
8. Belum ada mekanisme abstain untuk confidence rendah.
9. Belum ada pengujian terhadap input adversarial.
10. Belum ada deployment container atau cloud.

Keterbatasan tersebut tidak menghalangi tujuan UAS, tetapi perlu dicatat agar API tidak dipresentasikan sebagai layanan produksi penuh.

## 13. Kesimpulan

FastAPI berhasil mengubah pipeline klasifikasi sentimen menjadi layanan REST yang dapat diuji dan diaudit. Model dimuat sekali saat startup, request divalidasi menggunakan Pydantic, dan teks asli diteruskan ke pipeline tersimpan tanpa preprocessing manual tambahan.

Pengujian otomatis menunjukkan seluruh kontrak API telah terpenuhi. Smoke test Swagger UI juga membuktikan bahwa endpoint dapat menghasilkan prediksi negatif, netral, dan positif, sekaligus menolak teks kosong serta kode bahasa yang tidak didukung.

Dengan demikian, tahap implementasi API dinyatakan selesai untuk ruang lingkup proyek ini. Tahap berikutnya adalah dokumentasi penggunaan API pada `README.md`, penyusunan laporan akhir, dan pemeriksaan ulang kelengkapan repository.
