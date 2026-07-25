# PRD - UAS Machine Learning End-to-End

## Case C: Klasifikasi Sentimen Teks Bahasa Indonesia

---

# Bagian I - Simple PRD (Discovery Gate)

## 1. Nama dan konsep

**Nama kerja:** SentimenID API

**Konsep satu kalimat:** membangun sistem machine learning yang dapat mengklasifikasikan teks Bahasa Indonesia menjadi sentimen positif, netral, atau negatif melalui REST API yang dapat direproduksi dari nol.

## 2. Masalah, pihak yang dilayani, dan alternatif saat ini

Sebuah media atau marketplace menerima banyak ulasan/komentar berBahasa Indonesia. Pengelompokan sentimen secara manual memakan waktu, tidak konsisten, dan sulit diskalakan.

Pihak yang dilayani dalam konteks tugas adalah penguji UAS sebagai evaluator teknis; dalam konteks bisnis hipotetis, pihak yang diuntungkan adalah tim yang perlu memilah masukan pengguna.

Alternatif saat ini adalah pembacaan manual atau pencarian kata kunci sederhana. Keduanya tidak cukup konsisten untuk membedakan kalimat seperti "bagus" dan "tidak bagus".

## 3. Hipotesis nilai

Jika teks dinormalisasi secara terbatas, kata negasi dipertahankan, lalu direpresentasikan dengan TF-IDF unigram dan bigram dalam pipeline yang bebas leakage, maka baseline model linear dapat memberi klasifikasi tiga kelas yang dapat diuji, dijelaskan, dan disajikan lewat API.

Hipotesis ini bukan klaim bahwa sistem memahami bahasa seperti manusia. Sistem hanya diharapkan menjadi baseline klasifikasi sentimen yang layak untuk lingkup UAS.

## 4. Minimum credible outcome

Hasil minimum yang dianggap kredibel adalah:

* dataset publik berBahasa Indonesia, minimal 1.000 baris, memiliki minimal tiga label sentimen, dan belum pernah dipakai pada Modul 2-6;
* repo dapat di-clone, mengunduh/memuat data, melatih model, lalu menjalankan API tanpa langkah tersembunyi;
* minimal tiga model dibandingkan melalui cross-validation pada data latih saja;
* model terpilih disimpan sebagai pipeline utuh dan dilayani oleh FastAPI;
* sedikitnya enam automated test lulus, termasuk dua behavioral test; dan
* laporan serta video memenuhi ketentuan UAS.

## 5. Lingkup awal dan non-goal

**Dalam lingkup**

* Klasifikasi sentimen positif, netral, dan negatif untuk teks Bahasa Indonesia.
* EDA, pembersihan yang dapat dipertanggungjawabkan, training, evaluasi, FastAPI, dan pytest.
* Representasi teks TF-IDF dengan n-gram serta penanganan terbatas untuk kosakata informal dan negasi.

**Di luar lingkup**

* Deployment cloud, autentikasi pengguna, dashboard, aplikasi mobile, dan Docker.
* Fine-tuning IndoBERT/LLM atau klaim bahwa model memahami konteks secara semantik penuh.
* Pengumpulan data pribadi baru, dataset sintetis, atau pemakaian ulang dataset Modul 2-6.
* Menargetkan angka F1 tertentu tanpa bukti; nilai model dipilih dari validasi yang benar, bukan dari target kosmetik.

## 6. Kendala, risiko, asumsi, dan pertanyaan terbuka

|Kategori|Isi|Dampak / respons awal|
|-|-|-|
|Kendala|Proyek individu, waktu UAS 7 hari, laporan maksimal 10 halaman, video 3-5 menit|Scope dijaga ke baseline linear yang dapat diuji penuh.|
|Kendala|Waktu pengerjaan UAS terbatas|Prioritaskan seluruh item rubrik; tidak menambah dashboard, Docker, database, atau fitur kosmetik.|
|Kendala|Dataset sumber dan artefak model biner tidak dikomit ke Git|README harus menyediakan langkah reproduksi; metadata dan laporan audit tetap dikomit.|
|Bukti|Dataset Modul 2-6 seluruhnya dibuat sintetis oleh pemilik proyek|SmSA tidak pernah dipakai pada Modul 2-6; syarat non-pengulangan terpenuhi.|
|Risiko sedang|Cakupan lisensi upstream dapat disalahartikan|Proyek mempertahankan `LICENSE` Apache-2.0 dari repository resmi, menambahkan `DATASET_ATTRIBUTION.md`, dan tidak membuat klaim lisensi terpisah yang belum diverifikasi.|
|Risiko sedang|Pembersihan berlebihan menghapus sinyal sentimen, terutama negasi/emoji|Normalisasi dibatasi dan setiap aturan dijustifikasi melalui EDA.|
|Risiko sedang|Behavioral test rapuh setelah model dilatih ulang|Test membandingkan relasi perilaku, bukan angka probabilitas presisi.|
|Bukti|SmSA memiliki label positive, neutral, dan negative serta 12.260 baris berlabel|Diverifikasi oleh `src/load_data.py` saat memuat dua split berlabel dari repository resmi.|
|Keputusan|Dataset final adalah IndoNLU SmSA|Train dan validation resmi digabung; test resmi yang labelnya disamarkan tidak digunakan.|
|Keputusan|NIM pemilik adalah 1003240008|Nama proyek lokal: `UAS-ML-1003240008`.|

## 7. Status bukti dan validasi

1. **Selesai:** dataset dikonfirmasi belum pernah dipakai pada Modul 2-6.
2. **Selesai:** jumlah baris, label, format, sumber publik, atribusi, dan metode unduh terverifikasi.
3. **Selesai:** audit menemukan 81 exact duplicate, 45 placeholder token, dan 103 teks dengan pemanjangan huruf; tidak ditemukan missing value atau konflik label.
4. **Selesai:** split stratified 80:20 dikunci menjadi 9.743 train dan 2.436 test.
5. **Selesai:** preprocessing, training tiga kandidat, dan evaluasi final satu kali telah dilaksanakan.
6. **Diputuskan:** API menerima kontrak bahasa `id` tanpa mengklaim melakukan deteksi bahasa otomatis.
7. **Berjalan:** implementasi FastAPI, pengujian API/behavioral, README, laporan PDF, dan video.

## 8. Keputusan gate

**Keputusan: PROCEED ke Professional PRD.**

Alasan: Case C memenuhi seluruh bentuk deliverable yang diminta dan mempunyai jalur implementasi yang lebih sederhana daripada A/B tanpa menurunkan ketelitian teknis.

**Catatan v0.3:** D-01 sampai D-05 telah ditutup. IndoNLU SmSA tetap menjadi dataset final; tahap data hingga evaluasi telah selesai dan implementasi FastAPI menjadi fokus berikutnya.

---

# Bagian II - Professional PRD (Execution Baseline)

## 1. Kontrol dokumen

|Atribut|Nilai|
|-|-|
|Judul|PRD UAS Machine Learning End-to-End - SentimenID API|
|Versi|0.3 (evaluation checkpoint)|
|Tanggal|26 Juli 2026|
|Pemilik|Panji Arya Soma|
|Tim|Individu|
|Status|In progress - data, EDA, preprocessing, training, dan evaluasi selesai; FastAPI belum selesai|
|Proyek/repo|`UAS-ML-1003240008`|

### Changelog

|Versi|Tanggal|Perubahan|
|-|-|-|
|0.3|26 Juli 2026|Mencatat hasil audit, preprocessing, cross-validation, model terpilih, evaluasi final satu kali, artefak audit, lisensi, dan atribusi dataset.|
|0.2|25 Juli 2026|Mengunci IndoNLU SmSA, repo/NIM, Python 3.12.7, metode unduh terverifikasi, dan arsitektur seleksi model.|
|0.1|24 Juli 2026|Baseline awal untuk Case C, termasuk scope, risiko, requirement, dan acceptance criteria.|

## 2. Ringkasan eksekutif dan visi

SentimenID API adalah proyek UAS end-to-end untuk mengklasifikasikan masukan teks Bahasa Indonesia ke kelas positif, netral, atau negatif. Proyek membuktikan kemampuan membangun alur ML yang dapat direproduksi: sumber data jelas, EDA bertafsiran, pipeline bebas leakage, evaluasi disiplin, serving FastAPI, dan pengujian otomatis.

Visi proyek bukan membuat mesin pemahaman bahasa universal, melainkan membuat baseline yang jujur mengenai kemampuan dan keterbatasannya serta dapat dijalankan ulang oleh penguji.

## 3. Masalah dan alasan kebutuhan

Teks masukan pengguna berjumlah besar dan beragam. Proses manual membutuhkan waktu serta rentan inkonsisten, sedangkan pencocokan kata tunggal gagal menangkap frasa penting seperti "tidak bagus". Klasifikasi otomatis dapat membantu triage awal, tetapi hasilnya hanya dapat dipercaya bila data, pembersihan, evaluasi, dan perilaku API-nya dapat diaudit.

Kebutuhan proyek ditentukan oleh lembar UAS: data mentah sampai API, bukan sekadar notebook dan skor metrik.

## 4. Pihak terkait

|Pihak|Kepentingan|
|-|-|
|Panji Arya Soma|Pemilik dan pelaksana proyek; harus dapat menjelaskan setiap keputusan saat sidang.|
|Penguji/dosen|Menilai reproduktibilitas, ketepatan metodologi, API, test, dan penjelasan lisan.|
|Pengguna bisnis hipotetis|Memerlukan triage awal atas ulasan atau komentar Indonesia, bukan keputusan final otomatis.|

## 5. Tujuan, scope, dan batas pengiriman

### Tujuan

1. Menghasilkan klasifikasi sentimen tiga kelas dengan metrik utama F1-macro.
2. Membandingkan minimal tiga algoritma melalui 5-fold cross-validation pada data train saja.
3. Menyimpan pipeline utuh sebagai artefak yang dapat dipakai ulang oleh API.
4. Menyajikan prediksi, probabilitas kelas, dan confidence melalui FastAPI.
5. Membuktikan kontrak API dan perilaku model melalui automated test.

### Batas pengiriman

* Repo Git dengan struktur wajib `src/`, `app/`, `tests/`, `data/`, `models/`, `reports/`, requirements, `.gitignore`, dan README.
* Laporan PDF maksimal 10 halaman.
* Rekaman layar 3-5 menit yang memperlihatkan server, prediksi valid, request 422, dan pytest hijau.

### Non-goal

* Tidak ada deployment publik maupun klaim kesiapan produksi skala besar.
* Tidak ada pengumpulan data baru, pelabelan manual baru, atau data sintetis.
* Tidak ada transformer/LLM bila tidak diperlukan untuk memenuhi requirement UAS.

## 6. Alur utama dan requirement fungsional

|ID|Requirement|Kriteria penerimaan|
|-|-|-|
|FR-01|Data acquisition dan profiling|`src/load_data.py` memperoleh/memuat data ke `data/`, lalu mencetak jumlah baris/kolom, tipe kolom, dan missing value per kolom.|
|FR-02|Reproduksibilitas data|README menyebut URL, lisensi, atribusi, dan cara mengisi ulang `data/` dari clone bersih.|
|FR-03|EDA|`src/eda.py` menghasilkan minimal empat PNG dan setiap grafik di laporan memiliki tafsiran 2-3 kalimat.|
|FR-04|Kekotoran data|Minimal tiga kekotoran nyata didokumentasikan: temuan, cara deteksi, tindakan, dan alasannya.|
|FR-05|Prakiraan sebelum training|Tiga hipotesis yang dapat diuji ditulis sebelum training, lalu ditinjau ulang setelah evaluasi.|
|FR-06|Split dan preprocessing|`train_test_split(..., stratify=y, random_state=...)` terjadi sebelum TF-IDF/normalisasi yang dipelajari; seluruh transformasi berada di sklearn Pipeline.|
|FR-07|Perbandingan model|Minimal Multinomial Naive Bayes, Logistic Regression, dan Linear SVC dibandingkan melalui Stratified 5-fold CV pada train set.|
|FR-08|Model produksi|Semua kandidat yang dibandingkan sudah deployable dan memberi probabilitas kelas. Kandidat SVC menggunakan `CalibratedClassifierCV` yang membungkus pipeline normalizer, TF-IDF, dan Linear SVC secara utuh.|
|FR-09|Evaluasi akhir|`src/evaluate.py` menyentuh test set hanya setelah semua keputusan model dikunci; hasil mencakup F1-macro, classification report, confusion matrix, dan analisis kesalahan.|
|FR-10|Artefak|`models/model.joblib` berisi pipeline utuh; `models/metadata.json` merekam sumber data, konfigurasi, model, metrik CV, dan informasi reproduksi.|
|FR-11|API informasi|`GET /` menjelaskan layanan; `GET /health` memberi status dan `model_loaded`.|
|FR-12|API prediksi|`POST /predict-teks` menerima teks Indonesia dan mengembalikan label, probabilitas tiap kelas, confidence, dan versi model.|
|FR-13|Validasi|Request memiliki `text` dengan batas panjang dan `language` yang hanya menerima enum `id`; field hilang atau `language` tak didukung menghasilkan 422, bukan 500.|
|FR-14|Logging|API mencatat waktu, hasil label, dan confidence tanpa menyimpan teks lengkap yang mungkin sensitif.|
|FR-15|Test otomatis|`python -m pytest tests/ -v` menjalankan sedikitnya enam test dan semuanya lulus.|

### Kontrak API awal

```json
POST /predict-teks
{
  "text": "aplikasinya tidak bagus dan sering error",
  "language": "id"
}
```

```json
{
  "prediction": "negative",
  "confidence": 0.87,
  "probabilities": {
    "negative": 0.87,
    "neutral": 0.09,
    "positive": 0.04
  },
  "model_version": "<metadata-version>"
}
```

Nilai dalam contoh di atas hanyalah bentuk respons, bukan target hasil model.

## 7. Strategi data dan EDA

### Dataset final

Dataset final adalah **IndoNLU SmSA** (`smsa_doc-sentiment-prosa`) dari organisasi IndoNLP. `src/load_data.py` mengunduh `train_preprocess.tsv` dan `valid_preprocess.tsv` dari repository resmi, memverifikasi identitas blob Git keduanya, lalu menggabungkannya menjadi 12.260 baris berlabel. Test resmi IndoNLU tidak digunakan karena labelnya disamarkan.

Profil awal menghasilkan 7.151 positive, 3.830 negative, dan 1.279 neutral. Audit tidak menemukan missing value atau konflik label, serta menemukan 81 exact duplicate yang kemudian dihapus secara deterministik sebelum split.

Repository resmi IndoNLU menandai proyek dengan Apache License 2.0 dan meminta pengguna komponennya menyitir paper IndoNLU. Proyek ini menyertakan salinan `LICENSE` dari upstream dan `DATASET_ATTRIBUTION.md`. Tidak dibuat klaim lisensi terpisah untuk isi dataset di luar pemberitahuan yang diberikan repository upstream. Dataset dikonfirmasi bukan sintetis dan belum pernah dipakai pada Modul 2-6.

### Grafik EDA yang dihasilkan

1. `reports/label_distribution.png` untuk distribusi label train set.
2. `reports/missing_values_by_column.png` untuk nilai hilang per kolom.
3. `reports/text_length_by_label.png` untuk distribusi panjang teks per kelas.
4. `reports/top_terms_by_label.png` untuk istilah teratas pada setiap kelas.

Stopword removal hanya digunakan untuk visualisasi istilah dan tidak diterapkan pada pipeline modeling.

### Hasil verifikasi kualitas data

* missing text, missing label, blank text, dan konflik label: 0;
* exact duplicate: 81 baris dan dihapus secara deterministik sebelum split;
* duplicate-text rows: 151 baris dalam 70 grup, tanpa label yang bertentangan;
* placeholder token: 45 baris dan dipertahankan sebagai sinyal teks setelah normalisasi format;
* pemanjangan huruf: 103 baris dan ditangani oleh normalizer;
* teks sangat pendek, maksimal dua kata: 346 baris dan dipertahankan karena tetap dapat membawa sentimen;
* URL, mention, hashtag, dan emoji: 0.

### Tiga prakiraan sebelum model dilatih

1. Bigram yang memuat negasi, seperti `tidak bagus`, akan membantu mengurangi kebingungan antara sentimen positif dan negatif.
2. Kelas netral berpotensi lebih sering tertukar dengan positif/negatif karena banyak teks informasional memiliki kata evaluatif lemah.
3. Normalisasi ejaan informal terbatas akan membantu bila variasi slang cukup sering, tetapi normalisasi agresif dapat menghapus sinyal sentimen.

Ketiganya adalah hipotesis, bukan kesimpulan. Mereka harus ditinjau ulang dengan hasil CV, confusion matrix, dan error analysis.

### Checkpoint EDA

EDA telah selesai pada 12.260 data IndoNLU SmSA.

Temuan utama:

- 81 exact duplicate dihapus sebelum split.
- 45 baris mengandung placeholder token.
- 103 baris mengandung pemanjangan huruf.
- Tidak ditemukan missing value atau konflik label.
- Split terkunci: 9.743 train dan 2.436 test.

Hasil lengkap dan tafsiran grafik tersedia di
`reports/eda_findings.md`.

Tiga prakiraan sebelum training telah dikunci:

1. Bigram negasi diperkirakan membantu.
2. Kelas neutral diperkirakan paling sulit.
3. Normalisasi informal terbatas diperkirakan membantu, tetapi normalisasi agresif dapat merusak sinyal.

## 8. Desain modeling dan evaluasi

1. Audit schema, label, blank, exact duplicate, dan konflik label dilakukan sebagai kualifikasi data sebelum split; tahap ini tidak mempelajari representasi fitur.
2. Setelah kurasi baris yang dapat dipertanggungjawabkan, data dibagi 80:20 melalui `train_test_split(..., stratify=y, random_state=42)`.
3. Test set dikunci. EDA yang memengaruhi desain model, pemilihan slang, dan tuning hanya memakai train set.
4. Normalisasi yang dapat dipickling dan TF-IDF dengan kandidat unigram/bigram berada di dalam pipeline.
5. Tiga pencarian CV kecil dijalankan terpisah untuk Multinomial Naive Bayes, Logistic Regression, dan Calibrated Linear SVC karena struktur parameternya tidak identik.
6. Semua kandidat memakai fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` dan metrik utama **F1-macro** yang sama.
7. `CalibratedClassifierCV(method="sigmoid", cv=3)` membungkus pipeline SVC lengkap agar TF-IDF juga di-fit ulang di dalam setiap calibration fold.
8. Model final dipilih secara deterministik berdasarkan mean F1-macro tertinggi; simpangan baku yang lebih rendah dan nama model digunakan sebagai tie-breaker. Seluruh kandidat telah dibuat deployable dan mampu menghasilkan probabilitas.
9. Test set dipakai satu kali di `evaluate.py` setelah pilihan final dikunci.
10. Analisis kesalahan mencakup sedikitnya lima contoh penting, misalnya negasi, slang, teks ambigu, atau teks sangat pendek.

### Checkpoint Preprocessing

`IndonesianTextNormalizer` telah selesai dan diuji. Normalizer menangani lowercase, whitespace, placeholder, pemanjangan huruf, serta negasi informal tanpa menghapus negasi formal.

Normalizer mendukung list, Series, DataFrame satu kolom, dan NumPy array; menolak DataFrame multi-kolom; serta lolos serialisasi pickle dan joblib. Hasil lengkap tersedia di `reports/preprocessing_findings.md`.

Status test saat checkpoint: 9 test transformer dan 16 test proyek lulus.

### Checkpoint Training dan Cross-Validation

Training dijalankan hanya pada 9.743 baris train. Tiga kandidat dibandingkan menggunakan 5-fold `StratifiedKFold`, metrik utama F1-macro, serta pencarian parameter kecil untuk unigram dan unigram-bigram.

| Model | Mean F1-macro | Std F1-macro | Parameter terbaik |
|---|---:|---:|---|
| Calibrated Linear SVC | **0.850229** | 0.009328 | `C=0.5`, `ngram_range=(1, 2)` |
| Logistic Regression | 0.847877 | **0.007200** | `C=2.0`, `ngram_range=(1, 2)` |
| Multinomial Naive Bayes | 0.754436 | 0.014869 | `alpha=0.5`, `ngram_range=(1, 1)` |

Calibrated Linear SVC dipilih karena memperoleh mean F1-macro tertinggi. Selisihnya terhadap Logistic Regression hanya 0.002352, sehingga hasilnya dicatat sebagai kemenangan tipis, bukan perbedaan besar.

Artefak yang dihasilkan:

- `models/model.joblib`
- `models/metadata.json`
- `reports/model_selection_results.csv`
- `reports/training_findings.md`

Test set 2.436 baris tetap terkunci selama tahap ini. Status test saat checkpoint training: 5 test training dan 21 test proyek lulus.

### Checkpoint Evaluasi Final

Tahap evaluasi final telah selesai. Pipeline terpilih, yaitu Calibrated Linear SVC dengan TF-IDF unigram dan bigram, dievaluasi satu kali pada test set terkunci yang berjumlah 2.436 baris.

Evaluasi dilakukan melalui `src/evaluate.py` setelah kode evaluasi dan konfigurasi model dikunci. Test set tidak digunakan selama EDA, preprocessing, cross-validation, atau pemilihan model.

#### Hasil Evaluasi

| Metrik | Nilai |
|---|---:|
| Accuracy | 0.889163 |
| F1-macro | 0.855483 |
| F1-weighted | 0.888721 |
| Log loss | 0.312220 |
| Prediksi benar | 2.166 |
| Prediksi salah | 270 |
| Total test set | 2.436 |

Performa per kelas:

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| negative | 0.859230 | 0.851316 | 0.855254 | 760 |
| neutral | 0.805785 | 0.767717 | 0.786290 | 254 |
| positive | 0.918806 | 0.931083 | 0.924904 | 1.422 |

Kelas `neutral` memperoleh F1-score terendah dan menjadi kelas paling sulit. Kesalahan terbesar secara absolut adalah pertukaran antara kelas `negative` dan `positive`.

#### Evaluasi Prakiraan Awal

1. Prakiraan bahwa kelas neutral paling sulit terkonfirmasi.
2. Bigram mendapat dukungan karena dua model terbaik memilih unigram dan bigram, tetapi error analysis menunjukkan bahwa beberapa negasi masih gagal.
3. Manfaat normalisasi informal terbatas belum dapat dibuktikan secara kausal karena tidak dilakukan ablation study tanpa normalizer.

#### Temuan Error Analysis

Kesalahan utama ditemukan pada:

- negasi komposisional seperti `tidak mantap` dan `tidak mahal`;
- teks panjang dengan sentimen campuran;
- dominasi kata positif atau negatif yang menutupi kesimpulan akhir;
- teks politik atau lintas-domain;
- kelas dan label yang ambigu.

Lima kesalahan dengan confidence tertinggi telah dianalisis. Analisis lengkap tersedia di `reports/evaluation_findings.md`.

#### Artefak Evaluasi

Artefak yang telah dibuat:

- `reports/evaluation_summary.json`
- `reports/classification_report.csv`
- `reports/confusion_matrix.png`
- `reports/correct_prediction_examples.csv`
- `reports/error_analysis.csv`
- `reports/test_predictions.csv`
- `reports/evaluation_findings.md`

Status test set saat ini adalah `evaluated_once`. Model, parameter, normalizer, TF-IDF, dan split tidak boleh diubah berdasarkan hasil test set tersebut.

#### Status Pengujian

Seluruh unit test proyek setelah implementasi evaluasi telah lulus:

```text
25 passed
```

## 9. Pengujian dan validasi

### Test mekanis minimum

1. `GET /health` mengembalikan 200 dan menyatakan model termuat.
2. Request prediksi valid mengembalikan 200 dan struktur respons yang benar.
3. Request tanpa `text` mengembalikan 422.
4. Request dengan `language` selain `id` mengembalikan 422.

### Behavioral test minimum

1. Kalimat positif dan versi yang dinegasikan harus menunjukkan penurunan probabilitas positif atau kenaikan probabilitas negatif yang sesuai.
2. Keluhan yang jelas harus memiliki probabilitas negatif lebih tinggi daripada pujian yang jelas.

Pasangan kalimat final dipilih setelah sanity check model. Assertion hanya menguji hubungan perilaku, tidak angka probabilitas presisi, sehingga lebih tahan terhadap retraining deterministik atau penyesuaian kecil model.

### Validasi akhir

* Clone repo pada folder bersih, lalu ikuti README dari nol.
* Jalankan training untuk menghasilkan artefak lokal, kemudian `uvicorn app.main:app` dan buka `/docs`.
* Jalankan `python -m pytest tests/ -v`.
* Pastikan semua grafik di laporan memiliki tafsiran, bukan sekadar ditempel.

## 10. Kebutuhan kualitas dan operasional

|Area|Ketentuan|
|-|-|
|Reproduksibilitas|Python 3.12.7, pandas 3.0.5, dan scikit-learn 1.9.0 tercatat pada metadata. Versi FastAPI dan paket serving dipin setelah environment API tervalidasi.|
|Keandalan|API memuat artefak pada FastAPI lifespan, tidak melatih ulang saat request.|
|Validasi input|Pydantic membatasi tipe, panjang teks, dan bahasa yang didukung.|
|Privasi|Hanya data publik/anonymized. Logging tidak menyimpan isi teks lengkap tanpa kebutuhan.|
|Kompatibilitas|Target utama adalah eksekusi lokal melalui Python dan uvicorn; Docker tidak diperlukan.|
|Keterbatasan|Model hanya mendukung Bahasa Indonesia dan dapat gagal pada sarkasme, konteks panjang, code-switching, atau kosakata baru.|

## 11. Struktur artefak

```text
UAS-ML-1003240008/
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── eda.py
│   ├── transformers.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   ├── __init__.py          # pending
│   ├── schemas.py           # pending
│   └── main.py              # pending
├── tests/
│   ├── test_load_data.py
│   ├── test_eda.py
│   ├── test_transformers.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_api.py          # pending
│   └── test_behavior.py     # pending
├── data/
├── models/
│   ├── model.joblib         # lokal, diabaikan Git
│   └── metadata.json
├── reports/
│   ├── eda_findings.md
│   ├── preprocessing_findings.md
│   ├── training_findings.md
│   ├── evaluation_findings.md
│   └── artefak CSV/PNG/JSON
├── LICENSE
├── DATASET_ATTRIBUTION.md
├── requirements.txt
├── requirements-api.txt
├── .gitignore
├── PRD.md
└── README.md
```

Dataset sumber pada `data/` dan `models/model.joblib` diabaikan Git. Metadata ringan, kode, laporan, grafik, hasil evaluasi, lisensi, dan atribusi dikomit sebagai bukti audit. README nantinya harus dapat mengisi ulang data dan membangun ulang model dari clone bersih.

## 12. Dependensi, risiko, dan mitigasi

|Risiko|Pencegahan / mitigasi|
|-|-|
|Dataset SmSA pernah dipakai di Modul 2-6|Ditutup: dataset Modul 2-6 seluruhnya sintetis buatan pemilik; SmSA belum pernah digunakan.|
|Cakupan lisensi upstream disalahartikan|Pertahankan `LICENSE` Apache-2.0 dari repository resmi, sertakan `DATASET_ATTRIBUTION.md`, sitasi paper IndoNLU, dan hindari klaim lisensi tambahan yang belum diverifikasi.|
|Tidak menemukan tiga kekotoran nyata|Perluas inspeksi secara jujur atau pilih dataset yang lebih kaya variasi, bukan menciptakan kekotoran.|
|F1 bagus tetapi prediksi negasi buruk|Tinjau bigram, normalisasi, dan contoh error; jangan menyamarkan kelemahan di laporan.|
|API gagal karena artefak belum ada|README memberi urutan `load_data` -> `train` -> `uvicorn`; health endpoint memberi status jelas.|
|Requirements serving berubah|Pin versi API persis dan uji dari virtual environment bersih.|
|Test perilaku gagal setelah perubahan model|Tinjau sebab model berubah; jangan melemahkan assertion hanya agar hijau tanpa alasan.|

## 13. Keputusan terbuka dan blocker

|ID|Keputusan|Status|
|-|-|-|
|D-01|Konfirmasi IndoNLU SmSA belum dipakai di Modul 2-6|CLOSED - seluruh dataset Modul 2-6 sintetis buatan pemilik|
|D-02|Verifikasi URL, lisensi, atribusi, dan metode download dataset final|CLOSED - sumber resmi GitHub, `LICENSE`, `DATASET_ATTRIBUTION.md`, sitasi, dan metode pemuatan telah dicatat|
|D-03|Isi NIM untuk nama repo|CLOSED - 1003240008|
|D-04|Tetapkan versi Python dan dependensi berdasarkan environment aktual|PARTIALLY CLOSED - Python 3.12.7, pandas 3.0.5, dan scikit-learn 1.9.0 tercatat; versi FastAPI/serving dipin setelah API tervalidasi|
|D-05|Tentukan apakah kandidat SmSA dipakai atau diganti setelah inspeksi data|CLOSED - SmSA digunakan|

## 14. Urutan delivery

1. Selesaikan audit kualitas data, dokumentasikan sedikitnya tiga kekotoran nyata, lalu lakukan kurasi baris deterministik.
2. Kunci split stratified 80:20 dan tiga prakiraan sebelum training.
3. Buat EDA train-only, grafik, tafsiran, dan keputusan preprocessing.
4. Implementasikan serta uji custom text normalizer.
5. Training dan perbandingan tiga kandidat deployable melalui 5-fold CV pada train set.
6. Kunci model, lakukan evaluasi test set sekali, simpan artefak dan metadata.
7. Bangun FastAPI, validasi Pydantic, logging, dan contoh curl.
8. Lengkapi minimal empat mechanical test dan dua behavioral test.
9. Finalisasi README, pin serving requirements, lakukan clean-clone rehearsal, laporan, dan rekaman demo.

## 15. Definition of done

Proyek selesai hanya jika seluruh requirement UAS dapat dibuktikan, bukan sekadar ada filenya: data sah dan dapat diulang, EDA bertafsiran, pipeline bebas leakage, model/API berjalan, minimal enam test lulus, README dapat diikuti dari clone bersih, laporan memenuhi batas halaman, dan demo memperlihatkan prediksi valid maupun 422.

