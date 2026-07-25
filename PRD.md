# PRD - UAS Machine Learning End-to-End

## Case C: Klasifikasi Sentimen Teks Bahasa Indonesia

---

# Bagian I - Simple PRD (Discovery Gate)

## 1. Nama dan konsep

**Nama kerja:** SentimenID API

**Konsep satu kalimat:** membangun sistem machine learning yang dapat mengklasifikasikan teks bahasa Indonesia menjadi sentimen positif, netral, atau negatif melalui REST API yang dapat direproduksi dari nol.

## 2. Masalah, pihak yang dilayani, dan alternatif saat ini

Sebuah media atau marketplace menerima banyak ulasan/komentar berbahasa Indonesia. Pengelompokan sentimen secara manual memakan waktu, tidak konsisten, dan sulit diskalakan.

Pihak yang dilayani dalam konteks tugas adalah penguji UAS sebagai evaluator teknis; dalam konteks bisnis hipotetis, pihak yang diuntungkan adalah tim yang perlu memilah masukan pengguna.

Alternatif saat ini adalah pembacaan manual atau pencarian kata kunci sederhana. Keduanya tidak cukup konsisten untuk membedakan kalimat seperti "bagus" dan "tidak bagus".

## 3. Hipotesis nilai

Jika teks dinormalisasi secara terbatas, kata negasi dipertahankan, lalu direpresentasikan dengan TF-IDF unigram dan bigram dalam pipeline yang bebas leakage, maka baseline model linear dapat memberi klasifikasi tiga kelas yang dapat diuji, dijelaskan, dan disajikan lewat API.

Hipotesis ini bukan klaim bahwa sistem memahami bahasa seperti manusia. Sistem hanya diharapkan menjadi baseline klasifikasi sentimen yang layak untuk lingkup UAS.

## 4. Minimum credible outcome

Hasil minimum yang dianggap kredibel adalah:

- dataset publik berbahasa Indonesia, minimal 1.000 baris, memiliki minimal tiga label sentimen, dan belum pernah dipakai pada Modul 2-6;
- repo dapat di-clone, mengunduh/memuat data, melatih model, lalu menjalankan API tanpa langkah tersembunyi;
- minimal tiga model dibandingkan melalui cross-validation pada data latih saja;
- model terpilih disimpan sebagai pipeline utuh dan dilayani oleh FastAPI;
- sedikitnya enam automated test lulus, termasuk dua behavioral test; dan
- laporan serta video memenuhi ketentuan UAS.

## 5. Lingkup awal dan non-goal

**Dalam lingkup**

- Klasifikasi sentimen positif, netral, dan negatif untuk teks bahasa Indonesia.
- EDA, pembersihan yang dapat dipertanggungjawabkan, training, evaluasi, FastAPI, dan pytest.
- Representasi teks TF-IDF dengan n-gram serta penanganan terbatas untuk kosakata informal dan negasi.

**Di luar lingkup**

- Deployment cloud, autentikasi pengguna, dashboard, aplikasi mobile, dan Docker.
- Fine-tuning IndoBERT/LLM atau klaim bahwa model memahami konteks secara semantik penuh.
- Pengumpulan data pribadi baru, dataset sintetis, atau pemakaian ulang dataset Modul 2-6.
- Menargetkan angka F1 tertentu tanpa bukti; nilai model dipilih dari validasi yang benar, bukan dari target kosmetik.

## 6. Kendala, risiko, asumsi, dan pertanyaan terbuka

| Kategori | Isi | Dampak / respons awal |
|---|---|---|
| Kendala | Proyek individu, waktu UAS 7 hari, laporan maksimal 10 halaman, video 3-5 menit | Scope dijaga ke baseline linear yang dapat diuji penuh. |
| Kendala | Deadline operasional tersisa pada malam 25 Juli 2026 | Prioritaskan seluruh item rubrik; tidak menambah dashboard, Docker, database, atau fitur kosmetik. |
| Kendala | Data dan model tidak dikomit ke Git | README harus menyediakan langkah reproduksi; model dibuat sebelum API dijalankan. |
| Bukti | Dataset Modul 2-6 seluruhnya dibuat sintetis oleh pemilik proyek | SmSA tidak pernah dipakai pada Modul 2-6; syarat non-pengulangan terpenuhi. |
| Risiko sedang | Lisensi dataset dan lisensi repository dapat tertukar | README membedakan lisensi dataset card IndoNLU (MIT) dari lisensi repository/code IndoNLU (Apache-2.0). |
| Risiko sedang | Pembersihan berlebihan menghapus sinyal sentimen, terutama negasi/emoji | Normalisasi dibatasi dan setiap aturan dijustifikasi melalui EDA. |
| Risiko sedang | Behavioral test rapuh setelah model dilatih ulang | Test membandingkan relasi perilaku, bukan angka probabilitas presisi. |
| Bukti | SmSA memiliki label positive, neutral, dan negative serta 12.260 baris berlabel | Diverifikasi oleh `src/load_data.py` dengan identitas blob sumber. |
| Keputusan | Dataset final adalah IndoNLU SmSA | Train dan validation resmi digabung; test resmi yang labelnya disamarkan tidak digunakan. |
| Keputusan | NIM pemilik adalah 1003240008 | Nama proyek lokal: `UAS-ML-1003240008`. |

## 7. Status bukti dan validasi

1. **Selesai:** dataset dikonfirmasi belum pernah dipakai pada Modul 2-6.
2. **Selesai:** jumlah baris, label, format, sumber publik, lisensi, dan metode unduh terverifikasi.
3. **Selesai:** audit kualitas data menemukan tiga kekotoran nyata yang berbeda: 81 exact duplicate, 45 baris dengan placeholder token, dan 103 baris dengan huruf berulang.
4. **Selesai:** data modeling dikurasi secara deterministik menjadi 12.179 baris, lalu dibagi stratified 80:20 menjadi 9.743 train dan 2.436 test dengan `random_state=42`.
5. **Selesai:** empat grafik EDA aktual telah dihasilkan dari train set atau audit full-data sesuai fungsi masing-masing, dan lima unit test EDA lulus.
6. **Diputuskan:** API menerima kontrak bahasa `id` tanpa mengklaim melakukan deteksi bahasa otomatis.

## 8. Keputusan gate

**Keputusan: PROCEED ke Professional PRD.**

Alasan: Case C memenuhi seluruh bentuk deliverable yang diminta dan mempunyai jalur implementasi yang lebih sederhana daripada A/B tanpa menurunkan ketelitian teknis.

**Catatan v0.3:** tahap data dan EDA telah dikunci. Tiga temuan kualitas data terdokumentasi, split kanonik sudah dibentuk, empat grafik aktual tersedia, dan tiga prakiraan sebelum training tetap dipertahankan untuk diuji pada Tahap 3.

---

# Bagian II - Professional PRD (Execution Baseline)

## 1. Kontrol dokumen

| Atribut | Nilai |
|---|---|
| Judul | PRD UAS Machine Learning End-to-End - SentimenID API |
| Versi | 0.3 (EDA checkpoint) |
| Tanggal | 25 Juli 2026 |
| Pemilik | Panji Arya Soma |
| Tim | Individu |
| Status | In progress - data dan EDA selesai; text normalizer berikutnya |
| Proyek/repo | `UAS-ML-1003240008` |

### Changelog

| Versi | Tanggal | Perubahan |
|---|---|---|
| 0.3 | 25 Juli 2026 | Mengunci hasil audit dan EDA aktual: 81 exact duplicate, 45 placeholder token, 103 elongated-word rows, 12.179 data modeling, split 9.743/2.436, empat PNG aktual, serta `tests/test_eda.py` dengan lima test lulus. |
| 0.2 | 25 Juli 2026 | Mengunci IndoNLU SmSA, repo/NIM, Python 3.12.7, metode unduh terverifikasi, dan arsitektur seleksi model; pin package serving tetap menjadi pekerjaan implementasi. |
| 0.1 | 24 Juli 2026 | Baseline awal untuk Case C, termasuk scope, risiko, requirement, dan acceptance criteria. |

## 2. Ringkasan eksekutif dan visi

SentimenID API adalah proyek UAS end-to-end untuk mengklasifikasikan masukan teks bahasa Indonesia ke kelas positif, netral, atau negatif. Proyek membuktikan kemampuan membangun alur ML yang dapat direproduksi: sumber data jelas, EDA bertafsiran, pipeline bebas leakage, evaluasi disiplin, serving FastAPI, dan pengujian otomatis.

Visi proyek bukan membuat mesin pemahaman bahasa universal, melainkan membuat baseline yang jujur mengenai kemampuan dan keterbatasannya serta dapat dijalankan ulang oleh penguji.

## 3. Masalah dan alasan kebutuhan

Teks masukan pengguna berjumlah besar dan beragam. Proses manual membutuhkan waktu serta rentan inkonsisten, sedangkan pencocokan kata tunggal gagal menangkap frasa penting seperti "tidak bagus". Klasifikasi otomatis dapat membantu triage awal, tetapi hasilnya hanya dapat dipercaya bila data, pembersihan, evaluasi, dan perilaku API-nya dapat diaudit.

Kebutuhan proyek ditentukan oleh lembar UAS: data mentah sampai API, bukan sekadar notebook dan skor metrik.

## 4. Pihak terkait

| Pihak | Kepentingan |
|---|---|
| Panji Arya Soma | Pemilik dan pelaksana proyek; harus dapat menjelaskan setiap keputusan saat sidang. |
| Penguji/dosen | Menilai reproduktibilitas, ketepatan metodologi, API, test, dan penjelasan lisan. |
| Pengguna bisnis hipotetis | Memerlukan triage awal atas ulasan atau komentar Indonesia, bukan keputusan final otomatis. |

## 5. Tujuan, scope, dan batas pengiriman

### Tujuan

1. Menghasilkan klasifikasi sentimen tiga kelas dengan metrik utama F1-macro.
2. Membandingkan minimal tiga algoritma melalui 5-fold cross-validation pada data train saja.
3. Menyimpan pipeline utuh sebagai artefak yang dapat dipakai ulang oleh API.
4. Menyajikan prediksi, probabilitas kelas, dan confidence melalui FastAPI.
5. Membuktikan kontrak API dan perilaku model melalui automated test.

### Batas pengiriman

- Repo Git dengan struktur wajib `src/`, `app/`, `tests/`, `data/`, `models/`, `reports/`, requirements, `.gitignore`, dan README.
- Laporan PDF maksimal 10 halaman.
- Rekaman layar 3-5 menit yang memperlihatkan server, prediksi valid, request 422, dan pytest hijau.

### Non-goal

- Tidak ada deployment publik maupun klaim kesiapan produksi skala besar.
- Tidak ada pengumpulan data baru, pelabelan manual baru, atau data sintetis.
- Tidak ada transformer/LLM bila tidak diperlukan untuk memenuhi requirement UAS.

## 6. Alur utama dan requirement fungsional

| ID | Requirement | Kriteria penerimaan |
|---|---|---|
| FR-01 | Data acquisition dan profiling | `src/load_data.py` memperoleh/memuat data ke `data/`, lalu mencetak jumlah baris/kolom, tipe kolom, dan missing value per kolom. |
| FR-02 | Reproduksibilitas data | README menyebut URL, lisensi, atribusi, dan cara mengisi ulang `data/` dari clone bersih. |
| FR-03 | EDA | `src/eda.py` menghasilkan minimal empat PNG dan setiap grafik di laporan memiliki tafsiran 2-3 kalimat. |
| FR-04 | Kekotoran data | Minimal tiga kekotoran nyata didokumentasikan: temuan, cara deteksi, tindakan, dan alasannya. |
| FR-05 | Prakiraan sebelum training | Tiga hipotesis yang dapat diuji ditulis sebelum training, lalu ditinjau ulang setelah evaluasi. |
| FR-06 | Split dan preprocessing | `train_test_split(..., stratify=y, random_state=...)` terjadi sebelum TF-IDF/normalisasi yang dipelajari; seluruh transformasi berada di sklearn Pipeline. |
| FR-07 | Perbandingan model | Minimal Multinomial Naive Bayes, Logistic Regression, dan Linear SVC dibandingkan melalui Stratified 5-fold CV pada train set. |
| FR-08 | Model produksi | Semua kandidat yang dibandingkan sudah deployable dan memberi probabilitas kelas. Kandidat SVC menggunakan `CalibratedClassifierCV` yang membungkus pipeline normalizer, TF-IDF, dan LinearSVC secara utuh. |
| FR-09 | Evaluasi akhir | `src/evaluate.py` menyentuh test set hanya setelah semua keputusan model dikunci; hasil mencakup F1-macro, classification report, confusion matrix, dan analisis kesalahan. |
| FR-10 | Artefak | `models/model.joblib` berisi pipeline utuh; `models/metadata.json` merekam sumber data, konfigurasi, model, metrik CV, dan informasi reproduksi. |
| FR-11 | API informasi | `GET /` menjelaskan layanan; `GET /health` memberi status dan `model_loaded`. |
| FR-12 | API prediksi | `POST /predict-teks` menerima teks Indonesia dan mengembalikan label, probabilitas tiap kelas, confidence, dan versi model. |
| FR-13 | Validasi | Request memiliki `text` dengan batas panjang dan `language` yang hanya menerima enum `id`; field hilang atau `language` tak didukung menghasilkan 422, bukan 500. |
| FR-14 | Logging | API mencatat waktu, hasil label, dan confidence tanpa menyimpan teks lengkap yang mungkin sensitif. |
| FR-15 | Test otomatis | `python -m pytest tests/ -v` menjalankan sedikitnya enam test dan semuanya lulus. |

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

Profil aktual menghasilkan 7.151 positive, 3.830 negative, dan 1.279 neutral. Tidak ditemukan missing value standar, blank text, konflik label, URL, mention, maupun hashtag. Audit menemukan 81 exact duplicate, 45 baris dengan placeholder token, 103 baris dengan huruf berulang, serta 346 teks dengan panjang maksimal dua kata.

Exact duplicate dihapus secara deterministik sebelum split agar pasangan `text-label` identik tidak tersebar ke train dan test. Setelah deduplikasi tersisa 12.179 baris, kemudian dibagi stratified 80:20 menjadi 9.743 train dan 2.436 test dengan `random_state=42`. Placeholder token dan huruf berulang tidak dihapus pada tahap audit; keduanya akan ditangani secara konsisten oleh text normalizer di dalam pipeline. Teks maksimal dua kata dipertahankan karena pendek tidak otomatis berarti tidak valid.

Dataset card resmi IndoNLU menyatakan lisensi benchmark dataset adalah MIT, sedangkan repository/code IndoNLU menyatakan Apache-2.0. README wajib mencatat keduanya secara terpisah agar tidak menganggap lisensi code sebagai lisensi data. Dataset dikonfirmasi bukan sintetis dan belum pernah dipakai pada Modul 2-6.

### Grafik EDA yang diimplementasikan

1. `reports/label_distribution.png` — distribusi tiga label pada train set.
2. `reports/missing_values_by_column.png` — jumlah nilai hilang per kolom pada dataset audit.
3. `reports/text_length_by_label.png` — boxplot panjang teks per label pada train set.
4. `reports/top_terms_by_label.png` — tiga panel istilah teratas untuk negative, neutral, dan positive setelah stopword removal khusus visualisasi.

Keempat PNG tersebut memenuhi minimum UAS. Grafik ciri linguistik tiga kelas digabung dalam satu figure agar tetap mudah dibandingkan tanpa menambah file kosmetik. Setiap grafik masih wajib diberi tafsiran 2–3 kalimat di laporan final.

### Temuan kualitas data aktual dan keputusan

| Temuan | Jumlah | Deteksi | Keputusan |
|---|---:|---|---|
| Exact duplicate | 81 baris | `df.duplicated().sum()` | Hapus pasangan `text-label` identik sebelum split untuk mencegah kebocoran antarpartisi. |
| Placeholder token | 45 baris | Regex `__[a-z_]+__` | Pertahankan baris; normalisasi token secara konsisten di dalam pipeline karena dapat membawa sinyal emosi. |
| Huruf berulang | 103 baris | Regex `([a-z])\1{2,}` | Pertahankan baris; lakukan normalisasi terbatas di dalam pipeline agar bentuk seperti `enaaak` tidak menjadi vocabulary terpisah. |
| Teks maksimal dua kata | 346 baris | Jumlah token `<= 2` | Dipertahankan dan diperlakukan sebagai risiko konteks terbatas, bukan otomatis data kotor. |

Audit juga mengonfirmasi nilai nol untuk missing text/label, blank text, konflik label, leading/trailing space, repeated whitespace, unknown-like text, URL, mention, dan hashtag. Nilai nol tetap dicatat sebagai bukti pemeriksaan, tetapi tidak diklaim sebagai kekotoran yang ditemukan.

### Tiga prakiraan sebelum model dilatih

1. Bigram yang memuat negasi, seperti `tidak bagus`, akan membantu mengurangi kebingungan antara sentimen positif dan negatif.
2. Kelas netral berpotensi lebih sering tertukar dengan positif/negatif karena banyak teks informasional memiliki kata evaluatif lemah.
3. Normalisasi ejaan informal terbatas akan membantu bila variasi slang cukup sering, tetapi normalisasi agresif dapat menghapus sinyal sentimen.

Ketiganya adalah hipotesis, bukan kesimpulan. Mereka harus ditinjau ulang dengan hasil CV, confusion matrix, dan error analysis.

## 8. Desain modeling dan evaluasi

1. Audit schema, label, blank, exact duplicate, dan konflik label dilakukan sebagai kualifikasi data sebelum split; tahap ini tidak mempelajari representasi fitur.
2. Setelah exact duplicate dihapus, 12.179 baris dibagi 80:20 melalui `train_test_split(..., stratify=y, random_state=42)`, menghasilkan 9.743 train dan 2.436 test.
3. Test set dikunci. EDA yang memengaruhi desain model, pemilihan slang, dan tuning hanya memakai train set; audit schema dan kualitas non-learned tetap boleh memakai dataset penuh.
4. Normalisasi yang dapat dipickling dan TF-IDF dengan kandidat unigram/bigram berada di dalam pipeline.
5. Tiga pencarian CV kecil dijalankan terpisah untuk Multinomial Naive Bayes, Logistic Regression, dan calibrated LinearSVC karena struktur parameternya tidak identik.
6. Semua kandidat memakai fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` dan metrik utama **F1-macro** yang sama.
7. `CalibratedClassifierCV(method="sigmoid", cv=3)` membungkus pipeline SVC lengkap agar TF-IDF juga di-fit ulang di dalam setiap calibration fold.
8. Model final dipilih berdasarkan rata-rata CV, simpangan baku, error pattern, kemampuan memberi probabilitas, dan kesederhanaan; bukan skor tertinggi secara buta.
9. Test set dipakai satu kali di `evaluate.py` setelah pilihan final dikunci.
10. Analisis kesalahan mencakup sedikitnya lima contoh penting, misalnya negasi, slang, teks ambigu, atau teks sangat pendek.

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

- Clone repo pada folder bersih, lalu ikuti README dari nol.
- Jalankan training untuk menghasilkan artefak lokal, kemudian `uvicorn app.main:app` dan buka `/docs`.
- Jalankan `python -m pytest tests/ -v`.
- Pastikan semua grafik di laporan memiliki tafsiran, bukan sekadar ditempel.

## 10. Kebutuhan kualitas dan operasional

| Area | Ketentuan |
|---|---|
| Reproduksibilitas | Python 3.12.7 dikunci sebagai runtime proyek. Versi pandas, scikit-learn, dan FastAPI dicatat di README; `requirements-api.txt` dipin persis setelah environment serving tervalidasi. |
| Keandalan | API memuat artefak pada FastAPI lifespan, tidak melatih ulang saat request. |
| Validasi input | Pydantic membatasi tipe, panjang teks, dan bahasa yang didukung. |
| Privasi | Hanya data publik/anonymized. Logging tidak menyimpan isi teks lengkap tanpa kebutuhan. |
| Kompatibilitas | Target utama adalah eksekusi lokal melalui Python dan uvicorn; Docker tidak diperlukan. |
| Keterbatasan | Model hanya mendukung bahasa Indonesia dan dapat gagal pada sarkasme, konteks panjang, code-switching, atau kosakata baru. |

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
│   ├── __init__.py
│   ├── schemas.py
│   └── main.py
├── tests/
│   ├── test_load_data.py
│   ├── test_eda.py
│   ├── test_transformers.py
│   ├── test_api.py
│   └── test_behavior.py
├── data/
├── models/
├── reports/
├── requirements.txt
├── requirements-api.txt
├── .gitignore
├── PRD.md
└── README.md
```

`data/` dan artefak model besar diabaikan Git. README harus dapat mengisi ulang data dan membangun ulang model. Metadata ringan dapat dikecualikan dari ignore bila diperlukan sebagai bukti konfigurasi; keputusan tepatnya diverifikasi saat implementasi agar tetap sejalan dengan instruksi UAS.

## 12. Dependensi, risiko, dan mitigasi

| Risiko | Pencegahan / mitigasi |
|---|---|
| Dataset SmSA pernah dipakai di Modul 2-6 | Ditutup: dataset Modul 2-6 seluruhnya sintetis buatan pemilik; SmSA belum pernah digunakan. |
| Lisensi dataset tertukar dengan lisensi repository | Cantumkan MIT untuk dataset berdasarkan dataset card resmi dan Apache-2.0 secara terpisah untuk repository/code IndoNLU. |
| Tiga kekotoran nyata tidak cukup kuat | Ditutup: audit mengonfirmasi exact duplicate, placeholder token, dan huruf berulang sebagai tiga temuan berbeda; teks sangat pendek hanya dicatat sebagai risiko tambahan. |
| F1 bagus tetapi prediksi negasi buruk | Tinjau bigram, normalisasi, dan contoh error; jangan menyamarkan kelemahan di laporan. |
| API gagal karena artefak belum ada | README memberi urutan `load_data` -> `train` -> `uvicorn`; health endpoint memberi status jelas. |
| Requirements serving berubah | Pin versi API persis dan uji dari virtual environment bersih. |
| Test perilaku gagal setelah perubahan model | Tinjau sebab model berubah; jangan melemahkan assertion hanya agar hijau tanpa alasan. |

## 13. Keputusan terbuka dan blocker

| ID | Keputusan | Status |
|---|---|---|
| D-01 | Konfirmasi IndoNLU SmSA belum dipakai di Modul 2-6 | CLOSED - seluruh dataset Modul 2-6 sintetis buatan pemilik |
| D-02 | Verifikasi URL, lisensi, atribusi, dan metode download dataset final | CLOSED - sumber resmi, lisensi terpisah, dan identitas blob dicatat |
| D-03 | Isi NIM untuk nama repo | CLOSED - 1003240008 |
| D-04 | Tetapkan versi Python dan dependensi berdasarkan environment aktual | RESOLVED FOR EXECUTION - Python 3.12.7 dikunci; versi package serving dipin setelah API tervalidasi |
| D-05 | Tentukan apakah kandidat SmSA dipakai atau diganti setelah inspeksi data | CLOSED - SmSA digunakan |
| D-06 | Kunci hasil audit, kurasi, split, dan grafik EDA | CLOSED - 81 duplicate dihapus; 12.179 data modeling dibagi 9.743/2.436; empat PNG dan lima test EDA tersedia |

## 14. Urutan delivery

1. **Selesai:** audit kualitas data, tiga kekotoran nyata, dan kurasi baris deterministik.
2. **Selesai:** split stratified 80:20 dan tiga prakiraan sebelum training dikunci.
3. **Selesai pada sisi kode:** EDA train-only, empat grafik, dan keputusan preprocessing awal. Tafsiran 2–3 kalimat per grafik masih menjadi pekerjaan laporan.
4. **Berikutnya:** implementasikan serta uji custom text normalizer.
5. Training dan perbandingan tiga kandidat deployable melalui 5-fold CV pada train set.
6. Kunci model, lakukan evaluasi test set sekali, simpan artefak dan metadata.
7. Bangun FastAPI, validasi Pydantic, logging, dan contoh curl.
8. Lengkapi minimal empat mechanical test dan dua behavioral test.
9. Finalisasi README, pin serving requirements, lakukan clean-clone rehearsal, laporan, dan rekaman demo.

## 15. Definition of done

Proyek selesai hanya jika seluruh requirement UAS dapat dibuktikan, bukan sekadar ada filenya: data sah dan dapat diulang, EDA bertafsiran, pipeline bebas leakage, model/API berjalan, minimal enam test lulus, README dapat diikuti dari clone bersih, laporan memenuhi batas halaman, dan demo memperlihatkan prediksi valid maupun 422.
