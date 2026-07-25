# Evaluation Findings — SentimenID API

**Proyek:** UAS Machine Learning End-to-End  
**Kasus:** Case C — Klasifikasi Sentimen Teks Bahasa Indonesia  
**Dataset:** IndoNLU SmSA  
**Model final:** `calibrated_linear_svc`  
**Versi model:** `sentimenid-20260725T181613Z`  
**Status:** Evaluasi test set selesai satu kali; model dan konfigurasi tidak boleh dituning ulang menggunakan hasil ini

---

## 1. Tujuan Evaluasi

Tahap ini mengukur kemampuan generalisasi pipeline final pada test set yang sebelumnya dikunci selama EDA, preprocessing, cross-validation, dan model selection.

Evaluasi dilakukan setelah model final dipilih berdasarkan mean F1-macro cross-validation pada train set. Test set tidak digunakan untuk memilih algoritma, menentukan parameter, memperbaiki normalizer, atau mengubah representasi TF-IDF.

Status evaluasi yang tercatat pada `reports/evaluation_summary.json` adalah:

```text
test_set_status = evaluated_once
```

---

## 2. Konfigurasi Evaluasi

| Komponen | Nilai |
|---|---|
| Data modeling setelah deduplikasi | 12.179 baris |
| Train set | 9.743 baris |
| Test set | 2.436 baris |
| Metode split | Stratified 80:20 |
| Random state | 42 |
| Model | Calibrated Linear SVC |
| Representasi | TF-IDF unigram + bigram |
| Kalibrasi probabilitas | Sigmoid, 3-fold |
| Urutan kelas | negative, neutral, positive |

Model yang dievaluasi adalah pipeline utuh yang telah disimpan dalam `models/model.joblib`, termasuk normalizer, TF-IDF, Linear SVC, dan kalibrasi probabilitas.

---

## 3. Metrik Final

| Metrik | Nilai |
|---|---:|
| Accuracy | **0,889163** |
| F1-macro | **0,855483** |
| F1-weighted | **0,888721** |
| Log loss | **0,312220** |
| Prediksi benar | **2.166** |
| Prediksi salah | **270** |
| Total test set | **2.436** |

Accuracy sebesar 0,8892 berarti sekitar 88,92% data test diprediksi dengan benar. Namun, F1-macro tetap menjadi metrik utama karena memberikan bobot yang sama kepada setiap kelas, termasuk kelas `neutral` yang jumlahnya jauh lebih sedikit.

F1-weighted mendekati accuracy karena skor tersebut memberi bobot berdasarkan jumlah data tiap kelas. Nilainya terutama dipengaruhi oleh performa kelas `positive`, yang memiliki support terbesar.

---

## 4. Konsistensi Cross-Validation dan Test Set

Mean F1-macro Calibrated Linear SVC pada cross-validation adalah:

```text
mean CV F1-macro = 0,850229
final test F1-macro = 0,855483
selisih = +0,005254
```

Skor test sekitar 0,53 poin persentase lebih tinggi daripada mean cross-validation. Selisih kecil ini menunjukkan bahwa performa final tetap konsisten dengan estimasi CV dan tidak mengalami penurunan besar ketika diterapkan pada partisi yang sebelumnya tidak digunakan.

Temuan ini tidak membuktikan bahwa model bebas dari semua bentuk overfitting. Kesimpulan yang dapat dipertanggungjawabkan hanya bahwa tidak terdapat performance collapse antara estimasi CV dan evaluasi test final.

---

## 5. Performa per Kelas

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| negative | 0,859230 | 0,851316 | 0,855254 | 760 |
| neutral | 0,805785 | 0,767717 | **0,786290** | 254 |
| positive | **0,918806** | **0,931083** | **0,924904** | 1.422 |

### 5.1 Kelas negative

Model mengidentifikasi sekitar 85,13% teks negative dengan benar. Dari 760 data negative, 647 diprediksi tepat, sedangkan 113 lainnya salah diklasifikasikan.

Precision dan recall negative relatif seimbang. Model tidak hanya cukup baik dalam menemukan sentimen negatif, tetapi juga tidak terlalu sering memberi label negative kepada kelas lain. Meski demikian, 87 data negative diprediksi sebagai positive, sehingga kegagalan terbesar pada kelas ini merupakan pembalikan polaritas langsung.

### 5.2 Kelas neutral

Kelas `neutral` memiliki F1-score dan recall paling rendah. Dari 254 data neutral, hanya 195 yang diprediksi benar. Sebanyak 59 data atau sekitar 23,23% salah diklasifikasikan.

Kesalahan neutral terbagi hampir seimbang:

```text
neutral → negative = 29
neutral → positive = 30
```

Hal ini menunjukkan bahwa kelas neutral berada di antara dua kelas polar dan tidak mempunyai batas leksikal sejelas positive atau negative. Jumlah data yang lebih sedikit juga membatasi variasi pola neutral yang dapat dipelajari model.

### 5.3 Kelas positive

Kelas positive memperoleh performa terbaik. Dari 1.422 data, 1.324 diprediksi benar. Recall sebesar 0,9311 menunjukkan bahwa sebagian besar sentimen positif berhasil ditemukan.

Performa tinggi ini perlu dibaca bersama distribusi data. Positive merupakan kelas terbesar, sehingga model menerima lebih banyak contoh dan mempunyai cakupan kosakata yang lebih luas. Skor tinggi tidak boleh langsung diterjemahkan sebagai kemampuan memahami sentimen positif secara semantik.

---

## 6. Confusion Matrix

Confusion matrix final:

| True \ Predicted | negative | neutral | positive |
|---|---:|---:|---:|
| negative | **647** | 26 | 87 |
| neutral | 29 | **195** | 30 |
| positive | 77 | 21 | **1.324** |

Pola utama:

1. **Negative → positive merupakan kesalahan tunggal terbesar**, sebanyak 87 kasus.
2. **Positive → negative terjadi pada 77 kasus**, sehingga pertukaran polaritas dua arah menjadi masalah utama.
3. Neutral lebih sering salah secara proporsional, meskipun jumlah absolutnya lebih kecil.
4. Prediksi ke kelas neutral relatif sedikit, sesuai distribusi prediksi akhir.

Distribusi label aktual dan prediksi:

| Kelas | Aktual | Prediksi | Selisih |
|---|---:|---:|---:|
| negative | 760 | 753 | -7 |
| neutral | 254 | 242 | -12 |
| positive | 1.422 | 1.441 | +19 |

Secara agregat, distribusi prediksi tidak menyimpang jauh dari distribusi aktual. Namun, model sedikit lebih sering memilih kelas positive dan sedikit lebih jarang memilih neutral.

---

## 7. Analisis Lima Kesalahan dengan Confidence Tertinggi

Tabel berikut memakai cuplikan singkat agar laporan tetap terbaca. Teks lengkap dan seluruh probabilitas tersedia di `reports/error_analysis.csv` dan `reports/test_predictions.csv`.

| No. | Cuplikan teks | Aktual | Prediksi | Confidence | Interpretasi |
|---:|---|---|---|---:|---|
| 1 | “...dagingnya enak... banyak pilihan... pelayanan sering buruk dan mengecewakan...” | negative | positive | 0,989383 | Ulasan campuran dan panjang. Banyak kata positif mendominasi fitur meskipun kesimpulan keseluruhan bersifat negatif. |
| 2 | “tidak mantap” | negative | positive | 0,986408 | Kegagalan negasi komposisional. Kata `mantap` sangat kuat sebagai sinyal positive dan bigram belum cukup mengoreksinya. |
| 3 | “tempatnya nyaman... tapi rasa makanan tidak semewah harganya... overpriced” | negative | positive | 0,985519 | Model menangkap pujian pada tempat dan suasana lebih kuat daripada keluhan harga dan rasa. |
| 4 | “suasana resto sangat santai dan nyaman... rasa rata-rata mendekati tidak enak” | negative | positive | 0,982953 | Sentimen lokal positif pada suasana menutupi penilaian negatif terhadap makanan. |
| 5 | “...sabotase jelas di luar konteks ajaran islam... islam itu rahmatan lil alamin” | positive | negative | 0,976750 | Kosakata konflik seperti `sabotase` dan konteks politik/agama mendorong prediksi negative meskipun label keseluruhan positive. |

Kesalahan ini dipilih berdasarkan confidence tertinggi, bukan karena pasti merupakan lima contoh paling representatif. Justru confidence tinggi menunjukkan kasus ketika model salah tetapi sangat yakin, sehingga lebih penting untuk audit.

---

## 8. Pola Kesalahan Utama

### 8.1 Negasi tidak bersifat seragam

Error analysis memperlihatkan dua arah kegagalan:

```text
tidak mantap  → seharusnya negative, diprediksi positive
tidak mahal   → seharusnya positive, diprediksi negative
tidak malas   → seharusnya positive, diprediksi negative
tidak aib     → seharusnya positive, diprediksi negative
tidak cinta   → seharusnya negative, diprediksi positive
```

Kata `tidak` tidak otomatis membuat sentimen menjadi negative. Dampaknya bergantung pada makna kata setelahnya. Bigram membantu menyediakan konteks dua kata, tetapi model linear tetap bergantung pada pola frekuensi yang tersedia pada train set.

### 8.2 Teks panjang dan sentimen campuran

Beberapa ulasan mengandung pujian dan keluhan sekaligus. Model sering memberi bobot tinggi kepada kata seperti `enak`, `nyaman`, `baik`, dan `murah`, lalu mengabaikan kesimpulan negatif yang muncul pada bagian akhir.

Masalah ini menunjukkan keterbatasan representasi bag-of-ngrams:

- urutan global teks tidak dimodelkan;
- posisi kesimpulan tidak diberi prioritas;
- hubungan kontras seperti `tetapi`, `namun`, dan `hanya saja` hanya ditangkap secara lokal;
- semakin panjang teks, semakin banyak fitur yang saling bersaing.

### 8.3 Ambiguitas dan kemungkinan noise label

Beberapa contoh tampak mengandung label yang dapat diperdebatkan. Misalnya, teks neutral yang menyatakan “sangat memuaskan” terlihat memiliki sinyal positif yang kuat. Ada pula teks berlabel negative yang hampir seluruh isinya berupa pujian.

Temuan tersebut dapat mengindikasikan:

- konteks yang tidak lengkap;
- label berdasarkan sumber atau skema anotasi tertentu;
- sentimen campuran;
- noise atau inkonsistensi anotasi.

Namun, tidak semua kesalahan boleh langsung disebut label noise. Pemeriksaan manual hanya menunjukkan kemungkinan, bukan bukti bahwa label asli salah.

### 8.4 Perbedaan domain

Dataset mencakup ulasan restoran, komentar politik, layanan telekomunikasi, dan topik lain. Kata yang polar dalam satu domain belum tentu memiliki fungsi sama dalam domain lain.

Contohnya, kosakata konflik dalam teks politik dapat mendorong prediksi negative, walaupun sikap penulis terhadap gagasan tertentu sebenarnya positive. Keragaman domain meningkatkan kegunaan dataset, tetapi juga memperbesar kemungkinan model mempelajari shortcut leksikal.

### 8.5 Kelas neutral tidak memiliki sinyal eksplisit yang konsisten

Teks neutral dapat berupa berita, informasi faktual, deskripsi lokasi, atau kalimat yang memuat kata polar tetapi tidak mengekspresikan sentimen utama. Akibatnya, kelas ini tidak dapat dikenali hanya dari ketiadaan kata positif dan negatif.

---

## 9. Evaluasi Prakiraan Sebelum Training

### 9.1 Kelas neutral kemungkinan paling sulit

**Status: terkonfirmasi.**

F1 neutral sebesar 0,786290, lebih rendah daripada:

```text
F1 negative = 0,855254
F1 positive = 0,924904
```

Recall neutral juga paling rendah, yaitu 0,767717. Sebanyak 23,23% data neutral salah diprediksi. Dengan demikian, bukti per kelas dan confusion matrix mendukung prakiraan awal.

### 9.2 Bigram membantu menangani negasi

**Status: mendapat dukungan, tetapi tidak terbukti secara kausal.**

Calibrated Linear SVC dan Logistic Regression sama-sama memilih `ngram_range=(1, 2)` sebagai konfigurasi terbaik dalam pencarian parameter. Hal tersebut menunjukkan bahwa penambahan bigram berguna bagi dua model terbaik.

Namun, eksperimen yang tersimpan tidak melaporkan selisih skor terbaik unigram versus bigram untuk model final secara terpisah. Error analysis juga memperlihatkan bahwa beberapa negasi masih gagal. Oleh karena itu, klaim yang sah adalah bahwa bigram dipilih dan konsisten dengan kebutuhan konteks negasi, bukan bahwa bigram menyelesaikan negasi sepenuhnya.

### 9.3 Normalisasi informal terbatas membantu

**Status: belum dapat dibuktikan secara kausal.**

Normalizer bekerja secara teknis, tersimpan di dalam pipeline, dan mempertahankan negasi. Namun, tidak ada ablation study yang membandingkan:

```text
pipeline dengan normalizer
versus
pipeline tanpa normalizer
```

Karena itu, hasil final tidak cukup untuk mengukur kontribusi independen normalisasi terhadap F1-macro.

---

## 10. Implikasi Penggunaan

Model cukup layak sebagai baseline untuk triage atau klasifikasi awal teks, tetapi tidak boleh diperlakukan sebagai penilai sentimen yang selalu benar.

Dalam penggunaan operasional:

- prediksi dengan confidence tinggi tetap dapat salah;
- kelas neutral memerlukan kehati-hatian lebih besar;
- teks panjang atau campuran lebih rentan mengalami pembalikan label;
- keputusan penting sebaiknya menyediakan jalur pemeriksaan manusia;
- probabilitas model harus dibaca sebagai keyakinan statistik, bukan kepastian semantik.

Model cocok untuk demonstrasi API dan pemrosesan awal, bukan untuk keputusan otomatis berisiko tinggi tanpa validasi tambahan.

---

## 11. Keterbatasan

1. Dataset tidak seimbang dan kelas neutral jauh lebih sedikit.
2. TF-IDF tidak memahami urutan global, ironi, implikatur, atau konteks panjang.
3. Bigram hanya menangkap konteks lokal dua token.
4. Tidak dilakukan ablation study khusus untuk normalizer.
5. Tidak dilakukan pengujian lintas-dataset atau out-of-domain.
6. Probabilitas berasal dari kalibrasi model, tetapi belum diuji dengan reliability diagram atau expected calibration error.
7. Beberapa label atau teks tampak ambigu, tetapi tidak dilakukan relabeling manual.
8. Test set telah digunakan satu kali dan tidak boleh dipakai untuk tuning ulang.

---

## 12. Rencana Perbaikan di Luar Scope UAS

Perbaikan berikut dicatat sebagai pekerjaan masa depan, bukan alasan untuk mengubah model setelah melihat test set:

- menambah eksperimen ablation untuk normalizer;
- membandingkan unigram dan bigram secara eksplisit;
- menambah fitur negation scope;
- mencoba character n-gram untuk variasi informal;
- membuat reliability diagram untuk evaluasi kalibrasi;
- mengevaluasi model pada dataset eksternal;
- menggunakan model kontekstual seperti IndoBERT pada eksperimen terpisah;
- melakukan audit manual terhadap label ambigu.

Eksperimen lanjutan harus menggunakan validation set atau dataset baru. Test set final proyek ini tidak boleh dijadikan dasar tuning.

---

## 13. Artefak Evaluasi

```text
reports/
├── evaluation_summary.json
├── classification_report.csv
├── confusion_matrix.png
├── correct_prediction_examples.csv
├── error_analysis.csv
├── test_predictions.csv
└── evaluation_findings.md
```

Fungsi artefak:

| File | Fungsi |
|---|---|
| `evaluation_summary.json` | Ringkasan metrik, versi model, split, dan runtime |
| `classification_report.csv` | Precision, recall, F1, dan support per kelas |
| `confusion_matrix.png` | Visualisasi pola kesalahan antar-kelas |
| `correct_prediction_examples.csv` | Contoh prediksi benar dengan confidence tinggi |
| `error_analysis.csv` | Kesalahan dengan confidence tertinggi untuk audit |
| `test_predictions.csv` | Seluruh prediksi test beserta probabilitas dan teks |
| `evaluation_findings.md` | Interpretasi hasil evaluasi dan batas klaim |

---

## 14. Status Checkpoint

- Model final telah dikunci sebelum test dibuka.
- Test set telah dievaluasi satu kali.
- F1-macro final telah dicatat.
- Classification report telah dibuat.
- Confusion matrix telah dibuat.
- Contoh prediksi benar dan salah telah disimpan.
- Lima kesalahan terburuk telah dianalisis.
- Tiga prakiraan awal telah diperiksa.
- Artefak audit lengkap telah disimpan.
- Tuning ulang berdasarkan test set dilarang.
- Tahap berikutnya: implementasi dan pengujian REST API.
