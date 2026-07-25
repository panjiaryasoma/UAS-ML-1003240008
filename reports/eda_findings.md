# EDA Findings — SentimenID API

**Proyek:** UAS Machine Learning End-to-End  
**Kasus:** Case C — Klasifikasi Sentimen Teks Bahasa Indonesia  
**Dataset:** IndoNLU SmSA  
**Status:** Checkpoint EDA selesai dan prakiraan sebelum training telah dikunci  
**Tanggal:** 25 Juli 2026  

---

## 1. Tujuan Dokumen

Dokumen ini mencatat hasil aktual exploratory data analysis (EDA), temuan kualitas data, keputusan kurasi, interpretasi grafik, serta tiga prakiraan yang ditetapkan sebelum proses training dimulai.

Dokumen ini menjadi bukti bahwa keputusan preprocessing dan modeling dibuat berdasarkan data train dan hasil audit yang dapat direproduksi, bukan disusun setelah melihat hasil evaluasi model.

---

## 2. Ringkasan Dataset

Dataset yang digunakan adalah **IndoNLU SmSA** untuk klasifikasi sentimen Bahasa Indonesia dengan tiga label:

- `positive`
- `negative`
- `neutral`

Data berasal dari dua file resmi:

- `train_preprocess.tsv`
- `valid_preprocess.tsv`

Keduanya digabung oleh `src/load_data.py`.

### 2.1 Profil awal

| Informasi | Nilai |
|---|---:|
| Jumlah baris awal | 12.260 |
| Jumlah kolom | 2 |
| Kolom fitur | `text` |
| Kolom target | `label` |
| Missing value pada `text` | 0 |
| Missing value pada `label` | 0 |
| Jumlah label | 3 |

### 2.2 Distribusi target

| Label | Jumlah | Persentase |
|---|---:|---:|
| positive | 7.151 | 58,33% |
| negative | 3.830 | 31,24% |
| neutral | 1.279 | 10,43% |
| **Total** | **12.260** | **100%** |

Distribusi kelas tidak seimbang karena kelas `positive` mendominasi, sedangkan `neutral` hanya sekitar 10% dari seluruh data. Oleh karena itu, **accuracy tidak akan digunakan sebagai metrik utama** karena dapat terlihat tinggi walaupun model buruk pada kelas minoritas.

Metrik utama yang akan digunakan adalah **F1-macro**, sehingga precision dan recall setiap kelas dipertimbangkan dengan bobot yang sama.

---

## 3. Pemeriksaan Wajib

Pemeriksaan berikut dijalankan oleh `src/eda.py`:

```python
df.isna().sum()
df.describe(include="all")
df["label"].value_counts(dropna=False)
df.duplicated().sum()
```

Selain itu, audit tambahan dilakukan untuk mendeteksi:

- teks kosong atau hanya berisi whitespace;
- duplikasi teks;
- teks identik dengan label berbeda;
- spasi berulang;
- nilai mirip `unknown`;
- placeholder token;
- pemanjangan huruf;
- teks sangat pendek;
- URL, mention, dan hashtag.

---

## 4. Temuan Kualitas Data

### 4.1 Ringkasan audit

| Temuan | Jumlah |
|---|---:|
| Exact duplicate rows | 81 |
| Duplicate text rows | 151 |
| Duplicate text groups | 70 |
| Conflicting label rows | 0 |
| Conflicting text groups | 0 |
| Placeholder token rows | 45 |
| Elongated word rows | 103 |
| Very short text, maksimal 2 kata | 346 |
| Missing text | 0 |
| Missing label | 0 |
| Blank atau whitespace-only text | 0 |
| Leading/trailing space | 0 |
| Repeated whitespace | 0 |
| Unknown-like text | 0 |
| URL | 0 |
| Mention | 0 |
| Hashtag | 0 |

Tiga indikator duplikasi tidak dianggap sebagai tiga kekotoran berbeda. Ketiganya menjelaskan satu masalah yang sama, yaitu keberadaan teks identik atau berulang.

---

## 5. Tiga Kekotoran Data Nyata

### 5.1 Exact duplicate

**Temuan:** terdapat 81 baris yang identik pada seluruh kolom.

**Metode deteksi:**

```python
df.duplicated().sum()
```

**Tindakan:** pasangan `text` dan `label` yang identik dihapus secara deterministik sebelum split.

**Alasan:** apabila salinan teks yang sama tersebar ke train dan test set, model dapat terlihat lebih baik karena pernah melihat teks identik saat training. Deduplikasi ini merupakan kurasi integritas baris, bukan learned preprocessing, karena tidak mempelajari vocabulary, statistik fitur, atau representasi dari test set.

---

### 5.2 Placeholder token

**Temuan:** terdapat 45 baris yang mengandung token seperti `__laugh__`.

**Metode deteksi:**

```python
text.str.contains(r"__[a-z_]+__", case=False)
```

**Tindakan:** baris tidak dihapus. Placeholder akan dinormalisasi secara konsisten oleh custom text normalizer.

**Alasan:** placeholder dapat membawa informasi emosi. Menghapus seluruh baris berisiko membuang sinyal sentimen, sedangkan membiarkannya tanpa normalisasi dapat memperbesar fragmentasi vocabulary.

---

### 5.3 Pemanjangan huruf

**Temuan:** terdapat 103 baris dengan pemanjangan huruf, misalnya bentuk seperti `enaaak` atau `mantapppp`.

**Metode deteksi:**

```python
re.compile(r"([a-z])\1{2,}", flags=re.IGNORECASE)
```

**Tindakan:** pemanjangan huruf akan dinormalisasi secara terbatas di dalam custom transformer.

**Alasan:** TF-IDF dapat menganggap `enak`, `enaaak`, dan `enaaaak` sebagai fitur berbeda. Normalisasi terbatas dapat mengurangi fragmentasi vocabulary, tetapi aturan tidak boleh terlalu agresif karena pengulangan huruf juga dapat membawa penekanan emosional.

---

## 6. Temuan Tambahan yang Tidak Langsung Dihapus

### 6.1 Teks sangat pendek

Ditemukan 346 teks dengan panjang maksimal dua kata.

Teks tersebut **tidak langsung dihapus** karena teks pendek tetap dapat membawa sinyal sentimen yang jelas, misalnya:

- `tidak bagus`
- `sangat kecewa`
- `enak banget`

Temuan ini diperlakukan sebagai risiko keterbatasan konteks dan akan diperiksa kembali pada tahap error analysis.

### 6.2 Tidak ada konflik label

Tidak ditemukan teks identik yang memiliki label berbeda. Dengan demikian, tidak diperlukan aturan arbitrer untuk memilih salah satu label.

### 6.3 Tidak ada missing value standar

Kolom `text` dan `label` tidak memiliki nilai hilang. Karena itu, imputasi tidak diperlukan untuk dataset ini.

---

## 7. Keputusan Kurasi dan Split

Setelah exact duplicate dihapus:

| Informasi | Jumlah |
|---|---:|
| Data awal | 12.260 |
| Exact duplicate dihapus | 81 |
| Data modeling | 12.179 |
| Train set | 9.743 |
| Test set terkunci | 2.436 |

Split dilakukan dengan konfigurasi:

```python
train_test_split(
    modeling_df,
    test_size=0.2,
    stratify=modeling_df["label"],
    random_state=42,
)
```

### Aturan penggunaan partisi

- Train set digunakan untuk EDA yang memengaruhi modeling.
- Train set digunakan untuk cross-validation, pemilihan model, dan tuning.
- Test set dikunci.
- Test set hanya boleh digunakan sekali di `src/evaluate.py` setelah keputusan model final ditetapkan.
- TF-IDF dan preprocessing yang dipelajari harus berada di dalam `sklearn Pipeline`.

---

## 8. Kolom yang Dibuang Sebelum Modeling

Dataset hanya memiliki dua kolom:

- `text` sebagai fitur;
- `label` sebagai target.

Tidak ditemukan:

- ID unik;
- kolom konstan;
- kolom metadata tambahan;
- kolom yang membocorkan target.

Dengan demikian, tidak ada kolom tambahan yang harus dibuang. Kolom `label` dipisahkan sebagai target, bukan digunakan sebagai fitur.

---

## 9. Interpretasi Grafik

### 9.1 Distribusi label sentimen pada train set

**File:** `reports/label_distribution.png`

Grafik menunjukkan kelas `positive` sebagai kelas dominan, disusul `negative`, sedangkan `neutral` merupakan kelas paling sedikit. Ketimpangan ini berarti accuracy dapat memberikan gambaran yang terlalu optimistis apabila model terlalu sering memilih kelas dominan.

Konsekuensi modeling adalah penggunaan stratified split, stratified cross-validation, serta F1-macro sebagai metrik utama agar performa kelas `neutral` tidak tertutup oleh besarnya kelas `positive`.

---

### 9.2 Jumlah nilai hilang per kolom

**File:** `reports/missing_values_by_column.png`

Grafik menunjukkan nilai hilang pada kolom `text` dan `label` sama-sama nol. Dengan demikian, tidak diperlukan imputasi untuk tahap modeling.

Walaupun hasilnya nol, pemeriksaan ini tetap penting karena membuktikan bahwa tidak ada missing value standar yang perlu ditangani sebelum pipeline dibangun.

---

### 9.3 Panjang teks per label

**File:** `reports/text_length_by_label.png`

Kelas `positive` memiliki median panjang teks dan variasi paling besar, sedangkan kelas `neutral` cenderung lebih pendek. Kelas `negative` berada di antara keduanya, tetapi distribusi ketiga kelas tetap saling tumpang tindih.

Perbedaan panjang teks dapat memengaruhi jumlah fitur TF-IDF yang tersedia, tetapi panjang teks tidak boleh dianggap sebagai penentu sentimen. Temuan ini akan diperiksa kembali melalui error analysis, terutama pada teks yang sangat pendek.

---

### 9.4 Istilah teratas per kelas

**File:** `reports/top_terms_by_label.png`

Kelas `negative` sangat didominasi kata `tidak`, sedangkan kelas `positive` banyak memuat istilah seperti `enak`, `tempat`, `makanan`, dan `makan`. Kelas `neutral` memiliki banyak istilah informasional dan politik seperti `demokrat`, `pilkada`, `partai`, dan `jakarta`.

Kata `tidak` muncul pada lebih dari satu kelas. Hal ini menunjukkan bahwa unigram saja berpotensi kehilangan konteks karena `tidak bagus` dan `tidak mengecewakan` mempunyai arah sentimen berbeda. Temuan ini mendukung penggunaan kombinasi unigram dan bigram.

---

## 10. Prakiraan Sebelum Training

### Prakiraan 1 — Bigram negasi akan membantu

Kata negasi seperti `tidak` muncul pada kelas `negative`, `neutral`, dan `positive`. Unigram `tidak` tidak cukup untuk menentukan arah sentimen karena makna bergantung pada kata setelahnya.

TF-IDF dengan kombinasi unigram dan bigram diperkirakan akan mengurangi kekeliruan antara kelas `positive` dan `negative`, terutama pada frasa seperti:

- `tidak bagus`
- `tidak buruk`
- `tidak mengecewakan`

Prakiraan ini akan diuji melalui cross-validation, confusion matrix, dan error analysis.

---

### Prakiraan 2 — Kelas neutral kemungkinan paling sulit

Kelas `neutral` memiliki jumlah data paling sedikit dan banyak teksnya bersifat informasional. Sebagian teks neutral juga dapat mengandung kata evaluatif lemah yang tumpang tindih dengan kelas positive atau negative.

Karena itu, kelas `neutral` diperkirakan memiliki recall atau F1 terendah dan paling sering tertukar dengan dua kelas lainnya.

Prakiraan ini akan diuji melalui classification report dan confusion matrix.

---

### Prakiraan 3 — Normalisasi informal terbatas akan membantu

Normalisasi placeholder dan pemanjangan huruf diperkirakan dapat mengurangi fragmentasi vocabulary. Bentuk seperti `enak`, `enaaak`, dan `enaaaak` seharusnya tidak selalu diperlakukan sebagai tiga fitur yang sepenuhnya berbeda.

Namun, normalisasi agresif dapat menghapus penekanan emosional atau mengubah kata yang sebenarnya valid. Oleh karena itu, normalisasi akan dibuat terbatas, dapat dipickling, dan ditempatkan di dalam sklearn Pipeline.

Prakiraan ini akan diuji melalui perbandingan hasil cross-validation dan analisis kesalahan model.

---

## 11. Implikasi untuk Tahap Modeling

Berdasarkan EDA, keputusan awal untuk tahap berikutnya adalah:

1. Menggunakan `F1-macro` sebagai metrik utama.
2. Menggunakan stratified 5-fold cross-validation.
3. Mempertahankan kata negasi seperti `tidak`, `bukan`, dan `belum`.
4. Menggunakan TF-IDF dengan kandidat unigram dan bigram.
5. Membuat custom text normalizer untuk:
   - lowercase;
   - trim whitespace;
   - menyatukan repeated whitespace;
   - menormalisasi placeholder;
   - menormalisasi pemanjangan huruf secara terbatas.
6. Tidak menghapus teks sangat pendek secara otomatis.
7. Tidak menyentuh test set selama pemilihan model dan tuning.

---

## 12. Artefak EDA

Artefak yang dihasilkan:

```text
reports/
├── data_quality_summary.csv
├── eda_findings.md
├── label_distribution.png
├── missing_values_by_column.png
├── text_length_by_label.png
└── top_terms_by_label.png
```

Perintah reproduksi:

```powershell
python -m pytest tests/test_eda.py -v
python -m src.eda
```

Hasil test saat checkpoint ini:

```text
5 passed
```

---

## 13. Status Checkpoint

- Audit dataset: selesai.
- Minimal tiga kekotoran nyata: terpenuhi.
- Empat grafik EDA: selesai.
- Interpretasi grafik: selesai.
- Deduplikasi deterministik: selesai.
- Split train/test terkunci: selesai.
- Tiga prakiraan sebelum training: terkunci.
- Tahap berikutnya: implementasi dan pengujian custom text normalizer.
