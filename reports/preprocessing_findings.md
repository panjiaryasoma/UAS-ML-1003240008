# Preprocessing Findings — SentimenID API

**Proyek:** UAS Machine Learning End-to-End  
**Kasus:** Case C — Klasifikasi Sentimen Teks Bahasa Indonesia  
**Komponen:** `IndonesianTextNormalizer`  
**Status:** Selesai dan tervalidasi  
**Lokasi implementasi:** `src/transformers.py`  
**Lokasi pengujian:** `tests/test_transformers.py`

---

## 1. Tujuan

Tahap preprocessing bertujuan menormalkan variasi teks Bahasa Indonesia secara terbatas tanpa menghapus sinyal sentimen, terutama kata negasi.

Seluruh transformasi dirancang untuk ditempatkan di dalam `sklearn Pipeline` agar proses training, cross-validation, evaluasi, dan serving menggunakan aturan preprocessing yang sama. Pendekatan ini juga mencegah data leakage karena normalisasi dan TF-IDF dijalankan di dalam pipeline pada setiap fold.

---

## 2. Custom Transformer

Preprocessing diimplementasikan melalui class:

```python
src.transformers.IndonesianTextNormalizer
```

Class tersebut mengikuti kontrak:

```python
BaseEstimator
TransformerMixin
```

Dengan demikian, normalizer kompatibel dengan:

- `sklearn Pipeline`;
- proses cross-validation;
- serialisasi model;
- pemakaian ulang pada API.

Method `fit()` tidak mempelajari statistik apa pun dari data dan hanya mengembalikan `self`. Seluruh perubahan teks dilakukan melalui method `transform()`.

---

## 3. Aturan Normalisasi

Normalizer melakukan proses berikut.

### 3.1 Lowercase

Seluruh teks diubah menjadi huruf kecil.

Contoh:

```text
MAKANAN INI ENAK
```

menjadi:

```text
makanan ini enak
```

### 3.2 Perapian whitespace

Normalizer:

- menghapus spasi di awal dan akhir;
- menggabungkan spasi, tab, atau whitespace berulang menjadi satu spasi.

Contoh:

```text
"  makanan   ini	 enak  "
```

menjadi:

```text
"makanan ini enak"
```

### 3.3 Normalisasi placeholder

Placeholder dengan pola seperti:

```text
__laugh__
__sad__
```

dinormalisasi menjadi:

```text
laugh
sad
```

Placeholder tidak dihapus seluruhnya karena masih dapat membawa informasi emosi.

### 3.4 Normalisasi pemanjangan huruf

Huruf yang berulang tiga kali atau lebih dinormalisasi menjadi satu karakter.

Contoh:

```text
enaaak
mantapppp
sediiih
```

menjadi:

```text
enak
mantap
sedih
```

Normalisasi ini bertujuan mengurangi fragmentasi vocabulary pada TF-IDF.

### 3.5 Normalisasi slang negasi

Variasi negasi informal berikut dipetakan menjadi `tidak`:

```text
enggak
nggak
ngga
gak
ga
gk
tdk
```

Penggantian dilakukan menggunakan batas kata sehingga potongan karakter di dalam kata lain tidak ikut berubah.

### 3.6 Mempertahankan negasi formal

Kata negasi formal tidak dihapus, termasuk:

```text
tidak
bukan
belum
```

Keputusan ini penting karena menghapus negasi dapat membuat frasa seperti:

```text
bagus
tidak bagus
```

terlihat terlalu mirip bagi model.

---

## 4. Stopword

Stopword removal tidak digunakan pada data modeling.

Daftar stopword hanya dipakai untuk visualisasi istilah teratas pada tahap EDA. Kata negasi tidak dimasukkan ke daftar stopword.

Dengan demikian, pipeline tetap mempertahankan kata yang berpotensi mengubah polaritas sentimen.

---

## 5. Bentuk Input yang Didukung

`IndonesianTextNormalizer` telah diuji untuk menerima:

- Python `list`;
- `pandas.Series`;
- `pandas.DataFrame` satu kolom;
- NumPy array dua dimensi dengan satu kolom.

Contoh bentuk yang valid:

```python
["bagus", "tidak bagus"]
```

```python
pd.Series(["bagus", "tidak bagus"])
```

```python
pd.DataFrame({"text": ["bagus", "tidak bagus"]})
```

```python
np.array([["bagus"], ["tidak bagus"]], dtype=object)
```

---

## 6. Validasi Input

DataFrame dengan lebih dari satu kolom ditolak secara eksplisit menggunakan `ValueError`.

Contoh input yang ditolak:

```python
pd.DataFrame({
    "text": ["bagus"],
    "extra": ["data"],
})
```

Pembatasan ini mencegah input ambigu masuk ke pipeline.

Nilai berikut juga ditangani:

- `None`;
- `NaN`;
- `pd.NA`.

Nilai tersebut diubah menjadi string kosong tanpa memodifikasi objek input asli.

---

## 7. Serialisasi

Normalizer telah diuji melalui:

- `pickle.dumps()` dan `pickle.loads()`;
- `joblib.dump()` dan `joblib.load()`.

Hasil pengujian menunjukkan bahwa class dapat disimpan dan dimuat kembali tanpa mengubah perilaku transformasinya.

Hal ini penting karena artefak final UAS harus menyimpan pipeline utuh, bukan hanya classifier telanjang.

---

## 8. Hasil Pengujian

File pengujian:

```text
tests/test_transformers.py
```

Cakupan pengujian:

1. lowercase, trim, dan whitespace;
2. placeholder;
3. pemanjangan huruf;
4. slang negasi;
5. negasi formal tetap dipertahankan;
6. missing value;
7. input list;
8. input Series;
9. input DataFrame satu kolom;
10. input NumPy array satu kolom;
11. penolakan DataFrame multi-kolom;
12. kompatibilitas pickle;
13. round trip joblib;
14. input asli tidak dimutasi.

Hasil checkpoint:

```text
9 test transformer passed
16 test proyek passed
```

Perintah reproduksi:

```powershell
python -m pytest tests/test_transformers.py -v
python -m pytest tests/ -v
```

---

## 9. Alasan Metodologis

Normalisasi dilakukan secara terbatas karena variasi informal dapat menyebabkan fragmentasi vocabulary.

Sebagai contoh:

```text
enak
enaaak
enaaaak
```

dapat dianggap sebagai tiga fitur berbeda oleh TF-IDF walaupun maknanya serupa.

Namun, normalisasi tidak dibuat terlalu agresif karena:

- pengulangan huruf dapat membawa penekanan emosional;
- placeholder dapat mengandung informasi sentimen;
- penggantian slang yang terlalu luas dapat mengubah kata yang sebenarnya valid;
- stemming atau penghapusan kata secara agresif dapat merusak konteks negasi.

Karena itu, aturan baru hanya boleh ditambahkan bila didukung oleh temuan data atau pola kesalahan model.

---

## 10. Batasan

Keberhasilan teknis normalizer belum membuktikan bahwa normalisasi meningkatkan F1-macro.

Yang sudah terbukti pada checkpoint ini adalah:

- class bekerja sesuai kontrak;
- bentuk input ditangani dengan benar;
- negasi dipertahankan;
- class kompatibel dengan sklearn dan joblib.

Dampak normalisasi terhadap performa model masih harus diuji melalui:

- 5-fold cross-validation;
- perbandingan model;
- confusion matrix;
- error analysis.

---

## 11. Implikasi untuk Training

Pipeline kandidat akan mengikuti urutan:

```text
IndonesianTextNormalizer
→ TfidfVectorizer
→ classifier
```

TF-IDF akan menguji kandidat:

- unigram;
- kombinasi unigram dan bigram.

Classifier yang akan dibandingkan:

- Multinomial Naive Bayes;
- Logistic Regression;
- calibrated Linear SVC.

Seluruh preprocessing tetap berada di dalam pipeline agar tidak terjadi data leakage selama cross-validation.

---

## 12. Status Checkpoint

- Custom transformer: selesai.
- Lowercase dan whitespace: selesai.
- Placeholder normalization: selesai.
- Elongated-word normalization: selesai.
- Slang negation mapping: selesai.
- Negation preservation: selesai.
- Validasi bentuk input: selesai.
- DataFrame multi-kolom ditolak: selesai.
- Pickle compatibility: selesai.
- Joblib round trip: selesai.
- Test transformer: 9 passed.
- Seluruh test proyek: 16 passed.
- Tahap berikutnya: `tests/test_train.py` dan `src/train.py`.
