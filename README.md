# Dashboard Moderasi Komentar Judol

Aplikasi demonstrasi artefak Tugas Akhir: deteksi komentar judi online (judol) di YouTube yang disamarkan dengan
karakter Unicode, menggunakan **Feature Fusion iterasi 3 (P9)**.

## Isi aplikasi

1. **Dashboard benchmark.** Macro-F1 sepuluh peserta pada skenario S0–S4, profil ketahanan, perbandingan akurasi S0 dengan skor terburuk, serta FPR dan recall.
2. **Deteksi dan antrean moderasi.** Satu komentar diklasifikasi oleh P9 dan dua pembanding (P3 LR char n-gram, P1 kata kunci). Berkas CSV menghasilkan antrean tinjauan untuk moderator.
3. **Generator skenario stres.** Komentar diubah dengan fungsi penyamaran yang sama dengan pembentukan skenario uji, lalu diklasifikasi ketiga model.
4. **Ablasi, target, dan proposisi.** Varian V1–V7, status target T1–T4 (iterasi 2 dan 3), dan status proposisi Pr1–Pr3.
5. **Tentang protokol.**

## Asal model dan angka

- `models/p9.joblib` dan `models/p3.joblib` dilatih ulang dengan kode eksperimen asli (seed 11) melalui `tools/latih_model.py`.
  Prediksi kedua model **identik** dengan prediksi seed 11 yang dilaporkan pada TA, untuk semua skenario S0–S4 (`models/info_model.json`).
- `src/` adalah salinan tanpa perubahan dari kode eksperimen yang dibutuhkan untuk prediksi.
- `hasil/*.csv` adalah tabel hasil eksperimen (rata-rata seed 11, 911, 99; IK 95% bootstrap 1.000 kali). Karena dashboard memakai rata-rata
  tiga seed, angkanya dapat berbeda sedikit dari skor satu model seed 11 (misalnya Macro-F1 S4 P9: 0,914 rata-rata, 0,910 seed 11).
- `data/confusables.txt` adalah tabel UTS #39 dari Unicode, Inc. (lihat ketentuan penggunaan di kepala berkas).

Dataset tidak disertakan. Dataset yang dipakai adalah dataset komentar judi online YouTube dari Kaggle.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Versi scikit-learn dikunci karena model joblib hanya termuat dengan versi yang sama dengan saat pelatihan.

## Catatan

Model hanya memberi saran penandaan. Keputusan akhir tetap ada pada moderator.
