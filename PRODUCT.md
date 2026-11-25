# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
- Penguji seminar proposal / sidang TA dan dosen pembimbing (ARK), melihat demo di proyektor ruang sidang yang dioperasikan penulis (2–3 menit).
- Pembimbing membuka aplikasi sendiri lewat link (repo GitHub private, Streamlit Community Cloud) tanpa penjelasan langsung.

## Product Purpose
Demonstrasi artefak Tugas Akhir (DSRM, tahap demonstration): deteksi komentar judi online (judol) di YouTube yang disamarkan dengan karakter Unicode, memakai Feature Fusion iterasi 3 (P9). Sukses = penguji memahami dalam satu tampilan bahwa model pembanding (P1 kata kunci, P3 LR char n-gram) jebol pada komentar tersamar, sedangkan P9 tetap menangkapnya, dan bahwa angkanya sama dengan Bab IV.

## Positioning
Benchmark ketahanan S0–S4, bukan sekadar akurasi: "akurasi S0 ≠ ketahanan". Model di aplikasi dilatih ulang dengan kode eksperimen asli (seed 11) dan prediksinya identik dengan hasil yang dilaporkan.

## Capabilities and Constraints
- Tiga bagian saja: (1) Coba deteksi (P1/P3/P9 berdampingan + alasan), (2) Uji penyamaran (satu komentar diubah ke F1–F10, S2, S3), (3) Ringkasan hasil (satu grafik inti + status T1–T4).
- Dibuang dari demo: antrean CSV, halaman ablasi lengkap, tabel panjang, halaman latih model.
- Stack: Streamlit (diminta ARK), Python; angka dari `hasil/*.csv`, model dari `models/`.
- Istilah: peserta P1–P10, proposisi Pr1–Pr3, skenario S0–S4, keluarga F1–F10, target T1–T4.
- Skor P9/P3 = fungsi keputusan linear, bukan peluang.

## Brand Commitments
- Warna Telkom University sebagai aksen (merah Telkom).
- Bahasa Indonesia baku tetapi ringkas.

## Evidence on Hand
- `hasil/*.csv` (rata-rata 3 seed), `models/info_model.json` (bukti prediksi identik seed 11).
- Tidak ada pengguna nyata/moderator; jangan mengklaim dipakai platform.

## Product Principles
1. Satu pesan per layar; dapat dibaca dari belakang ruang sidang.
2. Perbandingan antarmodel selalu berdampingan, bukan satu model sendirian.
3. Jujur: tampilkan juga kelemahan (F3 homoglyph, S4) bila muncul.
4. Model memberi saran; keputusan tetap pada moderator.
