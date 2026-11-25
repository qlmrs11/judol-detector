# -*- coding: utf-8 -*-
"""Leksikon dan pola penyamaran judol yang dipelajari dari data nyata (audit 9 Nov 2026, dibantu LLM).

Temuan audit pada 2.945 komentar judol (data bersih, sebelum split):
- 1.694 (58%) memuat karakter font aneh, tetapi hanya sekitar 19% token per komentar (median 14%) yang ber-font.
  Kalimat di sekitarnya ditulis dengan huruf biasa; yang disamarkan adalah NAMA SITUS dan kadang ISTILAH JUDOL.
- Token ber-font terbanyak: alexis17 (424), sgi88 (166), agustoto (83), weton88 (70), dora77 (66), dewadora (56),
  mandalika77 (54), pulauwin (51), pulau777 (43), berkah99, axl777, probet855, miya88, king328 (klng328), ahmadtoto,
  manut88, pluto88, bukit888, bowo77, suhu328.
- Bentuk penyamaran pada nama situs: (a) campuran gaya per huruf, misalnya "AGuSToTo" dengan gaya matematis berbeda
  tiap huruf; (b) full-width; (c) huruf Cyrillic/Yunani yang mirip; (d) huruf berspasi ("D O R A 7 7", "P U L A U W I N");
  (e) hiasan di sekitar nama ("ALEXIS⭐17⭐", "░MANUT88░"); (f) zero-width di dalam kata ("gacu<ZW>r").
- Istilah judol yang sering muncul: jepe, wd, maxwin, gacor/gacir/gac0r, wede, jp, scatter, cuan, rezeki, modal, hoki.
  Istilah umumnya ditulis biasa atau dengan ejaan plesetan (gac0r, wede), jarang ber-font.
- Komentar normal ber-font (90 dari 6.018) umumnya seluruh komentar ditulis dalam satu gaya (mis. monospace "𝙾𝚔 𝚝𝚎𝚖𝚎𝚗")
  atau memuat huruf pangkat ("omon²"); ini dasar skenario S3 (seluruh komentar normal diberi font).
"""
import re

# nama situs (huruf kecil), diturunkan dari data; bentuk generik huruf+angka ditangani pola GENERIK
BRANDS = ['alexis17', 'sgi88', 'agustoto', 'weton88', 'dora77', 'dewadora', 'mandalika77', 'pulauwin', 'pulau777', 'berkah99',
          'axl777', 'axl77', 'probet855', 'miya88', 'king328', 'ahmadtoto', 'manut88', 'pluto88', 'bukit888', 'bowo77', 'suhu328',
          'aero88', 'asia99', 'betul88', 'dongo88', 'gacha168', 'ggwp88', 'hoki777', 'luto88', 'metrowin88', 'nut88', 'owo77',
          'simba69', 'yuk69', 'visi4d', 'kyt4d', 'pawpaw4d', 'alexis']
# istilah judol yang dapat ikut disamarkan (lebih jarang daripada nama situs)
TERMS = ['jepe', 'wd', 'maxwin', 'gacor', 'gacir', 'wede', 'jackpot', 'scatter', 'slot']
GENERIK = r'[a-z]{3,}\d{2,4}'

_alt = '|'.join(sorted(map(re.escape, BRANDS), key=len, reverse=True))
BRAND_RE = re.compile(rf'(?<![A-Za-z0-9])(?:{_alt}|{GENERIK})(?![A-Za-z0-9])', re.I)
TERM_RE = re.compile(rf"(?<![A-Za-z0-9])(?:{'|'.join(TERMS)})(?![A-Za-z0-9])", re.I)

# kemungkinan penyamaran tambahan (diamati pada data nyata)
P_SPASI = 0.15      # huruf berspasi
P_HIASAN = 0.15     # hiasan simbol di sekitar nama situs
HIASAN = ['⭐', '░', '✌']
P_ISTILAH = 0.35    # peluang istilah judol ikut disamarkan pada komentar yang juga memuat nama situs
