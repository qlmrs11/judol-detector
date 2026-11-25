"""
================================================================================
UNICODE FAMILY DETECTOR — Rule-based taksonomi F1–F10
================================================================================
Mengapa rule-based, bukan machine learning?
  Tujuan family detection adalah MENGETAHUI taktik penyamaran mana yang
  dipakai, agar hold-out split bisa dibangun. Family adalah properti objektif
  yang bisa dibaca langsung dari code point (menurut blok Unicode), bukan
  target prediksi. Menggunakan ML untuk mendeteksi family akan menimbulkan
  circular dependency: model dilatih dengan label yang dihasilkan model lain.

  Referensi: UTS#39 (Unicode Security Mechanisms) &
             guide_sempro.md §5 — Family <-> Unicode Block Mapping.

Catatan PENTING tentang F5:
  Family F5 (Zero-Width) terindeksi kasus artefak scraping YouTube, bukan
  serangan. Pola umum: U+200B + U+00A0 (NBSP) + @username + U+00A0.
  Modul ini mendeteksi F5 apa adanya (fakta), lalu menandai apakah baris tersebut
  juga mengandung pola mention. Keputusan bersih/tidak di ambiguity dilakukan
  di Notebook 02 dengan temuan audit — bukan diasumsikan di sini.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable

# ---------------------------------------------------------------------------
# Definisi blok (start, end, label) — UTS#39 & Unicode Character Database
# ---------------------------------------------------------------------------
FAMILY_RANGES: dict[str, list[tuple[int, int]]] = {
    # v2 (5 Nov 2026): rentang disamakan dengan Tabel II-2 proposal; versi lama di unicode_family_v1_arsip.py.
    # Koreksi v2: regional indicator/bendera (U+1F1E6-1F1FF) bukan huruf -> dikeluarkan (emoji, benign);
    # huruf persegi U+1F130-1F189 adalah Enclosed Alphanumeric Supplement -> pindah ke F4;
    # bentuk geometris U+25A0-25FF (hiasan, bukan huruf) dan katakana/hangul halfwidth U+FF61-FFDC (bukan Latin) dikeluarkan;
    # tanda x dan bagi Latin-1 (U+00D7, U+00F7) dikeluarkan dari F9 karena dipakai wajar pada komentar normal.
    # F1: Mathematical Alphanumeric Symbols
    "F1": [(0x1D400, 0x1D7FF)],
    # F2: Fullwidth ASCII variants
    "F2": [(0xFF10, 0xFF19), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)],  # hanya angka dan huruf Latin full-width (tanda baca full-width dipakai wajar pada teks CJK)
    # F3: Cyrillic + Cyrillic Supplement
    "F3": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    # F4: Enclosed Alphanumerics + huruf Latin persegi (Enclosed Alphanumeric Supplement)
    "F4": [(0x2460, 0x24FF), (0x1F130, 0x1F189)],
    # F5: Zero-width & karakter format tak terlihat
    "F5": [(0x200B, 0x200F), (0x2060, 0x2064), (0xFEFF, 0xFEFF)],
    # F6: Superscripts and Subscripts + huruf pengubah (Spacing Modifier Letters, Phonetic Extensions)
    "F6": [
        (0x02B0, 0x02FF),
        (0x2070, 0x209F),
        (0x1D2C, 0x1DBF),
        (0x2C7C, 0x2C7D),
    ],
    # F7: Combining Diacritical Marks (+ Extended, + for Symbols)
    "F7": [(0x0300, 0x036F), (0x1AB0, 0x1AFF), (0x1DC0, 0x1DFF), (0x20D0, 0x20FF)],
    # F8: Greek and Coptic + Greek Extended
    "F8": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    # F9: Mathematical Operators
    "F9": [(0x2200, 0x22FF)],
}

# Codepoint zero-width terpisah agar bisa dihitung independen dari family F5.
ZERO_WIDTH_CODEPOINTS: frozenset[int] = frozenset(
    cp
    for ranges in [FAMILY_RANGES["F5"]]
    for lo, hi in ranges
    for cp in range(lo, hi + 1)
)

NBSP = 0x00A0

FAMILY_NAMES: dict[str, str] = {
    "F1": "Mathematical Alphanumerics",
    "F2": "Full-width Latin",
    "F3": "Cyrillic Look-Alike",
    "F4": "Enclosed Alphanumerics",
    "F5": "Zero-Width / Non-Spacing",
    "F6": "Superscript & Subscript",
    "F7": "Combining Diacritic Marks",
    "F8": "Greek Look-Alike",
    "F9": "Math Operators",
    "F10": "Mixed (lintas blok)",
    "NORMAL": "Tanpa penyamaran blok",
    "MIXED": "Kombinasi >1 family (perlu verifikasi)",
}

# Unicode block yang 'benign' dan TIDAK boleh dianggap sinyal serangan:
# emoji, NBSP, CJK/ideograf (dipakai di komentar Indonesia), Arab dll.
_BENIGN_BLOCKS: list[tuple[int, int]] = [
    (0x00A0, 0x00A0),  # NBSP
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x1F000, 0x1FAFF),  # emoji & pictographs
    (0x2600, 0x27BF),  # misc symbols & dingbats (sering dipakai spam)
    (0x20000, 0x2FA1F),  # CJK ext
]


def _in_ranges(cp: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def _is_emoji_cp(cp: int) -> bool:
    return 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or cp in (0xFE0F, 0x2640, 0x2642, 0x2194) or 0x1F3FB <= cp <= 0x1F3FF


def _is_emoji_zwj(text: str, i: int) -> bool:
    """True bila U+200D pada posisi i diapit karakter emoji (emoji ZWJ sequence)."""
    prev = ord(text[i - 1]) if i > 0 else 0
    nxt = ord(text[i + 1]) if i + 1 < len(text) else 0
    return _is_emoji_cp(prev) and _is_emoji_cp(nxt)


def classify_characters(text: str) -> tuple[list[str], bool, bool]:
    """
    Analisis satu string.

    Returns
    -------
    families : list[str]
        Family yang terdeteksi (urutan, tanpa duplikat). Kosong bila NORMAL.
    has_zero_width : bool
        True bila ada codepoint zero-width/invisible.
    has_benign : bool
        True bila ada karakter dari blok benign (emoji/NBSP/CJK).

    Catatan: karakter dari blok benign TETAP ikut menghitung sebagai 'ada',
    tetapi tidak dipetakan ke family serangan (F1-F9). Ini menjaga agar
    fitur 'unicode_ratio' tidak menyalahartikan emoji sebagai serangan.
    """
    families: list[str] = []
    has_zero_width = False
    has_benign = False

    if not text:
        return [], False, False

    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp < 128:
            continue  # ASCII -> bukan penyamaran
        if cp == 0x200D and _is_emoji_zwj(text, i):
            has_benign = True  # ZWJ di dalam rangkaian emoji (mis. 🤦‍♀️) adalah bagian sah emoji, bukan penyamaran
            continue

        matched = False
        for fam, ranges in FAMILY_RANGES.items():
            if _in_ranges(cp, ranges):
                if fam == "F5":
                    has_zero_width = True
                if fam not in families:
                    families.append(fam)
                matched = True
                break

        if not matched and _in_ranges(cp, _BENIGN_BLOCKS):
            has_benign = True

    return families, has_zero_width, has_benign


def assign_family_label(
    families: list[str],
    has_zero_width: bool,
) -> str:
    """
    Terjemahkan daftar family menjadi SATU label untuk tabel eksperimen.

    Aturan:
      - 0 family            -> 'NORMAL'
      - 1 family            -> nama family tersebut (mis. 'F1')
      - >1 family            -> 'MIXED'
      - hanya zero-width    -> 'F5'
    """
    active = list(families)
    if not active:
        return "F5" if has_zero_width else "NORMAL"
    if len(active) == 1:
        return active[0]
    return "MIXED"


def detect_youtube_mention(text: str) -> bool:
    """
    Deteksi artefak scraping YouTube pada blok F5.

    Pola nyata di dataset: chr(0x200B) + chr(0xA0) + '@username' + chr(0xA0)
    Contoh: '\\u200b\\xa0@izurzuhri\\xa0'
    Ini BUKAN unicode obfuscation -> harus dipisahkan dari F5.
    """
    if not text:
        return False
    # Dua varian: dengan dan tanpa U+200B di depan
    return ("\x00a0@" in text or "\xa0@" in text) or (
        "\u200b\xa0@" in text
    )


def detect_family_series(texts: Iterable[str]) -> list[dict]:
    """
    Jalankan klasifikasi family pada seluruh kolom teks.

    Returns list of dict per baris:
      {family_label, families_detected, n_families,
       has_zero_width, has_benign, is_youtube_mention, n_non_ascii}
    """
    results = []
    for text in texts:
        text = text if isinstance(text, str) else ""
        families, has_zw, has_benign = classify_characters(text)
        label = assign_family_label(families, has_zw)
        n_non_ascii = sum(1 for ch in text if ord(ch) > 127)
        results.append(
            {
                "family_label": label,
                "families_detected": "+".join(families) if families else "NORMAL",
                "n_families": len(families),
                "has_zero_width": has_zw,
                "has_benign": has_benign,
                "is_youtube_mention": detect_youtube_mention(text),
                "n_non_ascii": n_non_ascii,
            }
        )
    return results


def normalize_nfkc(text: str) -> str:
    """
    Normalisasi Unicode NFKC (Compatibility Composition).

    Peran: membuat cabang pembanding 'normalisasi saja' untuk menguji
    apakah F1 (Mathematical Alphanumerics) benar-benar bisa dilumpuhkan NFKC.
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_tr39_skeleton(text: str) -> str:
    """
    Placeholder pemetaan skeleton TR-39.

    CATATAN PENTING: skeleton confusable resmi Unicode berada di file
    'Confusable.txt' (delimiter ';' berbasis codepoint). Modul ini menyediakan
    antarmuka agar Notebook 02 dapat mengimplementasikan skeleton sungguhan
    setelah file tersebut diunduh. Untuk sekarang mengembalikan NFKC sebagai
    fallback agar pipeline tetap dapat dijalankan.

    STATUS PER 19 Okt 2026 - SUDAH DIJALANKAN:
      - Tabel 'Confusable.txt' BELUM diunduh.
      - Cabang view "skel" TIDAK dipakai pada skenario manapun
        (A1/A2/A3/B/C/D semuanya memakai view "raw" atau "nfkc").
      - Semua hasil eksperimen pada guide_sempro.md §10 TIDAK tersentuh
        skeleton TR-39; normalisasi yang diuji hanyalah NFKC.
      - Implementasi skeleton sungguhan = pekerjaan lanjutan (Bab 6 Saran).
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


# ---------------------------------------------------------------------------
# PEMBERSIHAN ATURAN PENULIS (proposal Bab III.4.2, v2 5 Nov 2026)
# ---------------------------------------------------------------------------
import re as _re

_MENTION = _re.compile('[\u200b\u2060\ufeff]*\xa0?@[^\s\xa0]+\xa0?')
_MENTION_TANPA_AT = _re.compile('^[\u200b\u2060\ufeff]+\xa0[^\s\xa0]+\xa0')
_LEADING_ZW = _re.compile('^[\u200b\u2060\ufeff]+')


def clean_comment(text: str) -> str:
    """Aturan pembersihan minimal: mention (beserta karakter zero-width/NBSP pembungkusnya hasil format
    pengambilan data) diganti token '@pengguna'; zero-width di awal komentar (sisa format balasan) dibuang;
    spasi tepi dirapikan. Huruf besar-kecil, emoji, simbol, angka, tanda baca, dan zero-width di dalam kata
    TIDAK diubah."""
    t = text if isinstance(text, str) else ''
    t = _MENTION_TANPA_AT.sub('@pengguna ', t)
    t = _MENTION.sub(' @pengguna ', t)
    t = _LEADING_ZW.sub('', t)
    return _re.sub(' {2,}', ' ', t).strip()
