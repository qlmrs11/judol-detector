# -*- coding: utf-8 -*-
"""Normalisasi teks (NFKC, skeleton TR39) dan fitur statistik Unicode (12 fitur, 20 kolom; Bab III.4.3).
Fitur statistik selalu dihitung dari teks MENTAH, karena sinyal penyamaran hilang bila teks dinormalisasi dulu (III.4.3)."""
import math, re, unicodedata, os
from collections import Counter
import numpy as np
from .unicode_family import FAMILY_RANGES

HERE = os.path.dirname(os.path.abspath(__file__))
CONF_PATH = os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'raw', 'confusables.txt'))
_CONF = None


def _load_confusables():
    """Peta confusables.txt (UTS#39 18.0.0): kode sumber -> string prototipe."""
    global _CONF
    if _CONF is not None: return _CONF
    m = {}
    with open(CONF_PATH, encoding='utf-8-sig') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line: continue
            parts = [p.strip() for p in line.split(';')]
            if len(parts) < 3: continue
            src = ''.join(chr(int(x, 16)) for x in parts[0].split())
            tgt = ''.join(chr(int(x, 16)) for x in parts[1].split())
            if len(src) == 1: m[src] = tgt
    _CONF = m
    return m


def nfkc(t):
    return unicodedata.normalize('NFKC', t if isinstance(t, str) else '')


def skeleton(t):
    """skeleton(X) = NFD(map_prototipe(NFD(X))) menurut UTS#39 (III.4.2: 'm' -> 'rn', '1' -> 'l')."""
    c = _load_confusables()
    d = unicodedata.normalize('NFD', t if isinstance(t, str) else '')
    d = ''.join(c.get(ch, ch) for ch in d)
    return unicodedata.normalize('NFD', d)


def nfkc_skeleton(t):
    return skeleton(nfkc(t))


NORMALIZERS = {'none': lambda t: t, 'nfkc': nfkc, 'nfkc_skel': nfkc_skeleton}

# ---------------- fitur statistik Unicode ----------------
_FAM = [f'F{i}' for i in range(1, 10)]
_ZW = set(range(0x200B, 0x2010)) | set(range(0x2060, 0x2065)) | {0xFEFF}
_URL = re.compile(r'https?://|www\.', re.I)
_MENT = re.compile(r'@\w+')
STAT_COLS = (['entropi', 'rasio_unicode', 'rasio_ascii'] + [f'blok_{f}' for f in _FAM] +
             ['n_zerowidth', 'rasio_panjang_norm', 'rasio_digit', 'rasio_simbol', 'n_emoji', 'ada_url', 'n_mention', 'lebar_rata2'])


def _fam_of(cp):
    for f in _FAM:
        for lo, hi in FAMILY_RANGES[f]:
            if lo <= cp <= hi: return f
    return None


def _is_emoji(cp):
    return 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF


def stat_features(t):
    t = t if isinstance(t, str) else ''
    n = max(len(t), 1)
    cps = [ord(c) for c in t]
    cnt = Counter(cps)
    ent = -sum((v / n) * math.log2(v / n) for v in cnt.values()) if t else 0.0
    nonascii = sum(1 for c in cps if c > 127)
    fam = Counter(_fam_of(c) for c in cps if c > 127)
    zw = sum(1 for c in cps if c in _ZW)
    digit = sum(1 for ch in t if ch.isdigit())
    sym = sum(1 for ch in t if not ch.isalnum() and not ch.isspace() and ord(ch) < 0x1F000)
    emo = sum(1 for c in cps if _is_emoji(c))
    width = sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in t) / n
    row = [ent, nonascii / n, (n - nonascii) / n] + [fam.get(f, 0) / n for f in _FAM] + \
          [zw, len(nfkc(t)) / n, digit / n, sym / n, emo, 1.0 if _URL.search(t) else 0.0, len(_MENT.findall(t)), width]
    return row


def stat_matrix(texts):
    X = np.array([stat_features(t) for t in texts], dtype=np.float64)
    for j, c in enumerate(STAT_COLS):          # kolom hitungan diperhalus agar skalanya wajar
        if c in ('n_zerowidth', 'n_emoji', 'n_mention'): X[:, j] = np.log1p(X[:, j])
    return X
