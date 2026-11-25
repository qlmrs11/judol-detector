# -*- coding: utf-8 -*-
"""Konverter Unicode deterministik (dengan seed) untuk data sintetis skenario S1-S3 (Bab III.4.5, Tabel III-9)."""
import random, string
from .textnorm import skeleton, nfkc

LOW, UP, DIG = string.ascii_lowercase, string.ascii_uppercase, '0123456789'


def _seq(start, chars):
    return {c: chr(start + i) for i, c in enumerate(chars)}


def _math_styles():
    """Gaya Mathematical Alphanumerics tanpa lubang kodepoin (tebal, miring-tebal, sans, sans-tebal, sans-miring, mono)."""
    base = [(0x1D400, 0x1D41A, 0x1D7CE), (0x1D468, 0x1D482, 0x1D7CE), (0x1D5A0, 0x1D5BA, 0x1D7E2),
            (0x1D5D4, 0x1D5EE, 0x1D7EC), (0x1D608, 0x1D622, 0x1D7E2), (0x1D670, 0x1D68A, 0x1D7F6)]
    out = []
    for u, l, d in base:
        m = {}; m.update(_seq(u, UP)); m.update(_seq(l, LOW)); m.update(_seq(d, DIG)); out.append(m)
    return out


F1_STYLES = _math_styles()
F2 = {}; F2.update(_seq(0xFF21, UP)); F2.update(_seq(0xFF41, LOW)); F2.update(_seq(0xFF10, DIG))
F3 = {c: chr(int(h, 16)) for c, h in dict(
    a='0430', c='0441', e='0435', o='043E', p='0440', x='0445', y='0443', i='0456', s='0455', j='0458',
    A='0410', B='0412', C='0421', E='0415', H='041D', K='041A', M='041C', O='041E', P='0420', T='0422', X='0425').items()}
F4 = {}; F4.update(_seq(0x24B6, UP)); F4.update(_seq(0x24D0, LOW)); F4.update({str(i): chr(0x2460 + i - 1) for i in range(1, 10)}); F4['0'] = chr(0x24EA)
F4_SQ = _seq(0x1F130, UP)   # huruf kapital persegi (Enclosed Alphanumeric Supplement)
F6 = {c: chr(int(h, 16)) for c, h in dict(
    a='1D43', b='1D47', c='1D9C', d='1D48', e='1D49', f='1DA0', g='1D4D', h='02B0', i='2071', j='02B2', k='1D4F', l='02E1',
    m='1D50', n='207F', o='1D52', p='1D56', r='02B3', s='02E2', t='1D57', u='1D58', v='1D5B', w='02B7', x='02E3', y='02B8', z='1DBB').items()}
F6.update({str(i): chr(0x2080 + i) for i in range(10)})
# F9: hanya padanan huruf yang sah menurut confusables.txt (UTS#39): E, T, U, l, v. Blok U+2200-22FF memang miskin pengganti huruf.
F9 = {'E': '⋿', 'T': '⊤', 'U': '∪', 'u': '∪', 'l': '∣', 'I': '∣', 'v': '∨'}
F8 = {c: chr(int(h, 16)) for c, h in dict(a='03B1', b='03B2', e='03B5', i='03B9', k='03BA', n='03B7', o='03BF', p='03C1', t='03C4', u='03C5', v='03BD', w='03C9', x='03C7', y='03B3').items()}


def _one(ch, fam, rng):
    if fam == 'F1':
        return rng.choice(F1_STYLES).get(ch)
    if fam == 'F2': return F2.get(ch)
    if fam == 'F3': return F3.get(ch)
    if fam == 'F4':
        if ch in UP and rng.random() < 0.3: return F4_SQ[ch]
        return F4.get(ch)
    if fam == 'F6': return F6.get(ch)
    if fam == 'F8': return F8.get(ch)
    if fam == 'F9': return F9.get(ch)
    return None


MIX = ['F1', 'F2', 'F3', 'F4', 'F6']


def convert(text, fam, prob, rng):
    """Ganti huruf/angka Latin ke keluarga `fam` dengan peluang `prob` per karakter yang dapat dipetakan.
    F10 = lintas blok: tiap karakter diganti dengan keluarga acak dari MIX (dibentuk sintetis)."""
    out = []
    for ch in text:
        if (ch in LOW or ch in UP or ch in DIG) and rng.random() < prob:
            f = rng.choice(MIX) if fam == 'F10' else fam
            r = _one(ch, f, rng)
            out.append(r if r else ch)
        else:
            out.append(ch)
    return ''.join(out)


LEET = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 'A': '4', 'E': '3', 'I': '1', 'O': '0'}


def leetspeak(text):
    return ''.join(LEET.get(c, c) for c in text)


_ZW = {'​', '‌', '‍', '⁠', '﻿'}


def restore_plain(text):
    """Kembalikan penyamaran ke huruf biasa (S2): buang zero-width, NFKC, lalu petakan prototipe confusable
    bila hasilnya huruf/angka ASCII. Karakter lain (emoji, tanda baca) dipertahankan."""
    t = nfkc(''.join(c for c in text if c not in _ZW))
    out = []
    for ch in t:
        if ord(ch) < 128: out.append(ch); continue
        sk = skeleton(ch)
        out.append(sk if sk and all(ord(x) < 128 for x in sk) and sk.isalnum() else ch)
    return ''.join(out)


# ---------------- penyamaran tingkat token (nama situs dan istilah judol) ----------------
from .lexicon_judol import BRAND_RE, TERM_RE, P_SPASI, P_HIASAN, HIASAN, P_ISTILAH
import re as _re


def disguise_token(tok, fam, prob, rng):
    """Samarkan satu token (nama situs atau istilah) ke keluarga `fam`; prob = peluang tiap huruf/angka yang dapat dipetakan diganti.
    Meniru data nyata: kadang huruf berspasi dan kadang diberi hiasan simbol."""
    out = convert(tok, fam, prob, rng)
    if out == tok: return tok
    if rng.random() < P_SPASI: out = ' '.join(out)
    if rng.random() < P_HIASAN:
        h = rng.choice(HIASAN); out = h + out + h
    return out


def target_spans(text, include_terms=True):
    """Posisi nama situs (dan istilah judol) berhuruf ASCII polos pada teks."""
    sp = [(m.start(), m.end(), 'brand') for m in BRAND_RE.finditer(text)]
    if include_terms: sp += [(m.start(), m.end(), 'istilah') for m in TERM_RE.finditer(text)]
    sp.sort(); res, last = [], -1
    for s, e, k in sp:
        if s >= last: res.append((s, e, k)); last = e
    return res


def disguise_targets(text, fam, prob, rng):
    """Samarkan HANYA nama situs (selalu) dan istilah judol (dengan peluang P_ISTILAH); kalimat lain tetap huruf biasa."""
    out, last = [], 0
    for s, e, k in target_spans(text):
        out.append(text[last:s])
        out.append(disguise_token(text[s:e], fam, prob, rng) if (k == 'brand' or rng.random() < P_ISTILAH) else text[s:e]); last = e
    out.append(text[last:]); return ''.join(out)


def salient_token(text):
    """Token yang disamarkan pada komentar normal: kata ASCII terpanjang (>= 4 huruf), mis. nama produk."""
    c = [m for m in _re.finditer(r'[A-Za-z]{4,}', text)]
    return max(c, key=lambda m: m.end() - m.start()) if c else None


def disguise_salient(text, fam, prob, rng):
    m = salient_token(text)
    return text if m is None else text[:m.start()] + disguise_token(m.group(), fam, prob, rng) + text[m.end():]


def leet_targets(text, rng):
    """S2: sebagian nama situs dan istilah judol ditulis leetspeak (al3x1s17, j3p3, g4c0r)."""
    out, last = [], 0
    for s, e, k in target_spans(text):
        out.append(text[last:s]); out.append(leetspeak(text[s:e])); last = e
    out.append(text[last:]); return ''.join(out)
