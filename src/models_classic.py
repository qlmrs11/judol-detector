# -*- coding: utf-8 -*-
"""Peserta benchmark berbasis scikit-learn (Tabel III-3 no. 1, 2, 3, 8) dan varian ablasi V1-V7 (Tabel III-4).
Hiperparameter dipilih pada data validasi berdasarkan Macro-F1 (III.4.7); ruang pencarian mengikuti Tabel III-5."""
import itertools, re, time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from .textnorm import NORMALIZERS, stat_matrix, nfkc

C_GRID = [0.01, 0.1, 1, 10]


class Feats:
    """Penyusun fitur: normalisasi -> TF-IDF (kata/karakter) [+ fitur statistik Unicode dari teks mentah]."""
    def __init__(self, norm='none', analyzer='char', ngram=(3, 5), stats=False, text=True, max_features=None, kw=False,
                 anomali=False, min_df=1, max_df=1.0):
        # kw: False | True (2 bendera lama) | 'alyudani' (keyword score + keyword flag, Alyudani dkk. 2026 Tabel 6)
        # anomali: fitur anomali dan perilaku Nagalay dkk. (2025) Tabel 2, dihitung dari teks asli
        self.norm, self.analyzer, self.ngram, self.stats, self.text, self.mf, self.kw = norm, analyzer, ngram, stats, text, max_features, kw
        self.anomali, self.min_df, self.max_df = anomali, min_df, max_df

    def _t(self, texts): return [NORMALIZERS[self.norm](t) for t in texts]

    def fit(self, texts):
        if self.text:
            self.vec = TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram, max_features=self.mf, sublinear_tf=True,
                                       min_df=self.min_df, max_df=self.max_df, dtype=np.float32)
            self.vec.fit(self._t(texts))
        if self.stats:
            self.sc = MaxAbsScaler().fit(stat_matrix(texts))
        return self

    def transform(self, texts):
        parts = []
        if self.text: parts.append(self.vec.transform(self._t(texts)))
        if self.stats: parts.append(sparse.csr_matrix(self.sc.transform(stat_matrix(texts)).astype(np.float32)))
        if self.kw == 'alyudani': parts.append(sparse.csr_matrix(alyudani_matrix(texts)))
        elif self.kw: parts.append(sparse.csr_matrix(keyword_matrix(texts)))
        if self.anomali: parts.append(sparse.csr_matrix(nagalay_matrix(texts)))
        return sparse.hstack(parts).tocsr() if len(parts) > 1 else parts[0]


KW = re.compile(r'\b(4d|wd|jepe|maxwin|visi4d|kyt4d|alexis17)\b')
SITE = re.compile(r'[a-z]{3,}\d{2,4}')


def keyword_flags(t):
    s = t.lower()
    return [1.0 if KW.search(s) else 0.0, 1.0 if SITE.search(s) else 0.0]


def keyword_matrix(texts): return np.array([keyword_flags(t) for t in texts], dtype=np.float32)


# --- Alyudani dkk. (2026) Tabel 6: strong keywords bobot 2, weak keywords bobot 0,5.
# Dua fitur: keyword score (total bobot kata kunci) dan keyword flag (ada kata kunci kuat = indikasi kuat promosi).
# Praproses makalah (normalisasi Unicode, de-leet, dst.) TIDAK diterapkan: seluruh peserta memakai data hasil praproses penulis.
ALY_STRONG = ['slot', 'maxwin', 'wd', 'depo', 'scatter', 'jp', 'jackpot', 'zeus', 'togel']
ALY_WEAK = ['bonus', 'pola', 'gacor', 'cuan', 'situs', 'link', 'bio', 'modal', 'profil', 'spin']
_ALY_S = re.compile(r'\b(' + '|'.join(ALY_STRONG) + r')\b'); _ALY_W = re.compile(r'\b(' + '|'.join(ALY_WEAK) + r')\b')


def alyudani_feats(t):
    s = t.lower(); ns, nw = len(_ALY_S.findall(s)), len(_ALY_W.findall(s))
    return [2.0 * ns + 0.5 * nw, 1.0 if ns else 0.0]


def alyudani_matrix(texts): return np.array([alyudani_feats(t) for t in texts], dtype=np.float32)


# --- Nagalay dkk. (2025) Tabel 2: fitur anomali dan perilaku dari teks ASLI. Tabel makalah mendeskripsikan 10 fitur
# (teks menyebut 11); yang dideskripsikan diimplementasikan semuanya. Daftar kata kunci tidak dilaporkan lengkap (contoh: gacor, wd),
# sehingga dipakai daftar terbitan Alyudani dkk. (2026) Tabel 6. Emoji dihitung dengan rentang blok emoji (pustaka emoji tidak tersedia).
_TS = re.compile(r'\b\d{1,2}:\d{2}\b'); _MEREK = re.compile(r'[A-Za-z]+[0-9]+'); _ULANG = re.compile(r'([0-9])\1')
_NAG_KW = re.compile(r'\b(' + '|'.join(ALY_STRONG + ALY_WEAK) + r')\b')


def nagalay_feats(t):
    n = max(len(t), 1); huruf = [c for c in t if c.isalpha()]
    aneh = sum(1 for c in t if not c.isalnum() and not c.isspace())
    kapital = 100.0 * sum(1 for c in huruf if c.isupper()) / max(len(huruf), 1)
    angka = sum(1 for c in t if '0' <= c <= '9')
    fullwidth = 1.0 if any(0xFF01 <= ord(c) <= 0xFF5E for c in t) else 0.0
    emo = sum(1 for c in t if 0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF)
    return [aneh, kapital, angka, fullwidth, 1.0 if _TS.search(t) else 0.0, len(set(t)) / n,
            1.0 if _MEREK.search(t) else 0.0, len(_NAG_KW.findall(t.lower())), 1.0 if _ULANG.search(t) else 0.0, emo]


def nagalay_matrix(texts): return np.array([nagalay_feats(t) for t in texts], dtype=np.float32)


class Model:
    """Pembungkus: fitur + pengklasifikasi (+ SMOTE hanya pada data latih)."""
    def __init__(self, feats, clf, smote=False, seed=11):
        self.f, self.clf, self.smote, self.seed = feats, clf, smote, seed

    def fit(self, texts, y, Xcache=None):
        X = Xcache if Xcache is not None else self.f.fit(texts).transform(texts)
        y = np.asarray(y)
        if self.smote:
            X, y = SMOTE(random_state=self.seed).fit_resample(X, y)
        self.clf.fit(X, y); return self

    def predict(self, texts): return self.clf.predict(self.f.transform(texts))


def macro_f1(y, p): return f1_score(y, p, average='macro', zero_division=0)


def make_clf(kind, C=1, seed=11, balanced=False, trees=100, depth=None):
    cw = 'balanced' if balanced else None
    if kind == 'lr': return LogisticRegression(C=C, max_iter=2000, class_weight=cw, solver='liblinear', random_state=seed)
    if kind == 'svm': return LinearSVC(C=C, class_weight=cw, random_state=seed, max_iter=5000)
    if kind == 'rf': return RandomForestClassifier(n_estimators=trees, max_depth=depth, class_weight=cw, n_jobs=-1, random_state=seed)
    raise ValueError(kind)


def search(train, val, configs, seed, log=None, tag=''):
    """configs: iterable dict(feats=dict, clf=dict, smote=bool). Cache fitur per konfigurasi fitur."""
    best, cache = None, {}
    rows = []
    for c in configs:
        key = repr(sorted(c['feats'].items()))
        if key not in cache:
            f = Feats(**c['feats']).fit(train.text.tolist()); cache = {key: (f, f.transform(train.text.tolist()))} if len(cache) > 3 else {**cache, key: (f, f.transform(train.text.tolist()))}
        f, Xtr = cache[key]
        t0 = time.time()
        m = Model(f, make_clf(seed=seed, **c['clf']), smote=c.get('smote', False), seed=seed).fit(train.text.tolist(), train.label, Xcache=Xtr)
        s = macro_f1(val.label, m.predict(val.text.tolist()))
        rows.append(dict(tag=tag, **{f'f_{k}': v for k, v in c['feats'].items()}, **{f'c_{k}': v for k, v in c['clf'].items()}, smote=c.get('smote', False), val_macro_f1=round(s, 4), detik=round(time.time() - t0, 1)))
        if best is None or s > best[0]: best = (s, m, c)
    return best, rows


def grid_linear(feats_opts, clfs=('lr', 'svm'), balanced=True, smote=False):
    for fo, k, C in itertools.product(feats_opts, clfs, C_GRID):
        yield dict(feats=fo, clf=dict(kind=k, C=C, balanced=balanced), smote=smote)


# ---------------- peserta ----------------
class Keyword:
    """No. 1 (Edvra dkk., 2026): pencocokan kata kunci kondisi As-Is pada teks mentah huruf kecil, tanpa pelatihan."""
    def predict(self, texts): return np.array([int(bool(KW.search(t.lower()) or SITE.search(t.lower()))) for t in texts])


def fit_participant(pid, train, val, seed):
    """Kembalikan (model, info). pid: 'P1'..'P3','P8','V1'..'V7'."""
    ch, wd = (3, 5), (1, 3)
    if pid == 'P1': return Keyword(), dict(config='daftar istilah dan pola nama situs', val_macro_f1=macro_f1(val.label, Keyword().predict(val.text.tolist())))
    if pid == 'P2':   # Nagalay dkk. (2025): Random Forest, fitur hibrida (TF-IDF kata + fitur anomali), SMOTE pada data latih
        # 23 Nov 2026: fitur anomali Nagalay (Tabel 2 makalah) menggantikan fitur statistik Unicode penulis
        cfg = [dict(feats=dict(norm='none', analyzer='word', ngram=(1, 1), anomali=True, max_features=5000), clf=dict(kind='rf', trees=t, depth=d), smote=True)
               for t, d in itertools.product((100, 300), (None, 30))]
        cat = 'Random Forest + SMOTE'
    elif pid == 'P3':  # Alyudani dkk. (2026): LR, TF-IDF character n-gram + fitur kata kunci, SMOTE; C=2 (makalah) ikut sebagai kandidat
        # 23 Nov 2026: kata kunci berbobot Alyudani (Tabel 6), TF-IDF min_df 2 / max_df 0,95 (Tabel 7), class_weight balanced + SMOTE
        cfg = [dict(feats=dict(norm='none', analyzer='char', ngram=ng, kw='alyudani', min_df=2, max_df=0.95), clf=dict(kind='lr', C=C, balanced=True), smote=True)
               for ng, C in itertools.product([ch, wd], C_GRID + [2])]
        cat = 'Logistic Regression + SMOTE'
    elif pid == 'P8' or pid == 'V6':   # Feature Fusion = V6: NFKC + TF-IDF char + fitur statistik Unicode
        cfg = list(grid_linear([dict(norm='nfkc', analyzer='char', ngram=n, stats=True) for n in (wd, ch)]))
        if pid == 'P8':
            cfg += [dict(feats=dict(norm='nfkc', analyzer='char', ngram=n, stats=True, max_features=20000), clf=dict(kind='rf', trees=t, depth=d, balanced=True))
                    for n, t, d in itertools.product((wd, ch), (100, 300), (None, 30))]
        cat = 'Feature Fusion'
    else:
        spec = {'V1': dict(norm='none', analyzer='word', ngram=(1, 1), stats=False), 'V2': dict(norm='none', analyzer='char', ngram=ch, stats=False),
                'V3': dict(norm='nfkc', analyzer='char', ngram=ch, stats=False), 'V4': dict(norm='none', text=False, stats=True),
                'V5': dict(norm='none', analyzer='char', ngram=ch, stats=True), 'V7': dict(norm='nfkc_skel', analyzer='char', ngram=ch, stats=True)}[pid]
        opts = [spec] if pid in ('V1', 'V4') else [dict(spec, ngram=n) for n in (wd, ch)]
        if pid == 'V1': opts = [dict(spec, ngram=n) for n in ((1, 1), (1, 2))]
        cfg = list(grid_linear(opts)); cat = f'Ablasi {pid}'
    (s, m, c), rows = search(train, val, cfg, seed, tag=pid)
    return m, dict(config=str({**c['feats'], **c['clf'], 'smote': c.get('smote', False)}), val_macro_f1=round(s, 4), kategori=cat, search=rows)
