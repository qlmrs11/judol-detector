# -*- coding: utf-8 -*-
"""Latih ulang model aplikasi memakai kode eksperimen asli (seed 11), lalu periksa bahwa prediksinya
identik dengan prediksi yang dilaporkan pada Bab IV.

Model yang dibuat:
- P9  Feature Fusion iterasi 3 (artefak): konfigurasi terpilih dari iterasi3_konfigurasi_terpilih.json
- P3  Logistic Regression char n-gram (pembanding terkuat pada S0)
P1 (pencocokan kata kunci) tidak perlu dilatih; aplikasi memakai kelas Keyword langsung.

Skrip ini dijalankan SEKALI di laptop, tidak di server. Data dan kode eksperimen tidak ikut ke repo.
Pemakaian (dari folder APLIKASI):
    python tools/latih_model.py [--eks ../05_EKSPERIMEN/eksperimen2] [--seed 11]
"""
import argparse, json, os, pickle, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--eks', default=os.path.join(REPO, '..', '05_EKSPERIMEN', 'eksperimen2'))
ap.add_argument('--seed', type=int, default=11)
a = ap.parse_args()
EKS = os.path.abspath(a.eks)
sys.path.insert(0, EKS)   # paket 'src' di sini = kode eksperimen asli

import joblib, numpy as np, pandas as pd, sklearn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from src.prep import CFG, ROOT  # noqa: E402
from src import iterasi3, models_classic as mc  # noqa: E402

P = lambda *x: os.path.join(ROOT, *x)
df = pd.read_pickle(P(CFG['paths']['processed'], 'data.pkl'))
S = pickle.load(open(P(CFG['paths']['processed'], 'scenarios.pkl'), 'rb'))
train, val = df[df.split == 'train'], df[df.split == 'val']
cfg = json.load(open(P(CFG['paths']['tables'], 'iterasi3_konfigurasi_terpilih.json'), encoding='utf-8'))
acuan_f1 = pd.read_csv(P(CFG['paths']['tables'], f'hasil_metrik_seed{a.seed}.csv'))


def periksa(pid, model):
    """Bandingkan prediksi model baru dengan prediksi tersimpan dari eksperimen (results/preds/seed11)."""
    lama = pickle.load(open(P(CFG['paths']['preds'], f'seed{a.seed}', f'{pid}.pkl'), 'rb'))['pred']
    rows = []
    for k, v in S.items():
        p = model.predict(v.text.tolist())
        f1 = f1_score(v.label.values, p, average='macro', zero_division=0)
        ac = acuan_f1[(acuan_f1.peserta == pid) & (acuan_f1.skenario == k)].macro_f1
        rows.append(dict(skenario=k, identik=bool(np.array_equal(p, lama[k])), macro_f1=round(f1, 4),
                         macro_f1_bab4=round(float(ac.iloc[0]), 4) if len(ac) else None))
    r = pd.DataFrame(rows); print(r.to_string(index=False)); return r


hasil = {}
for pid in ('P9', 'P3'):
    t0 = time.time()
    if pid == 'P9':
        model = iterasi3.P9(cfg, a.seed).fit(df).m
        konfig = cfg
    else:
        model, info = mc.fit_participant('P3', train, val, a.seed)
        konfig = info['config']
    dt = round(time.time() - t0, 1)
    print(f'\n[{pid}] dilatih dalam {dt} detik')
    r = periksa(pid, model)
    out = os.path.join(REPO, 'models', f'{pid.lower()}.joblib')
    joblib.dump(model, out, compress=3)
    hasil[pid] = dict(konfigurasi=konfig, seed=a.seed, detik_latih=dt, ukuran_mb=round(os.path.getsize(out) / 2**20, 1),
                      semua_prediksi_identik=bool(r.identik.all()), per_skenario=r.to_dict('records'))
    print(f'[{pid}] disimpan: {out} ({hasil[pid]["ukuran_mb"]} MB) | identik dengan Bab IV: {hasil[pid]["semua_prediksi_identik"]}')

hasil['_versi'] = dict(python=sys.version.split()[0], scikit_learn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__)
json.dump(hasil, open(os.path.join(REPO, 'models', 'info_model.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str)
print('\nRingkasan disimpan: models/info_model.json')
