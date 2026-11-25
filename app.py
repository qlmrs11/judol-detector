"""Demo artefak TA: deteksi komentar judi online (judol) YouTube yang disamarkan dengan karakter Unicode.

Model: P9 Feature Fusion iterasi 3 (artefak), P3 LR char n-gram dan P1 kata kunci (pembanding).
- models/*.joblib dilatih ulang dengan kode eksperimen asli (seed 11); prediksinya identik dengan
  prediksi seed 11 yang dilaporkan (models/info_model.json, tools/latih_model.py).
- hasil/*.csv adalah tabel hasil eksperimen (rata-rata seed 11, 911, 99).
Jalankan: streamlit run app.py
"""
import html
import os
import random
import sys

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import textnorm as tn  # noqa: E402

tn.CONF_PATH = os.path.join(HERE, "data", "confusables.txt")  # tabel UTS #39 di dalam repo
from src import convert as cv  # noqa: E402
from src import models_classic as mc  # noqa: E402
from src import unicode_family as uf  # noqa: E402

MERAH, TINTA, ABU = "#b3121b", "#1c1b1a", "#8a8782"
MODEL = {"P1": ("P1", "Kata kunci", "kondisi sekarang"),
         "P3": ("P3", "LR char n-gram", "pembanding"),
         "P9": ("P9", "Feature Fusion", "artefak")}
NAMA_KELUARGA = {"F1": "huruf matematis", "F2": "huruf lebar penuh", "F3": "Cyrillic mirip Latin",
                 "F4": "huruf berbingkai", "F5": "karakter tak terlihat", "F6": "superscript/subscript",
                 "F7": "tanda diakritik", "F8": "Yunani mirip Latin", "F9": "operator matematika"}
SKEN = {"S0": "S0", "S1": "S1", "S2": "S2", "S3_1": "S3 1%", "S3_3": "S3 3%", "S3_5": "S3 5%", "S4": "S4"}
STRES = ["S1", "S2", "S3_1", "S3_3", "S3_5", "S4"]
CONTOH = {   # 1-5: dari huruf biasa sampai paling aneh; nama situs fiktif, tidak ada di data latih
    "1 · Huruf biasa": "bosen kerja terus? coba main di rajacuan88, tiap hari ada yg jp",
    "2 · Leetspeak": "pagi² udah wd 3jt dr r4j4cu4n88, modal 20rb doang gan",
    "3 · Lebar penuh": "yg blm kenal ｒａｊａｃｕａｎ８８ rugi bgt sumpah, depo dikit bisa maxwin",
    "4 · Tebal + superscript": "𝐫𝐚𝐣𝐚𝐜𝐮𝐚𝐧𝟖𝟖 lagi ᵍᵃᶜᵒʳ ᵖᵃʳᵃʰ malam ini, buruan cek bio",
    "5 · Campuran": "ⓡа​jа​сᴜаⓝ⑧⑧ ʙᴀɴᴊɪʀ ᴊᴘ ᴛɪᴀᴘ ʜᴀʀɪ 🔥🔥",
    "Normal ber-font": "makasih bang, 𝓿𝓲𝓭𝓮𝓸𝓷𝔂𝓪 bermanfaat banget 🙏",
    "Cyrillic (kelemahan P9)": "роlа gасоr hаrі іnі сеk dі dеwа4d bоѕ",
}
koma = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")

st.set_page_config(page_title="Deteksi Komentar Judol Tersamar", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(f"""<style>
.block-container {{max-width: 1160px; padding-top: 0 !important; padding-bottom: 4rem;}}
[data-testid="stHeader"] {{background: transparent;}}
[data-testid="stHeader"] * {{color: #fff;}}
/* pita judul merah selebar layar */
.kepala {{background: {MERAH}; color: #fff; margin: -1.1rem calc(50% - 50vw) 1.6rem; padding: 3.6rem calc(50vw - 50%) 2rem;}}
.kepala h1 {{color: #fff !important; margin: 0 0 .5rem; padding: 0 !important;}}
.kepala p {{color: #fbe3e4; font-size: 1.08rem; max-width: 66ch; margin: 0;}}
.kepala ::selection {{background: #fff; color: {MERAH};}}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{display: none;}}
h1 {{font-size: 2.35rem !important; letter-spacing: -0.02em; line-height: 1.15 !important; padding-bottom: .2rem !important;}}
h2 {{font-size: 1.45rem !important; letter-spacing: -0.01em; margin-top: 1.6rem !important;}}
::selection {{background: #f6d4d6; color: {TINTA};}}
/* navigasi: tab bergaris bawah */
.st-key-nav [data-testid="stButtonGroup"] {{gap: 0; border-bottom: 1px solid #dcdad6; width: 100%;}}
.st-key-nav button {{border: 0 !important; border-radius: 0 !important; background: transparent !important; box-shadow: none !important;
  padding: .55rem 1.2rem .7rem 0 !important; margin-right: 1.4rem; color: #6b6864 !important; border-bottom: 3px solid transparent !important; margin-bottom: -1px;}}
.st-key-nav button p {{font-size: 1.05rem !important;}}
.st-key-nav button[aria-checked="true"] {{color: {TINTA} !important; border-bottom-color: {MERAH} !important;}}
.st-key-nav button[aria-checked="true"] p {{font-weight: 700;}}
.st-key-nav button:hover {{color: {TINTA} !important;}}
.st-key-nav button:focus-visible {{outline: 2px solid {MERAH}; outline-offset: 2px;}}
/* tombol contoh */
.st-key-kotak_contoh button {{background: #fff !important; border-color: #dcdad6 !important;}}
.st-key-kotak_contoh button[aria-checked="true"] {{background: {TINTA} !important; border-color: {TINTA} !important;}}
.st-key-kotak_contoh button[aria-checked="true"] p {{color: #fff !important;}}
/* kotak komentar */
.stTextArea [data-baseweb="textarea"] {{background: #fff !important; border: 1px solid #cfccc7 !important;}}
.stTextArea textarea {{background: #fff !important; font-size: 1.3rem !important; line-height: 1.5 !important; color: {TINTA} !important;}}
.stTextArea textarea::placeholder {{color: #a3a09b !important; opacity: 1;}}
.stTextArea [data-baseweb="textarea"]:focus-within {{border-color: {MERAH} !important;}}
table.uji tr.grup td {{padding: 1.1rem 0 .35rem; font-size: .8rem; font-weight: 700; color: #6b6864; border-bottom: 1px solid #cfccc7 !important;}}
table.uji tr.beda td {{background: #fbf1f1 !important;}}
.peringatan {{background: #fbf1f1; padding: .7rem .9rem; border-radius: 4px; color: #6e1a1f; max-width: 80ch; font-size: .95rem;}}
.lead {{color: #4a4845; font-size: 1.05rem; max-width: 68ch; margin: 0 0 1.4rem;}}
.catatan {{color: #5d5a56; font-size: .88rem; max-width: 80ch; line-height: 1.45;}}

/* pita karakter: tiap karakter seperti sel pada tabel kode Unicode */
.pita {{display: flex; flex-wrap: wrap; gap: 4px 3px; margin: .6rem 0 .4rem;}}
.sel {{display: flex; flex-direction: column; align-items: center; min-width: 2.35rem; padding: .35rem .2rem .3rem;
       border: 1px solid #dcdad6; border-radius: 4px; background: #fff;}}
.sel b {{font-weight: 400; font-size: 1.55rem; line-height: 1.15; font-family: "Atkinson Hyperlegible", sans-serif;}}
.sel i {{font-style: normal; font-family: "JetBrains Mono", monospace; font-size: .58rem; color: {ABU}; letter-spacing: .01em; margin-top: .15rem;}}
.sel em {{font-style: normal; font-family: "JetBrains Mono", monospace; font-size: .62rem; font-weight: 600; color: transparent; margin-top: .05rem;}}
.sel.tersamar {{background: #fbeced; border-color: #e9b3b6;}}
.sel.tersamar i {{color: #8f2a2f;}}
.sel.tersamar em {{color: {MERAH};}}
.jeda {{width: .9rem;}}
.dibaca {{font-family: "JetBrains Mono", monospace; font-size: .95rem; color: {TINTA};}}
.dibaca span {{color: {ABU}; font-family: "Atkinson Hyperlegible", sans-serif; font-size: .85rem; margin-right: .6rem;}}

/* putusan tiga model: blok berwarna penuh */
.putusan {{display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin: 1.2rem 0 .6rem;}}
.putusan > div {{padding: 1rem 1.15rem 1.1rem; border-radius: 6px; background: #eeedea; color: {TINTA};}}
.putusan > div.judol {{background: {MERAH}; color: #fff;}}
.putusan .nama {{font-size: .95rem; opacity: .85;}}
.putusan .vonis {{font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1; margin: .4rem 0 .3rem;}}
.putusan .skor {{font-family: "JetBrains Mono", monospace; font-size: .8rem; opacity: .8;}}
.putusan > div.p9 {{outline: 2px solid {TINTA}; outline-offset: 3px;}}

/* komentar bergaya kolom komentar video */
.yt {{display: flex; gap: .9rem; padding: 1.1rem 1.2rem; border: 1px solid #e3e1dd; border-radius: 10px; background: #fff; margin: .4rem 0 1.2rem; max-width: 820px;}}
.yt-av {{flex: none; width: 2.6rem; height: 2.6rem; border-radius: 50%; display: grid; place-items: center; font-weight: 700; color: #fff; font-size: 1.05rem;}}
.yt-kepala {{font-size: .9rem; color: #4a4845;}}
.yt-kepala b {{color: {TINTA}; margin-right: .45rem;}}
.yt-teks {{font-size: 1.35rem; line-height: 1.45; margin: .25rem 0 .55rem; color: {TINTA}; word-break: break-word;}}
.yt-aksi {{display: flex; align-items: center; gap: .45rem; color: #4a4845; font-size: .85rem;}}
.yt-aksi svg {{display: block;}}
.yt-aksi .balas {{margin-left: .9rem; font-weight: 700;}}

/* tabel uji penyamaran dan target */
table.uji {{width: 100%; border-collapse: collapse; border: 0 !important; font-size: .95rem; margin-top: .6rem;}}
table.uji th, table.uji td {{border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; background: transparent !important;}}
table.uji th {{text-align: left; font-weight: 700; font-size: .82rem; color: #4a4845; border-bottom: 2px solid {TINTA} !important; padding: .45rem .6rem .45rem 0;}}
table.uji td {{border-bottom: 1px solid #e7e5e1 !important; padding: .5rem .6rem .5rem 0; vertical-align: top;}}
table.uji td.teks {{font-family: "Atkinson Hyperlegible", sans-serif; font-size: 1.05rem;}}
table.uji td.hasil {{white-space: nowrap; font-weight: 700;}}
table.uji td.gagal {{color: {MERAH};}}
table.uji td.ok {{color: {TINTA}; font-weight: 400;}}
table.uji td.angka {{font-family: "JetBrains Mono", monospace; font-size: .88rem; white-space: nowrap;}}
table.uji tr.asli td {{color: {ABU};}}
.rekap {{display: flex; gap: 2.5rem; flex-wrap: wrap; margin: 1rem 0 .4rem; font-size: 1.05rem;}}
.rekap b {{font-size: 2rem; letter-spacing: -0.02em; display: block; line-height: 1.1;}}
.rekap .p9 b {{color: {MERAH};}}

@media (max-width: 720px) {{
  .kepala {{padding-top: 3.2rem;}}
  .st-key-nav button {{margin-right: .5rem; padding-right: .6rem !important;}}
  .yt-teks {{font-size: 1.15rem;}}
  .putusan {{grid-template-columns: 1fr;}}
}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model dan data
@st.cache_resource
def muat_model():
    return {"P1": mc.Keyword(), "P3": joblib.load(os.path.join(HERE, "models", "p3.joblib")),
            "P9": joblib.load(os.path.join(HERE, "models", "p9.joblib"))}


@st.cache_data
def muat_hasil():
    baca = lambda n: pd.read_csv(os.path.join(HERE, "hasil", n))
    return baca("hasil_metrik_final.csv"), baca("status_target_iterasi3_final.csv"), baca("status_target_final.csv")


MODELS = muat_model()


def prediksi(pid, texts):
    """(label, skor). Skor = fungsi keputusan model linear, > 0 berarti judol; P1 tidak punya skor."""
    texts = [uf.clean_comment(str(t)) for t in texts]   # pembersihan yang sama dengan data latih
    m = MODELS[pid]
    if pid == "P1":
        p = m.predict(texts)
        return p, None
    s = m.clf.decision_function(m.f.transform(texts))
    return (s > 0).astype(int), s


def keluarga_char(ch):
    fams, zw, _ = uf.classify_characters(ch)
    return fams[0] if fams else ("F5" if zw else None)


def keluarga_teks(t):
    fams, zw, _ = uf.classify_characters(t)
    return uf.assign_family_label(fams, zw)


def pita(text, batas=90):
    sel = []
    for ch in text[:batas]:
        if ch == " ":
            sel.append('<span class="jeda"></span>')
            continue
        f = keluarga_char(ch)
        glyph = "·" if f == "F5" else html.escape(ch)
        sel.append(f'<span class="sel{" tersamar" if f else ""}" title="{NAMA_KELUARGA.get(f, "")}">'
                   f'<b>{glyph}</b><i>{ord(ch):04X}</i><em>{f or "–"}</em></span>')
    lebih = f'<p class="catatan">… {len(text) - batas} karakter berikutnya tidak ditampilkan.</p>' if len(text) > batas else ""
    return f'<div class="pita">{"".join(sel)}</div>{lebih}'


SUKA = ('<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" '
        'stroke-linejoin="round" aria-hidden="true"><path d="M7 10v10H4V10h3Zm0 0 4-7c1.4 0 2.4 1 2.1 2.5L12.6 9H19a2 2 0 0 1 '
        '2 2.3l-1.2 6.9A2 2 0 0 1 17.8 20H7"/></svg>')
TIDAK_SUKA = SUKA.replace('aria-hidden="true">', 'aria-hidden="true" style="transform:rotate(180deg)">')
AKUN = {"1 · Huruf biasa": ("@rezeki.harian", "#5b6b7a", "3 jam yang lalu", 12),
        "2 · Leetspeak": ("@gan_sultan", "#7a5b3e", "5 jam yang lalu", 4),
        "3 · Lebar penuh": ("@info.cuan", "#3e6b55", "1 hari yang lalu", 21),
        "4 · Tebal + superscript": ("@malam.hoki", "#4a4f7a", "2 jam yang lalu", 9),
        "5 · Campuran": ("@banjir.jp", "#6b3e5e", "40 menit yang lalu", 37),
        "Normal ber-font": ("@dina.nonton", "#3e5f6b", "6 jam yang lalu", 58),
        "Cyrillic (kelemahan P9)": ("@pola.harian", "#6b5b3e", "1 jam yang lalu", 6)}


def kartu_komentar(text, akun="@kamu", warna="#5d5a56", waktu="baru saja", suka=0):
    return (f'<div class="yt"><div class="yt-av" style="background:{warna}">{html.escape(akun[1:2].upper())}</div><div>'
            f'<div class="yt-kepala"><b>{html.escape(akun)}</b>{waktu}</div><div class="yt-teks">{html.escape(text)}</div>'
            f'<div class="yt-aksi">{SUKA}<span>{suka or ""}</span>{TIDAK_SUKA}<span class="balas">Balas</span></div></div></div>')


def analisis(teks, akun=None):
    bersih = uf.clean_comment(teks)
    st.markdown(kartu_komentar(bersih, *(akun or ())), unsafe_allow_html=True)
    blok = []
    for pid in MODEL:
        p, s = prediksi(pid, [bersih])
        judol = p[0] == 1
        kode, nama, peran = MODEL[pid]
        skor = "aturan, tanpa skor" if s is None else f"skor {koma(s[0], 2)}"
        kelas = ("judol" if judol else "normal") + (" p9" if pid == "P9" else "")
        blok.append(f'<div class="{kelas}"><div class="nama"><b>{kode} {nama}</b><br>{peran}</div>'
                    f'<div class="vonis">{"Judol" if judol else "Normal"}</div><div class="skor">{skor}</div></div>')
    st.markdown(f'<div class="putusan">{"".join(blok)}</div>', unsafe_allow_html=True)
    st.markdown('<p class="catatan">Skor adalah nilai fungsi keputusan model linear: di atas 0 ditandai judol. '
                'Skor bukan peluang. Model hanya memberi saran; keputusan tetap pada moderator.</p>', unsafe_allow_html=True)

    n_tersamar = sum(1 for ch in bersih if keluarga_char(ch))
    kel = keluarga_teks(bersih)
    ket = ("tidak ada karakter tersamar" if kel == "NORMAL" else
           f"{n_tersamar} karakter tersamar, keluarga {kel}" + (f" ({NAMA_KELUARGA[kel]})" if kel in NAMA_KELUARGA else ""))
    st.markdown(f"## Karakter yang ditulis  <span style='font-size:1rem;font-weight:400;color:#5d5a56'>· {ket}</span>",
                unsafe_allow_html=True)
    st.markdown(pita(bersih), unsafe_allow_html=True)
    st.markdown(f'<p class="dibaca"><span>dibaca P9 setelah NFKC</span>{html.escape(tn.nfkc(bersih))}</p>',
                unsafe_allow_html=True)
    with st.expander("20 kolom fitur Unicode yang dihitung P9 dari teks mentah"):
        f = pd.DataFrame({"fitur": tn.STAT_COLS, "nilai": np.round(tn.stat_features(bersih), 3)})
        st.dataframe(f, hide_index=True, width="stretch", height=420)


# ------------------------------------------------------------------ kepala
st.markdown('<div class="kepala"><h1>Deteksi komentar judol yang disamarkan</h1><p>Komentar promosi judi online di YouTube '
            'sering menulis nama situsnya dengan huruf Unicode yang mirip huruf biasa. Aplikasi ini membandingkan tiga model '
            'pada komentar yang sama.</p></div>', unsafe_allow_html=True)
with st.container(key="nav"):
    bagian = st.segmented_control("Bagian", ["Contoh", "Coba sendiri", "Uji penyamaran", "Ringkasan hasil"], default="Contoh",
                                  label_visibility="collapsed")
bagian = bagian or "Contoh"

# ================================================================== 1a. contoh
if bagian == "Contoh":
    with st.container(key="kotak_contoh"):
        pilih = st.pills("Contoh 1–5 berurutan dari huruf biasa sampai paling aneh. Nama situs rajacuan88 fiktif dan tidak ada di data latih.",
                         list(CONTOH), default="5 · Campuran", key="contoh")
    pilih = pilih or "5 · Campuran"
    analisis(CONTOH[pilih], AKUN[pilih])

# ================================================================== 1b. coba sendiri
elif bagian == "Coba sendiri":
    txt = st.text_area("Tulis atau tempel satu komentar", key="teks", height=100,
                       placeholder="misalnya: main di rajacuan88 pasti jp")
    if txt.strip():
        analisis(txt)
    else:
        st.markdown('<p class="catatan">Hasil ketiga model muncul di sini setelah komentar ditulis. '
                    'Tekan Ctrl+Enter untuk memeriksa.</p>', unsafe_allow_html=True)

# ================================================================== 2. uji penyamaran
elif bagian == "Uji penyamaran":
    st.markdown('<p class="catatan" style="font-size:1rem">Satu komentar berhuruf biasa diubah ke setiap keluarga penyamaran '
                'dengan fungsi yang sama dengan pembentuk skenario uji, lalu diperiksa ketiga model.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1.3])
    sumber = c1.text_area("Komentar sumber (huruf biasa)", "yang mau cuan tiap hari langsung aja ke alexis17", height=90)
    jenis = c2.radio("Komentar ini sebenarnya", ["judol", "normal"])
    seed = 11
    rng = random.Random(seed)
    ubah = ((lambda t, f, p: cv.disguise_targets(t, f, p, rng)) if jenis == "judol"
            else (lambda t, f, p: cv.disguise_salient(t, f, p, rng)))
    kosong = (not cv.target_spans(sumber)) if jenis == "judol" else (cv.salient_token(sumber) is None)
    if kosong:
        st.markdown('<p class="peringatan">' + ("Komentar ini tidak menyebut nama situs atau istilah judol, jadi tidak ada bagian yang "
                    "disamarkan. Coba komentar promosi yang menyebut nama situs, misalnya <i>main di rajacuan88 pasti jp</i>."
                    if jenis == "judol" else "Komentar ini tidak punya kata berhuruf Latin minimal 4 huruf untuk disamarkan.")
                    + '</p>', unsafe_allow_html=True)
    baris = [("asli", "", sumber)]
    for fam in ["F1", "F2"]:
        baris.append((f"{fam} berat", "Keluarga yang dikenal saat latih", ubah(sumber, fam, 1.0)))
    for fam in ["F3", "F4", "F6", "F9", "F10"]:
        for nama, p in (("ringan", 0.5), ("berat", 1.0)):
            baris.append((f"{fam} {nama}", "S1 · keluarga yang belum pernah dilihat model", ubah(sumber, fam, p)))
    if jenis == "judol":
        polos = cv.restore_plain(sumber)
        baris.append(("leetspeak", "S2 · penyerang meninggalkan font", cv.leet_targets(polos, rng)))
    else:
        baris += [(f"seluruh {f}", "S3 · komentar normal ber-font", cv.convert(sumber, f, 1.0, rng)) for f in ["F1", "F2"]]

    teks = [b[2] for b in baris]
    hasil = {pid: prediksi(pid, teks)[0] for pid in MODEL}
    benar = 1 if jenis == "judol" else 0
    varian = range(1, len(baris))
    tangkap = {pid: int(sum(hasil[pid][i] == benar for i in varian)) for pid in MODEL}
    kata = "tertangkap" if jenis == "judol" else "benar (normal)"
    rekap = "".join(f'<div class="{"p9" if pid == "P9" else ""}"><b>{tangkap[pid]} / {len(varian)}</b>'
                    f'{MODEL[pid][0]} {MODEL[pid][1]}</div>' for pid in MODEL)
    st.markdown(f'<div class="rekap">{rekap}</div><p class="catatan">Jumlah versi tersamar yang {kata}.</p>',
                unsafe_allow_html=True)

    def sel(pid, i):
        ok = hasil[pid][i] == benar
        if jenis == "judol":
            return f'<td class="hasil {"ok" if ok else "gagal"}">{"tertangkap" if ok else "lolos"}</td>'
        return f'<td class="hasil {"ok" if ok else "gagal"}">{"normal" if ok else "salah tanda"}</td>'

    rows = []
    grup_lalu = None
    for i, (nama, ket, t) in enumerate(baris):
        if ket and ket != grup_lalu:
            rows.append(f'<tr class="grup"><td colspan="5">{ket}</td></tr>'); grup_lalu = ket
        beda = int(hasil["P3"][i]) != int(hasil["P9"][i])   # sorot hanya bila P3 dan P9 berbeda
        kelas = "asli" if i == 0 else ("beda" if beda else "")
        rows.append(f'<tr class="{kelas}"><td><b>{nama}</b></td>'
                    f'<td class="teks">{html.escape(t)}</td>{sel("P1", i)}{sel("P3", i)}{sel("P9", i)}</tr>')
    st.markdown('<table class="uji"><thead><tr><th>Penyamaran</th><th>Teks yang diperiksa</th><th>P1</th><th>P3</th><th>P9</th>'
                f'</tr></thead><tbody>{"".join(rows)}</tbody></table>', unsafe_allow_html=True)
    st.markdown('<p class="catatan" style="margin-top:.8rem">Ringan: sekitar separuh huruf sasaran diganti. Berat: seluruh huruf '
                'sasaran diganti. Baris berwarna: P3 dan P9 berbeda putusan. Penyamaran acak dengan seed 11, sama seperti eksperimen.</p>', unsafe_allow_html=True)

# ================================================================== 3. ringkasan hasil
else:
    M, T3, T2 = muat_hasil()
    P = M[M.peserta.str.match(r"^P\d+$")].copy()
    P["Skenario"] = P.skenario.map(SKEN)
    s0 = P[P.skenario == "S0"].set_index("peserta").macro_f1
    stres = P[P.skenario.isin(STRES)]
    idx = stres.groupby("peserta").macro_f1.idxmin()
    buruk = stres.loc[idx].set_index("peserta")[["macro_f1", "Skenario"]]
    nama = P.drop_duplicates("peserta").set_index("peserta").nama
    d = pd.DataFrame({"S0": s0, "Terburuk": buruk.macro_f1, "di": buruk.Skenario, "nama": nama}).reset_index(names="peserta")
    d["label"] = d.peserta + " " + d.nama
    d["artefak"] = d.peserta == "P9"
    urut = d.sort_values("Terburuk", ascending=False).label.tolist()

    st.markdown("## Akurasi pada data biasa belum tentu tahan penyamaran")
    st.markdown('<p class="catatan" style="font-size:1rem">Setiap baris: Macro-F1 pada data uji biasa (S0, titik kosong) dan '
                'pada skenario terburuk S1–S4 (titik penuh). Garis yang panjang berarti model jatuh jauh saat komentar disamarkan.</p>',
                unsafe_allow_html=True)
    warna = alt.condition("datum.artefak", alt.value(MERAH), alt.value(TINTA))
    y = alt.Y("label:N", sort=urut, title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13, labelPadding=8, ticks=False, domain=False))
    x = alt.X("Terburuk:Q", scale=alt.Scale(domain=[0.45, 1.0], nice=False), title="Macro-F1",
              axis=alt.Axis(format=".2f", grid=True, gridColor="#ecebe8", labelFontSize=12, titleFontSize=12))
    tip = ["label", alt.Tooltip("S0:Q", format=".3f"), alt.Tooltip("Terburuk:Q", format=".3f"), alt.Tooltip("di:N", title="skenario terburuk")]
    base = alt.Chart(d).encode(y=y, tooltip=tip)
    garis = base.mark_rule(strokeWidth=2.5).encode(x=x, x2="S0:Q", color=warna)
    titik_s0 = base.mark_point(size=110, filled=False, strokeWidth=2, opacity=1).encode(x="S0:Q", color=warna)
    titik_b = base.mark_circle(size=130, opacity=1).encode(x="Terburuk:Q", color=warna)
    tulis = base.mark_text(align="right", dx=-10, fontSize=11, color="#5d5a56").encode(x="Terburuk:Q", text="di:N")
    st.altair_chart((garis + titik_s0 + titik_b + tulis).properties(height=430).configure_view(stroke=None)
                    .configure_axis(labelFont="Atkinson Hyperlegible", titleFont="Atkinson Hyperlegible", labelColor=TINTA),
                    width="stretch")

    st.markdown("## Target P9 pada iterasi 3")
    t1 = T3[T3.target == "T1"].iloc[0]
    t2 = T3[T3.ukuran == "Macro-F1 terburuk pada S1-S4"].iloc[0]
    t3 = T3[T3.ukuran == "FPR kelas normal"]
    t3r = T3[T3.ukuran == "Ringkasan T3"].iloc[0]
    t4 = T3[T3.target == "T4"].iloc[0]
    persen = lambda v: f"{100 * v:.2f}%".replace(".", ",")
    target = [
        ("T1", "Macro-F1 data biasa (S0)", koma(t1.feature_fusion), f"{t1.pembanding} {koma(t1.nilai_pembanding)}",
         "tertinggal paling banyak 2 poin", t1.status),
        ("T2", "Macro-F1 skenario terburuk", f"{koma(t2.feature_fusion)}<br><span class='catatan'>IK 95% {koma(t2.ff_lo)}–{koma(t2.ff_hi)}</span>",
         f"{t2.pembanding} {koma(t2.nilai_pembanding)}<br><span class='catatan'>IK 95% {koma(t2.pembanding_lo)}–{koma(t2.pembanding_hi)}</span>",
         "lebih tinggi bermakna, selisih ≥ 2 poin", t2.status),
        ("T3", "Salah tanda komentar normal (S0, S3)", f"paling tinggi {persen(t3.feature_fusion.max())}", "–", "paling besar 5%", t3r.status),
        ("T4", "Judol polos/leetspeak tertangkap (S2)", koma(t4.feature_fusion), f"{t4.pembanding} {koma(t4.nilai_pembanding)}",
         "tertinggal paling banyak 2 poin", t4.status),
    ]
    rows = "".join(f'<tr><td><b>{a}</b></td><td>{b}</td><td class="angka">{c}</td><td class="angka">{e}</td><td>{f}</td>'
                   f'<td class="hasil {"ok" if g == "Terpenuhi" else "gagal"}">{g.lower()}</td></tr>'
                   for a, b, c, e, f, g in target)
    st.markdown('<table class="uji"><thead><tr><th></th><th>Ukuran</th><th>P9</th><th>Pembanding</th><th>Syarat</th><th>Status</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    ok2 = int((T2[T2.target.isin(["T1", "T4"]) | T2.ukuran.isin(["Macro-F1 terburuk pada S1-S4", "Ringkasan T3"])].status == "Terpenuhi").sum())
    st.markdown(f'<p class="catatan" style="margin-top:.9rem">Pada iterasi 2 (P8, tanpa augmentasi) hanya {ok2} dari 4 target terpenuhi. '
                'Angka adalah rata-rata seed 11, 911, dan 99 dengan interval kepercayaan bootstrap 1.000 kali. Model di aplikasi '
                'memakai seed 11; prediksinya identik dengan prediksi seed 11 yang dilaporkan.</p>', unsafe_allow_html=True)
