from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.tree import plot_tree

st.set_page_config(page_title="Fuzzy Pendidikan Jatim", layout="wide")

FOLDER = Path(__file__).parent / "model"
FILE_MODEL = FOLDER / "model_pca_fcm_dt.joblib"
FILE_HASIL = FOLDER / "hasil_fcm_dt.csv"

VARIABEL = pd.DataFrame([
    ("RLS", "Rata-rata lama sekolah penduduk usia 25+", "tahun", 0.0, 16.0),
    ("PTT_SD", "% penduduk 10+ berpendidikan tertinggi SD/sederajat", "%", 0.0, 100.0),
    ("PTT_SMP", "% penduduk 10+ berpendidikan tertinggi SMP/sederajat", "%", 0.0, 100.0),
    ("PTT_SMA", "% penduduk 10+ berpendidikan tertinggi SMA/sederajat", "%", 0.0, 100.0),
    ("APM_SD", "Angka partisipasi murni SD (usia 7-12)", "%", 0.0, 100.0),
    ("APM_SMP", "Angka partisipasi murni SMP (usia 13-15)", "%", 0.0, 100.0),
    ("APM_SMA", "Angka partisipasi murni SMA (usia 16-18)", "%", 0.0, 100.0),
    ("AMH", "Angka melek huruf penduduk usia 10+", "%", 0.0, 100.0),
], columns=["Variabel", "Keterangan", "Satuan", "Min", "Maks"])
BATAS_TRANSISI = 0.6

# ------------------------------------------------------------------ tema Nusantara
SERIF = 'Georgia, "Times New Roman", serif'
SANS = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
MONO = 'ui-monospace, "SFMono-Regular", Consolas, monospace'
T = dict(pri="#A4471F", on="#FBF6EE", bg="#FBF6EE", bg2="#F1E6D6", line="#E2D2BB",
         tx="#3B2A1E", tx2="#7A5A3A", c=["#8C2F1B", "#D69E2E", "#4E7D5B"], hf=SERIF, nf=SANS)


def pasang_css(t):
    st.markdown(f"""<style>
:root{{--pri:{t['pri']};--on:{t['on']};--bg:{t['bg']};--bg2:{t['bg2']};--line:{t['line']};--tx:{t['tx']};--tx2:{t['tx2']};
--c1:{t['c'][0]};--c2:{t['c'][1]};--c3:{t['c'][2]};--hf:{t['hf']};--nf:{t['nf']}}}
.stApp,[data-testid="stHeader"]{{background:var(--bg);color:var(--tx)}}
.stApp,.stApp p,.stApp li,.stApp label,[data-testid="stSidebar"],[data-testid="stSidebar"] p,[data-testid="stCaptionContainer"]{{color:var(--tx)}}
[data-testid="stCaptionContainer"] p{{color:var(--tx2)!important}}
[data-testid="stSidebar"]{{background:var(--bg2);border-right:1px solid var(--line)}}
[data-testid="stSidebar"] hr,.stApp hr{{border-color:var(--line)}}
h1,h2,h3,h4{{font-family:var(--hf)!important;color:var(--tx)!important}}
h1{{font-size:2rem!important}}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{{border-radius:8px;border:1px solid var(--pri);color:var(--pri);background:transparent}}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"],[data-testid="stFormSubmitButton"]>button{{background:var(--pri)!important;color:var(--on)!important;border:0}}
[data-baseweb="input"],[data-baseweb="input"] div,[data-baseweb="base-input"],[data-baseweb="select"]>div,[data-baseweb="select"] div,.stNumberInput button{{background:var(--bg)!important;border-color:var(--line)!important;color:var(--tx)!important}}
.stApp input,.stApp textarea{{background:var(--bg)!important;color:var(--tx)!important;-webkit-text-fill-color:var(--tx)!important;font-family:var(--nf)!important}}
.stNumberInput button svg{{fill:var(--tx)!important}}
[data-baseweb="popover"] li,[data-baseweb="popover"] ul{{background:var(--bg)!important;color:var(--tx)!important}}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p{{color:var(--tx)!important}}
[data-testid="stForm"],[data-testid="stExpander"] details{{border:1px solid var(--line);border-radius:10px;background:var(--bg)}}
[data-testid="stFileUploaderDropzone"]{{background:var(--bg2);border:1px dashed var(--line)}}
.stTabs [data-baseweb="tab"]{{color:var(--tx2)}}
.stTabs [aria-selected="true"]{{color:var(--pri)!important}}
.stTabs [data-baseweb="tab-highlight"]{{background:var(--pri)}}
code{{color:var(--pri)!important;background:var(--bg2)!important}}
.stApp .muted{{color:var(--tx2)!important}}
.status{{background:var(--bg);border:1px solid var(--line);border-left:4px solid var(--pri);border-radius:0;padding:10px 12px;font-size:.9rem}}
.kartu{{background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:16px 18px;height:100%}}
.kartu .l{{font-size:.85rem;color:var(--tx2)}}
.kartu .v{{font-size:1.55rem;font-weight:700;font-family:var(--nf);line-height:1.3}}
.tag{{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.8rem;font-weight:600;color:#fff}}
.ub{{display:flex;align-items:center;gap:10px;margin:7px 0}}
.ub .lb{{width:200px;font-size:.9rem}}
.ub .tr{{flex:1;height:16px;background:var(--bg2);border:1px solid var(--line);border-radius:5px;overflow:hidden}}
.ub .fl{{height:100%}}
.ub .vl{{width:60px;text-align:right;font-family:var(--nf)}}
.gr{{height:26px;border-radius:8px;position:relative;margin:10px 0 2px;background:linear-gradient(90deg,var(--c1),var(--c2),var(--c3))}}
.gr .pin{{position:absolute;top:-6px;width:4px;height:38px;background:var(--tx);border-radius:2px}}
.grl{{display:flex;justify-content:space-between;font-size:.8rem;color:var(--tx2)}}
.langkah{{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}}
.langkah div{{background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:12px 14px;font-size:.92rem}}
.langkah b.no{{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;background:var(--pri);color:var(--on);font-size:.8rem;margin-right:8px}}
.tbl{{width:100%;border-collapse:collapse;font-size:.88rem}}
.tbl th,.tbl td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;color:var(--tx)}}
.tbl th{{background:var(--bg2);color:var(--tx2);font-weight:600}}
.tbl .n{{text-align:right;font-family:var(--nf)}}
.wadah{{border:1px solid var(--line);border-radius:10px;overflow:auto}}
.pesan{{padding:10px 14px;border-radius:8px;margin:10px 0;font-size:.92rem}}
.pesan.w{{background:#fff4dc;color:#7a4f00!important}}.pesan.e{{background:#fdecec;color:#8a1f1f!important}}.pesan.o{{background:var(--bg2)}}
.jalur{{font-family:{MONO};background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:.88rem}}
[role="radiogroup"] label>div:first-child{{background:var(--bg)!important;border-color:var(--pri)!important}}
[role="radiogroup"] label>div:first-child>div{{background:var(--pri)!important}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,[role="radiogroup"] p{{color:var(--tx)}}
.stApp .logo{{font-family:var(--hf);font-weight:700;font-size:1.15rem;color:var(--pri)}}
</style>""", unsafe_allow_html=True)


def html(s: str):
    """Tampilkan HTML (baris dirapatkan supaya tidak dibaca sebagai markdown)."""
    st.markdown("".join(baris.strip() for baris in s.splitlines()), unsafe_allow_html=True)


def f2(x, d=2):
    """Format angka gaya Indonesia: 1.234,56"""
    return f"{x:,.{d}f}".replace(",", "#").replace(".", ",").replace("#", ".")


# ------------------------------------------------------------------ muat model
@st.cache_resource
def muat_model():
    if not FILE_MODEL.exists():
        return None, None
    model = joblib.load(FILE_MODEL)
    hasil = pd.read_csv(FILE_HASIL) if FILE_HASIL.exists() else None
    return model, hasil


model, hasil = muat_model()

# ------------------------------------------------------------------ sidebar
pasang_css(T)
with st.sidebar:
    html('<div class="logo">🎓 Fuzzy Pendidikan Jatim</div>')
    st.caption("PCA + Fuzzy C-Means + Decision Tree")
    hal = st.radio("Menu", ["Beranda", "Prediksi", "Hasil Model", "Evaluasi", "Informasi"],
                   label_visibility="collapsed")

if model is None:
    st.error(f"File model tidak ditemukan: `{FILE_MODEL}`. Letakkan `model_pca_fcm_dt.joblib` "
             "dan `hasil_fcm_dt.csv` di folder `model/` (satu folder dengan app.py).")
    st.stop()

FITUR = list(model["fitur"])
C = int(model["c"])
M = float(model["m"])
LABEL = {2: ["Rendah", "Tinggi"], 3: ["Rendah", "Sedang", "Tinggi"]}.get(C, [f"Tingkat {i + 1}" for i in range(C)])
NAMA = {i + 1: f"Cluster {i + 1} – Capaian {n}" for i, n in enumerate(LABEL)}
WARNA = {i + 1: T["c"][i % 3] for i in range(C)}
WARNA_C = {f"Cluster {k}": w for k, w in WARNA.items()}

with st.sidebar:
    st.divider()
    html(f'<div class="status"><b>Model dimuat</b><br><span class="muted">{C} cluster · '
         f'{model["pca"].n_components_} komponen PCA · m = {M:g}</span></div>')
    st.divider()
    st.caption("Final Project Teknologi Logika Fuzzy (Kelas D) · Kelompok 4 · Informatika Universitas Udayana")


# ------------------------------------------------------------------ komponen tampilan
def tag(k):
    return f'<span class="tag" style="background:{WARNA[k]}">{LABEL[k - 1]}</span>'


def kartu(label, nilai, ket="", warna=None):
    gaya = f' style="color:{warna}"' if warna else ""
    return f'<div class="kartu"><div class="l">{label}</div><div class="v"{gaya}>{nilai}</div><div class="l">{ket}</div></div>'


def baris_kartu(isi):
    kol = st.columns(len(isi))
    for k, x in zip(kol, isi):
        with k:
            html(x)


def bar_keanggotaan(u):
    html("".join(f'<div class="ub"><div class="lb">{NAMA[i + 1]}</div><div class="tr"><div class="fl" '
                 f'style="width:{u[i] * 100:.1f}%;background:{WARNA[i + 1]}"></div></div>'
                 f'<div class="vl">{f2(u[i], 3)}</div></div>' for i in range(C)))


def bar_gradasi(u):
    posisi = float(np.dot(u, np.linspace(0, 100, C)))      # rata-rata berbobot derajat keanggotaan
    html(f'<div class="gr"><div class="pin" style="left:calc({posisi:.1f}% - 2px)"></div></div>'
         f'<div class="grl"><span>Rendah</span><span>Sedang</span><span>Tinggi</span></div>')


def tabel(df, angka=(), tinggi=None):
    """Tabel HTML bergaya tema. `angka` = kolom yang rata kanan."""
    kepala = "".join(f'<th class="{"n" if c in angka else ""}">{c}</th>' for c in df.columns)
    isi = "".join("<tr>" + "".join(f'<td class="{"n" if c in angka else ""}">{v}</td>' for c, v in r.items()) + "</tr>"
                  for _, r in df.iterrows())
    gaya = f' style="max-height:{tinggi}px"' if tinggi else ""
    html(f'<div class="wadah"{gaya}><table class="tbl"><tr>{kepala}</tr>{isi}</table></div>')


def pesan(teks, jenis="o"):
    html(f'<div class="pesan {jenis}">{teks}</div>')


def gaya_plot(fig, tinggi=None):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=T["tx"], family=T["nf"]), title_font=dict(family=T["hf"]),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["tx"])), margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(gridcolor=T["line"], zerolinecolor=T["line"], tickfont=dict(color=T["tx2"]), title_font=dict(color=T["tx"]))
    fig.update_yaxes(gridcolor=T["line"], zerolinecolor=T["line"], tickfont=dict(color=T["tx2"]), title_font=dict(color=T["tx"]))
    if tinggi:
        fig.update_layout(height=tinggi)
    return fig


# ------------------------------------------------------------------ fungsi proses
def validasi(df: pd.DataFrame):
    """Cek kolom, tipe angka, nilai kosong, dan rentang. Return (df_bersih | None, daftar_error)."""
    err = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    kurang = [f for f in FITUR if f not in df.columns]
    if kurang:
        return None, [f"Kolom wajib tidak ada: {', '.join(kurang)}"]
    if "kabkota" not in df.columns:
        df.insert(0, "kabkota", [f"Wilayah {i + 1}" for i in range(len(df))])
    for f in FITUR:
        df[f] = pd.to_numeric(df[f].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    kosong = df[df[FITUR].isna().any(axis=1)]
    if len(kosong):
        err.append(f"{len(kosong)} baris berisi nilai kosong / bukan angka (baris ke: "
                   f"{', '.join(str(i + 1) for i in kosong.index[:10])}).")
    for _, v in VARIABEL.iterrows():
        luar = df[(df[v.Variabel] < v.Min) | (df[v.Variabel] > v.Maks)]
        if len(luar):
            err.append(f"{v.Variabel}: {len(luar)} nilai di luar rentang {v.Min:g}–{v.Maks:g} {v.Satuan}.")
    return (None if err else df), err


def proses(df: pd.DataFrame) -> dict:
    """Input -> z-score -> PCA -> jarak ke pusat -> derajat keanggotaan FCM -> Decision Tree."""
    X = df[FITUR]
    Z = model["scaler"].transform(X)
    Y = model["pca"].transform(Z)
    V = model["V"]
    D = np.linalg.norm(Y[:, None, :] - V[None], axis=2) + 1e-12
    U = 1 / ((D[:, :, None] / D[:, None, :]) ** (2 / (M - 1))).sum(2)
    return {"Z": Z, "Y": Y, "D": D, "U": U, "cluster_fcm": U.argmax(1) + 1,
            "cluster_dt": model["dt"].predict(X)}


def jalur_dt(x: pd.DataFrame):
    t = model["dt"].tree_
    langkah = []
    for nid in model["dt"].decision_path(x).indices:
        if t.children_left[nid] == t.children_right[nid]:
            continue
        f, thr = FITUR[t.feature[nid]], t.threshold[nid]
        val = x.iloc[0][f]
        langkah.append(f"{f} = {f2(val)} {'≤' if val <= thr else '>'} {f2(thr)}")
    return langkah


prof = model["profil"]

# ================================================================== BERANDA
if hal == "Beranda":
    html('<p class="muted" style="margin:0">Final Project Teknologi Logika Fuzzy · Kelompok 4</p>')
    st.title("Pengelompokan Kabupaten/Kota di Jawa Timur Berdasarkan Kualitas Pendidikan")
    st.markdown("Aplikasi ini menentukan kelompok kualitas pendidikan sebuah wilayah dari 8 indikator pendidikan "
                "dengan kombinasi **PCA**, **Fuzzy C-Means**, dan **Decision Tree**. Setiap wilayah memperoleh "
                "**derajat keanggotaan** pada tiap kelompok, sehingga wilayah yang berada di antara dua kelompok "
                "dapat dikenali. Model dilatih dengan data 38 kabupaten/kota tahun 2022–2024 (BPS Jawa Timur).")
    st.subheader(f"{C} kelompok hasil Fuzzy C-Means")
    kol = st.columns(C)
    for i, k in enumerate(kol):
        p = prof.loc[i + 1]
        with k:
            html(f'<div class="kartu" style="border-top:4px solid {WARNA[i + 1]}">'
                 f'<span class="tag" style="background:{WARNA[i + 1]}">Cluster {i + 1}</span>'
                 f'<h3 style="margin:8px 0 4px">Capaian {LABEL[i]}</h3>'
                 f'<div class="muted">Rata-rata lama sekolah <b style="color:var(--tx)">{f2(p.RLS)} th</b><br>'
                 f'Tamatan SMA {f2(p.PTT_SMA, 1)}% · Melek huruf {f2(p.AMH, 1)}%<br>'
                 f'{int(p["jumlah (wilayah-tahun)"])} data wilayah-tahun</div></div>')
    st.subheader("Cara kerja")
    html('<div class="langkah">'
         '<div><b class="no">1</b><b>Input</b><br><span class="muted">8 indikator pendidikan</span></div>'
         '<div><b class="no">2</b><b>Z-score</b><br><span class="muted">Disamakan skalanya</span></div>'
         '<div><b class="no">3</b><b>PCA</b><br><span class="muted">Diringkas jadi 3 komponen</span></div>'
         '<div><b class="no">4</b><b>Fuzzy C-Means</b><br><span class="muted">Derajat keanggotaan</span></div>'
         '<div><b class="no">5</b><b>Decision Tree</b><br><span class="muted">Aturan IF-THEN</span></div></div>')
    st.write("")
    pesan("Buka menu <b>Prediksi</b> di sidebar untuk mulai.")

# ================================================================== PREDIKSI
elif hal == "Prediksi":
    st.title("Prediksi Kelompok Wilayah")
    with st.expander("Penjelasan variabel input"):
        tabel(VARIABEL.assign(Rentang=lambda d: d.Min.map("{:g}".format) + "–" + d.Maks.map("{:g}".format))
              [["Variabel", "Keterangan", "Satuan", "Rentang"]], angka=["Rentang"])
    tab1, tab2 = st.tabs(["Input manual (1 wilayah)", "Upload CSV (banyak wilayah)"])

    # ---------------- input manual
    with tab1:
        contoh = hasil[hasil.tahun == hasil.tahun.max()] if hasil is not None else None
        pilihan = ["-"] + (sorted(contoh.kabkota) if contoh is not None else [])
        isi = st.selectbox("Isi otomatis dari data 2024 (opsional)", pilihan)
        awal = contoh[contoh.kabkota == isi].iloc[0] if isi != "-" else None
        with st.form("manual"):
            nama = st.text_input("Nama wilayah", isi if isi != "-" else "Wilayah baru")
            kol = st.columns(4)
            nilai = {}
            for i, (_, v) in enumerate(VARIABEL.iterrows()):
                if awal is not None:
                    default = float(awal[v.Variabel])
                elif hasil is not None:
                    default = float(hasil[v.Variabel].median())     # median data latih sebagai nilai awal
                else:
                    default = 0.0
                nilai[v.Variabel] = kol[i % 4].number_input(f"{v.Variabel} ({v.Satuan})", v.Min, v.Maks,
                                                            round(default, 2), 0.01, help=v.Keterangan,
                                                            key=f"{v.Variabel}_{isi}")
            kirim = st.form_submit_button("Prediksi", type="primary")

        if kirim:
            x = pd.DataFrame([nilai])
            _, err = validasi(x)
            if err:
                for e in err:
                    pesan(e, "e")
                st.stop()
            r = proses(x)
            u, cf, cd = r["U"][0], int(r["cluster_fcm"][0]), int(r["cluster_dt"][0])
            transisi = u.max() < BATAS_TRANSISI
            baris_kartu([
                kartu("Fuzzy C-Means", f"Capaian {LABEL[cf - 1]}", f"Cluster {cf} · derajat keanggotaan {f2(u.max(), 3)}", WARNA[cf]),
                kartu("Decision Tree", f"Capaian {LABEL[cd - 1]}", "Kelas dari aturan IF-THEN", WARNA[cd]),
                kartu("Status", "Transisi" if transisi else "Jelas",
                      f"u maks {'<' if transisi else '≥'} {f2(BATAS_TRANSISI, 1)}"),
            ])
            if transisi:
                pesan(f"<b>{nama}</b> berstatus <b>transisi</b>: keanggotaannya terbagi di antara dua kelompok.", "w")
            if cf != cd:
                pesan("FCM dan Decision Tree berbeda. Ini biasa terjadi di perbatasan antar-cluster: FCM menilai dari "
                      "jarak ke pusat cluster, Decision Tree dari ambang aturan.", "w")

            st.subheader(f"Derajat keanggotaan — {nama}")
            bar_keanggotaan(u)
            st.markdown("##### Posisi wilayah di antara kelompok")
            bar_gradasi(u)

            st.subheader("Tahapan proses")
            st.markdown("##### 1. Standardisasi z-score")
            st.caption("z = (x − rata-rata data latih) / simpangan baku data latih")
            tabel(pd.DataFrame({"Variabel": FITUR, "Nilai asli": [f2(v) for v in x.iloc[0].values],
                                "Rata-rata latih": [f2(v) for v in model["scaler"].mean_],
                                "Simp. baku latih": [f2(v) for v in model["scaler"].scale_],
                                "z-score": [f2(v, 3) for v in r["Z"][0]]}),
                  angka=["Nilai asli", "Rata-rata latih", "Simp. baku latih", "z-score"])
            st.markdown("##### 2. Proyeksi PCA")
            st.caption("Skor komponen utama = z-score × loading")
            evr = model["pca"].explained_variance_ratio_
            kol_pc = [f"PC{i + 1} ({f2(evr[i] * 100, 1)}%)" for i in range(len(r["Y"][0]))]
            tabel(pd.DataFrame([[f2(v, 4) for v in r["Y"][0]]], columns=kol_pc), angka=kol_pc)
            st.markdown("##### 3–4. Jarak ke pusat cluster dan derajat keanggotaan FCM")
            st.latex(r"u_k = \left[\sum_{j=1}^{c}\left(\frac{d_k}{d_j}\right)^{\frac{2}{m-1}}\right]^{-1},\quad m = 2")
            tabel(pd.DataFrame({"Cluster": [NAMA[i + 1] for i in range(C)],
                                "Jarak ke pusat (d)": [f2(v, 4) for v in r["D"][0]],
                                "Derajat keanggotaan (u)": [f2(v, 4) for v in u]}),
                  angka=["Jarak ke pusat (d)", "Derajat keanggotaan (u)"])
            st.markdown("##### 5. Jalur aturan Decision Tree")
            html(f'<div class="jalur">{"  →  ".join(jalur_dt(x) + [f"Cluster {cd}"])}</div>')

    # ---------------- upload CSV
    with tab2:
        st.markdown("Kolom wajib: 8 variabel di atas. Kolom `kabkota` (nama wilayah) opsional. "
                    "Pemisah koma atau titik koma.")
        st.download_button("Unduh template CSV", pd.DataFrame(columns=["kabkota"] + FITUR).to_csv(index=False),
                           "template_input.csv", "text/csv")
        f = st.file_uploader("Pilih file CSV", type=["csv"])
        if f is not None:
            try:
                data = pd.read_csv(f, sep=None, engine="python")
            except Exception as e:
                pesan(f"File tidak bisa dibaca: {e}", "e")
                st.stop()
            data, err = validasi(data)
            if err:
                for e in err:
                    pesan(e, "e")
                st.stop()
            r = proses(data)
            out = data[["kabkota"] + FITUR].copy()
            for i in range(C):
                out[f"u_C{i + 1}"] = r["U"][:, i].round(4)
            out["cluster_FCM"] = r["cluster_fcm"]
            out["cluster_DT"] = r["cluster_dt"]
            out["status"] = np.where(r["U"].max(1) < BATAS_TRANSISI, "transisi", "jelas")
            pesan(f"{len(out)} wilayah berhasil diproses.")
            st.download_button("Unduh hasil prediksi", out.to_csv(index=False), "hasil_prediksi.csv",
                               "text/csv", type="primary")
            kol_u = [f"u_C{i + 1}" for i in range(C)]
            tampil = pd.DataFrame({"Kab/Kota": out.kabkota,
                                   **{k: out[k].map(lambda v: f2(v, 3)) for k in kol_u},
                                   "Cluster FCM": out.cluster_FCM.map(tag), "Cluster DT": out.cluster_DT.map(tag),
                                   "Status": out.status})
            tabel(tampil, angka=kol_u, tinggi=420)
            hitung = out.cluster_FCM.value_counts().reindex(range(1, C + 1), fill_value=0)
            fig = go.Figure(go.Bar(x=[NAMA[k] for k in hitung.index], y=hitung.values, text=hitung.values,
                                   marker_color=[WARNA[k] for k in hitung.index]))
            st.plotly_chart(gaya_plot(fig.update_layout(title="Jumlah wilayah per cluster"), 320), width="stretch")

# ================================================================== HASIL MODEL
elif hal == "Hasil Model":
    st.title("Hasil Model")
    if hasil is None:
        pesan("File <code>model/hasil_fcm_dt.csv</code> tidak ditemukan.", "w")
        st.stop()
    html('<p class="muted">Hasil pengelompokan 114 data (38 kabupaten/kota × 2022–2024).</p>')

    st.subheader("Profil tiap cluster")
    tp = pd.DataFrame({"Cluster": [tag(i) for i in prof.index],
                       **{f: prof[f].map(f2).values for f in FITUR},
                       "Jumlah": prof["jumlah (wilayah-tahun)"].astype(int).values})
    tabel(tp, angka=FITUR + ["Jumlah"])
    st.markdown(f"Cluster diurutkan dari capaian terendah ke tertinggi berdasarkan rata-rata lama sekolah. "
                f"Kelompok **capaian rendah** (RLS {f2(prof.loc[1, 'RLS'])} tahun, tamatan SMA "
                f"{f2(prof.loc[1, 'PTT_SMA'], 1)}%) menjadi prioritas utama intervensi kebijakan pendidikan, "
                f"sedangkan kelompok **capaian tinggi** didominasi wilayah perkotaan.")

    st.subheader("Peta cluster pada ruang PCA")
    Y = model["pca"].transform(model["scaler"].transform(hasil[FITUR]))
    evr = model["pca"].explained_variance_ratio_
    fig = go.Figure()
    for k in range(1, C + 1):
        g = hasil[hasil.cluster == k]
        idx = g.index
        fig.add_trace(go.Scatter(
            x=Y[idx, 0], y=Y[idx, 1], mode="markers", name=NAMA[k],
            marker=dict(color=WARNA[k], size=6 + 10 * g.u_maks, opacity=.8,
                        line=dict(width=np.where(g.status == "transisi", 2, 0), color=T["tx"])),
            text=g.kabkota + " (" + g.tahun.astype(str) + ")<br>u = " + g.u_maks.round(3).astype(str),
            hoverinfo="text"))
    fig.add_trace(go.Scatter(x=model["V"][:, 0], y=model["V"][:, 1], mode="markers+text", name="Pusat cluster",
                             text=[f"C{i + 1}" for i in range(C)], textposition="top right",
                             marker=dict(symbol="x", size=16, color=T["tx"])))
    fig.update_layout(xaxis_title=f"PC1 ({f2(evr[0] * 100, 1)}%)", yaxis_title=f"PC2 ({f2(evr[1] * 100, 1)}%)")
    st.plotly_chart(gaya_plot(fig, 520), width="stretch")
    st.caption("Ukuran titik = derajat keanggotaan tertinggi · titik bergaris tepi = transisi · X = pusat cluster")

    st.subheader("Anggota cluster per tahun")
    tahun = st.selectbox("Tahun", sorted(hasil.tahun.unique(), reverse=True))
    t = hasil[hasil.tahun == tahun]
    kol = st.columns(C)
    for i, k in enumerate(kol):
        g = t[t.cluster == i + 1].sort_values("u_maks", ascending=False)
        baris = "".join(f'<tr><td>{r.kabkota.replace("Kabupaten", "Kab.")}'
                        f'{" <span class=muted>(transisi)</span>" if r.status == "transisi" else ""}</td>'
                        f'<td class="n">{f2(r.u_maks, 3)}</td></tr>' for r in g.itertuples())
        with k:
            html(f'<div class="kartu" style="background:var(--bg)"><span class="tag" style="background:{WARNA[i + 1]}">'
                 f'{LABEL[i]} · {len(g)}</span><table class="tbl" style="margin-top:8px">{baris}</table></div>')

    st.subheader("Aturan IF-THEN Decision Tree")
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=T["bg"])
    plot_tree(model["dt"], feature_names=FITUR, class_names=[f"Cluster {k}" for k in model["dt"].classes_],
              filled=True, rounded=True, fontsize=9, ax=ax)
    for teks in ax.texts:
        teks.set_color("#1C2333")
    st.pyplot(fig)
    plt.close(fig)

# ================================================================== EVALUASI
elif hal == "Evaluasi":
    st.title("Evaluasi Model")
    if hasil is None:
        pesan("File <code>model/hasil_fcm_dt.csv</code> tidak ditemukan.", "w")
        st.stop()
    U = hasil[[f"u_C{i + 1}" for i in range(C)]].to_numpy()
    Y = model["pca"].transform(model["scaler"].transform(hasil[FITUR]))
    V, n = model["V"], len(U)
    pc = (U ** 2).sum() / n
    pe = -(U * np.log(U + 1e-12)).sum() / n
    mpc = 1 - C / (C - 1) * (1 - pc)
    dmin = min(((V[a] - V[b]) ** 2).sum() for a in range(C) for b in range(a + 1, C))
    xb = ((U ** M) * ((Y[:, None, :] - V[None]) ** 2).sum(2)).sum() / (n * dmin)
    sil = silhouette_score(Y, hasil.cluster)

    st.subheader(f"1. Kualitas clustering FCM (c = {C})")
    baris_kartu([kartu(l, f2(v, 4), f"terbaik = {a}") for l, v, a in
                 [("PC", pc, "maks"), ("PE", pe, "min"), ("MPC", mpc, "maks"), ("Xie-Beni", xb, "min"),
                  ("Silhouette", sil, "maks")]])
    st.caption(f"Jumlah cluster c = {C} dipilih di notebook dengan voting 5 indeks validitas untuk c = 2–6. "
               "MPC, Xie-Beni, dan Silhouette memilih c = 3; PC dan PE cenderung monoton memihak c kecil.")
    if "tabel_c" in model:
        st.dataframe(model["tabel_c"].round(4), width="stretch")

    st.subheader("2. Decision Tree")
    acc = accuracy_score(hasil.cluster, hasil.cluster_dt)
    cv = f"{f2(np.mean(model['cv_skor']) * 100)}%" if "cv_skor" in model else "91,15%"
    baris_kartu([kartu("Akurasi 5-fold cross validation", cv, "dari notebook"),
                 kartu("Akurasi data latih", f"{f2(acc * 100)}%", "konsistensi aturan terhadap FCM"),
                 kartu("Wilayah transisi", f"{(hasil.status == 'transisi').sum()} dari {n}", "u maks < 0,6")])
    st.write("")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("##### Confusion matrix (FCM vs DT)")
        cm = confusion_matrix(hasil.cluster, hasil.cluster_dt)
        isi = "".join(f"<tr><th>FCM C{a + 1}</th>" + "".join(
            f'<td class="n" style="{"font-weight:700;background:var(--bg2)" if a == b else ""}">{cm[a, b]}</td>'
            for b in range(C)) + "</tr>" for a in range(C))
        html(f'<div class="wadah"><table class="tbl"><tr><th></th>'
             f'{"".join(f"<th class=n>DT C{b + 1}</th>" for b in range(C))}</tr>{isi}</table></div>')
    with k2:
        st.markdown("##### Feature importance")
        imp = pd.Series(model["dt"].feature_importances_, index=FITUR).sort_values(ascending=False)
        html("".join(f'<div class="ub"><div class="lb" style="width:90px">{f}</div><div class="tr">'
                     f'<div class="fl" style="width:{v * 100:.1f}%;background:var(--pri)"></div></div>'
                     f'<div class="vl">{f2(v, 3)}</div></div>' for f, v in imp.items()))

    st.subheader("3. Kelebihan dan keterbatasan")
    st.markdown(
        f"- **Kelebihan:** derajat keanggotaan menunjukkan wilayah transisi ({(hasil.status == 'transisi').sum()} "
        "dari 114 data); PCA meringkas indikator yang saling berkorelasi; aturan Decision Tree mudah dipahami; "
        "hasil FCM stabil terhadap inisialisasi acak.\n"
        "- **Keterbatasan:** hanya 38 wilayah per tahun; rasio murid-guru tidak tersedia di BPS 2022–2024; "
        "clustering tidak punya label kebenaran sehingga akurasi Decision Tree hanya mengukur konsistensi "
        "terhadap FCM; model perlu dilatih ulang di notebook jika ada data tahun baru.")

# ================================================================== INFORMASI
elif hal == "Informasi":
    st.title("Informasi Sistem")
    st.subheader("Metode")
    evr_total = model["pca"].explained_variance_ratio_.sum() * 100
    metode = [("Z-score", "z = (x − x̄) / s", "Menyamakan skala tahun dan persen."),
              ("PCA", "C = ZᵀZ / (n−1),  Cw = λw,  Y = ZWₖ",
               f"{model['pca'].n_components_} komponen, varians {f2(evr_total)}%."),
              ("Fuzzy C-Means", "J = ΣΣ uᵢₖᵐ ‖xᵢ − vₖ‖²", f"m = {M:g}, ε = 10⁻⁵, maks 100 iterasi."),
              ("Decision Tree", "Gini = 1 − Σ pₖ²",
               f"Kedalaman {model['dt'].max_depth}, minimal {model['dt'].min_samples_leaf} data per daun.")]
    kol = st.columns(4)
    for k, (j, rumus, ket) in zip(kol, metode):
        with k:
            html(f'<div class="kartu"><h3 style="margin:0 0 6px">{j}</h3>'
                 f'<div style="font-family:var(--nf)">{rumus}</div><div class="muted" style="margin-top:6px">{ket}</div></div>')
    st.markdown("**Rumus lengkap Fuzzy C-Means**")
    st.latex(r"v_k = \frac{\sum_i u_{ik}^m x_i}{\sum_i u_{ik}^m},\qquad "
             r"u_{ik} = \left[\sum_j\left(\frac{\lVert x_i-v_k\rVert}{\lVert x_i-v_j\rVert}\right)^{\frac{2}{m-1}}\right]^{-1}")

    st.subheader("Parameter model")
    tabel(pd.DataFrame({
        "Parameter": ["Jumlah cluster (c)", "Fuzzifier (m)", "Komponen PCA", "Varians dijelaskan PCA",
                      "Kedalaman pohon", "Minimal data per daun", "Batas transisi"],
        "Nilai": [C, f"{M:g}", model["pca"].n_components_, f"{f2(evr_total)}%",
                  model["dt"].max_depth, model["dt"].min_samples_leaf, f2(BATAS_TRANSISI, 1)]}), angka=["Nilai"])
    st.subheader("Variabel")
    tabel(VARIABEL.assign(Rentang=lambda d: d.Min.map("{:g}".format) + "–" + d.Maks.map("{:g}".format))
          [["Variabel", "Keterangan", "Satuan", "Rentang"]], angka=["Rentang"])
    st.subheader("Petunjuk penggunaan")
    st.markdown("1. Buka **Prediksi**.\n"
                "2. Pilih **Input manual** (isi 8 indikator, atau pilih wilayah untuk isi otomatis) lalu klik "
                "**Prediksi**; atau pilih **Upload CSV** sesuai template untuk banyak wilayah sekaligus.\n"
                "3. Lihat kelompok, derajat keanggotaan, dan tahapan proses. Hasil CSV dapat diunduh.\n"
                "4. **Hasil Model** dan **Evaluasi** menampilkan hasil dan kualitas model dari data latih.\n\n"
                "**Menjalankan aplikasi:** `pip install -r requirements.txt` lalu `streamlit run app.py`. "
                "File model harus berada di folder `model/` (satu folder dengan `app.py`).")
    html('<p class="muted">Sumber data: BPS Provinsi Jawa Timur — RLS penduduk 25+ dan publikasi '
         '<i>Statistik Pendidikan Provinsi Jawa Timur</i> 2022–2024. Acuan: Hermawan &amp; Sulaiman (2026); '
         'Riyadi &amp; Fauziah (2024); Dave (1996).</p>')