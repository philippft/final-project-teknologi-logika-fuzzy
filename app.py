"""
GUI Pengelompokan Kabupaten/Kota di Jawa Timur Berdasarkan Kualitas Pendidikan
PCA + Fuzzy C-Means + Decision Tree

Aplikasi ini TIDAK melatih ulang model. Model hasil training di notebook
(model/model_pca_fcm_dt.joblib) dimuat, lalu dipakai untuk memproses input user.

Jalankan:  streamlit run app.py
"""
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

st.set_page_config(page_title="Fuzzy Pendidikan Jatim", page_icon="🎓", layout="wide")

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
WARNA = px.colors.qualitative.Set2


# ------------------------------------------------------------------ muat model
@st.cache_resource
def muat_model():
    if not FILE_MODEL.exists():
        return None, None
    model = joblib.load(FILE_MODEL)
    hasil = pd.read_csv(FILE_HASIL) if FILE_HASIL.exists() else None
    return model, hasil


model, hasil = muat_model()
if model is None:
    st.error(f"File model tidak ditemukan: `{FILE_MODEL}`. Letakkan `model_pca_fcm_dt.joblib` "
             "dan `hasil_fcm_dt.csv` di folder `model/` (satu folder dengan app.py).")
    st.stop()

FITUR = list(model["fitur"])
C = int(model["c"])
M = float(model["m"])
NAMA = {2: ["Rendah", "Tinggi"], 3: ["Rendah", "Sedang", "Tinggi"]}.get(C, [f"Tingkat {i + 1}" for i in range(C)])
NAMA = {i + 1: f"Cluster {i + 1} – Capaian {n}" for i, n in enumerate(NAMA)}
WARNA_C = {f"Cluster {i + 1}": WARNA[i] for i in range(C)}


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
        langkah.append(f"{f} = {val:.2f} {'≤' if val <= thr else '>'} {thr:.2f}")
    return langkah


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🎓 Fuzzy Pendidikan Jatim")
    st.caption("PCA + Fuzzy C-Means + Decision Tree")
    hal = st.radio("Menu", ["🏠 Beranda", "🔮 Prediksi", "📊 Hasil Model", "✅ Evaluasi", "ℹ️ Informasi"],
                   label_visibility="collapsed")
    st.divider()
    st.success(f"Model dimuat: {C} cluster")
    st.caption(f"{model['pca'].n_components_} komponen PCA · m = {M:g}")
    st.divider()
    st.caption("Final Project Teknologi Logika Fuzzy (Kelas D) · Kelompok 4 · Informatika Universitas Udayana")

# ================================================================== BERANDA
if hal == "🏠 Beranda":
    st.title("🎓 Pengelompokan Kabupaten/Kota di Jawa Timur Berdasarkan Kualitas Pendidikan")
    st.markdown("##### Kombinasi Principal Component Analysis, Fuzzy C-Means, dan Decision Tree")
    st.write("Aplikasi ini menentukan **kelompok kualitas pendidikan** sebuah wilayah dari 8 indikator pendidikan. "
             "Hasilnya berupa **derajat keanggotaan** (seberapa besar wilayah termasuk tiap kelompok) dan "
             "**aturan IF-THEN** yang menjelaskan alasannya. Model dilatih dengan data 38 kabupaten/kota "
             "Jawa Timur tahun 2022–2024 (BPS).")
    kol = st.columns(C)
    prof = model["profil"]
    for i, k in enumerate(kol):
        k.metric(NAMA[i + 1], f"RLS {prof.loc[i + 1, 'RLS']:.2f} th",
                 f"{int(prof.loc[i + 1, 'jumlah (wilayah-tahun)'])} data", delta_color="off")
    st.subheader("Cara kerja")
    st.markdown("1. **Input** 8 indikator pendidikan (manual atau upload CSV)\n"
                "2. **Standardisasi z-score** dengan rata-rata & simpangan baku data latih\n"
                "3. **PCA**: diproyeksikan ke komponen utama yang sama dengan saat training\n"
                "4. **Fuzzy C-Means**: dihitung derajat keanggotaan terhadap pusat cluster hasil training\n"
                "5. **Decision Tree**: menentukan kelas dan menampilkan aturan yang dilalui")
    st.info("Buka menu **🔮 Prediksi** untuk mulai.")

# ================================================================== PREDIKSI
elif hal == "🔮 Prediksi":
    st.title("🔮 Prediksi Kelompok Wilayah")
    with st.expander("📖 Penjelasan variabel input"):
        st.dataframe(VARIABEL, hide_index=True, width="stretch")
    tab1, tab2 = st.tabs(["✏️ Input manual (1 wilayah)", "⬆️ Upload CSV (banyak wilayah)"])

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
            kirim = st.form_submit_button("🔮 Prediksi", type="primary")

        if kirim:
            x = pd.DataFrame([nilai])
            _, err = validasi(x)
            if err:
                for e in err:
                    st.error(e)
                st.stop()
            r = proses(x)
            u, cf, cd = r["U"][0], int(r["cluster_fcm"][0]), int(r["cluster_dt"][0])
            a, b = st.columns(2)
            a.success(f"**Fuzzy C-Means:** {NAMA[cf]}  \nDerajat keanggotaan tertinggi: **{u.max():.3f}**")
            b.info(f"**Decision Tree:** {NAMA[cd]}")
            if u.max() < BATAS_TRANSISI:
                st.warning(f"**{nama}** berstatus **transisi** (derajat keanggotaan tertinggi < {BATAS_TRANSISI}): "
                           "wilayah ini berada di antara dua kelompok.")
            if cf != cd:
                st.warning("FCM dan Decision Tree berbeda. Biasanya terjadi pada wilayah di perbatasan antar-cluster: "
                           "FCM menilai dari jarak ke pusat cluster, Decision Tree dari ambang aturan.")
            fig = px.bar(x=[NAMA[i + 1] for i in range(C)], y=u, text=np.round(u, 3),
                         color=[f"Cluster {i + 1}" for i in range(C)], color_discrete_map=WARNA_C,
                         labels={"x": "", "y": "Derajat keanggotaan"}, title=f"Derajat keanggotaan — {nama}")
            fig.update_layout(showlegend=False, yaxis_range=[0, 1], height=340)
            st.plotly_chart(fig, width="stretch")

            st.subheader("Tahapan proses")
            st.markdown("**1. Standardisasi z-score** — z = (x − rata-rata data latih) / simpangan baku data latih")
            st.dataframe(pd.DataFrame({"Nilai asli": x.iloc[0].values, "Rata-rata latih": model["scaler"].mean_,
                                       "Simpangan baku latih": model["scaler"].scale_, "z-score": r["Z"][0]},
                                      index=FITUR).round(3), width="stretch")
            st.markdown("**2. Proyeksi PCA** — skor komponen utama = z-score × loading")
            st.dataframe(pd.DataFrame([r["Y"][0]], columns=[f"PC{i + 1}" for i in range(len(r["Y"][0]))],
                                      index=["Skor"]).round(4), width="stretch")
            st.markdown("**3. Jarak ke pusat cluster & 4. derajat keanggotaan FCM**")
            st.latex(r"u_k = \left[\sum_{j=1}^{c}\left(\frac{d_k}{d_j}\right)^{\frac{2}{m-1}}\right]^{-1}")
            st.dataframe(pd.DataFrame({"Jarak ke pusat (d)": r["D"][0], "Derajat keanggotaan (u)": u},
                                      index=[NAMA[i + 1] for i in range(C)]).round(4), width="stretch")
            st.markdown("**5. Jalur aturan Decision Tree**")
            st.code("  →  ".join(jalur_dt(x) + [f"Cluster {cd}"]), language="text")

    # ---------------- upload CSV
    with tab2:
        st.write("Kolom wajib: 8 variabel di atas. Kolom `kabkota` (nama wilayah) opsional.")
        st.download_button("Unduh template CSV", pd.DataFrame(columns=["kabkota"] + FITUR).to_csv(index=False),
                           "template_input.csv", "text/csv")
        f = st.file_uploader("Pilih file CSV", type=["csv"])
        if f is not None:
            try:
                data = pd.read_csv(f, sep=None, engine="python")
            except Exception as e:
                st.error(f"File tidak bisa dibaca: {e}")
                st.stop()
            data, err = validasi(data)
            if err:
                for e in err:
                    st.error(e)
                st.stop()
            r = proses(data)
            out = data[["kabkota"] + FITUR].copy()
            for i in range(C):
                out[f"u_C{i + 1}"] = r["U"][:, i].round(4)
            out["cluster_FCM"] = [NAMA[c] for c in r["cluster_fcm"]]
            out["cluster_DT"] = [NAMA[c] for c in r["cluster_dt"]]
            out["status"] = np.where(r["U"].max(1) < BATAS_TRANSISI, "transisi", "jelas")
            st.success(f"{len(out)} wilayah berhasil diproses.")
            st.dataframe(out, hide_index=True, width="stretch")
            st.plotly_chart(px.histogram(out, x="cluster_FCM", color="cluster_FCM",
                                         title="Jumlah wilayah per cluster").update_layout(showlegend=False),
                            width="stretch")
            st.download_button("⬇️ Unduh hasil prediksi", out.to_csv(index=False), "hasil_prediksi.csv",
                               "text/csv", type="primary")

# ================================================================== HASIL MODEL
elif hal == "📊 Hasil Model":
    st.title("📊 Hasil Model (data latih 2022–2024)")
    if hasil is None:
        st.warning("File `model/hasil_fcm_dt.csv` tidak ditemukan.")
        st.stop()
    st.subheader("Profil tiap cluster")
    prof = model["profil"].copy()
    prof.insert(0, "Nama", [NAMA[i] for i in prof.index])
    st.dataframe(prof.round(2), width="stretch")
    st.markdown("\n".join(
        f"- **{NAMA[i]}**: rata-rata lama sekolah {prof.loc[i, 'RLS']:.2f} tahun, tamatan SMA "
        f"{prof.loc[i, 'PTT_SMA']:.1f}%, APM SMA {prof.loc[i, 'APM_SMA']:.1f}%, melek huruf {prof.loc[i, 'AMH']:.1f}%."
        for i in prof.index) + f"\n\n**{NAMA[1]}** menjadi prioritas utama intervensi kebijakan pendidikan.")

    st.subheader("Peta cluster pada ruang PCA")
    Y = model["pca"].transform(model["scaler"].transform(hasil[FITUR]))
    plot = hasil.assign(PC1=Y[:, 0], PC2=Y[:, 1], Cluster="Cluster " + hasil.cluster.astype(str))
    fig = px.scatter(plot, x="PC1", y="PC2", color="Cluster", symbol="status", color_discrete_map=WARNA_C,
                     hover_data=["kabkota", "tahun", "u_maks"], height=500)
    fig.add_trace(go.Scatter(x=model["V"][:, 0], y=model["V"][:, 1], mode="markers", name="Pusat cluster",
                             marker=dict(symbol="x", size=16, color="black")))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Anggota cluster per tahun")
    tahun = st.selectbox("Tahun", sorted(hasil.tahun.unique(), reverse=True))
    t = hasil[hasil.tahun == tahun]
    kol = st.columns(C)
    for i, k in enumerate(kol):
        g = t[t.cluster == i + 1].sort_values("u_maks", ascending=False)
        k.markdown(f"**{NAMA[i + 1]}** ({len(g)})")
        k.dataframe(g[["kabkota", "u_maks", "status"]].rename(columns={"kabkota": "Kab/Kota", "u_maks": "u"}),
                    hide_index=True, width="stretch", height=420)

    st.subheader("Aturan IF-THEN Decision Tree")
    fig, ax = plt.subplots(figsize=(13, 5))
    plot_tree(model["dt"], feature_names=FITUR, class_names=[f"Cluster {k}" for k in model["dt"].classes_],
              filled=True, rounded=True, fontsize=9, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ================================================================== EVALUASI
elif hal == "✅ Evaluasi":
    st.title("✅ Evaluasi Model")
    if hasil is None:
        st.warning("File `model/hasil_fcm_dt.csv` tidak ditemukan.")
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

    st.subheader("1. Kualitas clustering FCM")
    k = st.columns(5)
    for kol, (lab, val, ket) in zip(k, [("PC", pc, "maks"), ("PE", pe, "min"), ("MPC", mpc, "maks"),
                                        ("Xie-Beni", xb, "min"), ("Silhouette", sil, "maks")]):
        kol.metric(lab, f"{val:.4f}", f"terbaik = {ket}", delta_color="off")
    st.caption(f"Jumlah cluster c = {C} dipilih di notebook dengan voting 5 indeks validitas untuk c = 2–6 "
               "(MPC, Xie-Beni, dan Silhouette memilih c = 3).")
    if "tabel_c" in model:
        st.dataframe(model["tabel_c"].round(4), width="stretch")

    st.subheader("2. Decision Tree (konsistensi aturan terhadap label FCM)")
    acc = accuracy_score(hasil.cluster, hasil.cluster_dt)
    a, b = st.columns(2)
    a.metric("Akurasi pada data latih", f"{acc * 100:.2f}%")
    b.metric("Akurasi 5-fold cross validation (dari notebook)",
             f"{np.mean(model['cv_skor']) * 100:.2f}%" if "cv_skor" in model else "91,15%")
    cm = pd.DataFrame(confusion_matrix(hasil.cluster, hasil.cluster_dt),
                      index=[f"FCM C{i + 1}" for i in range(C)], columns=[f"DT C{i + 1}" for i in range(C)])
    k1, k2 = st.columns(2)
    k1.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion matrix"),
                    width="stretch")
    imp = pd.Series(model["dt"].feature_importances_, index=FITUR).sort_values()
    k2.plotly_chart(px.bar(imp, orientation="h", title="Feature importance").update_layout(showlegend=False),
                    width="stretch")

    st.subheader("3. Kelebihan & keterbatasan")
    st.markdown(
        f"- **Kelebihan:** derajat keanggotaan menunjukkan wilayah transisi ({(hasil.status == 'transisi').sum()} "
        "dari 114 data); PCA meringkas indikator yang saling berkorelasi; aturan Decision Tree mudah dipahami; "
        "hasil FCM stabil terhadap inisialisasi acak.\n"
        "- **Keterbatasan:** hanya 38 wilayah per tahun; rasio murid-guru tidak tersedia di BPS 2022–2024; "
        "clustering tidak punya label kebenaran sehingga akurasi Decision Tree hanya mengukur konsistensi "
        "terhadap FCM; model perlu dilatih ulang di notebook jika ada data tahun baru.")

# ================================================================== INFORMASI
elif hal == "ℹ️ Informasi":
    st.title("ℹ️ Informasi Sistem")
    st.subheader("Metode")
    st.markdown("**Standardisasi z-score** menyamakan skala variabel:")
    st.latex(r"z = \frac{x - \bar{x}}{s}")
    st.markdown("**PCA** meringkas 8 indikator menjadi komponen utama (varians kumulatif ≥ 80%):")
    st.latex(r"C = \frac{1}{n-1}Z^{T}Z,\quad C w = \lambda w,\quad Y = Z W_k")
    st.markdown("**Fuzzy C-Means** mengelompokkan dengan derajat keanggotaan:")
    st.latex(r"J_m = \sum_i\sum_k u_{ik}^m \lVert x_i - v_k\rVert^2,\quad "
             r"v_k = \frac{\sum_i u_{ik}^m x_i}{\sum_i u_{ik}^m},\quad "
             r"u_{ik} = \left[\sum_j\left(\frac{\lVert x_i-v_k\rVert}{\lVert x_i-v_j\rVert}\right)^{\frac{2}{m-1}}\right]^{-1}")
    st.markdown("**Decision Tree (CART)** dilatih dari label FCM untuk menghasilkan aturan IF-THEN:")
    st.latex(r"Gini = 1 - \sum_k p_k^2")
    st.subheader("Parameter model")
    st.dataframe(pd.DataFrame({
        "Parameter": ["Jumlah cluster (c)", "Fuzzifier (m)", "Komponen PCA", "Varians dijelaskan PCA",
                      "Kedalaman pohon", "Minimal data per daun", "Batas transisi"],
        "Nilai": [C, M, model["pca"].n_components_, f"{model['pca'].explained_variance_ratio_.sum() * 100:.2f}%",
                  model["dt"].max_depth, model["dt"].min_samples_leaf, BATAS_TRANSISI]}).astype(str),
        hide_index=True, width="stretch")
    st.subheader("Variabel")
    st.dataframe(VARIABEL, hide_index=True, width="stretch")
    st.markdown("**Sumber data:** BPS Provinsi Jawa Timur — tabel RLS penduduk 25+ dan publikasi "
                "*Statistik Pendidikan Provinsi Jawa Timur* 2022–2024.")
    st.subheader("Petunjuk penggunaan")
    st.markdown("1. Buka **🔮 Prediksi**.\n"
                "2. Pilih **Input manual** (isi 8 indikator, atau pilih wilayah untuk isi otomatis) lalu klik "
                "**Prediksi**; atau pilih **Upload CSV** sesuai template untuk banyak wilayah sekaligus.\n"
                "3. Lihat kelompok, derajat keanggotaan, dan tahapan proses. Hasil CSV dapat diunduh.\n"
                "4. **📊 Hasil Model** dan **✅ Evaluasi** menampilkan hasil dan kualitas model dari data latih.\n\n"
                "**Menjalankan aplikasi:** `pip install -r requirements.txt` lalu `streamlit run app.py`. "
                "File model harus berada di folder `model/` (satu folder dengan `app.py`).")
