import math
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

# Configuración general de la página en modo ancho
st.set_page_config(
    page_title="Windep OS - Suite Cuantitativa",
    page_icon="⚽",
    layout="wide"
)

SHEET_ID = "1wXHz_CaJ7rx-e2wha3lWwQKpV1rcVjlQfKiliHLAPBU"
URL_SHEET = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
    f"/gviz/tq?tqx=out:csv&sheet=GOLES"
)
MIN_PJ = 6
REPS_RADAR = 10000

# --------------------------------------------------------
# 1. CARGA Y PREPROCESAMIENTO DE DATOS (CON CACHÉ DE STREAMLIT)
# --------------------------------------------------------
@st.cache_data(ttl=600)
def cargar_datos():
    try:
        df = pd.read_csv(URL_SHEET)
    except Exception:
        url_alt = (
            f"https://docs.google.com/spreadsheets/d/"
            f"{SHEET_ID}/gviz/tq?tqx=out:csv"
        )
        try:
            df = pd.read_csv(url_alt)
        except Exception:
            df = pd.read_csv("basedatos_windep.csv")

    df = df.dropna(subset=["Local", "Visitante"]).copy()
    c_l, c_v = (
        df["Local"].astype(str).str.lower(),
        df["Visitante"].astype(str).str.lower()
    )
    df = df[~c_l.isin(["nan", "none", ""])]
    df = df[~c_v.isin(["nan", "none", ""])]

    num_cols = [
        "Goles_Loc", "Goles_Vis", "xG_Loc", "xG_Vis",
        "Goles_Loc_1T", "Goles_Vis_1T",
        "xG_Loc_1T", "xG_Vis_1T"
    ]
    for c in num_cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = (
                    df[c].astype(str).str.replace(",", ".")
                    .str.strip()
                )
            df[c] = pd.to_numeric(
                df[c], errors="coerce"
            ).fillna(0.0)

    for c in ["Liga", "Local", "Visitante"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    col_f = "Fecha" if "Fecha" in df.columns else "FECHA"
    if col_f in df.columns:
        df[col_f] = pd.to_datetime(
            df[col_f], dayfirst=True, errors="coerce"
        )
        df = df.sort_values(by=col_f).reset_index(drop=True)

    df["Tot_FT"] = df["Goles_Loc"] + df["Goles_Vis"]
    df["Tot_1T"] = df["Goles_Loc_1T"] + df["Goles_Vis_1T"]
    return df, col_f

def procesar_historial(df, col_f):
    c_l = [
        col_f, "Liga", "Local", "Goles_Loc", "xG_Loc",
        "Goles_Vis", "xG_Vis", "Goles_Loc_1T", "xG_Loc_1T",
        "Goles_Vis_1T", "xG_Vis_1T", "Tot_FT", "Tot_1T"
    ]
    loc = df[c_l].rename(columns={
        "Local": "Equipo", "Goles_Loc": "Goles_FT",
        "xG_Loc": "xG_FT", "Goles_Vis": "Goles_Contra_FT",
        "xG_Vis": "xG_Contra_FT", "Goles_Loc_1T": "Goles_1T",
        "xG_Loc_1T": "xG_1T",
        "Goles_Vis_1T": "Goles_Contra_1T",
        "xG_Vis_1T": "xG_Contra_1T"
    })
    c_v = [
        col_f, "Liga", "Visitante", "Goles_Vis", "xG_Vis",
        "Goles_Loc", "xG_Loc", "Goles_Vis_1T", "xG_Vis_1T",
        "Goles_Loc_1T", "xG_Loc_1T", "Tot_FT", "Tot_1T"
    ]
    vis = df[c_v].rename(columns={
        "Visitante": "Equipo", "Goles_Vis": "Goles_FT",
        "xG_Vis": "xG_FT", "Goles_Loc": "Goles_Contra_FT",
        "xG_Loc": "xG_Contra_FT", "Goles_Vis_1T": "Goles_1T",
        "xG_Vis_1T": "xG_1T",
        "Goles_Loc_1T": "Goles_Contra_1T",
        "xG_Loc_1T": "xG_Contra_1T"
    })

    hist = pd.concat([loc, vis]).dropna(subset=["Equipo"])
    hist = hist[hist["Equipo"] != ""].sort_values(
        by=["Equipo", col_f]
    )

    cnt = hist.groupby("Equipo").size()
    rob = cnt[cnt >= MIN_PJ].index
    hist = hist[hist["Equipo"].isin(rob)].copy()

    metricas = [
        "Goles_FT", "xG_FT", "Goles_Contra_FT",
        "xG_Contra_FT", "Goles_1T", "xG_1T",
        "Goles_Contra_1T", "xG_Contra_1T"
    ]
    for m in metricas:
        hist["EMA_" + m] = hist.groupby(
            "Equipo"
        )[m].transform(
            lambda x: x.ewm(span=8, adjust=False).mean()
        )

    ea = hist.groupby("Equipo").tail(1).copy()
    ea["Deuda_FT"] = ea["EMA_Goles_FT"] - ea["EMA_xG_FT"]
    ea["Deuda_1T"] = ea["EMA_Goles_1T"] - ea["EMA_xG_1T"]
    return hist, ea

DF_RAW, COL_FECHA = cargar_datos()
HISTORIAL, ESTADO_ACTUAL = procesar_historial(DF_RAW, COL_FECHA)
LEAGUES = ["Todas"] + sorted(
    ESTADO_ACTUAL["Liga"].dropna().unique().tolist()
)
TEAMS = sorted(
    ESTADO_ACTUAL["Equipo"].dropna().unique().tolist()
)

st.title("WINDEP OS - Suite Cuantitativa")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Sequías FT", 
    "⏱️ 0-0 al Descanso", 
    "🏆 Radar Combinadas", 
    "🎲 Calculadora 1X2"
])

# --------------------------------------------------------
# PESTAÑA 1: SEQUÍAS FT Y POST 0-0
# --------------------------------------------------------
with tab1:
    st.subheader("Análisis de Sequías Full Time y Post 0-0")
    f_m1 = st.selectbox(
        "Filtro de Bloques:",
        ["Todos los Bloques", "🔴 Solo Bloque Rojo", "🟠 Solo Bloque Naranja", "🟡 Solo Bloque Amarillo"],
        key="filtro_m1"
    )
    
    if st.button("Escanear Sequías FT", type="primary"):
        rojo, naranja, amarillo = [], [], []

        for eq, grp in HISTORIAL.groupby("Equipo"):
            lg = grp["Liga"].iloc[-1]
            racha = 0
            xg_acum = 0.0
            p00 = (grp["Tot_FT"].iloc[-1] == 0)

            for _, f in grp.iloc[::-1].iterrows():
                if f["Goles_FT"] == 0:
                    racha += 1
                    xg_acum += float(f["xG_FT"])
                else:
                    break

            if racha >= 1 or p00:
                item = {
                    "Equipo": eq.upper(), "Liga": lg,
                    "PJ_Blanco": racha, "xG_Acum": xg_acum,
                    "Vino_0-0": "SÍ" if p00 else "NO"
                }
                c_r = (racha >= 3 or (racha >= 2 and xg_acum >= 2.0))
                c_n = racha >= 2 or (p00 and xg_acum >= 1.0)
                if c_r:
                    item["Bloque"] = "🔴 ROJO"
                    rojo.append(item)
                elif c_n:
                    item["Bloque"] = "🟠 NARANJA"
                    naranja.append(item)
                else:
                    item["Bloque"] = "🟡 AMARILLO"
                    amarillo.append(item)

        sort_k = lambda x: (x["PJ_Blanco"], x["xG_Acum"])
        rojo = sorted(rojo, key=sort_k, reverse=True)
        naranja = sorted(naranja, key=sort_k, reverse=True)
        amarillo = sorted(amarillo, key=sort_k, reverse=True)

        if f_m1 == "🔴 Solo Bloque Rojo":
            datos = rojo
        elif f_m1 == "🟠 Solo Bloque Naranja":
            datos = naranja
        elif f_m1 == "🟡 Solo Bloque Amarillo":
            datos = amarillo
        else:
            datos = rojo + naranja + amarillo

        df_out = pd.DataFrame(datos)
        if df_out.empty:
            st.info("Sin equipos para el filtro seleccionado.")
        else:
            st.markdown(f"**🔴 Rojo: {len(rojo)}** | **🟠 Naranja: {len(naranja)}** | **🟡 Amarillo: {len(amarillo)}** | Total: **{len(datos)}**")
            st.dataframe(df_out.style.format({"xG_Acum": "{:.2f}"}), use_container_width=True)

# --------------------------------------------------------
# PESTAÑA 2: 0-0 AL DESCANSO (HT)
# --------------------------------------------------------
with tab2:
    st.subheader("Escáner de Sequías al Descanso (0-0 HT)")
    f_m2 = st.selectbox(
        "Filtro Bloques HT:",
        ["Todos los Bloques HT", "🔴 Solo Bloque Rojo HT", "🟠 Solo Bloque Naranja HT", "🟡 Solo Bloque Amarillo HT"],
        key="filtro_m2"
    )
    
    if st.button("Escanear 0-0 al Descanso", type="secondary"):
        rojo_ht, naranja_ht, amarillo_ht = [], [], []

        for eq in TEAMS:
            c_eq = ((DF_RAW["Local"] == eq) | (DF_RAW["Visitante"] == eq))
            p_eq = DF_RAW[c_eq]
            if len(p_eq) == 0:
                continue

            lg = p_eq["Liga"].iloc[-1]
            racha_ht = 0
            xg_ht_acum = 0.0

            for _, f in p_eq.iloc[::-1].iterrows():
                g_ht = f["Goles_Loc_1T"] + f["Goles_Vis_1T"]
                x_ht = f["xG_Loc_1T"] + f["xG_Vis_1T"]
                if g_ht == 0:
                    racha_ht += 1
                    xg_ht_acum += float(x_ht)
                else:
                    break

            if racha_ht >= 1:
                item = {
                    "Equipo": eq.upper(), "Liga": lg,
                    "Racha_00_HT": racha_ht,
                    "xG_HT_Acum": xg_ht_acum
                }
                c_r = (racha_ht >= 3 and xg_ht_acum >= 2.0)
                if c_r:
                    item["Bloque"] = "🔴 ROJO"
                    rojo_ht.append(item)
                elif racha_ht >= 2:
                    item["Bloque"] = "🟠 NARANJA"
                    naranja_ht.append(item)
                else:
                    item["Bloque"] = "🟡 AMARILLO"
                    amarillo_ht.append(item)

        sort_k = lambda x: (x["Racha_00_HT"], x["xG_HT_Acum"])
        rojo_ht = sorted(rojo_ht, key=sort_k, reverse=True)
        naranja_ht = sorted(naranja_ht, key=sort_k, reverse=True)
        amarillo_ht = sorted(amarillo_ht, key=sort_k, reverse=True)

        if f_m2 == "🔴 Solo Bloque Rojo HT":
            datos = rojo_ht
        elif f_m2 == "🟠 Solo Bloque Naranja HT":
            datos = naranja_ht
        elif f_m2 == "🟡 Solo Bloque Amarillo HT":
            datos = amarillo_ht
        else:
            datos = rojo_ht + naranja_ht + amarillo_ht

        df_out = pd.DataFrame(datos)
        if df_out.empty:
            st.info("Sin rachas 0-0 HT para el filtro.")
        else:
            st.markdown(f"**🔴 Rojo HT: {len(rojo_ht)}** | **🟠 Naranja HT: {len(naranja_ht)}** | **🟡 Amarillo HT: {len(amarillo_ht)}**")
            st.dataframe(df_out.style.format({"xG_HT_Acum": "{:.2f}"}), use_container_width=True)

# --------------------------------------------------------
# PESTAÑA 3: RADAR DE COMBINADAS MULTI-MERCADO
# --------------------------------------------------------
with tab3:
    st.subheader("Radar de Combinadas y Mercados Múltiples")
    col1, col2 = st.columns(2)
    with col1:
        w_liga_m3 = st.selectbox("Liga:", LEAGUES, key="liga_m3")
    with col2:
        w_perfil_m3 = st.selectbox(
            "Estrategia:",
            ["1. Conservador (1.20 - 1.55)", "2. Valor Medio (1.60 - 2.10)", "3. Cuotas Altas (2.15 - 3.50+)", "4. Rebote +EV (xG > Goles)", "5. Top 25 Absoluto (>70%)"],
            key="perfil_m3"
        )
        
    if st.button("Generar Radar Combinadas", type="primary"):
        with st.spinner("Simulando 10,000 universos Monte Carlo..."):
            np.random.seed(42)
            base = ESTADO_ACTUAL if w_liga_m3 == "Todas" else ESTADO_ACTUAL[ESTADO_ACTUAL["Liga"] == w_liga_m3]
            resultados = []
            reps = REPS_RADAR

            for _, row in base.iterrows():
                eq = str(row["Equipo"]).strip()
                if eq.lower() in ["nan", "none", ""]:
                    continue
                lg = row["Liga"]
                deuda = float(row["Deuda_FT"])

                if deuda <= -0.25:
                    reg_tag = "🟢 REBOTE +EV"
                elif deuda >= 0.25:
                    reg_tag = "⚠️ BURBUJA"
                else:
                    reg_tag = "🔵 ESTABLE"

                l_1t_f = max(0.01, float(row["EMA_xG_1T"]))
                l_1t_c = max(0.01, float(row["EMA_xG_Contra_1T"]))
                l_ft_f = max(0.01, float(row["EMA_xG_FT"]))
                l_ft_c = max(0.01, float(row["EMA_xG_Contra_FT"]))

                s_1t_f = np.random.poisson(l_1t_f, reps)
                s_1t_c = np.random.poisson(l_1t_c, reps)
                s_ft_f = np.random.poisson(l_ft_f, reps)
                s_ft_c = np.random.poisson(l_ft_c, reps)

                s_1t_tot = s_1t_f + s_1t_c
                s_ft_tot = s_ft_f + s_ft_c

                p_05_ht_p = (s_1t_tot > 0).mean() * 100.0
                p_05_ht_e = (s_1t_f > 0).mean() * 100.0
                p_15_ht_p = (s_1t_tot > 1).mean() * 100.0
                p_15_ft_p = (s_ft_tot > 1).mean() * 100.0
                p_25_ft_p = (s_ft_tot > 2).mean() * 100.0
                p_35_ft_p = (s_ft_tot > 3).mean() * 100.0
                p_u25_p = (s_ft_tot <= 2).mean() * 100.0
                p_g_ft_e = (s_ft_f > 0).mean() * 100.0
                p_eq15_e = (s_ft_f > 1).mean() * 100.0
                p_btts = ((s_ft_f > 0) & (s_ft_c > 0)).mean() * 100.0
                p_bttsn = ((s_ft_f == 0) | (s_ft_c == 0)).mean() * 100.0
                p_win_e = (s_ft_f > s_ft_c).mean() * 100.0
                p_1x_e = (s_ft_f >= s_ft_c).mean() * 100.0

                sc_mat = np.column_stack((s_ft_f, s_ft_c))
                u_sc, cnts = np.unique(sc_mat, axis=0, return_counts=True)
                b_idx = np.argmax(cnts)
                m_ex = f"{u_sc[b_idx][0]}-{u_sc[b_idx][1]}"
