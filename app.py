import math
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

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
# 1. CARGA Y PREPROCESAMIENTO CON EXTRACCIÓN LIMPIA
# --------------------------------------------------------
@st.cache_data(ttl=600)
def cargar_datos_maestros():
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

    # Construcción segura del historial sin conflictos de índices
    registros = []
    for _, row in df.iterrows():
        fec = row[col_f] if col_f in df.columns else pd.NaT
        lig = row["Liga"] if "Liga" in df.columns else "General"
        
        # Partido como Local
        registros.append({
            "Fecha": fec, "Liga": lig, "Equipo": str(row["Local"]).strip(),
            "Goles_FT": float(row["Goles_Loc"]), "xG_FT": float(row["xG_Loc"]),
            "Goles_Contra_FT": float(row["Goles_Vis"]), "xG_Contra_FT": float(row["xG_Vis"]),
            "Goles_1T": float(row["Goles_Loc_1T"]), "xG_1T": float(row["xG_Loc_1T"]),
            "Goles_Contra_1T": float(row["Goles_Vis_1T"]), "xG_Contra_1T": float(row["xG_Vis_1T"]),
            "Tot_FT": float(row["Tot_FT"]), "Tot_1T": float(row["Tot_1T"])
        })
        # Partido como Visitante
        registros.append({
            "Fecha": fec, "Liga": lig, "Equipo": str(row["Visitante"]).strip(),
            "Goles_FT": float(row["Goles_Vis"]), "xG_FT": float(row["xG_Vis"]),
            "Goles_Contra_FT": float(row["Goles_Loc"]), "xG_Contra_FT": float(row["xG_Loc"]),
            "Goles_1T": float(row["Goles_Vis_1T"]), "xG_1T": float(row["xG_Vis_1T"]),
            "Goles_Contra_1T": float(row["Goles_Loc_1T"]), "xG_Contra_1T": float(row["xG_Loc_1T"]),
            "Tot_FT": float(row["Tot_FT"]), "Tot_1T": float(row["Tot_1T"])
        })

    hist = pd.DataFrame(registros).dropna(subset=["Equipo"])
    hist = hist[(hist["Equipo"] != "") & (hist["Equipo"].str.lower() != "nan")].sort_values(
        by=["Equipo", "Fecha"]
    ).reset_index(drop=True)

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
    return df, col_f, hist, ea

DF_RAW, COL_FECHA, HISTORIAL, ESTADO_ACTUAL = cargar_datos_maestros()
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
# PESTAÑA 1: SEQUÍAS FT
# --------------------------------------------------------
with tab1:
    st.subheader("Análisis de Sequías Full Time y Post 0-0")
    f_m1 = st.selectbox(
        "Filtro de Bloques:",
        ["Todos los Bloques", "🔴 Solo Bloque Rojo", "🟠 Solo Bloque Naranja", "🟡 Solo Bloque Amarillo"],
        key="filtro_m1"
    )
    
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
    st.markdown(f"**🔴 Rojo: {len(rojo)}** | **🟠 Naranja: {len(naranja)}** | **🟡 Amarillo: {len(amarillo)}** | Total: **{len(datos)}**")
    if df_out.empty:
        st.info("Sin equipos para el filtro seleccionado.")
    else:
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

    rojo_ht = sorted(rojo_ht, key=lambda x: (x["Racha_00_HT"], x["xG_HT_Acum"]), reverse=True)
    naranja_ht = sorted(naranja_ht, key=lambda x: (x["Racha_00_HT"], x["xG_HT_Acum"]), reverse=True)
    amarillo_ht = sorted(amarillo_ht, key=lambda x: (x["Racha_00_HT"], x["xG_HT_Acum"]), reverse=True)

    if f_m2 == "🔴 Solo Bloque Rojo HT":
        datos_ht = rojo_ht
    elif f_m2 == "🟠 Solo Bloque Naranja HT":
        datos_ht = naranja_ht
    elif f_m2 == "🟡 Solo Bloque Amarillo HT":
        datos_ht = amarillo_ht
    else:
        datos_ht = rojo_ht + naranja_ht + amarillo_ht

    df_out_ht = pd.DataFrame(datos_ht)
    st.markdown(f"**🔴 Rojo HT: {len(rojo_ht)}** | **🟠 Naranja HT: {len(naranja_ht)}** | **🟡 Amarillo HT: {len(amarillo_ht)}**")
    if df_out_ht.empty:
        st.info("Sin rachas 0-0 HT para el filtro.")
    else:
        st.dataframe(df_out_ht.style.format({"xG_HT_Acum": "{:.2f}"}), use_container_width=True)

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
        with st.spinner("⏳ Simulando 10,000 universos Monte Carlo... Por favor espere."):
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
                p_m_ex = (cnts[b_idx] / reps) * 100.0

                mkts = [
                    ("+0.5 HT (PARTIDO)", p_05_ht_p), ("GOL HT (EQUIPO)", p_05_ht_e),
                    ("+1.5 HT (PARTIDO)", p_15_ht_p), ("+1.5 FT (PARTIDO)", p_15_ft_p),
                    ("+2.5 FT (PARTIDO)", p_25_ft_p), ("+3.5 FT (PARTIDO)", p_35_ft_p),
                    ("-2.5 FT (UNDER)", p_u25_p), ("GOL FT (EQUIPO)", p_g_ft_e),
                    ("EQUIPO +1.5 FT", p_eq15_e), ("AMBOS MARCAN", p_btts),
                    ("AMBOS NO MARCAN", p_bttsn), ("VICTORIA (1X2)", p_win_e),
                    ("DOBLE OP. 1X", p_1x_e), (f"MARCADOR: {m_ex}", p_m_ex)
                ]

                for m_nom, prob in mkts:
                    if prob > 0:
                        cj = 100.0 / prob
                        resultados.append({
                            "Equipo": eq.upper(), "Liga": lg,
                            "Mercado": m_nom, "Certeza_%": prob,
                            "Cuota_Min": cj, "Regresion": reg_tag,
                            "Deuda_xG": deuda
                        })

            df_res = pd.DataFrame(resultados)
            if df_res.empty:
                st.info("Sin oportunidades encontradas.")
            else:
                if "Conservador" in w_perfil_m3:
                    cond = (df_res["Cuota_Min"] >= 1.20) & (df_res["Cuota_Min"] <= 1.55)
                    top = df_res[cond].sort_values(by="Certeza_%", ascending=False)
                elif "Medio" in w_perfil_m3:
                    cond = (df_res["Cuota_Min"] >= 1.60) & (df_res["Cuota_Min"] <= 2.10)
                    top = df_res[cond].sort_values(by="Certeza_%", ascending=False)
                elif "Altas" in w_perfil_m3:
                    cond = (df_res["Cuota_Min"] >= 2.15) & (df_res["Cuota_Min"] <= 3.60)
                    top = df_res[cond].sort_values(by="Certeza_%", ascending=False)
                elif "Rebote" in w_perfil_m3:
                    m_of = ["GOL FT (EQUIPO)", "EQUIPO +1.5 FT", "GOL HT (EQUIPO)", "+0.5 HT (PARTIDO)", "+1.5 FT (PARTIDO)", "+2.5 FT (PARTIDO)", "VICTORIA (1X2)", "DOBLE OP. 1X"]
                    cond = (df_res["Regresion"] == "🟢 REBOTE +EV") & (df_res["Certeza_%"] >= 50.0) & (df_res["Mercado"].isin(m_of))
                    top = df_res[cond].sort_values(by=["Deuda_xG", "Certeza_%"], ascending=[True, False])
                else:
                    cond = df_res["Certeza_%"] > 70.0
                    top = df_res[cond].sort_values(by="Certeza_%", ascending=False)

                top25 = top.drop_duplicates(subset=["Equipo"]).head(25).reset_index(drop=True)
                top25.index = top25.index + 1
                st.dataframe(top25.style.format({"Certeza_%": "{:.1f}%", "Cuota_Min": "{:.2f}", "Deuda_xG": "{:+.2f}"}), use_container_width=True)

# --------------------------------------------------------
# PESTAÑA 4: CALCULADORA 1X2 Y MARCADORES EXACTOS
# --------------------------------------------------------
with tab4:
    st.subheader("Calculadora 1X2, Deudas, Rachas y Marcadores Exactos")
    col1, col2 = st.columns(2)
    with col1:
        w_home_1x2 = st.selectbox("Local:", TEAMS, key="home_1x2")
    with col2:
        w_away_1x2 = st.selectbox("Visitante:", TEAMS, key="away_1x2")
        
    if st.button("Calcular Diagnóstico 1X2", type="primary"):
        if w_home_1x2 == w_away_1x2:
            st.error("Elige dos equipos diferentes.")
        else:
            e1 = ESTADO_ACTUAL[ESTADO_ACTUAL["Equipo"] == w_home_1x2].iloc[0]
            e2 = ESTADO_ACTUAL[ESTADO_ACTUAL["Equipo"] == w_away_1x2].iloc[0]
            lg = e1["Liga"]

            def calcular_racha_equipo(eq_nombre):
                h_eq = HISTORIAL[HISTORIAL["Equipo"] == eq_nombre]
                if h_eq.empty:
                    return 0, 0
                racha_ft, racha_ht = 0, 0
                for _, f in h_eq.iloc[::-1].iterrows():
                    if f["Goles_FT"] == 0: racha_ft += 1
                    else: break
                for _, f in h_eq.iloc[::-1].iterrows():
                    if f["Goles_1T"] == 0: racha_ht += 1
                    else: break
                return racha_ft, racha_ht

            racha_l_ft, racha_l_ht = calcular_racha_equipo(w_home_1x2)
            racha_v_ft, racha_v_ht = calcular_racha_equipo(w_away_1x2)

            deuda_l_ft = float(e1["Deuda_FT"])
            deuda_l_ht = float(e1["Deuda_1T"])
            deuda_v_ft = float(e2["Deuda_FT"])
            deuda_v_ht = float(e2["Deuda_1T"])

            p_lg = DF_RAW[DF_RAW["Liga"] == lg]
            ha_f = max(1.05, min(1.60, p_lg["xG_Loc"].mean() / max(0.5, p_lg["xG_Vis"].mean()))) if len(p_lg) >= 15 else 1.25

            h_m = DF_RAW[DF_RAW["Local"] == w_home_1x2]
            a_m = DF_RAW[DF_RAW["Visitante"] == w_away_1x2]

            xg_f_h = h_m["xG_Loc"].mean() if len(h_m) >= 3 else e1["EMA_xG_FT"]
            xg_c_h = h_m["xG_Vis"].mean() if len(h_m) >= 3 else e1["EMA_xG_Contra_FT"]
            xg_f_a = a_m["xG_Vis"].mean() if len(a_m) >= 3 else e2["EMA_xG_FT"]
            xg_c_a = a_m["xG_Loc"].mean() if len(a_m) >= 3 else e2["EMA_xG_Contra_FT"]

            xg_el = 0.65 * xg_f_h + 0.35 * e1["EMA_xG_FT"]
            xg_evc = 0.65 * xg_c_a + 0.35 * e2["EMA_xG_Contra_FT"]
            xg_ev = 0.65 * xg_f_a + 0.35 * e2["EMA_xG_FT"]
            xg_elc = 0.65 * xg_c_h + 0.35 * e1["EMA_xG_Contra_FT"]

            sq_ha = math.sqrt(ha_f)
            lam_l = max(0.10, ((xg_el + xg_evc) / 2.0) * sq_ha)
            lam_v = max(0.10, ((xg_ev + xg_elc) / 2.0) / sq_ha)
            lam_tot = lam_l + lam_v

            p_1, p_x, p_2 = 0.0, 0.0, 0.0
            scores = []
            for i in range(7):
                pi = (math.exp(-lam_l) * (lam_l**i)) / math.factorial(i)
                for j in range(7):
                    pj = (math.exp(-lam_v) * (lam_v**j)) / math.factorial(j)
                    p_ij = pi * pj
                    scores.append((f"{i}-{j}", p_ij))
                    if i > j: p_1 += p_ij
                    elif i == j: p_x += p_ij
                    else: p_2 += p_ij

            scores.sort(key=lambda x: x[1], reverse=True)
            top5 = scores[:5]

            p_00 = math.exp(-lam_tot) * 100.0
            p_anti00 = 100.0 - p_00

            st.markdown(f"### 🏟️ {w_home_1x2} vs {w_away_1x2} ({lg})")
            st.write(f"Goles Proyectados: **{w_home_1x2} ({lam_l:.2f})** - **{w_away_1x2} ({lam_v:.2f})** | Total: **{lam_tot:.2f}**")
            
            st.info(
                f"**📊 ANÁLISIS DE DEUDAS Y RACHAS SIN MARCAR:**\n\n"
                f"• **{w_home_1x2}:** Deuda FT: {deuda_l_ft:+.2f} | Deuda 1T: {deuda_l_ht:+.2f} | Racha sin marcar FT: **{racha_l_ft} PJ** | Racha sin marcar 1T: **{racha_l_ht} PJ**\n\n"
                f"• **{w_away_1x2}:** Deuda FT: {deuda_v_ft:+.2f} | Deuda 1T: {deuda_v_ht:+.2f} | Racha sin marcar FT: **{racha_v_ft} PJ** | Racha sin marcar 1T: **{racha_v_ht} PJ**"
            )

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("1 (Local)", f"{p_1*100:.1f}%", f"Cuota: {1.0/p_1:.2f}")
            with col_res2:
                st.metric("X (Empate)", f"{p_x*100:.1f}%", f"Cuota: {1.0/p_x:.2f}")
            with col_res3:
                st.metric("2 (Visita)", f"{p_2*100:.1f}%", f"Cuota: {1.0/p_2:.2f}")

            st.markdown("#### Top Marcadores Exactos:")
            cols_sc = st.columns(5)
            for idx, (sc, p) in enumerate(top5):
                with cols_sc[idx]:
                    st.metric(f"Marcador {sc}", f"{p*100:.1f}%", f"Cuota: {1/p:.2f}")

            st.success(f"⚡ **SEÑAL ANTI 0-0 ({p_anti00:.1f}%):** Riesgo 0-0 bajo. Ideal para seguimiento en directo.")
