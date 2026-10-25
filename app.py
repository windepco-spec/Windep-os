# WINDEP 360 PRO — Adaptación inicial Streamlit
# Motor de datos y Poisson extraídos del archivo original SUITE_PREMIUM.
# No modifica el cuaderno de Colab.

import math
import unicodedata
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

SHEET_ID = "1wXHz_CaJ7rx-e2wha3lWwQKpV1rcVjlQfKiliHLAPBU"
URL_SHEET = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
    f"/gviz/tq?tqx=out:csv&sheet=GOLES"
)
MIN_PJ = 6
REPS_RADAR = 10000

def _norm_k(s):
    return "".join(
        ch for ch in unicodedata.normalize("NFD", str(s))
        if unicodedata.category(ch) != "Mn"
    ).strip().lower()

def cargar_datos():
    try:
        df = pd.read_csv(URL_SHEET)
    except Exception as exc:
        url_alt = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv"
        try:
            df = pd.read_csv(url_alt)
        except Exception:
            raise RuntimeError(
                "No se pudo leer la hoja GOLES de Google Sheets. "
                "Verifica que la hoja sea accesible mediante enlace."
            ) from exc

    df = df.dropna(subset=["Local", "Visitante"]).copy()
    c_l = df["Local"].astype(str).str.lower()
    c_v = df["Visitante"].astype(str).str.lower()
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
                df[c] = df[c].astype(str).str.replace(",", ".").str.strip()
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in ["Liga", "Local", "Visitante"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    n_todos = df["Local"].tolist() + df["Visitante"].tolist()
    n_freq = {}
    for nom in n_todos:
        n_freq[nom] = n_freq.get(nom, 0) + 1
    n_map = {}
    orden = sorted(n_freq.keys(), key=lambda x: n_freq[x], reverse=True)
    for nom in orden:
        k = _norm_k(nom)
        if k not in n_map:
            n_map[k] = nom
    df["Local"] = df["Local"].apply(lambda x: n_map.get(_norm_k(x), x))
    df["Visitante"] = df["Visitante"].apply(lambda x: n_map.get(_norm_k(x), x))

    col_f = "Fecha" if "Fecha" in df.columns else "FECHA"
    if col_f in df.columns:
        df[col_f] = pd.to_datetime(df[col_f], dayfirst=True, errors="coerce")
        df = df.sort_values(by=col_f).reset_index(drop=True)

    df["Tot_FT"] = df["Goles_Loc"] + df["Goles_Vis"]
    df["Tot_1T"] = df["Goles_Loc_1T"] + df["Goles_Vis_1T"]
    return df, col_f

def procesar_historial_multi_span(df, col_f):
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
    hist = hist[hist["Equipo"] != ""].sort_values(by=["Equipo", col_f])
    cnt = hist.groupby("Equipo").size()
    rob = cnt[cnt >= MIN_PJ].index
    hist = hist[hist["Equipo"].isin(rob)].copy()

    metricas = [
        "Goles_FT", "xG_FT", "Goles_Contra_FT",
        "xG_Contra_FT", "Goles_1T", "xG_1T",
        "Goles_Contra_1T", "xG_Contra_1T"
    ]

    cache_estado = {}
    for sp in [6, 8, 12]:
        for m in metricas:
            hist[f"EMA_{m}"] = hist.groupby("Equipo")[m].transform(
                lambda x: x.ewm(span=sp, adjust=False).mean()
            )
        ea = hist.groupby("Equipo").tail(1).copy()
        xg_1t_eff = np.where(
            ea["EMA_xG_1T"] > 0.01,
            ea["EMA_xG_1T"],
            ea["EMA_xG_FT"] * 0.445
        )
        xg_c_1t_eff = np.where(
            ea["EMA_xG_Contra_1T"] > 0.01,
            ea["EMA_xG_Contra_1T"],
            ea["EMA_xG_Contra_FT"] * 0.445
        )
        ea["EMA_xG_1T_Eff"] = xg_1t_eff
        ea["EMA_xG_Contra_1T_Eff"] = xg_c_1t_eff
        ea["Deuda_FT"] = ea["EMA_Goles_FT"] - ea["EMA_xG_FT"]
        ea["Deuda_1T"] = np.where(
            ea["EMA_xG_1T"] > 0.01,
            ea["EMA_Goles_1T"] - ea["EMA_xG_1T"],
            ea["EMA_Goles_1T"] - (ea["EMA_xG_FT"] * 0.445)
        )
        cache_estado[sp] = ea

    # Restaurar métricas de span=8 en HISTORIAL para módulos 1-4 y 6
    for m in metricas:
        hist[f"EMA_{m}"] = hist.groupby("Equipo")[m].transform(
            lambda x: x.ewm(span=8, adjust=False).mean()
        )

    return hist, cache_estado


@st.cache_data(ttl=300, show_spinner="Cargando y procesando el historial Windep...")
def cargar_motor():
    df, col_fecha = cargar_datos()
    historial, cache_estado = procesar_historial_multi_span(df, col_fecha)
    return df, historial, cache_estado

def calcular_poisson_partido(eq_l, eq_v, span=8, hibrido=True, mult_lambda=1.0):
    estado_df = CACHE_ESTADO_ACTUAL.get(span, ESTADO_ACTUAL)
    e1 = estado_df[estado_df["Equipo"] == eq_l].iloc[0]
    e2 = estado_df[estado_df["Equipo"] == eq_v].iloc[0]
    lg = e1["Liga"]

    p_lg = DF_RAW[DF_RAW["Liga"] == lg]
    if len(p_lg) >= 15:
        ha_r = p_lg["xG_Loc"].mean() / max(0.5, p_lg["xG_Vis"].mean())
        ha_f = max(1.05, min(1.60, ha_r))
    else:
        ha_f = 1.25

    h_m = DF_RAW[DF_RAW["Local"] == eq_l]
    a_m = DF_RAW[DF_RAW["Visitante"] == eq_v]

    xg_f_h = h_m["xG_Loc"].mean() if len(h_m) >= 3 else float(e1["EMA_xG_FT"])
    g_f_h = h_m["Goles_Loc"].mean() if len(h_m) >= 3 else float(e1["EMA_Goles_FT"])
    xg_c_h = h_m["xG_Vis"].mean() if len(h_m) >= 3 else float(e1["EMA_xG_Contra_FT"])
    g_c_h = h_m["Goles_Vis"].mean() if len(h_m) >= 3 else float(e1["EMA_Goles_Contra_FT"])
    xg_f_a = a_m["xG_Vis"].mean() if len(a_m) >= 3 else float(e2["EMA_xG_FT"])
    g_f_a = a_m["Goles_Vis"].mean() if len(a_m) >= 3 else float(e2["EMA_Goles_FT"])
    xg_c_a = a_m["xG_Loc"].mean() if len(a_m) >= 3 else float(e2["EMA_xG_Contra_FT"])
    g_c_a = a_m["Goles_Loc"].mean() if len(a_m) >= 3 else float(e2["EMA_Goles_Contra_FT"])

    if hibrido:
        hyb_f_h = 0.70 * xg_f_h + 0.30 * g_f_h
        hyb_c_h = 0.70 * xg_c_h + 0.30 * g_c_h
        hyb_f_a = 0.70 * xg_f_a + 0.30 * g_f_a
        hyb_c_a = 0.70 * xg_c_a + 0.30 * g_c_a

        hyb_ema_el = 0.70 * float(e1["EMA_xG_FT"]) + 0.30 * float(e1["EMA_Goles_FT"])
        hyb_ema_evc = 0.70 * float(e2["EMA_xG_Contra_FT"]) + 0.30 * float(e2["EMA_Goles_Contra_FT"])
        hyb_ema_ev = 0.70 * float(e2["EMA_xG_FT"]) + 0.30 * float(e2["EMA_Goles_FT"])
        hyb_ema_elc = 0.70 * float(e1["EMA_xG_Contra_FT"]) + 0.30 * float(e1["EMA_Goles_Contra_FT"])
    else:
        hyb_f_h = xg_f_h
        hyb_c_h = xg_c_h
        hyb_f_a = xg_f_a
        hyb_c_a = xg_c_a

        hyb_ema_el = float(e1["EMA_xG_FT"])
        hyb_ema_evc = float(e2["EMA_xG_Contra_FT"])
        hyb_ema_ev = float(e2["EMA_xG_FT"])
        hyb_ema_elc = float(e1["EMA_xG_Contra_FT"])

    xg_el = 0.65 * hyb_f_h + 0.35 * hyb_ema_el
    xg_evc = 0.65 * hyb_c_a + 0.35 * hyb_ema_evc
    xg_ev = 0.65 * hyb_f_a + 0.35 * hyb_ema_ev
    xg_elc = 0.65 * hyb_c_h + 0.35 * hyb_ema_elc

    sq_ha = math.sqrt(ha_f)
    lam_l = max(0.10, ((xg_el + xg_evc) / 2.0) * sq_ha) * mult_lambda
    lam_v = max(0.10, ((xg_ev + xg_elc) / 2.0) / sq_ha) * mult_lambda
    lam_tot = lam_l + lam_v

    p_1, p_x, p_2 = 0.0, 0.0, 0.0
    scores = []
    matriz_5x5 = np.zeros((5, 5))
    for i in range(7):
        pi = math.exp(-lam_l) * (lam_l**i) / math.factorial(i)
        for j in range(7):
            pj = math.exp(-lam_v) * (lam_v**j) / math.factorial(j)
            p_ij = pi * pj
            scores.append((f"{i}-{j}", p_ij))
            if i < 5 and j < 5:
                matriz_5x5[i, j] = p_ij * 100.0
            if i > j:
                p_1 += p_ij
            elif i == j:
                p_x += p_ij
            else:
                p_2 += p_ij

    c_1 = 1.0 / p_1 if p_1 > 0 else 99.0
    c_x = 1.0 / p_x if p_x > 0 else 99.0
    c_2 = 1.0 / p_2 if p_2 > 0 else 99.0

    scores.sort(key=lambda x: x[1], reverse=True)
    top5 = scores[:5]

    p_gl = (1.0 - math.exp(-lam_l)) * 100.0
    p_gv = (1.0 - math.exp(-lam_v)) * 100.0
    p_gl_15 = (1.0 - math.exp(-lam_l) * (1.0 + lam_l)) * 100.0
    p_gv_15 = (1.0 - math.exp(-lam_v) * (1.0 + lam_v)) * 100.0

    lam_l_2t = lam_l * 0.555
    lam_v_2t = lam_v * 0.555
    p_gl_2t = (1.0 - math.exp(-lam_l_2t)) * 100.0
    p_gv_2t = (1.0 - math.exp(-lam_v_2t)) * 100.0

    p_05_ft = (1.0 - math.exp(-lam_tot)) * 100.0
    p_15_ft = (1.0 - math.exp(-lam_tot) * (1.0 + lam_tot)) * 100.0
    p_25_ft = (1.0 - math.exp(-lam_tot) * (1.0 + lam_tot + (lam_tot**2) / 2.0)) * 100.0
    p_35_ft = (1.0 - math.exp(-lam_tot) * (1.0 + lam_tot + (lam_tot**2) / 2.0 + (lam_tot**3) / 6.0)) * 100.0
    p_u25_ft = 100.0 - p_25_ft
    p_u35_ft = 100.0 - p_35_ft

    p_btts = ((1.0 - math.exp(-lam_l)) * (1.0 - math.exp(-lam_v))) * 100.0
    p_btts_no = 100.0 - p_btts

    lam_1t = lam_tot * 0.445
    lam_2t = lam_tot * 0.555
    p_05_ht = (1.0 - math.exp(-lam_1t)) * 100.0
    p_05_2t = (1.0 - math.exp(-lam_2t)) * 100.0

    p_1x = (p_1 + p_x) * 100.0
    p_x2 = (p_2 + p_x) * 100.0

    p_sin_emp = p_1 + p_2
    p_dnb_l = (p_1 / p_sin_emp) * 100.0 if p_sin_emp > 0 else 50.0
    p_dnb_v = (p_2 / p_sin_emp) * 100.0 if p_sin_emp > 0 else 50.0

    min_max_1t_150 = max(0, int(45.0 - (45.0 * 1.0986 / lam_1t))) if lam_1t > 1.0986 else 0
    min_max_2t_150 = max(45, int(90.0 - (45.0 * 1.0986 / lam_2t))) if lam_2t > 1.0986 else 45

    cj = lambda p: round(100.0 / p, 2) if p > 0.5 else 99.0

    return {
        "e1": e1, "e2": e2, "lg": lg, "ha_f": ha_f,
        "lam_l": lam_l, "lam_v": lam_v, "lam_tot": lam_tot,
        "lam_1t": lam_1t, "lam_2t": lam_2t,
        "p_1": p_1, "p_x": p_x, "p_2": p_2,
        "c_1": c_1, "c_x": c_x, "c_2": c_2,
        "p_1x": round(p_1x, 1), "c_1x": cj(p_1x),
        "p_x2": round(p_x2, 1), "c_x2": cj(p_x2),
        "p_dnb_l": round(p_dnb_l, 1), "c_dnb_l": cj(p_dnb_l),
        "p_dnb_v": round(p_dnb_v, 1), "c_dnb_v": cj(p_dnb_v),
        "p_gl": round(p_gl, 1), "c_gl": cj(p_gl),
        "p_gv": round(p_gv, 1), "c_gv": cj(p_gv),
        "p_gl_15": round(p_gl_15, 1), "c_gl_15": cj(p_gl_15),
        "p_gv_15": round(p_gv_15, 1), "c_gv_15": cj(p_gv_15),
        "p_gl_2t": round(p_gl_2t, 1), "c_gl_2t": cj(p_gl_2t),
        "p_gv_2t": round(p_gv_2t, 1), "c_gv_2t": cj(p_gv_2t),
        "p_05_ft": round(p_05_ft, 1), "c_05_ft": cj(p_05_ft),
        "p_15_ft": round(p_15_ft, 1), "c_15_ft": cj(p_15_ft),
        "p_25_ft": round(p_25_ft, 1), "c_25_ft": cj(p_25_ft),
        "p_35_ft": round(p_35_ft, 1), "c_35_ft": cj(p_35_ft),
        "p_u25_ft": round(p_u25_ft, 1), "c_u25_ft": cj(p_u25_ft),
        "p_u35_ft": round(p_u35_ft, 1), "c_u35_ft": cj(p_u35_ft),
        "p_btts": round(p_btts, 1), "c_btts": cj(p_btts),
        "p_btts_no": round(p_btts_no, 1), "c_btts_no": cj(p_btts_no),
        "p_05_ht": round(p_05_ht, 1), "c_05_ht": cj(p_05_ht),
        "p_05_2t": round(p_05_2t, 1), "c_05_2t": cj(p_05_2t),
        "min_max_1t_150": min_max_1t_150,
        "min_max_2t_150": min_max_2t_150,
        "top5": top5,
        "matriz_5x5": matriz_5x5
    }


# ---------------------------- INTERFAZ STREAMLIT ----------------------------
st.set_page_config(page_title="WINDEP 360 PRO", page_icon="⚽", layout="wide")
st.title("⚽ WINDEP 360 PRO")
st.caption("Motor estadístico Windep · Poisson híbrido original · EMA · xG")

try:
    DF_RAW, HISTORIAL, CACHE_ESTADO_ACTUAL = cargar_motor()
    ESTADO_ACTUAL = CACHE_ESTADO_ACTUAL[8]
    TEAMS = sorted(ESTADO_ACTUAL["Equipo"].dropna().unique().tolist())
    if not TEAMS:
        st.error("No se encontraron equipos con al menos 6 partidos.")
        st.stop()

    tab_calc, tab_base, tab_diag = st.tabs([
        "⚽ Calculadora", "🗂️ Base histórica", "🧪 Diagnóstico"
    ])

    with tab_calc:
        c1, c2, c3 = st.columns(3)
        c1.metric("Partidos en la base", f"{len(DF_RAW):,}")
        c2.metric("Equipos procesados", f"{len(TEAMS):,}")
        c3.metric("Ligas", f"{DF_RAW['Liga'].nunique():,}")

        st.subheader("Calculadora matemática de partidos")
        st.write("Selecciona dos equipos. El motor utiliza las fórmulas del cuaderno original.")
        col1, col2 = st.columns(2)
        local = col1.selectbox("Equipo local", TEAMS, index=0, key="calc_local")
        visitantes = [t for t in TEAMS if t != local]
        visita = col2.selectbox("Equipo visitante", visitantes, index=0 if visitantes else None, key="calc_visita")

        span = st.selectbox("Ventana EMA", [6, 8, 12], index=1, key="calc_span")
        hibrido = st.toggle("Modelo híbrido xG + goles (original)", value=True, key="calc_hibrido")
        multiplicador = st.number_input("Multiplicador de lambda", min_value=0.10, max_value=2.00, value=1.00, step=0.05, key="calc_mult")

        if st.button("Calcular probabilidades", type="primary", use_container_width=True):
            if local == visita:
                st.error("Selecciona dos equipos diferentes.")
            else:
                res = calcular_poisson_partido(local, visita, span=span, hibrido=hibrido, mult_lambda=float(multiplicador))
                st.markdown(f"### {local} vs {visita}")
                st.caption(f"Liga detectada por el modelo: {res['lg']} · EMA span={span} · Híbrido={hibrido}")
                a, b, c = st.columns(3)
                a.metric("Gana local", f"{res['p_1']*100:.1f}%", f"Cuota justa {res['c_1']:.2f}")
                b.metric("Empate", f"{res['p_x']*100:.1f}%", f"Cuota justa {res['c_x']:.2f}")
                c.metric("Gana visita", f"{res['p_2']*100:.1f}%", f"Cuota justa {res['c_2']:.2f}")

                st.subheader("Mercados de goles")
                markets = [
                    ("Más de 0.5 goles FT", "p_05_ft", "c_05_ft"),
                    ("Más de 1.5 goles FT", "p_15_ft", "c_15_ft"),
                    ("Más de 2.5 goles FT", "p_25_ft", "c_25_ft"),
                    ("Más de 3.5 goles FT", "p_35_ft", "c_35_ft"),
                    ("Menos de 2.5 goles FT", "p_u25_ft", "c_u25_ft"),
                    ("Menos de 3.5 goles FT", "p_u35_ft", "c_u35_ft"),
                    ("Ambos marcan: Sí", "p_btts", "c_btts"),
                    ("Ambos marcan: No", "p_btts_no", "c_btts_no"),
                    ("Gol en primer tiempo", "p_05_ht", "c_05_ht"),
                    ("Gol local", "p_gl", "c_gl"),
                    ("Gol visitante", "p_gv", "c_gv"),
                    ("Local marca 2+", "p_gl_15", "c_gl_15"),
                    ("Visitante marca 2+", "p_gv_15", "c_gv_15"),
                ]
                tab = pd.DataFrame([{"Mercado": name, "Probabilidad (%)": res[p], "Cuota justa": res[o]} for name,p,o in markets])
                tab["Probabilidad (%)"] = tab["Probabilidad (%)"].map(lambda x: round(float(x),1))
                st.dataframe(tab.sort_values("Probabilidad (%)", ascending=False), use_container_width=True, hide_index=True)

                st.subheader("Marcadores exactos más probables")
                scores = pd.DataFrame(res["top5"], columns=["Marcador", "Probabilidad"])
                scores["Probabilidad (%)"] = (scores.pop("Probabilidad")*100).round(2)
                st.dataframe(scores, use_container_width=True, hide_index=True)

                st.subheader("Parámetros estimados por el motor")
                p1, p2, p3 = st.columns(3)
                p1.metric("Lambda local", f"{res['lam_l']:.3f}")
                p2.metric("Lambda visitante", f"{res['lam_v']:.3f}")
                p3.metric("Lambda total", f"{res['lam_tot']:.3f}")
                st.info("Probabilidades estimadas, no garantías. La cuota justa es el inverso de la probabilidad estimada, sin margen de la casa.")
        else:
            st.info("Selecciona los equipos y pulsa «Calcular probabilidades» para ejecutar el motor.")

    with tab_base:
        st.subheader("Base histórica conectada")
        st.caption("Vista de control de la fuente. No edita ni modifica Google Sheets.")
        st.write("Columnas disponibles:", ", ".join(map(str, DF_RAW.columns.tolist())))
        filtros = st.columns(3)
        ligas = sorted(DF_RAW["Liga"].dropna().astype(str).unique().tolist()) if "Liga" in DF_RAW.columns else []
        liga_sel = filtros[0].selectbox("Liga", ["Todas"] + ligas)
        if "Fecha" in DF_RAW.columns:
            fechas = pd.to_datetime(DF_RAW["Fecha"], errors="coerce", dayfirst=True)
            df_show = DF_RAW.copy()
            df_show["_fecha_dt"] = fechas
            if liga_sel != "Todas" and "Liga" in df_show.columns:
                df_show = df_show[df_show["Liga"].astype(str) == liga_sel]
            df_show = df_show.sort_values("_fecha_dt", ascending=False, na_position="last")
            df_show = df_show.drop(columns=["_fecha_dt"])
        else:
            df_show = DF_RAW.copy()
            if liga_sel != "Todas" and "Liga" in df_show.columns:
                df_show = df_show[df_show["Liga"].astype(str) == liga_sel]
        busqueda = filtros[1].text_input("Buscar equipo")
        if busqueda:
            mask = df_show.astype(str).apply(lambda col: col.str.contains(busqueda, case=False, na=False)).any(axis=1)
            df_show = df_show[mask]
        filtros[2].metric("Filas filtradas", f"{len(df_show):,}")
        st.dataframe(df_show.head(500), use_container_width=True, hide_index=True)
        st.caption("Se muestran como máximo 500 filas para mantener rápida la aplicación.")

    with tab_diag:
        st.subheader("Diagnóstico de datos y motor")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Filas", f"{len(DF_RAW):,}")
        d2.metric("Columnas", f"{len(DF_RAW.columns):,}")
        d3.metric("Celdas vacías", f"{int(DF_RAW.isna().sum().sum()):,}")
        d4.metric("Duplicados exactos", f"{int(DF_RAW.duplicated().sum()):,}")
        st.markdown("**Campos detectados**")
        st.dataframe(pd.DataFrame({
            "Columna": DF_RAW.columns.astype(str),
            "Tipo": [str(x) for x in DF_RAW.dtypes],
            "Nulos": [int(DF_RAW[c].isna().sum()) for c in DF_RAW.columns],
            "Valores únicos": [int(DF_RAW[c].nunique(dropna=True)) for c in DF_RAW.columns],
        }), use_container_width=True, hide_index=True)
        st.warning("Este diagnóstico describe la base cargada; no demuestra por sí solo que las probabilidades estén calibradas ni que exista rentabilidad.")

except Exception as exc:
    st.error("El motor no pudo iniciar.")
    st.exception(exc)
    st.caption("No se han modificado los datos de Google Sheets ni el cuaderno original.")
