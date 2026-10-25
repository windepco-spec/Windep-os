import streamlit as st
import pandas as pd

# WINDEP — prueba de conexión a la base histórica
SHEET_ID = "1wXHz_CaJ7rx-e2wha3lWwQKpV1rcVjlQfKiliHLAPBU"

URL_SHEET = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
    "/gviz/tq?tqx=out:csv&sheet=GOLES"
)

st.set_page_config(
    page_title="WINDEP 360 PRO",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ WINDEP 360 PRO")
st.caption("Conexión de prueba con la base histórica")

@st.cache_data(ttl=300)
def cargar_base():
    return pd.read_csv(URL_SHEET)

try:
    df = cargar_base()

    st.success("Conexión con Google Sheets establecida")
    st.metric("Registros leídos", len(df))
    st.metric("Columnas detectadas", len(df.columns))

    st.subheader("Columnas de la base")
    st.write(df.columns.tolist())

    st.subheader("Vista previa")
    st.dataframe(df.head(10), use_container_width=True)

except Exception as e:
    st.error("No se pudo leer la hoja GOLES.")
    st.caption(
        "Verifica que la hoja sea accesible sin iniciar sesión "
        "y que el nombre de la pestaña sea GOLES."
    )
    st.exception(e)
