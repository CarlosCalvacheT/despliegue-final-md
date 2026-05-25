
import pandas as pd
import pickle
import streamlit as st
from huggingface_hub import hf_hub_download
 
# ── Configuración de la página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Pit Stop Predictor",
    page_icon="🏎️",
    layout="centered"
)
 
st.title("🏎️ Predicción de Pit Stop en Fórmula 1")
st.markdown(
    "Ingresa los datos de la vuelta actual para predecir "
    "si el piloto hará pit stop en la siguiente vuelta."
)
 
# ── Carga del pipeline desde Hugging Face ──────────────────────────────────────
# @st.cache_resource garantiza que la descarga ocurre UNA SOLA VEZ
# y el pipeline queda en memoria para todas las sesiones siguientes.
# Esto es clave cuando el modelo es pesado.
@st.cache_resource
def cargar_pipeline():
    ruta = hf_hub_download(
        repo_id="CarlosC4/modelo-despliegue-f1",   # <-- cambia por tu repo
        filename="pipeline_f1.pkl"                  # el pipeline completo
    )
    return pickle.load(open(ruta, "rb"))
 
pipeline = cargar_pipeline()
 
# ── Entradas del usuario ───────────────────────────────────────────────────────
st.subheader("Datos de la vuelta actual")
 
col1, col2 = st.columns(2)
 
with col1:
    Compound = st.selectbox(
        "Compuesto de Neumático",
        ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    )
    Stint = st.slider("Stint (número de stint)", 1, 6, 2)
    TyreLife = st.slider("Vida del Neumático (vueltas)", 1, 60, 15)
    Position = st.slider("Posición en carrera", 1, 20, 8)
    LapTime = st.slider("Tiempo de vuelta (segundos)", 70.0, 150.0, 92.0, 0.1)
 
with col2:
    LapTime_Delta = st.slider("Delta tiempo de vuelta (s)", -10.0, 10.0, 0.3, 0.1)
    Cumul_Deg = st.slider("Degradación acumulada", -80.0, 10.0, -5.0, 0.5)
    RaceProgress = st.slider("Progreso de carrera (0-100%)", 0, 100, 45) / 100
    Position_Change = st.slider("Cambio de posición", -5, 5, 0)
 
# ── Construcción del DataFrame ─────────────────────────────────────────────────
# Compound llega como STRING — el pipeline lo codifica internamente.
# Ya no necesitas compound_map, min_max_scaler ni necesita_norm por separado.
dato = pd.DataFrame([{
    "Compound":               Compound,
    "Stint":                  Stint,
    "TyreLife":               TyreLife,
    "Position":               Position,
    "LapTime (s)":            LapTime,
    "LapTime_Delta":          LapTime_Delta,
    "Cumulative_Degradation": Cumul_Deg,
    "RaceProgress":           RaceProgress,
    "Normalized_TyreLife":    round(TyreLife / 58.0, 4),
    "Position_Change":        float(Position_Change)
}])
 
# ── Predicción ─────────────────────────────────────────────────────────────────
if st.button("🔍 Predecir Pit Stop", use_container_width=True):
    prediccion   = pipeline.predict(dato)[0]
    probabilidad = pipeline.predict_proba(dato)[0][1]
 
    st.divider()
 
    if prediccion == 1:
        st.error(
            f"🔴 **PIT STOP RECOMENDADO**\n\n"
            f"Probabilidad: **{probabilidad:.1%}**"
        )
    else:
        st.success(
            f"🟢 **CONTINUAR EN PISTA**\n\n"
            f"Probabilidad de pit stop: **{probabilidad:.1%}**"
        )
 
    # Barra de probabilidad visual
    st.progress(probabilidad, text=f"Confianza del modelo: {probabilidad:.1%}")
 
st.divider()
st.warning(
    "ℹ️ El modelo tiene un ROC-AUC aproximado de 0.92 en datos de prueba. "
    "Úsalo como apoyo a la decisión, no como regla absoluta."
)
