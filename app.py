import pandas as pd
import numpy as np
import pickle
import streamlit as st
from huggingface_hub import hf_hub_download
 
# ── Configuración de la página ─────────────────────────────────────────────────
st.set_page_config(page_title="F1 Pit Stop Predictor", page_icon="🏎️", layout="centered")
 
st.title("🏎️ Predicción de Pit Stop en Fórmula 1")
st.markdown("Ingresa los datos de la vuelta actual para predecir si el piloto hará pit stop.")
 
# ── Carga del scaler y modelo desde Hugging Face ───────────────────────────────
@st.cache_resource
def cargar_artefactos():
    ruta = hf_hub_download(
        repo_id="CarlosC4/modelo-despliegue-f1",
        filename="pipeline_f1.pkl"
    )
    artefactos = pickle.load(open(ruta, "rb"))
    return artefactos['scaler'], artefactos['modelo']
 
scaler, modelo = cargar_artefactos()
 
# ── Preprocesamiento (sin pickle, instancias frescas) ──────────────────────────
COMPOUND_MAP = {
    'SOFT': 3.0, 'MEDIUM': 2.0, 'HARD': 1.0,
    'INTERMEDIATE': 1.5, 'WET': 1.5
}
 
VARIABLES_MODELO = [
    'Compound', 'Stint', 'TyreLife', 'Position', 'LapTime (s)',
    'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress',
    'Normalized_TyreLife', 'Position_Change'
]
 
# ── Entradas del usuario ───────────────────────────────────────────────────────
st.subheader("Datos de la vuelta actual")
 
col1, col2 = st.columns(2)
 
with col1:
    Compound = st.selectbox("Compuesto de Neumático", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
    Stint    = st.slider("Stint (número de stint)", 1, 6, 2)
    TyreLife = st.slider("Vida del Neumático (vueltas)", 1, 60, 15)
    Position = st.slider("Posición en carrera", 1, 20, 8)
    LapTime  = st.slider("Tiempo de vuelta (segundos)", 70.0, 150.0, 92.0, 0.1)
 
with col2:
    LapTime_Delta   = st.slider("Delta tiempo de vuelta (s)", -10.0, 10.0, 0.3, 0.1)
    Cumul_Deg       = st.slider("Degradación acumulada", -80.0, 10.0, -5.0, 0.5)
    RaceProgress    = st.slider("Progreso de carrera (0-100%)", 0, 100, 45) / 100
    Position_Change = st.slider("Cambio de posición", -5, 5, 0)
 
# ── Predicción ─────────────────────────────────────────────────────────────────
if st.button("🔍 Predecir Pit Stop", use_container_width=True):
 
    # 1. Codificar Compound y calcular Normalized_TyreLife
    compound_num      = COMPOUND_MAP.get(Compound, 2.0)
    normalized_tyre   = round(TyreLife / 58.0, 4)
 
    # 2. Construir array en el orden exacto del modelo
    X_array = np.array([[
        compound_num,
        float(Stint),
        float(TyreLife),
        float(Position),
        float(LapTime),
        float(LapTime_Delta),
        float(Cumul_Deg),
        float(RaceProgress),
        normalized_tyre,
        float(Position_Change)
    ]])
 
    # 3. Escalar y predecir
    X_scaled     = scaler.transform(X_array)
    prediccion   = modelo.predict(X_scaled)[0]
    probabilidad = modelo.predict_proba(X_scaled)[0][1]
 
    st.divider()
 
    if prediccion == 1:
        st.error(f"🔴 **PIT STOP RECOMENDADO**\n\nProbabilidad: **{probabilidad:.1%}**")
    else:
        st.success(f"🟢 **CONTINUAR EN PISTA**\n\nProbabilidad de pit stop: **{probabilidad:.1%}**")
 
 
st.divider()
st.warning("ℹ️ El modelo presento una precisión 71% en datos de prueba.")
