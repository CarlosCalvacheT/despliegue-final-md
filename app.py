import pandas as pd
import numpy as np
import pickle
import streamlit as st
from huggingface_hub import hf_hub_download
 
# ── Configuración de la página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Pit Stop Predictor",
    page_icon="🏎️",
    layout="centered"
)
 
# ── CSS personalizado: tema F1 oscuro con rojo Ferrari ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8e8;
}
 
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #12121a 50%, #0a0a0f 100%);
}
 
/* Título principal */
.main-title {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: 2rem;
    color: #ff1801;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 2px solid #ff1801;
    padding-bottom: 0.4rem;
    margin-bottom: 0.2rem;
}
 
.sub-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    color: #888;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
 
/* Tarjetas de sección */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,24,1,0.2);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
}
 
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    color: #ff1801;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
 
/* Resultado PIT STOP */
.result-pit {
    background: linear-gradient(135deg, #1a0000, #2d0000);
    border: 2px solid #ff1801;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse-red 1.5s ease-in-out infinite;
}
 
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 15px rgba(255,24,1,0.3); }
    50%       { box-shadow: 0 0 35px rgba(255,24,1,0.7); }
}
 
.result-stay {
    background: linear-gradient(135deg, #001a00, #002d00);
    border: 2px solid #00e676;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,230,118,0.2);
}
 
.result-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
 
.result-prob {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    color: #aaa;
    letter-spacing: 0.1em;
}
 
.result-prob span {
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
}
 
/* Barra de probabilidad custom */
.prob-bar-container {
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    height: 8px;
    margin-top: 1rem;
    overflow: hidden;
}
 
.prob-bar-fill-red  { background: linear-gradient(90deg, #ff1801, #ff6b4a); border-radius: 4px; height: 8px; transition: width 0.8s ease; }
.prob-bar-fill-green{ background: linear-gradient(90deg, #00c853, #00e676); border-radius: 4px; height: 8px; transition: width 0.8s ease; }
 
/* Sliders */
.stSlider > div > div > div > div {
    background: #ff1801 !important;
}
 
/* Botón */
.stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, #ff1801, #cc1400) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
 
.stButton > button:hover {
    background: linear-gradient(135deg, #ff4433, #ff1801) !important;
    box-shadow: 0 4px 20px rgba(255,24,1,0.5) !important;
    transform: translateY(-1px) !important;
}
 
/* Selectbox y labels */
.stSelectbox label, .stSlider label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.85rem !important;
    color: #bbb !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
 
/* Divider */
hr { border-color: rgba(255,24,1,0.2) !important; }
 
/* Métricas en fila */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
 
.metric-box {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
}
 
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: #ff1801;
}
 
.metric-label {
    font-size: 0.7rem;
    color: #666;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
 
.compound-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)
 
# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏎️ F1 Pit Stop Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Strategy Intelligence · Real-time Decision Support</div>', unsafe_allow_html=True)
 
# ── Carga de artefactos ────────────────────────────────────────────────────────
@st.cache_resource
def cargar_artefactos():
    ruta = hf_hub_download(
        repo_id="CarlosC4/modelo-despliegue-f1",
        filename="pipeline_f1.pkl"
    )
    artefactos = pickle.load(open(ruta, "rb"))
    return artefactos['scaler'], artefactos['modelo']
 
with st.spinner("Cargando modelo..."):
    scaler, modelo = cargar_artefactos()
 
COMPOUND_MAP = {
    'SOFT': 3.0, 'MEDIUM': 2.0, 'HARD': 1.0,
    'INTERMEDIATE': 1.5, 'WET': 1.5
}
 
COMPOUND_COLOR = {
    'SOFT': '#ff1801', 'MEDIUM': '#ffd700', 'HARD': '#e8e8e8',
    'INTERMEDIATE': '#39b54a', 'WET': '#0067ff'
}
 
# ── Inputs ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">⬡ Configuración de Neumático</div>', unsafe_allow_html=True)
 
col1, col2, col3 = st.columns(3)
with col1:
    Compound = st.selectbox("Compuesto", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
with col2:
    Stint    = st.slider("Stint (Periodo de vueltas que recorre un piloto)", 1, 6, 2)
with col3:
    TyreLife = st.slider("Vuelta del neumático", 1, 60, 15)
 
st.markdown('<div class="section-label" style="margin-top:1rem">⬡ Estado en Carrera</div>', unsafe_allow_html=True)
 
col4, col5, col6 = st.columns(3)
with col4:
    Position     = st.slider("Posición", 1, 20, 8)
with col5:
    RaceProgress = st.slider("Progreso (%)", 0, 100, 45) / 100
with col6:
    Position_Change = st.slider("Δ Posición", -5, 5, 0)
 
st.markdown('<div class="section-label" style="margin-top:1rem">⬡ Rendimiento de Vuelta</div>', unsafe_allow_html=True)
 
col7, col8, col9 = st.columns(3)
with col7:
    LapTime      = st.slider("Tiempo de vuelta (s)", 70.0, 150.0, 92.0, 0.1)
with col8:
    LapTime_Delta = st.slider("Δ Tiempo (s)", -10.0, 10.0, 0.3, 0.1)
with col9:
    Cumul_Deg    = st.slider("Degradación acum.", -80.0, 10.0, -5.0, 0.5)
 
# ── Resumen visual antes de predecir ──────────────────────────────────────────
color = COMPOUND_COLOR.get(Compound, '#888')
normalized_tyre = round(TyreLife / 58.0, 4)
 
st.markdown(f"""
<div class="metric-row" style="margin-top:1.5rem">
  <div class="metric-box">
    <div class="metric-value" style="color:{color}">{Compound}</div>
    <div class="metric-label">Compuesto</div>
  </div>
  <div class="metric-box">
    <div class="metric-value">{TyreLife}</div>
    <div class="metric-label">Vueltas en goma</div>
  </div>
  <div class="metric-box">
    <div class="metric-value">P{Position}</div>
    <div class="metric-label">Posición actual</div>
  </div>
  <div class="metric-box">
    <div class="metric-value">{int(RaceProgress*100)}%</div>
    <div class="metric-label">Carrera completada</div>
  </div>
  <div class="metric-box">
    <div class="metric-value">{LapTime:.1f}s</div>
    <div class="metric-label">Último tiempo</div>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ── Botón y predicción ─────────────────────────────────────────────────────────
predict = st.button("ANALIZAR ESTRATEGIA")
 
if predict:
    compound_num = COMPOUND_MAP.get(Compound, 2.0)
 
    X_array = np.array([[
        compound_num, float(Stint), float(TyreLife), float(Position),
        float(LapTime), float(LapTime_Delta), float(Cumul_Deg),
        float(RaceProgress), normalized_tyre, float(Position_Change)
    ]])
 
    X_scaled     = scaler.transform(X_array)
    prediccion   = modelo.predict(X_scaled)[0]
    probabilidad = modelo.predict_proba(X_scaled)[0][1]
    pct          = int(probabilidad * 100)
    bar_width    = pct
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    if prediccion == 1:
        st.markdown(f"""
        <div class="result-pit">
            <div class="result-title" style="color:#ff1801">🔴 &nbsp; PIT STOP RECOMENDADO</div>
            <div class="result-prob">Probabilidad &nbsp; <span>{pct}%</span></div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill-red" style="width:{bar_width}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-stay">
            <div class="result-title" style="color:#00e676">🟢 &nbsp; CONTINUAR EN PISTA</div>
            <div class="result-prob">Probabilidad de pit stop &nbsp; <span>{pct}%</span></div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill-green" style="width:{100 - bar_width}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    # Contexto adicional
    st.markdown("<br>", unsafe_allow_html=True)
    degradacion_pct = abs(Cumul_Deg / 80 * 100)
 
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        delta_color = "#ff1801" if LapTime_Delta > 1.5 else "#00e676"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:{delta_color}">{LapTime_Delta:+.1f}s</div>
            <div class="metric-label">Variación tiempo</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        deg_color = "#ff1801" if degradacion_pct > 60 else "#ffd700" if degradacion_pct > 30 else "#00e676"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:{deg_color}">{degradacion_pct:.0f}%</div>
            <div class="metric-label">Nivel degradación</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        vueltas_restantes = int((1 - RaceProgress) * 58)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">~{vueltas_restantes}</div>
            <div class="metric-label">Vueltas restantes est.</div>
        </div>""", unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#444; font-size:0.75rem; font-family:\'Rajdhani\',sans-serif; letter-spacing:0.1em">'
    'MODELO CON UNA PRECISIÓN DEL 71% CON DATOS DE PRUEBA · APOYO A DECISIÓN ESTRATÉGICA'
    '</p>',
    unsafe_allow_html=True
)
