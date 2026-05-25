import pandas as pd
import numpy as np
import pickle
import streamlit as st
from huggingface_hub import hf_hub_download
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
 
# ── Clases del pipeline (deben estar aquí para que pickle pueda cargarlas) ─────
 
class LapTimeCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=150.0):
        self.threshold = threshold
 
    def fit(self, X, y=None):
        X_ = X.copy()
        if 'Race' in X_.columns:
            self.global_median_ = X_.groupby('Race')['LapTime (s)'].median().mean()
        else:
            self.global_median_ = X_['LapTime (s)'].median()
        return self
 
    def transform(self, X, y=None):
        X_ = X.copy()
        mask = X_['LapTime (s)'] > self.threshold
        if mask.any():
            if 'Race' in X_.columns:
                mediana_por_carrera = X_.groupby('Race')['LapTime (s)'].transform('median')
                X_.loc[mask, 'LapTime (s)'] = mediana_por_carrera[mask]
            else:
                X_.loc[mask, 'LapTime (s)'] = self.global_median_
        return X_
 
 
class FeatureEngineer(BaseEstimator, TransformerMixin):
    COMPOUND_MAP = {
        'SOFT': 3,
        'MEDIUM': 2,
        'HARD': 1,
        'INTERMEDIATE': 1.5,
        'WET': 1.5
    }
    MAX_TYRE_LIFE = 58.0
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
        X_ = X.copy()
        if X_['Compound'].dtype == object:
            X_['Compound'] = X_['Compound'].map(self.COMPOUND_MAP).fillna(2)
        X_['Normalized_TyreLife'] = (X_['TyreLife'] / self.MAX_TYRE_LIFE).round(4)
        if 'LapTime_Delta' not in X_.columns:
            X_['LapTime_Delta'] = X_['LapTime (s)'].diff().fillna(0)
        if 'Position_Change' not in X_.columns:
            X_['Position_Change'] = X_['Position'].diff().fillna(0)
        return X_
 
 
class ColumnSelector(BaseEstimator, TransformerMixin):
    VARIABLES_MODELO = [
        'Compound', 'Stint', 'TyreLife', 'Position', 'LapTime (s)',
        'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress',
        'Normalized_TyreLife', 'Position_Change'
    ]
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
        return X.reindex(columns=self.VARIABLES_MODELO, fill_value=0).astype(float)
 
 
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
@st.cache_resource
def cargar_pipeline():
    ruta = hf_hub_download(
        repo_id="CarlosC4/modelo-despliegue-f1",
        filename="pipeline_f1.pkl"
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
 
    st.progress(probabilidad, text=f"Confianza del modelo: {probabilidad:.1%}")
 
st.divider()
st.warning(
    "ℹ️ El modelo tiene un ROC-AUC aproximado de 94.99% en datos de prueba. "
    "Úsalo como apoyo a la decisión, no como regla absoluta."
)
