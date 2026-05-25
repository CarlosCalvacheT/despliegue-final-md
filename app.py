import pandas as pd
import numpy as np
import pickle
import streamlit as st
from huggingface_hub import hf_hub_download
from sklearn.base import BaseEstimator, TransformerMixin

# ── Clase requerida para deserializar el pipeline ──
class CompoundEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, compound_map):
        self.compound_map = compound_map
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['Compound'] = X['Compound'].map(self.compound_map)
        return X

# ── Cargar pipeline ──
@st.cache_resource
def cargar_modelo():
    ruta_modelo = hf_hub_download(
        repo_id="CarlosC4/modelo-despliegue-f1",
        filename="pipeline_modelo_f1.pkl"
    )
    return pickle.load(open(ruta_modelo, 'rb'))

pipeline = cargar_modelo()

# ── Interfaz ──
st.title('🏎️ Predicción de Pit Stop en Fórmula 1')
st.markdown(
    'Ingresa los datos de la vuelta actual para predecir '
    'si el piloto hará pit stop en la siguiente vuelta.'
)

# ── Entradas del usuario ──
Compound = st.selectbox('Compuesto de Neumático',
    ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'])
Stint           = st.slider('Stint (número de stint)', 1, 6, 2)
TyreLife        = st.slider('Vida del Neumático (vueltas)', 1, 60, 15)
Position        = st.slider('Posición en carrera', 1, 20, 8)
LapTime         = st.slider('Tiempo de vuelta (segundos)', 70.0, 150.0, 92.0, 0.1)
LapTime_Delta   = st.slider('Delta tiempo de vuelta (s)', -10.0, 10.0, 0.3, 0.1)
Cumul_Deg       = st.slider('Degradación acumulada', -80.0, 10.0, -5.0, 0.5)
RaceProgress    = st.slider('Progreso de carrera (0-100%)', 0, 100, 45) / 100
Position_Change = st.slider('Cambio de posición', -5, 5, 0)
Normalized_Tyre = round(TyreLife / 58.0, 4)

# ── DataFrame de entrada (datos crudos, sin preprocesar) ──
data = pd.DataFrame([[
    Compound, Stint, TyreLife, Position, LapTime,
    LapTime_Delta, Cumul_Deg, RaceProgress,
    Normalized_Tyre, float(Position_Change)
]], columns=[
    'Compound', 'Stint', 'TyreLife', 'Position', 'LapTime (s)',
    'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress',
    'Normalized_TyreLife', 'Position_Change'
])

# ── Predicción ──
if st.button('Predecir Pit Stop'):
    prediccion   = pipeline.predict(data)[0]
    probabilidad = pipeline.predict_proba(data)[0][1]

    if prediccion == 1:
        st.error(f'🔴 PIT STOP RECOMENDADO\n\nProbabilidad: {probabilidad:.1%}')
    else:
        st.success(f'🟢 CONTINUAR EN PISTA\n\nProbabilidad de pit stop: {probabilidad:.1%}')

nombre_modelo = type(pipeline.named_steps['model']).__name__
st.warning(f'El modelo ({nombre_modelo}) tiene un ROC-AUC aproximado de 0.92 en datos de prueba.')
