# =========================
# LIBRERÍAS
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from huggingface_hub import hf_hub_download

# =========================
# CARGAR MODELO DESDE HUGGING FACE
# =========================

@st.cache_resource
def cargar_modelo():

    ruta_modelo = hf_hub_download(
        repo_id="CarlosC4/modelo-despliegue-f1",
        filename="modelo-f1.pkl"
    )

    return pickle.load(open(ruta_modelo, 'rb'))

# Cargar objetos del modelo
modelo, min_max_scaler, variables, compound_map, necesita_norm, nombre_modelo = cargar_modelo()

# =========================
# INTERFAZ STREAMLIT
# =========================

st.title('🏎️ Predicción de Pit Stop en Fórmula 1')

st.markdown(
    'Ingresa los datos de la vuelta actual para predecir '
    'si el piloto hará pit stop en la siguiente vuelta.'
)

# =========================
# ENTRADAS DEL USUARIO
# =========================

Compound = st.selectbox(
    'Compuesto de Neumático',
    ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
)

Stint = st.slider(
    'Stint (número de stint)',
    min_value=1,
    max_value=6,
    value=2,
    step=1
)

TyreLife = st.slider(
    'Vida del Neumático (vueltas)',
    min_value=1,
    max_value=60,
    value=15,
    step=1
)

Position = st.slider(
    'Posición en carrera',
    min_value=1,
    max_value=20,
    value=8,
    step=1
)

LapTime = st.slider(
    'Tiempo de vuelta (segundos)',
    min_value=70.0,
    max_value=150.0,
    value=92.0,
    step=0.1
)

LapTime_Delta = st.slider(
    'Delta tiempo de vuelta (s)',
    min_value=-10.0,
    max_value=10.0,
    value=0.3,
    step=0.1
)

Cumul_Deg = st.slider(
    'Degradación acumulada',
    min_value=-80.0,
    max_value=10.0,
    value=-5.0,
    step=0.5
)

RaceProgress = st.slider(
    'Progreso de carrera (0-100%)',
    min_value=0,
    max_value=100,
    value=45,
    step=1
) / 100

Position_Change = st.slider(
    'Cambio de posición',
    min_value=-5,
    max_value=5,
    value=0,
    step=1
)

# Variable calculada
Normalized_Tyre = round(TyreLife / 58.0, 4)

# =========================
# CREAR DATAFRAME
# =========================

datos = [[
    Compound,
    Stint,
    TyreLife,
    Position,
    LapTime,
    LapTime_Delta,
    Cumul_Deg,
    RaceProgress,
    Normalized_Tyre,
    float(Position_Change)
]]

data = pd.DataFrame(datos, columns=[
    'Compound',
    'Stint',
    'TyreLife',
    'Position',
    'LapTime (s)',
    'LapTime_Delta',
    'Cumulative_Degradation',
    'RaceProgress',
    'Normalized_TyreLife',
    'Position_Change'
])

# =========================
# PREPROCESAMIENTO
# =========================

# Codificar variable categórica
data['Compound'] = data['Compound'].map(compound_map)

# Reordenar columnas igual que entrenamiento
data = data.reindex(columns=variables, fill_value=0)

# Normalizar si el modelo lo necesita
if necesita_norm:

    variables_numericas = [
        'Stint',
        'TyreLife',
        'Position',
        'LapTime (s)',
        'LapTime_Delta',
        'Cumulative_Degradation',
        'RaceProgress',
        'Normalized_TyreLife',
        'Position_Change',
        'Compound'
    ]

    data[variables_numericas] = min_max_scaler.transform(
        data[variables_numericas]
    )

# =========================
# PREDICCIÓN
# =========================

if st.button('Predecir Pit Stop'):

    prediccion = modelo.predict(data)[0]

    probabilidad = modelo.predict_proba(data)[0][1]

    if prediccion == 1:

        st.error(
            f'🔴 PIT STOP RECOMENDADO\n\n'
            f'Probabilidad: {probabilidad:.1%}'
        )

    else:

        st.success(
            f'🟢 CONTINUAR EN PISTA\n\n'
            f'Probabilidad de pit stop: {probabilidad:.1%}'
        )

# =========================
# INFORMACIÓN DEL MODELO
# =========================

st.warning(
    f'El modelo ({nombre_modelo}) '
    f'tiene un ROC-AUC aproximado de 0.9 en datos de prueba.'
)
