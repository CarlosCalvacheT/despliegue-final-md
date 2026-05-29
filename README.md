# 🏎️ Predicción de Pit Stop en Fórmula 1

Proyecto final de la asignatura **Minería de Datos** — Universidad Pontificia Bolivariana.

Modelo predictivo que estima si un piloto de F1 realizará un pit stop en la siguiente vuelta, utilizando datos históricos vuelta a vuelta. Desarrollado bajo la metodología **CRISP-DM** y desplegado en Streamlit.

🔗 **[Ver aplicación en vivo](https://despliegue-final-md-dprjqjkq6vtuzwkjrayk5g.streamlit.app/)**

---

## 👥 Integrantes

- Juan David Castañeda
- Carlos Alberto Calvache
- Susana Toro
- Jerónimo Gálvez

---

## 🎯 Objetivo

Construir un modelo de **clasificación binaria** que prediga si en la siguiente vuelta un piloto realizará un pit stop (`PitNextLap = 1`) o continuará en pista (`PitNextLap = 0`), y desplegarlo como herramienta interactiva en Streamlit.

---

## 📊 Datos

| Característica | Detalle |
|---|---|
| Registros | 101.371 |
| Variables | 16 |
| Variable objetivo | `PitNextLap` |
| Tipo de problema | Clasificación binaria |
| Datos reales | ✅ |

**Variables principales:** `Compound`, `Stint`, `TyreLife`, `Position`, `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`, `Normalized_TyreLife`, `Position_Change`

---

## ⚙️ Metodología CRISP-DM

1. **Comprensión del negocio** — Definición del problema de predicción de pit stop
2. **Comprensión de los datos** — Análisis de variables, estructura y variable objetivo
3. **Preparación de datos** — Limpieza, atípicos, variables derivadas y balanceo
4. **Modelamiento** — Modelos supervisados y de ensamble
5. **Evaluación** — ANOVA, Tukey, GridSearchCV y optimización bayesiana
6. **Despliegue** — Aplicación interactiva en Streamlit

---

## 🧹 Preparación de datos

- Tratamiento de 66 valores nulos en `Compound`
- Corrección de 112 atípicos en `LapTime (s)` (>150s) usando mediana por carrera
- Creación de variables derivadas
- División entrenamiento/prueba
- Balanceo aplicado únicamente al conjunto de entrenamiento

---

## 🤖 Modelos evaluados

**Supervisados:** Árbol de Decisión, KNN, Red Neuronal, SVM, SGD-SVM

**Ensamble:** Random Forest, Gradient Boosting, Bagging, AdaBoost

---

## 📈 Resultados

### Top 3 modelos iniciales

| Modelo | ROC-AUC | F1-score | Recall |
|---|---:|---:|---:|
| KNN | 0.9547 | 0.8015 | 0.8919 |
| Random Forest | 0.9499 | 0.7817 | 0.8675 |
| Red Neuronal | 0.9250 | 0.7275 | 0.8832 |

### Después de GridSearchCV

| Modelo | Mejores hiperparámetros | ROC-AUC |
|---|---|---:|
| KNN | `metric=manhattan`, `n_neighbors=7`, `weights=distance` | 0.9702 |
| **Random Forest** ✅ | `max_depth=None`, `min_samples_split=2`, `n_estimators=200` | **0.9778** |
| Red Neuronal | `activation=relu`, `hidden_layer_sizes=(128,64)`, `lr=0.001` | 0.9429 |

### Modelo final — Random Forest (GridSearchCV)

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9369 |
| Precision | 0.8745 |
| Recall | 0.8783 |
| F1-score | 0.8764 |
| ROC-AUC | **0.9778** |

> La optimización bayesiana fue evaluada pero resultó ligeramente inferior; se conservó GridSearchCV como modelo final.

---

## 🔧 Pipeline

El modelo final está encapsulado en un pipeline que integra: limpieza de atípicos → ingeniería de características → selección de columnas → transformaciones → modelo Random Forest. Esto garantiza que los datos nuevos pasen por las mismas transformaciones del entrenamiento.

---

## 🖥️ Despliegue

La app en Streamlit permite ingresar datos de una vuelta actual y devuelve:

- ✅ Continuar en pista **o** 🔴 Pit stop recomendado
- Probabilidad estimada de pit stop

**Inputs:** Compuesto, Stint, Vida del neumático, Posición, Tiempo de vuelta, Delta de tiempo, Degradación acumulada, Progreso de carrera, Cambio de posición.

---

## 🚀 Ejecución local

```bash
git clone https://github.com/CarlosCalvacheT/despliegue-final-md.git
cd despliegue-final-md
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Estructura del repositorio
despliegue-final-md/
├── app.py
├── requirements.txt
├── pipeline_modelo_f1.pkl
├── notebooks/
│   ├── 01_Preparacion_Datos.ipynb
│   ├── 02_Modelamiento_Evaluacion.ipynb
│   └── 03_Despliegue_Streamlit.ipynb
├── informe/
│   └── Informe_Final_F1_PitStop.pdf
└── datos/
└── dataset_f1.csv

---

## 📦 Librerías principales

`pandas` · `numpy` · `scikit-learn` · `streamlit` · `matplotlib` · `seaborn` · `scipy` · `statsmodels`
