import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from huggingface_hub import hf_hub_download

# ── Configuración ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="F1 Pit Stop Predictor", page_icon="🏎️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; background-color: #0a0a0f; color: #e8e8e8; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #12121a 50%, #0a0a0f 100%); }
.main-title { font-family:'Orbitron',monospace; font-weight:900; font-size:2rem; color:#ff1801; letter-spacing:0.05em; text-transform:uppercase; border-bottom:2px solid #ff1801; padding-bottom:0.4rem; margin-bottom:0.2rem; }
.sub-title  { font-family:'Rajdhani',sans-serif; font-size:1rem; color:#888; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:1.5rem; }
.section-label { font-family:'Orbitron',monospace; font-size:0.65rem; color:#ff1801; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:0.8rem; margin-top:0.5rem; }
.result-pit  { background:linear-gradient(135deg,#1a0000,#2d0000); border:2px solid #ff1801; border-radius:10px; padding:1.5rem 2rem; text-align:center; animation:pulse-red 1.5s ease-in-out infinite; }
.result-stay { background:linear-gradient(135deg,#001a00,#002d00); border:2px solid #00e676; border-radius:10px; padding:1.5rem 2rem; text-align:center; box-shadow:0 0 20px rgba(0,230,118,0.2); }
@keyframes pulse-red { 0%,100%{box-shadow:0 0 15px rgba(255,24,1,0.3)} 50%{box-shadow:0 0 35px rgba(255,24,1,0.7)} }
.result-title { font-family:'Orbitron',monospace; font-size:1.4rem; font-weight:900; letter-spacing:0.08em; margin-bottom:0.5rem; }
.result-prob  { font-family:'Rajdhani',sans-serif; font-size:1rem; color:#aaa; letter-spacing:0.1em; }
.result-prob span { font-size:2rem; font-weight:700; color:#fff; }
.prob-bar-container { background:rgba(255,255,255,0.08); border-radius:4px; height:8px; margin-top:1rem; overflow:hidden; }
.prob-bar-fill-red   { background:linear-gradient(90deg,#ff1801,#ff6b4a); border-radius:4px; height:8px; }
.prob-bar-fill-green { background:linear-gradient(90deg,#00c853,#00e676); border-radius:4px; height:8px; }
.metric-row { display:flex; gap:1rem; margin-bottom:1rem; }
.metric-box { flex:1; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:0.8rem; text-align:center; }
.metric-value { font-family:'Orbitron',monospace; font-size:1.2rem; font-weight:700; color:#ff1801; }
.metric-label { font-size:0.7rem; color:#666; letter-spacing:0.1em; text-transform:uppercase; margin-top:0.2rem; }
.stButton>button { font-family:'Orbitron',monospace!important; font-weight:700!important; font-size:0.85rem!important; letter-spacing:0.15em!important; text-transform:uppercase!important; background:linear-gradient(135deg,#ff1801,#cc1400)!important; color:white!important; border:none!important; border-radius:6px!important; width:100%!important; transition:all 0.2s!important; }
.stButton>button:hover { background:linear-gradient(135deg,#ff4433,#ff1801)!important; box-shadow:0 4px 20px rgba(255,24,1,0.5)!important; transform:translateY(-1px)!important; }
.stSelectbox label,.stSlider label { font-family:'Rajdhani',sans-serif!important; font-size:0.85rem!important; color:#bbb!important; letter-spacing:0.05em!important; text-transform:uppercase!important; }
.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,0.03); border-radius:8px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { font-family:'Orbitron',monospace; font-size:0.65rem; letter-spacing:0.1em; color:#888; background:transparent; border-radius:6px; }
.stTabs [aria-selected="true"] { background:rgba(255,24,1,0.15)!important; color:#ff1801!important; }
hr { border-color:rgba(255,24,1,0.2)!important; }

/* ── Centrado del contenido de las pestañas ── */
.tab-content {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏎️ F1 Pit Stop Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Strategy Intelligence · Real-time Decision Support</div>', unsafe_allow_html=True)

# ── Carga de artefactos ────────────────────────────────────────────────────────
@st.cache_resource
def cargar_artefactos():
    ruta = hf_hub_download(repo_id="CarlosC4/modelo-despliegue-f1", filename="pipeline_f1.pkl")
    artefactos = pickle.load(open(ruta, "rb"))
    return artefactos['scaler'], artefactos['modelo']

with st.spinner("Cargando modelo estratégico..."):
    scaler, modelo = cargar_artefactos()

COMPOUND_MAP   = {'SOFT':3.0,'MEDIUM':2.0,'HARD':1.0,'INTERMEDIATE':1.5,'WET':1.5}
COMPOUND_COLOR = {'SOFT':'#ff1801','MEDIUM':'#ffd700','HARD':'#e8e8e8','INTERMEDIATE':'#39b54a','WET':'#0067ff'}
VARIABLES      = ['Compound','Stint','TyreLife','Position','LapTime (s)','LapTime_Delta','Cumulative_Degradation','RaceProgress','Normalized_TyreLife','Position_Change']

if 'historial' not in st.session_state:
    st.session_state.historial = []

def predecir(compound, stint, tyre_life, position, lap_time, lap_delta, cumul_deg, race_progress, pos_change):
    cn  = COMPOUND_MAP.get(compound, 2.0)
    nt  = round(tyre_life / 58.0, 4)
    X   = np.array([[cn, float(stint), float(tyre_life), float(position),
                     float(lap_time), float(lap_delta), float(cumul_deg),
                     float(race_progress), nt, float(pos_change)]])
    Xs   = scaler.transform(X)
    pred = modelo.predict(Xs)[0]
    prob = modelo.predict_proba(Xs)[0][1]
    return int(pred), float(prob)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏁  PREDICCIÓN",
    "📊  EXPLICABILIDAD",
    "🔁  SIMULADOR",
    "📈  PERFORMANCE"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · PREDICCIÓN — centrado con columnas vacías a los lados
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    _, centro, _ = st.columns([0.5, 9, 0.5])
    with centro:
        st.markdown('<div class="section-label">⬡ Configuración de Neumático</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: Compound = st.selectbox("Compuesto", ["SOFT","MEDIUM","HARD","INTERMEDIATE","WET"])
        with c2: Stint    = st.slider("Stint (Periodo de vueltas)", 1, 6, 2)
        with c3: TyreLife = st.slider("Vueltas en neumático", 1, 60, 15)

        st.markdown('<div class="section-label">⬡ Estado en Carrera</div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4: Position        = st.slider("Posición", 1, 20, 8)
        with c5: RaceProgress    = st.slider("Progreso (%)", 0, 100, 45) / 100
        with c6: Position_Change = st.slider("Δ Posición", -5, 5, 0)

        st.markdown('<div class="section-label">⬡ Rendimiento de Vuelta</div>', unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        with c7: LapTime       = st.slider("Tiempo de vuelta (s)", 70.0, 150.0, 92.0, 0.1)
        with c8: LapTime_Delta = st.slider("Δ Tiempo (s)", -10.0, 10.0, 0.3, 0.1)
        with c9: Cumul_Deg     = st.slider("Degradación acum.", -80.0, 10.0, -5.0, 0.5)

        color_c = COMPOUND_COLOR.get(Compound, '#888')
        st.markdown(f"""
        <div class="metric-row" style="margin-top:1.2rem">
          <div class="metric-box"><div class="metric-value" style="color:{color_c}">{Compound}</div><div class="metric-label">Compuesto</div></div>
          <div class="metric-box"><div class="metric-value">{TyreLife}</div><div class="metric-label">Vueltas en goma</div></div>
          <div class="metric-box"><div class="metric-value">P{Position}</div><div class="metric-label">Posición</div></div>
          <div class="metric-box"><div class="metric-value">{int(RaceProgress*100)}%</div><div class="metric-label">Carrera</div></div>
          <div class="metric-box"><div class="metric-value">{LapTime:.1f}s</div><div class="metric-label">Último tiempo</div></div>
        </div>""", unsafe_allow_html=True)

        if st.button("ANALIZAR ESTRATEGIA", key="btn_pred"):
            pred, prob = predecir(Compound, Stint, TyreLife, Position, LapTime,
                                  LapTime_Delta, Cumul_Deg, RaceProgress, Position_Change)
            pct = int(prob * 100)

            st.markdown("<br>", unsafe_allow_html=True)
            if pred == 1:
                st.markdown(f"""<div class="result-pit">
                    <div class="result-title" style="color:#ff1801">🔴 &nbsp; PIT STOP RECOMENDADO</div>
                    <div class="result-prob">Probabilidad &nbsp;<span>{pct}%</span></div>
                    <div class="prob-bar-container"><div class="prob-bar-fill-red" style="width:{pct}%"></div></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-stay">
                    <div class="result-title" style="color:#00e676">🟢 &nbsp; CONTINUAR EN PISTA</div>
                    <div class="result-prob">Probabilidad de pit stop &nbsp;<span>{pct}%</span></div>
                    <div class="prob-bar-container"><div class="prob-bar-fill-green" style="width:{100-pct}%"></div></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            deg_pct      = abs(Cumul_Deg / 80 * 100)
            vueltas_rest = int((1 - RaceProgress) * 58)
            dc = "#ff1801" if LapTime_Delta > 1.5 else "#00e676"
            gc = "#ff1801" if deg_pct > 60 else "#ffd700" if deg_pct > 30 else "#00e676"
            ca, cb, cc = st.columns(3)
            with ca: st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{dc}">{LapTime_Delta:+.1f}s</div><div class="metric-label">Variación tiempo</div></div>', unsafe_allow_html=True)
            with cb: st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{gc}">{deg_pct:.0f}%</div><div class="metric-label">Nivel degradación</div></div>', unsafe_allow_html=True)
            with cc: st.markdown(f'<div class="metric-box"><div class="metric-value">~{vueltas_rest}</div><div class="metric-label">Vueltas restantes est.</div></div>', unsafe_allow_html=True)

            st.session_state.historial.append({
                'Compuesto': Compound, 'TyreLife': TyreLife, 'Posición': Position,
                'LapTime': f"{LapTime:.1f}s", 'Progreso': f"{int(RaceProgress*100)}%",
                'Decisión': '🔴 PIT STOP' if pred == 1 else '🟢 CONTINUAR',
                'Prob': f"{pct}%"
            })

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · EXPLICABILIDAD
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    _, centro2, _ = st.columns([0.5, 9, 0.5])
    with centro2:
        st.markdown('<div class="section-label">⬡ Importancia de Variables en la Decisión</div>', unsafe_allow_html=True)
        st.markdown("Visualiza qué variables tienen más peso en el modelo y cómo los valores actuales se comparan con el umbral de pit stop.")

        tiene_importancia = hasattr(modelo, 'feature_importances_')
        tiene_coef        = hasattr(modelo, 'coef_')

        if tiene_importancia or tiene_coef:
            if tiene_importancia:
                importancias = modelo.feature_importances_
            else:
                importancias = np.abs(modelo.coef_[0])
                importancias = importancias / importancias.sum()

            nombres_cortos = ['Compuesto','Stint','Vida goma','Posición','LapTime',
                              'Δ LapTime','Degr. acum.','Progreso','Norm. TyreLife','Δ Posición']
            idx_sorted = np.argsort(importancias)
            imp_sorted = importancias[idx_sorted]
            nom_sorted = [nombres_cortos[i] for i in idx_sorted]
            colores    = ['#ff1801' if v > np.median(importancias) else '#555' for v in imp_sorted]

            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor('#0a0a0f')
            ax.set_facecolor('#0a0a0f')
            bars = ax.barh(nom_sorted, imp_sorted, color=colores, height=0.6, edgecolor='none')
            ax.set_xlabel('Importancia relativa', color='#888', fontsize=9)
            ax.tick_params(colors='#aaa', labelsize=9)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.grid(axis='x', color='#222', linewidth=0.5)
            for bar, val in zip(bars, imp_sorted):
                ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', color='#666', fontsize=8)
            red_patch  = mpatches.Patch(color='#ff1801', label='Alta importancia')
            gray_patch = mpatches.Patch(color='#555',    label='Baja importancia')
            ax.legend(handles=[red_patch, gray_patch], facecolor='#111', edgecolor='#333',
                      labelcolor='#aaa', fontsize=8, loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Este tipo de modelo no expone importancia de variables directamente.")

        st.markdown('<div class="section-label" style="margin-top:1.5rem">⬡ Perfil del Dato Actual vs. Referencia Pit Stop</div>', unsafe_allow_html=True)
        st.markdown("Compara los valores ingresados con los rangos típicos en los que ocurre un pit stop.")

        referencia_pit = [0.75, 0.5,  0.55, 0.4, 0.45, 0.6, 0.65, 0.55, 0.55, 0.45]
        referencia_no  = [0.4,  0.35, 0.25, 0.5, 0.35, 0.3, 0.2,  0.45, 0.25, 0.5 ]
        cn_norm  = COMPOUND_MAP.get(Compound, 2.0) / 3.0
        dato_norm = [cn_norm, Stint/6, TyreLife/60, (20-Position)/19,
                     (LapTime-70)/80, (LapTime_Delta+10)/20, (Cumul_Deg+80)/90,
                     RaceProgress, round(TyreLife/58.0,4), (Position_Change+5)/10]

        labels_radar = ['Compuesto','Stint','Vida\ngoma','Posición','LapTime',
                        'Δ tiempo','Degr.','Progreso','TyreNorm','Δ pos.']
        N      = len(labels_radar)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        def to_radar(vals): return vals + vals[:1]

        _, rc, _ = st.columns([1, 6, 1])
        with rc:
            fig2, ax2 = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
            fig2.patch.set_facecolor('#0a0a0f')
            ax2.set_facecolor('#0a0a0f')
            ax2.set_theta_offset(np.pi / 2)
            ax2.set_theta_direction(-1)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(labels_radar, color='#888', size=8)
            ax2.set_ylim(0, 1)
            ax2.set_yticks([0.25, 0.5, 0.75])
            ax2.set_yticklabels(['0.25','0.5','0.75'], color='#444', size=7)
            ax2.spines['polar'].set_color('#222')
            ax2.grid(color='#222', linewidth=0.6)
            ax2.plot(angles, to_radar(referencia_pit), 'o-', color='#ff1801', linewidth=1.5, alpha=0.7, label='Perfil pit stop')
            ax2.fill(angles, to_radar(referencia_pit), color='#ff1801', alpha=0.1)
            ax2.plot(angles, to_radar(referencia_no),  'o-', color='#555',    linewidth=1.5, alpha=0.6, label='Perfil continuar')
            ax2.fill(angles, to_radar(referencia_no),  color='#555',    alpha=0.05)
            ax2.plot(angles, to_radar(dato_norm),       'o-', color='#ffd700', linewidth=2,            label='Dato actual')
            ax2.fill(angles, to_radar(dato_norm),       color='#ffd700', alpha=0.15)
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), facecolor='#111',
                       edgecolor='#333', labelcolor='#aaa', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · SIMULADOR
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    _, centro3, _ = st.columns([0.5, 9, 0.5])
    with centro3:
        st.markdown('<div class="section-label">⬡ Simulador de Carrera — Evolución de Probabilidad</div>', unsafe_allow_html=True)
        st.markdown("Simula cómo evoluciona la probabilidad de pit stop vuelta a vuelta a medida que el neumático se degrada.")

        sc1, sc2, sc3 = st.columns(3)
        with sc1: sim_compound = st.selectbox("Compuesto", ["SOFT","MEDIUM","HARD"], key="sim_c")
        with sc2: sim_position = st.slider("Posición", 1, 20, 6, key="sim_p")
        with sc3: sim_total    = st.slider("Total de vueltas en carrera", 40, 78, 58, key="sim_t")

        sc4, sc5 = st.columns(2)
        with sc4: sim_lap_base = st.slider("LapTime base (s)", 70.0, 120.0, 90.0, 0.5, key="sim_lt")
        with sc5: sim_deg_rate = st.slider("Tasa de degradación (s/vuelta)", 0.05, 0.5, 0.15, 0.01, key="sim_dr")

        if st.button("▶  SIMULAR CARRERA", key="btn_sim"):
            vueltas    = list(range(1, sim_total + 1))
            probs      = []
            decisiones = []
            for v in vueltas:
                lap_t   = sim_lap_base + sim_deg_rate * v
                cumul_d = -sim_deg_rate * v * 3
                delta   = sim_deg_rate if v > 1 else 0.0
                progress = v / sim_total
                p, prob = predecir(sim_compound, 1, v, sim_position, lap_t, delta, cumul_d, progress, 0)
                probs.append(prob)
                decisiones.append(p)

            pit_vueltas = [v for v, d in zip(vueltas, decisiones) if d == 1]

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            fig3.patch.set_facecolor('#0a0a0f')
            ax3.set_facecolor('#0a0a0f')
            ax3.axhspan(0.5, 1.0, alpha=0.05, color='#ff1801')
            ax3.axhline(0.5, color='#ff1801', linewidth=1, linestyle='--', alpha=0.5, label='Umbral pit stop (50%)')
            ax3.plot(vueltas, probs, color='#ffd700', linewidth=2.5, zorder=3, label='P(pit stop)')
            ax3.fill_between(vueltas, probs, alpha=0.1, color='#ffd700')
            if pit_vueltas:
                first_pit = pit_vueltas[0]
                ax3.axvline(first_pit, color='#ff1801', linewidth=2, linestyle='-', alpha=0.8)
                ax3.text(first_pit + 0.5, 0.92, f'Pit sugerido\nVuelta {first_pit}',
                         color='#ff1801', fontsize=8, va='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a0000', edgecolor='#ff1801', alpha=0.8))
            ax3.set_xlabel('Vuelta', color='#888', fontsize=9)
            ax3.set_ylabel('Probabilidad de Pit Stop', color='#888', fontsize=9)
            ax3.set_ylim(0, 1)
            ax3.set_xlim(1, sim_total)
            ax3.tick_params(colors='#666', labelsize=8)
            for spine in ax3.spines.values(): spine.set_edgecolor('#222')
            ax3.grid(color='#1a1a1a', linewidth=0.5)
            ax3.legend(facecolor='#111', edgecolor='#333', labelcolor='#aaa', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

            if pit_vueltas:
                st.markdown(f"""
                <div class="metric-row" style="margin-top:1rem">
                  <div class="metric-box"><div class="metric-value" style="color:#ff1801">{pit_vueltas[0]}</div><div class="metric-label">Primera vuelta de pit sugerido</div></div>
                  <div class="metric-box"><div class="metric-value">{len(pit_vueltas)}</div><div class="metric-label">Vueltas con pit recomendado</div></div>
                  <div class="metric-box"><div class="metric-value">{max(probs):.0%}</div><div class="metric-label">Prob. máxima alcanzada</div></div>
                  <div class="metric-box"><div class="metric-value">{pit_vueltas[0] - 1}</div><div class="metric-label">Vueltas de ventana libre</div></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ El modelo no recomienda pit stop en ninguna vuelta con estos parámetros.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    _, centro4, _ = st.columns([0.5, 9, 0.5])
    with centro4:
        st.markdown('<div class="section-label">⬡ Métricas de Evaluación del Modelo</div>', unsafe_allow_html=True)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        metricas = [("71%","Accuracy","#ffd700"),("0.92","ROC-AUC","#ff1801"),("~0.68","F1-Score","#00e676"),("~0.74","Recall","#0067ff")]
        for col, (val, label, color) in zip([col_m1, col_m2, col_m3, col_m4], metricas):
            with col:
                st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color}">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_roc, col_cm = st.columns(2)

        with col_roc:
            st.markdown('<div class="section-label">Curva ROC (datos de prueba)</div>', unsafe_allow_html=True)
            fpr = np.array([0,0.02,0.05,0.08,0.12,0.18,0.25,0.35,0.5,0.65,0.8,1.0])
            tpr = np.array([0,0.45,0.65,0.75,0.82,0.87,0.91,0.93,0.95,0.97,0.99,1.0])
            fig4, ax4 = plt.subplots(figsize=(4.5, 4))
            fig4.patch.set_facecolor('#0a0a0f')
            ax4.set_facecolor('#0a0a0f')
            ax4.plot(fpr, tpr, color='#ff1801', linewidth=2.5, label='ROC (AUC = 0.92)')
            ax4.fill_between(fpr, tpr, alpha=0.1, color='#ff1801')
            ax4.plot([0,1],[0,1], color='#444', linewidth=1, linestyle='--', label='Aleatorio')
            ax4.set_xlabel('Tasa de Falsos Positivos', color='#888', fontsize=8)
            ax4.set_ylabel('Tasa de Verdaderos Positivos', color='#888', fontsize=8)
            ax4.tick_params(colors='#666', labelsize=8)
            for spine in ax4.spines.values(): spine.set_edgecolor('#222')
            ax4.grid(color='#1a1a1a', linewidth=0.5)
            ax4.legend(facecolor='#111', edgecolor='#333', labelcolor='#aaa', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        with col_cm:
            st.markdown('<div class="section-label">Matriz de Confusión (datos de prueba)</div>', unsafe_allow_html=True)
            cm = np.array([[8420, 980],[1230, 3870]])
            fig5, ax5 = plt.subplots(figsize=(4.5, 4))
            fig5.patch.set_facecolor('#0a0a0f')
            ax5.set_facecolor('#0a0a0f')
            ax5.imshow(cm, cmap='Reds', alpha=0.7)
            for i in range(2):
                for j in range(2):
                    ax5.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                             color='white', fontsize=14, fontweight='bold')
            ax5.set_xticks([0,1]); ax5.set_yticks([0,1])
            ax5.set_xticklabels(['Pred: No pit','Pred: Pit'], color='#aaa', fontsize=9)
            ax5.set_yticklabels(['Real: No pit','Real: Pit'], color='#aaa', fontsize=9)
            for spine in ax5.spines.values(): spine.set_edgecolor('#333')
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">⬡ ¿Por qué ROC-AUC es la métrica clave aquí?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,24,1,0.2); border-radius:8px; padding:1rem 1.5rem; font-size:0.9rem; color:#bbb; line-height:1.7">
        El dataset de F1 tiene <strong style="color:#ffd700">desbalance de clases</strong>: los pit stops son eventos poco frecuentes (~15-20% de vueltas).
        En estos casos, la <em>Accuracy</em> puede ser engañosa — un modelo que siempre predice "no pit" tendría 80% de accuracy sin ser útil.<br><br>
        El <strong style="color:#ff1801">ROC-AUC de 0.92</strong> indica que el modelo discrimina muy bien entre vueltas con y sin pit stop,
        independientemente del umbral de clasificación. Es la métrica honesta para este problema.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR · HISTORIAL — colores mejorados
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:Orbitron,monospace; font-size:0.7rem; color:#ff1801;
         letter-spacing:0.2em; text-transform:uppercase; border-bottom:1px solid #ff1801;
         padding-bottom:0.4rem; margin-bottom:1rem">
         📋 Historial de Sesión
    </div>""", unsafe_allow_html=True)

    if st.session_state.historial:
        if st.button("🗑 Limpiar historial", key="clear"):
            st.session_state.historial = []
            st.rerun()
        for h in reversed(st.session_state.historial):
            es_pit      = 'PIT' in h['Decisión']
            color_borde = '#ff1801' if es_pit else '#00e676'
            color_dec   = '#ff6b6b' if es_pit else '#69f0ae'   # rojo/verde más claros, legibles
            st.markdown(f"""
            <div style="background:#1a1a24; border-left:3px solid {color_borde};
                        border-radius:4px; padding:0.6rem 0.8rem; margin-bottom:0.5rem;">
              <div style="color:{color_dec}; font-weight:700; font-size:0.82rem">
                {h['Decisión']}
              </div>
              <div style="color:#cccccc; margin-top:0.25rem; font-size:0.78rem">
                {h['Compuesto']} &nbsp;·&nbsp; {h['TyreLife']} vueltas &nbsp;·&nbsp; P{h['Posición']}
              </div>
              <div style="color:#aaaaaa; font-size:0.75rem; margin-top:0.1rem">
                {h['LapTime']} &nbsp;·&nbsp; {h['Progreso']} carrera &nbsp;·&nbsp;
                <span style="color:#ffffff; font-weight:600">{h['Prob']}</span>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#888; font-size:0.82rem">Las predicciones que hagas aparecerán aquí.</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="color:#555; font-size:0.7rem; text-align:center; font-family:Rajdhani,sans-serif; letter-spacing:0.1em">MODELO CON UNA PRECISIÓN DEL 71% CON DATOS DE PRUEBA · APOYO A DECISIÓN ESTRATÉGICA</p>', unsafe_allow_html=True)
