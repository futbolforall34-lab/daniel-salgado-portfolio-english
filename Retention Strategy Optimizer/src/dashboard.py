"""
Retention Strategy Optimizer — Dashboard
=========================================
People Analytics | Predicción y Prevención de Rotación de Empleados
Jesús Salgado — Portfolio Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retention Strategy Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PATHS ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
MODELS    = BASE / "models"
DATA_DIR  = BASE / "data/processed"

# ─── PALETTE ───────────────────────────────────────────────────────────────────
DANGER   = "#fc8181"
WARNING  = "#f6ad55"
SUCCESS  = "#68d391"
INFO     = "#63b3ed"
ACCENT   = "#e94560"
PURPLE   = "#b794f4"
DARK_BG  = "#0d1117"
CARD_BG  = "#161b27"
BORDER   = "#2d3748"
TEXT     = "#e2e8f0"
SUBTEXT  = "#718096"

# ─── GLOBAL MATPLOTLIB STYLE ───────────────────────────────────────────────────
def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   CARD_BG,
        "axes.edgecolor":   BORDER,
        "axes.labelcolor":  SUBTEXT,
        "xtick.color":      SUBTEXT,
        "ytick.color":      SUBTEXT,
        "grid.color":       BORDER,
        "grid.alpha":       0.6,
        "text.color":       TEXT,
        "legend.facecolor": CARD_BG,
        "legend.edgecolor": BORDER,
        "font.family":      "sans-serif",
        "axes.titlesize":   11,
        "axes.titlecolor":  TEXT,
    })

apply_dark_style()

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  .stApp { background-color: #0d1117; color: #e2e8f0; }
  section[data-testid="stSidebar"] { background-color: #0a0e1a; border-right: 1px solid #2d3748; }
  section[data-testid="stSidebar"] { color: #e2e8f0; }
  section[data-testid="stSidebar"] .stMarkdown,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] li,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div {
    color: #e2e8f0;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, header, footer, div[data-testid="stToolbar"], div[data-testid="stDecoration"] {
    visibility: hidden;
    display: none;
  }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, #0f1923 0%, #111827 40%, #0d1b2a 100%);
    border-left: 5px solid #e94560;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
  }
  .hero h1 { color: #fff; font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
  .hero p  { color: #718096; margin: 0.4rem 0 0; font-size: 0.95rem; }
  .hero .tag {
    display: inline-block;
    background: rgba(233,69,96,0.15);
    color: #e94560;
    border: 1px solid rgba(233,69,96,0.4);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.6rem;
    margin-right: 0.4rem;
  }

  /* ── KPI card ── */
  .kpi {
    background: #161b27;
    border: 1px solid #2d3748;
    border-top: 3px solid var(--kpi-color, #e94560);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
  }
  .kpi-val   { font-size: 1.9rem; font-weight: 800; color: #fff; line-height: 1; }
  .kpi-label { font-size: 0.72rem; color: #718096; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }
  .kpi-sub   { font-size: 0.8rem; color: #a0aec0; margin-top: 0.15rem; }

  /* ── Section header ── */
  .sec-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e94560;
    display: inline-block;
    margin-bottom: 1rem;
  }

  /* ── Insight box ── */
  .insight {
    background: #111827;
    border: 1px solid #2d3748;
    border-left: 4px solid #63b3ed;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    color: #cbd5e0;
    font-size: 0.88rem;
    line-height: 1.65;
    margin: 0.7rem 0;
  }
  .insight b { color: #e2e8f0; }

  /* ── Risk pill ── */
  .pill-crisis  { background:rgba(252,129,129,.15); color:#fc8181; border:1px solid #fc8181; border-radius:20px; padding:.25rem .7rem; font-size:.78rem; font-weight:600; }
  .pill-alert   { background:rgba(246,173, 85,.15); color:#f6ad55; border:1px solid #f6ad55; border-radius:20px; padding:.25rem .7rem; font-size:.78rem; font-weight:600; }
  .pill-stable  { background:rgba(104,211,145,.15); color:#68d391; border:1px solid #68d391; border-radius:20px; padding:.25rem .7rem; font-size:.78rem; font-weight:600; }

  /* ── Metrics override ── */
  div[data-testid="metric-container"] {
    background:#161b27; border:1px solid #2d3748; border-radius:10px; padding:.9rem;
  }

  /* ── Primary button ── */
  .stButton>button {
    background:linear-gradient(135deg,#e94560,#c0392b);
    color:#fff; border:none; border-radius:8px; font-weight:600; padding:.5rem 1.5rem;
    transition: all .2s;
  }
  .stButton>button:hover {
    background:linear-gradient(135deg,#ff5577,#e74c3c);
    transform:translateY(-1px);
    box-shadow:0 4px 15px rgba(233,69,96,.35);
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { background:#0a0e1a; border-bottom:1px solid #2d3748; gap:0; }
  .stTabs [data-baseweb="tab"]      { color:#718096; font-weight:500; padding:.7rem 1.3rem; }
  .stTabs [aria-selected="true"]    { color:#e94560; border-bottom:2px solid #e94560; }

  /* ── Select / slider ── */
  .stSelectbox>div>div, .stMultiSelect>div>div { background:#161b27; border-color:#2d3748; color:#e2e8f0; }
  .stSlider [data-testid="stThumbValue"]        { color:#e2e8f0; }

  /* ── Divider ── */
  hr { border-color:#2d3748 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando modelos...")
def load_resources():
    model      = joblib.load(MODELS / "turnover_predictor.pkl")
    salary_enc = joblib.load(MODELS / "salary_encoder.pkl")
    dept_cols  = joblib.load(MODELS / "dept_columns.pkl")
    df         = pd.read_csv(DATA_DIR / "dashboard_data.csv")
    return model, salary_enc, dept_cols, df


def prepare_features(df: pd.DataFrame, dept_cols, salary_enc) -> pd.DataFrame:
    """Encode df into the feature matrix the model expects.

    Feature order (must match training):
    11 numeric cols → dept dummies (10) → salary_encoded (last)
    """
    X = df.copy()

    # Salary encoding — LabelEncoder expects 1D Series
    try:
        X["salary_encoded"] = salary_enc.transform(X["salary"])
    except Exception:
        # Fallback: alphabetical LabelEncoder order (high=0, low=1, medium=2)
        X["salary_encoded"] = X["salary"].map({"high": 0, "low": 1, "medium": 2})

    # Department one-hot
    X_dept = pd.get_dummies(X["department"], prefix="dept")
    for col in dept_cols:
        if col not in X_dept.columns:
            X_dept[col] = 0
    X_dept = X_dept[dept_cols]

    num_cols = [
        "satisfaction_level", "last_evaluation", "number_project",
        "average_monthly_hours", "time_spend_company", "Work_accident",
        "promotion_last_5years", "top_performer_risk", "overload_flag",
        "stagnation_flag", "burnout_risk",
    ]
    # salary_encoded goes LAST — matches training column order
    return pd.concat([X[num_cols], X_dept, X[["salary_encoded"]]], axis=1)


@st.cache_data(show_spinner="Calculando scores de riesgo...")
def score_dataset(_model, _salary_enc, dept_cols, _df_hash, df):
    """Score full dataset and append risk columns."""
    X     = prepare_features(df, dept_cols, _salary_enc)
    proba = _model.predict_proba(X)[:, 1]
    out   = df.copy()
    out["risk_score"] = proba
    out["risk_tier"]  = pd.cut(
        proba,
        bins=[0, 0.30, 0.60, 1.01],
        labels=["Estable", "En Alerta", "En Crisis"],
        right=False,
    )
    return out


# ── Intervention simulator ─────────────────────────────────────────────────────

INTERVENTIONS = {
    "salary":       {"label": "💰 Aumento Salarial (low → medium, medium → high)", "cost": 500_000,  "color": INFO},
    "promotions":   {"label": "📈 Programa de Promociones (empleados estancados)",   "cost": 200_000,  "color": SUCCESS},
    "hours":        {"label": "⏰ Reducción de Horas (overloaded -15%)",              "cost": 150_000,  "color": WARNING},
    "satisfaction": {"label": "😊 Mejora de Satisfacción (+0.2 donde sat < 0.5)",    "cost": 100_000,  "color": PURPLE},
    "combined":     {"label": "🚀 Programa Combinado (todas las intervenciones)",     "cost": 950_000,  "color": ACCENT},
}
COST_PER_TURNOVER = 50_000


def apply_intervention(df: pd.DataFrame, key: str) -> pd.DataFrame:
    d = df.copy()
    if key in ("salary", "combined"):
        d.loc[d["salary"] == "low",    "salary"] = "medium"
        d.loc[d["salary"] == "medium", "salary"] = "high"
    if key in ("promotions", "combined"):
        mask = d["stagnation_flag"] == 1
        d.loc[mask, "promotion_last_5years"] = 1
        d.loc[mask, "stagnation_flag"]       = 0
    if key in ("hours", "combined"):
        mask = d["overload_flag"] == 1
        d.loc[mask, "average_monthly_hours"] = (
            d.loc[mask, "average_monthly_hours"] * 0.85
        ).astype(int)
        d["overload_flag"] = (d["average_monthly_hours"] > 220).astype(int)
        d["burnout_risk"]  = (
            (d["number_project"] >= 6) & (d["average_monthly_hours"] > 200)
        ).astype(int)
    if key in ("satisfaction", "combined"):
        mask = d["satisfaction_level"] < 0.5
        d.loc[mask, "satisfaction_level"] = np.minimum(
            d.loc[mask, "satisfaction_level"] + 0.2, 1.0
        )
        d["top_performer_risk"] = (
            (d["last_evaluation"] > 0.8) & (d["satisfaction_level"] < 0.5)
        ).astype(int)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

try:
    model, salary_enc, dept_cols, df_raw = load_resources()
    df = score_dataset(model, salary_enc, dept_cols, str(df_raw.shape), df_raw)
    LOADED = True
except FileNotFoundError as exc:
    LOADED = False
    LOAD_ERROR = exc

if not LOADED:
    st.error(
        f"**No se encontraron los archivos de modelos.**\n\n"
        f"Ejecuta los notebooks primero para generar los `.pkl`.\n\n"
        f"Error: `{LOAD_ERROR}`"
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🎯</div>
        <div style="color:#e2e8f0; font-weight:700; font-size:1.05rem; margin-top:.3rem;">Retention Strategy Optimizer</div>
        <div style="color:#718096; font-size:.78rem; margin-top:.2rem;">People Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    n_total    = len(df)
    n_left     = int(df["left"].sum())
    base_rate  = df["left"].mean()
    n_crisis   = int((df["risk_tier"] == "En Crisis").sum())
    n_alert    = int((df["risk_tier"] == "En Alerta").sum())
    n_stable   = int((df["risk_tier"] == "Estable").sum())

    st.markdown("**Resumen del Dataset**")
    st.markdown(f"- Empleados totales: **{n_total:,}**")
    st.markdown(f"- Tasa de rotación base: **{base_rate:.1%}**")
    st.markdown(f"- Costo anual estimado: **$176M USD**")
    st.markdown("---")

    st.markdown("**Segmentos de riesgo (modelo)**")
    st.markdown(f'<span class="pill-crisis">En Crisis</span> {n_crisis:,} empleados ({n_crisis/n_total:.1%})', unsafe_allow_html=True)
    st.markdown(f'<span class="pill-alert">En Alerta</span> {n_alert:,} empleados ({n_alert/n_total:.1%})', unsafe_allow_html=True)
    st.markdown(f'<span class="pill-stable">Estable</span>  {n_stable:,} empleados ({n_stable/n_total:.1%})', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Modelo**")
    st.markdown("- XGBoost  ·  ROC-AUC **0.987**")
    st.markdown("- F1 **0.850**  ·  Recall **86.3%**")
    st.markdown("- Threshold **0.38** (optimizado para RRHH)")
    st.markdown("---")
    st.caption("Dataset: Kaggle Employee Turnover Analytics · 14,999 registros")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>🎯 Retention Strategy Optimizer</h1>
  <p>Sistema predictivo end-to-end para prevenir rotación de empleados y cuantificar el impacto de cada intervención.</p>
  <span class="tag">People Analytics</span>
  <span class="tag">XGBoost · ROC-AUC 0.987</span>
  <span class="tag">SHAP Interpretability</span>
  <span class="tag">ROI Simulator</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Panorama General",
    "🔴 Segmentos de Riesgo",
    "🎯 Predictor Individual",
    "🧪 Simulador ROI",
    "📁 Lab de Datos",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PANORAMA GENERAL
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    kpi_data = [
        ("23.8%",   "Tasa de Rotación",     "3,522 empleados/año",    ACCENT),
        ("$176M",   "Costo Anual (USD)",     "Estimado a $50K/empleado", DANGER),
        ("84.7%",   "Top Performers en Riesgo", "eval>0.8 + sat<0.5", WARNING),
        ("2.1%",    "Promovidos en 5 años", "Estancamiento masivo",   INFO),
    ]
    for col, (val, label, sub, color) in zip([c1, c2, c3, c4], kpi_data):
        with col:
            st.markdown(
                f'<div class="kpi" style="--kpi-color:{color}">'
                f'<div class="kpi-val">{val}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="insight">💡 <b>Hallazgo central:</b> Existen <b>dos tipos de rotación</b> con causas y soluciones completamente distintas. '
                'Los empleados con <i>insatisfacción</i> (sat &lt; 0.4) salen por agotamiento y mala compensación. '
                'Los <i>top performers ignorados</i> (eval &gt; 0.8, sat &lt; 0.5) salen por falta de reconocimiento — y representan el 84.7% de rotación. '
                'Tratarlos con la misma intervención es un error costoso.</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # Chart 1: Rotación por departamento
    with col_l:
        st.markdown('<div class="sec-title">Rotación por Departamento</div>', unsafe_allow_html=True)
        dept_rot = df.groupby("department")["left"].mean().sort_values()
        colors   = [DANGER if v > 0.25 else WARNING if v > 0.18 else INFO for v in dept_rot.values]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(dept_rot.index, dept_rot.values, color=colors, edgecolor="none", height=0.6)
        ax.set_xlabel("Tasa de rotación", labelpad=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.axvline(base_rate, color=ACCENT, linestyle="--", linewidth=1.2, alpha=0.7, label=f"Media {base_rate:.1%}")
        ax.legend(fontsize=9)
        ax.grid(axis="x")
        ax.set_frame_on(False)
        for bar, val in zip(bars, dept_rot.values):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=8.5, color=TEXT)
        fig.tight_layout()
        st.pyplot(fig)

    # Chart 2: Scatter satisfacción vs evaluación
    with col_r:
        st.markdown('<div class="sec-title">Mapa de Riesgo: Satisfacción × Evaluación</div>', unsafe_allow_html=True)

        sample = df.sample(min(3000, len(df)), random_state=42)
        stayed = sample[sample["left"] == 0]
        left_  = sample[sample["left"] == 1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(stayed["satisfaction_level"], stayed["last_evaluation"],
                   c=INFO, alpha=0.25, s=8, label="Permanece")
        ax.scatter(left_["satisfaction_level"],  left_["last_evaluation"],
                   c=DANGER, alpha=0.4, s=10, label="Se fue")

        # Highlight top performer risk zone
        rect = mpatches.FancyBboxPatch((0.0, 0.8), 0.5, 0.21,
                                       boxstyle="round,pad=0.01",
                                       linewidth=1.5, edgecolor=WARNING,
                                       facecolor="none", linestyle="--", zorder=5)
        ax.add_patch(rect)
        ax.text(0.25, 1.03, "⚡ Top Performers", ha="center", fontsize=8.5,
                color=WARNING, fontweight="bold")

        ax.set_xlabel("Nivel de Satisfacción")
        ax.set_ylabel("Última Evaluación")
        ax.legend(markerscale=2, fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.35, 1.07)
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    col_l2, col_r2 = st.columns(2)

    # Chart 3: Horas mensuales vs rotación
    with col_l2:
        st.markdown('<div class="sec-title">Carga Laboral y Rotación</div>', unsafe_allow_html=True)
        bins   = [0, 160, 180, 200, 220, 250, 320]
        labels = ["<160", "160-180", "180-200", "200-220", "220-250", ">250"]
        df["hours_bin"] = pd.cut(df["average_monthly_hours"], bins=bins, labels=labels)
        hours_rot = df.groupby("hours_bin", observed=True)["left"].mean()

        fig, ax = plt.subplots(figsize=(7, 3.5))
        bar_colors = [DANGER if v > 0.3 else WARNING if v > 0.15 else SUCCESS for v in hours_rot.values]
        bars = ax.bar(hours_rot.index, hours_rot.values, color=bar_colors, edgecolor="none", width=0.6)
        ax.set_xlabel("Horas mensuales promedio")
        ax.set_ylabel("Tasa de rotación")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.axhline(base_rate, color=ACCENT, linestyle="--", linewidth=1.2, alpha=0.7)
        for bar, val in zip(bars, hours_rot.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", fontsize=9, color=TEXT)
        ax.grid(axis="y")
        ax.set_frame_on(False)
        fig.tight_layout()
        st.pyplot(fig)

    # Chart 4: Salary + salary x rotación
    with col_r2:
        st.markdown('<div class="sec-title">Rotación por Nivel Salarial</div>', unsafe_allow_html=True)
        sal_order = ["low", "medium", "high"]
        sal_rot   = df.groupby("salary")["left"].mean().reindex(sal_order)
        sal_n     = df["salary"].value_counts().reindex(sal_order)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = [DANGER, WARNING, SUCCESS]
        bars   = ax.bar(["Bajo", "Medio", "Alto"], sal_rot.values, color=colors, edgecolor="none", width=0.5)
        for bar, val, n in zip(bars, sal_rot.values, sal_n.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}\n(n={n:,})", ha="center", fontsize=9, color=TEXT)
        ax.set_ylabel("Tasa de rotación")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.axhline(base_rate, color=ACCENT, linestyle="--", linewidth=1.2, alpha=0.7, label=f"Media {base_rate:.1%}")
        ax.legend(fontsize=9)
        ax.grid(axis="y")
        ax.set_frame_on(False)
        fig.tight_layout()
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEGMENTOS DE RIESGO
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="sec-title">Segmentación por Score de Riesgo (XGBoost)</div>', unsafe_allow_html=True)

    # Tier counts
    tier_counts = df["risk_tier"].value_counts()
    tier_rot    = df.groupby("risk_tier", observed=True)["left"].mean()

    c1, c2, c3 = st.columns(3)
    for col, tier, pill_class in zip(
        [c1, c2, c3],
        ["En Crisis", "En Alerta", "Estable"],
        ["pill-crisis", "pill-alert", "pill-stable"],
    ):
        with col:
            count = tier_counts.get(tier, 0)
            rot   = tier_rot.get(tier, 0)
            st.markdown(
                f'<div class="kpi" style="--kpi-color:{DANGER if tier=="En Crisis" else WARNING if tier=="En Alerta" else SUCCESS}">'
                f'<div class="kpi-val">{count:,}</div>'
                f'<div class="kpi-label">{tier}</div>'
                f'<div class="kpi-sub">Rotación real: {rot:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # Donut chart
    with col_l:
        st.markdown('<div class="sec-title">Distribución de Riesgo</div>', unsafe_allow_html=True)
        labels = ["En Crisis", "En Alerta", "Estable"]
        sizes  = [tier_counts.get(t, 0) for t in labels]
        colors = [DANGER, WARNING, SUCCESS]

        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 3},
        )
        for t in autotexts:
            t.set_color(DARK_BG)
            t.set_fontweight("bold")
            t.set_fontsize(10)
        # Centre hole for donut
        centre_circle = plt.Circle((0, 0), 0.55, fc=CARD_BG)
        ax.add_patch(centre_circle)
        ax.text(0, 0, f"{n_total:,}\nempleados", ha="center", va="center",
                color=TEXT, fontsize=11, fontweight="bold")
        legend_handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
        ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
                  ncol=3, frameon=False, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)

    # Feature flags por tier
    with col_r:
        st.markdown('<div class="sec-title">Señales de Riesgo por Segmento</div>', unsafe_allow_html=True)
        flags = ["top_performer_risk", "burnout_risk", "stagnation_flag", "overload_flag"]
        flag_labels = ["Top Performer Riesgo", "Burnout", "Estancamiento", "Sobrecarga"]

        tier_flags = df.groupby("risk_tier", observed=True)[flags].mean()
        tier_flags = tier_flags.reindex(["En Crisis", "En Alerta", "Estable"])

        fig, ax = plt.subplots(figsize=(6, 4))
        x   = np.arange(len(flag_labels))
        w   = 0.25
        bar_colors = [DANGER, WARNING, SUCCESS]
        for i, (tier, color) in enumerate(zip(["En Crisis", "En Alerta", "Estable"], bar_colors)):
            vals = tier_flags.loc[tier].values if tier in tier_flags.index else [0] * 4
            ax.bar(x + i * w, vals, width=w, color=color, edgecolor="none", label=tier, alpha=0.9)

        ax.set_xticks(x + w)
        ax.set_xticklabels(flag_labels, fontsize=8.5)
        ax.set_ylabel("% de empleados con la señal")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(fontsize=9, frameon=True)
        ax.grid(axis="y")
        ax.set_frame_on(False)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # Risk score distribution
    st.markdown('<div class="sec-title">Distribución del Score de Riesgo</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    stayed_scores = df.loc[df["left"] == 0, "risk_score"]
    left_scores   = df.loc[df["left"] == 1, "risk_score"]

    ax.hist(stayed_scores, bins=60, color=INFO,   alpha=0.6, label="Permanece (real)", density=True)
    ax.hist(left_scores,   bins=60, color=DANGER, alpha=0.6, label="Se fue (real)",    density=True)
    ax.axvline(0.38, color=WARNING, linestyle="--", linewidth=1.5, label="Threshold 0.38")
    ax.axvline(0.60, color=ACCENT,  linestyle="--", linewidth=1.5, label="Threshold alto 0.60")
    ax.set_xlabel("Probabilidad de rotación predicha")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=9)
    ax.grid(axis="y")
    ax.set_frame_on(False)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="insight">📌 <b>Por qué threshold 0.38:</b> En RRHH, una falsa alarma cuesta una conversación preventiva. '
                'Un falso negativo cuesta perder un empleado valioso. Bajando el threshold a 0.38 capturamos el <b>86.3%</b> '
                'de las rotaciones reales con solo un 16.3% de falsas alarmas — una asimetría que justifica el ajuste.</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTOR INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="sec-title">Evaluador de Riesgo Individual</div>', unsafe_allow_html=True)
    st.markdown("Ingresa las características de un empleado para obtener su probabilidad de rotación y las recomendaciones de retención.")

    st.markdown("<br>", unsafe_allow_html=True)
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("**Datos del empleado**")

        satisfaction  = st.slider("Nivel de Satisfacción", 0.0, 1.0, 0.45, 0.05,
                                   help="0 = muy insatisfecho, 1 = muy satisfecho")
        evaluation    = st.slider("Última Evaluación", 0.0, 1.0, 0.82, 0.05,
                                   help="Puntuación de desempeño")
        monthly_hours = st.slider("Horas Mensuales Promedio", 96, 310, 265, 5)
        num_projects  = st.slider("Número de Proyectos", 2, 7, 6)
        tenure        = st.slider("Años de Antigüedad", 2, 10, 6)

        col_a, col_b = st.columns(2)
        with col_a:
            salary      = st.selectbox("Salario", ["low", "medium", "high"],
                                        format_func=lambda x: {"low": "Bajo", "medium": "Medio", "high": "Alto"}[x])
            department  = st.selectbox("Departamento", sorted(df["department"].unique()))
        with col_b:
            accident    = st.checkbox("Accidente laboral")
            promoted    = st.checkbox("Promovido en 5 años")

        predict_btn = st.button("🔮 Calcular Riesgo", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            # Derived features
            overload    = int(monthly_hours > 220)
            stagnation  = int(tenure >= 5 and not promoted)
            burnout     = int(num_projects >= 6 and monthly_hours > 200)
            top_perf    = int(evaluation > 0.8 and satisfaction < 0.5)

            row = pd.DataFrame([{
                "satisfaction_level":   satisfaction,
                "last_evaluation":      evaluation,
                "number_project":       num_projects,
                "average_monthly_hours": monthly_hours,
                "time_spend_company":   tenure,
                "Work_accident":        int(accident),
                "promotion_last_5years": int(promoted),
                "top_performer_risk":   top_perf,
                "overload_flag":        overload,
                "stagnation_flag":      stagnation,
                "burnout_risk":         burnout,
                "department":           department,
                "salary":               salary,
            }])

            X_row = prepare_features(row, dept_cols, salary_enc)
            proba = float(model.predict_proba(X_row)[:, 1][0])

            # Risk level
            if proba >= 0.70:
                risk_label = "RIESGO CRÍTICO"
                risk_color = DANGER
                risk_emoji = "🔴"
            elif proba >= 0.38:
                risk_label = "RIESGO ALTO"
                risk_color = WARNING
                risk_emoji = "🟠"
            elif proba >= 0.20:
                risk_label = "RIESGO MEDIO"
                risk_color = "#fbd38d"
                risk_emoji = "🟡"
            else:
                risk_label = "BAJO RIESGO"
                risk_color = SUCCESS
                risk_emoji = "🟢"

            # ── Gauge ────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={"aspect": "equal"})
            fig.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(DARK_BG)

            theta_start, theta_end = np.pi, 0
            zones = [(0.0, 0.20, SUCCESS), (0.20, 0.38, "#fbd38d"),
                     (0.38, 0.70, WARNING), (0.70, 1.0, DANGER)]
            for lo, hi, col in zones:
                theta_lo = theta_start + (theta_start - theta_end) * (lo - 0) / 1
                theta_hi = theta_start + (theta_start - theta_end) * (hi - 0) / 1
                t = np.linspace(theta_lo, theta_hi, 60)
                ax.fill_between(
                    np.concatenate([0.65 * np.cos(t), 1.0 * np.cos(t[::-1])]),
                    np.concatenate([0.65 * np.sin(t), 1.0 * np.sin(t[::-1])]),
                    color=col, alpha=0.75,
                )

            # Needle
            angle  = np.pi * (1 - proba)
            needle_x = [0, 0.8 * np.cos(angle)]
            needle_y = [0, 0.8 * np.sin(angle)]
            ax.plot(needle_x, needle_y, color=TEXT, linewidth=3, zorder=5)
            ax.add_patch(plt.Circle((0, 0), 0.07, color=TEXT, zorder=6))

            ax.text(0, -0.15, f"{proba:.1%}", ha="center", va="center",
                    fontsize=24, fontweight="bold", color=risk_color)
            ax.text(0, -0.45, risk_label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=risk_color)

            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.6, 1.2)
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)

            # ── Active flags ──────────────────────────────────────────────────
            st.markdown("**Señales de riesgo detectadas:**")
            flags_detected = []
            if top_perf:
                flags_detected.append(("⚡ Top Performer en Riesgo", DANGER, "Alta evaluación + baja satisfacción"))
            if burnout:
                flags_detected.append(("🔥 Burnout Activo", WARNING, f"≥6 proyectos + {monthly_hours} hrs/mes"))
            if stagnation:
                flags_detected.append(("📉 Estancamiento", "#b794f4", f"{tenure} años sin promoción"))
            if overload:
                flags_detected.append(("⏰ Sobrecarga", "#fbd38d", f"{monthly_hours} hrs/mes (>220)"))
            if not flags_detected:
                flags_detected.append(("✅ Sin señales críticas", SUCCESS, "Perfil estable"))

            for flag, color, detail in flags_detected:
                st.markdown(
                    f'<div style="background:#161b27;border:1px solid {color};border-left:4px solid {color};'
                    f'border-radius:8px;padding:.6rem 1rem;margin:.3rem 0;color:{color};font-weight:600;font-size:.88rem;">'
                    f'{flag} <span style="color:{SUBTEXT};font-weight:400;font-size:.82rem;"> — {detail}</span></div>',
                    unsafe_allow_html=True,
                )

            # ── Recommendations ───────────────────────────────────────────────
            st.markdown("**Acciones recomendadas:**")
            recs = []
            if top_perf:
                recs += ["📈 Plan de carrera formal con hitos concretos",
                         "💰 Revisión salarial y bono por desempeño",
                         "💬 One-on-one para identificar aspiraciones"]
            if burnout:
                recs += ["⏰ Redistribuir proyectos — reducir a máx. 4",
                         "🏖️ Revisión de carga de trabajo inmediata"]
            if stagnation:
                recs += ["🎯 Propuesta de promoción en los próximos 3 meses",
                         "📚 Programa de mentoría y desarrollo de liderazgo"]
            if not recs:
                recs = ["✅ Mantener prácticas actuales de gestión",
                        "📊 Monitoreo trimestral del engagement"]

            for rec in recs:
                st.markdown(f"- {rec}")
        else:
            st.markdown(
                '<div style="background:#161b27;border:2px dashed #2d3748;border-radius:12px;'
                'padding:3rem;text-align:center;color:#4a5568;">'
                '<div style="font-size:3rem;">🎯</div>'
                '<div style="margin-top:.8rem;font-size:1rem;font-weight:600;">Configura el perfil del empleado</div>'
                '<div style="font-size:.85rem;margin-top:.4rem;">y haz clic en Calcular Riesgo</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SIMULADOR ROI
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="sec-title">Simulador de Intervenciones — Impacto Económico</div>', unsafe_allow_html=True)
    st.markdown(
        "El simulador modifica las condiciones del dataset según la intervención elegida, "
        "re-puntúa a todos los empleados con el modelo XGBoost y calcula el ahorro generado."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    intervention_key = st.selectbox(
        "Selecciona la intervención:",
        list(INTERVENTIONS.keys()),
        format_func=lambda k: INTERVENTIONS[k]["label"],
    )

    sim_btn = st.button("🚀 Simular Intervención", type="primary")

    if sim_btn:
        iv        = INTERVENTIONS[intervention_key]
        df_mod    = apply_intervention(df_raw, intervention_key)
        X_mod     = prepare_features(df_mod, dept_cols, salary_enc)
        proba_mod = model.predict_proba(X_mod)[:, 1]

        base_rate_model = df["risk_score"].mean()
        new_rate_model  = proba_mod.mean()
        reduction_pct   = (base_rate_model - new_rate_model) / base_rate_model * 100

        n_emp         = len(df)
        base_cost     = n_emp * base_rate_model * COST_PER_TURNOVER
        new_cost      = n_emp * new_rate_model  * COST_PER_TURNOVER
        savings       = base_cost - new_cost
        investment    = iv["cost"]
        roi           = (savings - investment) / investment * 100

        # ── Metrics ──────────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rotación antes",     f"{base_rate_model:.1%}")
        m2.metric("Rotación después",   f"{new_rate_model:.1%}",    delta=f"{new_rate_model - base_rate_model:.1%}")
        m3.metric("Reducción",          f"-{reduction_pct:.1f}%",   delta=f"-{reduction_pct:.1f}%")
        m4.metric("Ahorro estimado",    f"${savings:,.0f} USD")

        # ── Charts ────────────────────────────────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="sec-title">Antes vs Después</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            rates   = [base_rate_model, new_rate_model]
            colors_ = [DANGER, SUCCESS]
            labels_ = ["Antes", "Después"]
            bars    = ax.bar(labels_, rates, color=colors_, edgecolor="none", width=0.45)
            ax.set_ylabel("Tasa de rotación predicha")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
            for bar, val in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.2%}", ha="center", fontsize=12, fontweight="bold", color=TEXT)
            ax.set_ylim(0, max(rates) * 1.25)
            ax.grid(axis="y")
            ax.set_frame_on(False)
            fig.tight_layout()
            st.pyplot(fig)

        with col_r:
            st.markdown('<div class="sec-title">Distribución de Scores</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.hist(df["risk_score"],  bins=50, color=DANGER, alpha=0.55, density=True, label="Situación actual")
            ax.hist(proba_mod,          bins=50, color=SUCCESS, alpha=0.55, density=True, label="Con intervención")
            ax.axvline(0.38, color=WARNING, linestyle="--", linewidth=1.5)
            ax.set_xlabel("Probabilidad de rotación")
            ax.set_ylabel("Densidad")
            ax.legend(fontsize=9)
            ax.grid(axis="y")
            ax.set_frame_on(False)
            fig.tight_layout()
            st.pyplot(fig)

        # ── Economic panel ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sec-title">Análisis Económico</div>', unsafe_allow_html=True)

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Costo base anual",     f"${base_cost:,.0f}")
        e2.metric("Costo con programa",   f"${new_cost:,.0f}")
        e3.metric("Inversión estimada",   f"${investment:,.0f}")
        e4.metric("ROI del programa",     f"{roi:,.0f}%")

        st.markdown(
            f'<div class="insight">💰 <b>Lectura:</b> Con la intervención <b>{iv["label"]}</b>, '
            f'el modelo predice una reducción del <b>{reduction_pct:.1f}%</b> en la tasa de rotación. '
            f'Con un costo de reemplazo de <b>$50,000 USD</b> por empleado, el ahorro estimado es de '
            f'<b>${savings:,.0f} USD anuales</b> con una inversión de <b>${investment:,.0f} USD</b> — '
            f'un ROI de <b>{roi:,.0f}%</b>.</div>',
            unsafe_allow_html=True,
        )

    else:
        # Summary table of all interventions (from NB05 results)
        st.markdown("**Resumen de todas las intervenciones (resultados de NB05):**")
        summary = pd.DataFrame({
            "Intervención":      [INTERVENTIONS[k]["label"] for k in INTERVENTIONS],
            "Reducción rotación": ["-34.9%", "-27.0%", "-22.4%", "-18.3%", "-53.8%"],
            "Inversión":         ["$500K",  "$200K",  "$150K",  "$100K",  "$950K"],
            "Ahorro estimado":   ["$61.5M", "$47.6M", "$39.5M", "$32.2M", "$94.8M"],
            "ROI":               ["12,230%","23,675%","26,267%","30,800%","9,879%"],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="insight">🔑 <b>Nota sobre el combinado:</b> El ROI del programa combinado (9,879%) es '
            'menor que el de intervenciones individuales porque los efectos no son simplemente aditivos — '
            'existe solapamiento entre segmentos. Sin embargo, genera el mayor ahorro absoluto: '
            '<b>$94.8M USD</b> con payback &lt; 4 días.</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LAB DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="sec-title">Explorador de Datos</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dept_filter = st.multiselect("Filtrar departamento", sorted(df["department"].unique()))
    with col_f2:
        sal_filter = st.multiselect("Filtrar salario", ["low", "medium", "high"])
    with col_f3:
        tier_filter = st.multiselect("Filtrar tier de riesgo", ["En Crisis", "En Alerta", "Estable"])

    df_view = df.copy()
    if dept_filter:
        df_view = df_view[df_view["department"].isin(dept_filter)]
    if sal_filter:
        df_view = df_view[df_view["salary"].isin(sal_filter)]
    if tier_filter:
        df_view = df_view[df_view["risk_tier"].isin(tier_filter)]

    st.markdown(f"**{len(df_view):,} registros** (de {len(df):,})")

    # Styled dataframe
    display_cols = [
        "satisfaction_level", "last_evaluation", "average_monthly_hours",
        "time_spend_company", "department", "salary",
        "top_performer_risk", "burnout_risk", "stagnation_flag",
        "risk_score", "risk_tier", "left",
    ]
    st.dataframe(
        df_view[display_cols]
        .sort_values("risk_score", ascending=False)
        .head(500)
        .style.format({"risk_score": "{:.3f}", "satisfaction_level": "{:.2f}", "last_evaluation": "{:.2f}"}),
        use_container_width=True,
        height=380,
    )

    st.markdown("---")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Estadísticas descriptivas**")
        st.dataframe(df_view[["satisfaction_level", "last_evaluation",
                               "average_monthly_hours", "time_spend_company",
                               "risk_score"]].describe().round(3),
                     use_container_width=True)
    with col_s2:
        st.markdown("**Distribución de variables derivadas**")
        flag_summary = pd.DataFrame({
            "Variable": ["top_performer_risk", "burnout_risk", "stagnation_flag", "overload_flag"],
            "Activos":  [int(df_view[f].sum()) for f in ["top_performer_risk", "burnout_risk", "stagnation_flag", "overload_flag"]],
            "% del total": [f"{df_view[f].mean():.1%}" for f in ["top_performer_risk", "burnout_risk", "stagnation_flag", "overload_flag"]],
            "Tasa rotación": [f"{df_view[df_view[f]==1]['left'].mean():.1%}" if df_view[f].sum() > 0 else "—"
                              for f in ["top_performer_risk", "burnout_risk", "stagnation_flag", "overload_flag"]],
        })
        st.dataframe(flag_summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#4a5568; font-size:.8rem; padding:.5rem 0;">'
    '<b style="color:#718096;">Retention Strategy Optimizer</b> · People Analytics Dashboard · '
    'Jesús Salgado · Dataset público Kaggle · XGBoost ROC-AUC 0.987'
    '</div>',
    unsafe_allow_html=True,
)
