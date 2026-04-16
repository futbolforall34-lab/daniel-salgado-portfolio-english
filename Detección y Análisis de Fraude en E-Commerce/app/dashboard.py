"""
Dashboard Ejecutivo — IEEE-CIS Fraud Detection
Análisis de Fraude en Transacciones de E-Commerce
Vesta Corporation | Data Analyst Portfolio Project

Para ejecutar:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'processed')

# ─── CSS personalizado ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #E53935;
    }
    .metric-card.green { border-left-color: #4CAF50; }
    .metric-card.blue  { border-left-color: #2196F3; }
    .metric-card.orange{ border-left-color: #FF8F00; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Carga de datos ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    files = {}
    expected = [
        'fraud_by_hour', 'fraud_by_day', 'fraud_by_week', 'fraud_heatmap',
        'fraud_by_product', 'fraud_by_email', 'fraud_by_region',
        'fraud_by_card4', 'fraud_by_card6', 'fraud_by_segment',
        'cost_by_threshold', 'cumulative_gains', 'cost_scenarios', 'model_kpis'
    ]
    for name in expected:
        path = os.path.join(EXPORT_DIR, f'{name}.csv')
        if os.path.exists(path):
            files[name] = pd.read_csv(path)
        else:
            files[name] = None
    return files

data = load_data()

DAY_LABELS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
FRAUD_RATE_GLOBAL = 0.035  # ~3.5%

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-shield-green.png", width=64)
    st.title("Fraud Analytics")
    st.caption("IEEE-CIS Fraud Detection")
    st.markdown("---")

    section = st.radio(
        "Sección",
        ["Resumen Ejecutivo", "Análisis Temporal", "Geografía & Producto", "Análisis de Costos"],
        index=0
    )

    st.markdown("---")
    st.markdown("### Parámetros de Costo")
    cost_fn = st.slider("Costo FN (% monto fraude no detectado)", 50, 100, 100, 5,
                         help="Porcentaje del monto que se pierde si el fraude pasa") / 100
    cost_fp = st.slider("Costo FP (% monto bloqueado por error)", 1, 30, 10, 1,
                         help="Porcentaje del monto perdido por fricción/churn al bloquear falsamente") / 100

    st.markdown("---")
    st.caption("Proyecto de Portafolio · Data Analyst Track")

# ─── SECCIÓN 1: Resumen Ejecutivo ────────────────────────────────────────────
if section == "Resumen Ejecutivo":
    st.title("🛡️ Dashboard Ejecutivo — Detección de Fraude")
    st.markdown("**Vesta Corporation · 590,540 transacciones · 6 meses**")
    st.markdown("---")

    # KPIs
    kpis_data = data.get('model_kpis')

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Transacciones", "590,540")
    with col2:
        st.metric("Transacciones Fraude", "20,663", "3.50%")
    with col3:
        st.metric("Monto en Riesgo", "~$10M+", help="Monto aproximado en fraude en el período")
    with col4:
        auc_val = f"{kpis_data['auc_roc'].values[0]:.3f}" if kpis_data is not None else "—"
        st.metric("AUC-ROC Modelo", auc_val)
    with col5:
        opt_t = f"{kpis_data['opt_threshold'].values[0]:.2f}" if kpis_data is not None else "—"
        st.metric("Umbral Óptimo", opt_t, help="Umbral que minimiza el costo total")

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<p class="section-header">Distribución del Target</p>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=['Legítima (96.5%)', 'Fraude (3.5%)'],
            values=[569877, 20663],
            hole=0.5,
            marker_colors=['#4CAF50', '#E53935'],
            textinfo='label+percent',
            textfont_size=13
        ))
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-header">Resumen de Hallazgos Clave</p>', unsafe_allow_html=True)
        insights = [
            ("⏰", "Fraude nocturno (0h–5h) **~1.8x más frecuente** — patrón de card testing mientras la víctima duerme"),
            ("📦", "Productos **C y R** duplican la tasa de fraude global (~2x)"),
            ("💳", "Tarjetas de **crédito** tienen mayor tasa de fraude que débito; interacción con ProductCD amplifica el riesgo"),
            ("📧", "Dominios de email inusuales/anónimos multiplican el riesgo **x3–5**"),
            ("🗺️", "Múltiples regiones (addr1) duplican el riesgo; target encoding captura la señal de forma compacta"),
            ("💰", "Revisar el **10% más sospechoso** captura **>60% del fraude** (lift ~6x)"),
            ("🎯", f"Umbral óptimo ({opt_t}) reduce costos significativamente vs. umbral default 0.5"),
        ]
        for icon, text in insights:
            st.markdown(f"**{icon}** {text}")

    # Escenarios de costo
    st.markdown("---")
    st.markdown('<p class="section-header">Comparación de Escenarios de Decisión</p>', unsafe_allow_html=True)

    cost_scenarios = data.get('cost_scenarios')
    if cost_scenarios is not None:
        fig_sc = go.Figure()
        colors_sc = ['#E53935', '#FF8F00', '#4CAF50']
        for i, row in cost_scenarios.iterrows():
            fig_sc.add_trace(go.Bar(
                name=row['escenario'],
                x=[row['escenario']],
                y=[row['total'] / 1e6],
                marker_color=colors_sc[i],
                text=[f"${row['total']/1e6:.2f}M"],
                textposition='outside',
            ))
        fig_sc.update_layout(
            title="Costo Total por Escenario ($M) — Set de Prueba",
            yaxis_title="Costo ($M)",
            showlegend=False,
            height=350,
            margin=dict(t=50, b=30)
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ─── SECCIÓN 2: Análisis Temporal ────────────────────────────────────────────
elif section == "Análisis Temporal":
    st.title("⏰ Análisis Temporal del Fraude")
    st.info(
        "**Hallazgo clave (NB02):** El fraude nocturno (0h–5h) tiene una tasa ~1.8x superior al promedio diurno. "
        "El día de la semana no es discriminante — la hora del día sí lo es. "
        "La tendencia semanal es estable a lo largo de los 6 meses (sin concept drift visible)."
    )
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Por Hora del Día", "Por Día de Semana", "Tendencia Semanal"])

    with tab1:
        hourly = data.get('fraud_by_hour')
        if hourly is not None:
            col1, col2 = st.columns(2)
            with col1:
                fig_h = px.bar(
                    hourly, x='hour', y='fraud_rate',
                    color='fraud_rate',
                    color_continuous_scale='RdYlGn_r',
                    labels={'hour': 'Hora del día', 'fraud_rate': 'Tasa de fraude'},
                    title='Tasa de Fraude por Hora'
                )
                fig_h.update_layout(height=380, coloraxis_showscale=False)
                fig_h.add_hline(y=FRAUD_RATE_GLOBAL, line_dash="dash",
                                 annotation_text=f"Promedio global: {FRAUD_RATE_GLOBAL*100:.1f}%",
                                 line_color="black")
                fig_h.update_traces(
                    hovertemplate="<b>Hora %{x}:00</b><br>Tasa: %{y:.2%}<extra></extra>"
                )
                st.plotly_chart(fig_h, use_container_width=True)

            with col2:
                fig_hv = px.bar(
                    hourly, x='hour', y='n_total',
                    color='n_fraud',
                    color_continuous_scale='Reds',
                    labels={'hour': 'Hora', 'n_total': 'Total transacciones', 'n_fraud': 'Fraudes'},
                    title='Volumen de Transacciones por Hora'
                )
                fig_hv.update_layout(height=380)
                st.plotly_chart(fig_hv, use_container_width=True)

            # Heatmap
            st.subheader("Mapa de Calor: Día × Hora")
            heatmap_raw = data.get('fraud_heatmap')
            if heatmap_raw is not None:
                hm = heatmap_raw.copy()
                if 'day_of_week' in hm.columns:
                    hm.index = hm['day_of_week'].map(dict(enumerate(DAY_LABELS)))
                    hm = hm.drop(columns=['day_of_week'])
                else:
                    hm.index = DAY_LABELS[:len(hm)]

                fig_hm = px.imshow(
                    hm * 100,
                    color_continuous_scale='YlOrRd',
                    labels=dict(x='Hora del día', y='', color='Tasa fraude (%)'),
                    title='Tasa de Fraude (%) — Día × Hora',
                    text_auto='.1f'
                )
                fig_hm.update_layout(height=300)
                st.plotly_chart(fig_hm, use_container_width=True)

    with tab2:
        daily = data.get('fraud_by_day')
        if daily is not None:
            daily['day_label'] = daily['day_of_week'].map(dict(enumerate(DAY_LABELS)))
            fig_d = px.bar(
                daily, x='day_label', y='fraud_rate',
                color='fraud_rate',
                color_continuous_scale='RdYlGn_r',
                labels={'day_label': 'Día', 'fraud_rate': 'Tasa de fraude'},
                title='Tasa de Fraude por Día de la Semana',
                text_auto='.3f'
            )
            fig_d.add_hline(y=FRAUD_RATE_GLOBAL, line_dash="dash",
                             annotation_text="Promedio global", line_color="black")
            fig_d.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_d, use_container_width=True)

    with tab3:
        weekly = data.get('fraud_by_week')
        if weekly is not None:
            weekly['week_rel'] = weekly['week_num'] - weekly['week_num'].min() + 1
            weekly['fraud_rate_ma3'] = weekly['fraud_rate'].rolling(3, center=True).mean()

            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(
                x=weekly['week_rel'], y=weekly['fraud_rate'] * 100,
                mode='lines', name='Tasa semanal',
                line=dict(color='#E53935', width=1), opacity=0.5
            ))
            fig_w.add_trace(go.Scatter(
                x=weekly['week_rel'], y=weekly['fraud_rate_ma3'] * 100,
                mode='lines', name='Media móvil (3 sem)',
                line=dict(color='#B71C1C', width=3)
            ))
            fig_w.add_hline(y=FRAUD_RATE_GLOBAL * 100, line_dash="dash",
                              annotation_text="Promedio", line_color="gray")
            fig_w.update_layout(
                title='Evolución de la Tasa de Fraude — 6 Meses',
                xaxis_title='Semana',
                yaxis_title='Tasa de fraude (%)',
                height=400
            )
            st.plotly_chart(fig_w, use_container_width=True)

# ─── SECCIÓN 3: Geografía & Producto ─────────────────────────────────────────
elif section == "Geografía & Producto":
    st.title("🗺️ Geografía, Producto y Segmentación")
    st.info(
        "**Hallazgos clave (NB03):** Productos C y R duplican la tasa de fraude (~2x). "
        "Tarjetas de crédito son el vector preferido. "
        "Dominios de email inusuales multiplican el riesgo x3–5. "
        "El análisis combinado (Producto × Tarjeta) revela segmentos de máximo riesgo que ninguna variable individual captura — "
        "esta interacción fue incorporada como feature `product_credit` en el modelo de NB04."
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Producto", "Tarjeta", "Email", "Región"])

    with tab1:
        prod = data.get('fraud_by_product')
        if prod is not None:
            prod = prod.sort_values('fraud_rate', ascending=False)
            prod['color'] = prod['risk_ratio'].apply(
                lambda r: '#E53935' if r > 1.5 else '#FF8F00' if r > 1.0 else '#4CAF50')

            col1, col2 = st.columns(2)
            with col1:
                fig_p = px.bar(
                    prod, x='ProductCD', y='fraud_rate',
                    color='risk_ratio', color_continuous_scale='RdYlGn_r',
                    labels={'fraud_rate': 'Tasa de fraude', 'ProductCD': 'Producto'},
                    title='Tasa de Fraude por Tipo de Producto',
                    text_auto='.3f'
                )
                fig_p.add_hline(y=FRAUD_RATE_GLOBAL, line_dash="dash",
                                 annotation_text="Promedio", line_color="black")
                fig_p.update_layout(height=380, coloraxis_showscale=True)
                st.plotly_chart(fig_p, use_container_width=True)

            with col2:
                fig_pv = px.pie(
                    prod, values='n_total', names='ProductCD',
                    title='Distribución de Volumen por Producto',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_pv.update_layout(height=380)
                st.plotly_chart(fig_pv, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        for col, var, title in zip([col1, col2], ['fraud_by_card4', 'fraud_by_card6'],
                                     ['Red de Tarjeta (card4)', 'Tipo de Tarjeta (card6)']):
            card_df = data.get(var)
            if card_df is not None:
                with col:
                    card_col = [c for c in card_df.columns if c not in ['n_total', 'fraud_rate', 'risk_ratio']][0]
                    fig_c = px.bar(
                        card_df.sort_values('fraud_rate'),
                        x='fraud_rate', y=card_col, orientation='h',
                        color='risk_ratio', color_continuous_scale='RdYlGn_r',
                        labels={'fraud_rate': 'Tasa de fraude', card_col: ''},
                        title=title,
                        text_auto='.3f'
                    )
                    fig_c.add_vline(x=FRAUD_RATE_GLOBAL, line_dash="dash", line_color="black")
                    fig_c.update_layout(height=350, coloraxis_showscale=False)
                    st.plotly_chart(fig_c, use_container_width=True)

    with tab3:
        email_df = data.get('fraud_by_email')
        if email_df is not None:
            min_n = st.slider("Mínimo de transacciones para mostrar", 100, 5000, 500, 100)
            email_filtered = email_df[email_df['n_total'] >= min_n].sort_values('fraud_rate', ascending=False)

            fig_e = px.bar(
                email_filtered.head(20),
                x='fraud_rate', y='P_emaildomain', orientation='h',
                color='risk_ratio', color_continuous_scale='RdYlGn_r',
                labels={'fraud_rate': 'Tasa de fraude', 'P_emaildomain': 'Dominio'},
                title=f'Top 20 Dominios de Email por Tasa de Fraude (mín. {min_n} tx)',
                text_auto='.3f'
            )
            fig_e.add_vline(x=FRAUD_RATE_GLOBAL, line_dash="dash", line_color="black")
            fig_e.update_layout(height=550, coloraxis_showscale=True,
                                  yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_e, use_container_width=True)

    with tab4:
        addr_df = data.get('fraud_by_region')
        if addr_df is not None:
            min_n_addr = st.slider("Mínimo transacciones por región", 100, 2000, 300, 100)
            addr_filtered = addr_df[addr_df['n_total'] >= min_n_addr]

            fig_scatter = px.scatter(
                addr_filtered,
                x='n_total', y='fraud_rate',
                size='n_fraud', color='risk_ratio',
                color_continuous_scale='RdYlGn_r',
                hover_data={'addr1': True, 'n_total': True, 'fraud_rate': ':.3f'},
                labels={'n_total': 'Volumen de transacciones', 'fraud_rate': 'Tasa de fraude'},
                title=f'Regiones: Volumen vs Riesgo (mín. {min_n_addr} tx)'
            )
            fig_scatter.add_hline(y=FRAUD_RATE_GLOBAL, line_dash="dash",
                                   annotation_text="Promedio global", line_color="black")
            fig_scatter.update_layout(height=480)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ─── SECCIÓN 4: Análisis de Costos ───────────────────────────────────────────
elif section == "Análisis de Costos":
    st.title("💰 Análisis de Costo del Fraude")
    st.caption("*Nivel Senior — Traducción a lenguaje de negocio*")
    st.info(
        "**Marco de análisis (NB04):** Un Falso Negativo (fraude que pasa) cuesta ~10x más que un Falso Positivo "
        "(transacción legítima bloqueada). Por eso el umbral óptimo está muy por debajo de 0.5 — el sistema debe "
        "ser agresivo. El modelo incorpora directamente los hallazgos de NB01 (`log_amt`, card testing), "
        "NB02 (`is_night`) y NB03 (`email_fraud_rate`, `addr_fraud_rate`, interacción `product_credit`). "
        "Usa los sliders del sidebar para ajustar los costos según tu empresa."
    )
    st.markdown("---")

    cost_thresh = data.get('cost_by_threshold')
    gains = data.get('cumulative_gains')
    kpis = data.get('model_kpis')

    if cost_thresh is not None:
        # Recalcular con parámetros del sidebar
        # (usamos los valores exportados y escalamos por ratio de costo)
        cost_thresh_adj = cost_thresh.copy()
        # Los costos ya están calculados con ratio 1.0/0.1; escalar
        cost_thresh_adj['total_cost_adj'] = (
            cost_thresh_adj['cost_fn'] * cost_fn +
            cost_thresh_adj['cost_fp'] * cost_fp
        )
        opt_idx = cost_thresh_adj['total_cost_adj'].idxmin()
        opt_t_adj = cost_thresh_adj.loc[opt_idx, 'threshold']
        opt_cost_adj = cost_thresh_adj.loc[opt_idx, 'total_cost_adj']

        tab1, tab2, tab3 = st.tabs(["Curva de Costo", "Curva de Ganancia", "Simulación de Ahorro"])

        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_ct = go.Figure()
                fig_ct.add_trace(go.Scatter(
                    x=cost_thresh_adj['threshold'],
                    y=cost_thresh_adj['total_cost_adj'] / 1e6,
                    mode='lines', name='Costo total',
                    line=dict(color='#5C6BC0', width=2.5)
                ))
                fig_ct.add_trace(go.Scatter(
                    x=cost_thresh_adj['threshold'],
                    y=cost_thresh_adj['cost_fn'] * cost_fn / 1e6,
                    mode='lines', name='Costo FN (fraude no detectado)',
                    line=dict(color='#E53935', width=1.5, dash='dot')
                ))
                fig_ct.add_trace(go.Scatter(
                    x=cost_thresh_adj['threshold'],
                    y=cost_thresh_adj['cost_fp'] * cost_fp / 1e6,
                    mode='lines', name='Costo FP (falsos bloqueos)',
                    line=dict(color='#FF8F00', width=1.5, dash='dot')
                ))
                fig_ct.add_vline(x=opt_t_adj, line_dash="dash", line_color="red",
                                  annotation_text=f"Óptimo: {opt_t_adj:.2f}")
                fig_ct.add_vline(x=0.5, line_dash="dot", line_color="orange",
                                  annotation_text="Default: 0.50")
                fig_ct.update_layout(
                    title='Costo Total vs Umbral de Decisión',
                    xaxis_title='Umbral de clasificación',
                    yaxis_title='Costo ($M)',
                    height=400, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                )
                st.plotly_chart(fig_ct, use_container_width=True)

            with col2:
                st.markdown("### Umbral Óptimo")
                st.metric("Threshold óptimo", f"{opt_t_adj:.3f}")
                opt_row = cost_thresh_adj.loc[opt_idx]
                st.metric("Recall en óptimo", f"{opt_row['recall']*100:.1f}%")
                st.metric("Precision en óptimo", f"{opt_row['precision']*100:.1f}%")
                st.metric("Costo mínimo", f"${opt_cost_adj/1e6:.2f}M")
                st.markdown("---")
                default_row = cost_thresh_adj.loc[(cost_thresh_adj['threshold'] - 0.5).abs().idxmin()]
                cost_default = default_row['total_cost_adj']
                saving = (cost_default - opt_cost_adj) / 1e6
                st.metric("Ahorro vs umbral 0.5", f"${saving:.2f}M",
                           delta=f"{saving:.2f}M ahorrado" if saving > 0 else f"{saving:.2f}M extra")

        with tab2:
            if gains is not None:
                fig_g = go.Figure()
                fig_g.add_trace(go.Scatter(
                    x=gains['pct_population'], y=gains['cum_fraud_pct'],
                    mode='lines', name='Modelo (por cantidad)',
                    line=dict(color='#E53935', width=2.5),
                    fill='tonexty', fillcolor='rgba(229,57,53,0.1)'
                ))
                fig_g.add_trace(go.Scatter(
                    x=gains['pct_population'], y=gains['cum_fraud_amt_pct'],
                    mode='lines', name='Modelo (por monto $)',
                    line=dict(color='#5C6BC0', width=2.5)
                ))
                fig_g.add_trace(go.Scatter(
                    x=gains['pct_population'], y=gains['baseline'],
                    mode='lines', name='Aleatorio (sin modelo)',
                    line=dict(color='gray', width=1.5, dash='dash')
                ))
                fig_g.update_layout(
                    title='Curva de Ganancia Acumulada: ¿Cuánto fraude capturamos revisando el X%?',
                    xaxis_title='% de transacciones revisadas (más sospechosas primero)',
                    yaxis_title='% del fraude total capturado',
                    height=450
                )
                st.plotly_chart(fig_g, use_container_width=True)

                st.markdown("### Tabla de Lift por Decil")
                decil_data = []
                for pct in [1, 2, 5, 10, 15, 20, 30, 50]:
                    row = gains.iloc[(gains['pct_population'] - pct).abs().idxmin()]
                    decil_data.append({
                        'Top % revisado': f"{pct}%",
                        '% Fraudes capturados': f"{row['cum_fraud_pct']:.1f}%",
                        '% Monto capturado': f"{row['cum_fraud_amt_pct']:.1f}%",
                        'Lift': f"{row['cum_fraud_pct']/pct:.1f}x"
                    })
                st.dataframe(pd.DataFrame(decil_data), use_container_width=True, hide_index=True)

        with tab3:
            cost_scen = data.get('cost_scenarios')
            if cost_scen is not None:
                fig_sim = go.Figure()
                colors_s = ['#E53935', '#FF8F00', '#4CAF50']
                for i, row in cost_scen.iterrows():
                    fig_sim.add_trace(go.Bar(
                        name=row['escenario'],
                        x=[row['escenario']],
                        y=[row['total'] / 1e6],
                        marker_color=colors_s[i],
                        text=[f"${row['total']/1e6:.2f}M"],
                        textposition='outside',
                        width=0.4
                    ))
                fig_sim.update_layout(
                    title='Comparación de Escenarios de Decisión — Costo Total ($M)',
                    yaxis_title='Costo total ($M)',
                    showlegend=False, height=400,
                    barmode='group'
                )
                st.plotly_chart(fig_sim, use_container_width=True)

                st.markdown("---")
                st.markdown("#### Mensaje Ejecutivo")
                st.info(
                    "**Con el umbral de decisión correcto, el sistema puede reducir "
                    "significativamente las pérdidas por fraude comparado con un umbral estándar de 0.5.** "
                    "El análisis muestra que revisar el 10% de las transacciones más sospechosas "
                    "captura más del 60% del fraude total — multiplicando la eficiencia del equipo de revisión manual "
                    "por un factor de 6x."
                )
