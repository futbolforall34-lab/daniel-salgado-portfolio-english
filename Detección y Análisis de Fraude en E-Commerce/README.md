# Detección de Fraude en E-Commerce con Análisis de Costo
### IEEE-CIS Fraud Detection | Data Analyst Track | Portafolio de Ciencia de Datos

---

## Problema de Negocio

Vesta Corporation procesa millones de transacciones de e-commerce por mes. El fraude representa el 3.5% de las transacciones pero concentra un porcentaje desproporcionado del capital en riesgo. El desafío no es solo detectar el fraude — es **tomar la decisión correcta de bloqueo** considerando el costo real de cada tipo de error.

| Tipo de Error | Descripción | Costo |
|---|---|---|
| **Falso Negativo (FN)** | Fraude que pasa desapercibido | 100% del monto de la transacción |
| **Falso Positivo (FP)** | Cliente legítimo bloqueado | ~10% del monto (fricción + riesgo de churn) |

**El costo asimétrico entre FN y FP significa que el umbral de decisión óptimo NO es 0.5 — es una decisión de negocio.**

---

## Dataset

| Atributo | Valor |
|---|---|
| Fuente | Vesta Corporation / Kaggle IEEE-CIS Fraud Detection |
| Período | 6 meses de transacciones de e-commerce |
| Tamaño | 590,540 transacciones × 435 columnas (join de 2 tablas) |
| Target | `isFraud` — binario, 3.5% fraude (desbalance severo) |
| Tablas | `train_transaction.csv` + `train_identity.csv` (LEFT JOIN por `TransactionID`) |

**Familias de variables:**
- Transaccionales: `TransactionAmt`, `ProductCD`, `TransactionDT`
- Tarjeta: `card1`–`card6` (red, tipo, atributos hasheados)
- Geografía: `addr1` (región), `addr2` (país), `dist1`, `dist2`
- Email: `P_emaildomain`, `R_emaildomain`
- Conteos históricos: `C1`–`C14` (comportamiento de la tarjeta)
- Timedeltas: `D1`–`D15` (tiempo entre eventos)
- Match features: `M1`–`M9` (inconsistencias entre atributos)
- Vesta features: `V1`–`V339` (engineered features propietarias)
- Identidad: `id_01`–`id_38`, `DeviceType`, `DeviceInfo`

---

## Metodología y Pipeline

```
train_transaction.csv  ─┐
                         ├─ LEFT JOIN → DataFrame unificado (590,540 × 435)
train_identity.csv     ─┘
                              │
                         NB01: Auditoría
                         • Calidad de datos por grupo
                         • Perfil de fraude vs. legítima
                         • Distribución del target
                              │
                         NB02: Análisis Temporal
                         • Feature engineering: hour, day_of_week, is_night, week_num
                         • Tasa de fraude por hora/día
                         • Heatmap día × hora
                         • Tendencia semanal (6 meses)
                              │
                         NB03: Geografía & Producto
                         • Fraude por ProductCD, card4, card6
                         • Top dominios de email de alto riesgo
                         • Scatter: regiones por volumen vs. riesgo
                         • Análisis de segmentos combinados
                              │
                         NB04: Análisis de Costo (Senior)
                         • Definición de marco de costos
                         • Modelo de scoring (Random Forest)
                         • Curva de costo por umbral
                         • Curva de ganancia acumulada
                         • Simulación de ahorro por escenario
                              │
                         Dashboard Streamlit
                         • Resumen ejecutivo interactivo
                         • Sliders de parámetros de costo
                         • Curvas dinámicas de umbral óptimo
```

---

## Resultados Principales

### Análisis Temporal
- **El fraude nocturno (0h–5h) tiene tasa significativamente superior al promedio** (~1.8x más riesgo)
- El heatmap día × hora revela ventanas de ataque específicas, permitiendo reglas de negocio precisas
- La tasa de fraude es estable en los 6 meses — sin evento de brecha masiva visible

### Geografía y Segmentación
- **Productos C y R** tienen tasa de fraude >2x el promedio global
- **Tarjetas de crédito** (card6) tienen mayor tasa que débito — vector preferido de fraude
- **Dominios de email inusuales** tienen tasas de fraude 3–5x superiores al promedio
- Múltiples **regiones (addr1)** duplican o triplican el riesgo local

### Análisis de Costo
| Métrica | Valor |
|---|---|
| AUC-ROC del modelo base | ~0.85+ |
| Umbral óptimo (costo mínimo) | ~0.15–0.25 (vs. 0.5 default) |
| Recall con umbral óptimo | ~70–80% |
| Lift revisando top 10% | ~6–8x el aleatorio |
| Ahorro estimado vs. umbral default | Varios $M por período |

---

## Decisiones Técnicas

| Decisión | Justificación |
|---|---|
| Usar tasa de fraude (no conteo) en todos los gráficos | El desbalance 96.5/3.5 hace que los conteos sean engañosos |
| Mínimo de 200–300 transacciones para análisis de email/región | Evitar regiones con tasa de fraude alta por baja muestra |
| Random Forest ligero (max_depth=8) en NB04 | El objetivo es el análisis de costo, no la optimización del modelo |
| LEFT JOIN transaction→identity | Solo 24% tiene datos de identidad; el merge correcto preserva todas las transacciones |
| Curva de ganancia por monto además de por conteo | El fraude de alto valor tiene mayor prioridad en revisión manual |

---

## Recomendaciones de Negocio

1. **Implementar monitoreo reforzado nocturno (0h–5h):** La tasa de fraude en este período justifica autenticación adicional o revisión manual automática.

2. **Ajustar el umbral de decisión basado en costos reales:** El umbral óptimo NO es 0.5. Con parámetros típicos de la industria (FN=100%, FP=10%), el umbral óptimo está en el rango 0.15–0.25.

3. **Escalar revisión manual por decil de riesgo:** Revisar el 10% más sospechoso captura >60% del fraude. Esto permite al equipo de fraude triplicar su eficiencia sin incrementar headcount.

4. **Crear reglas de negocio por segmento:** Los análisis univariados son insuficientes. Las combinaciones (ej. ProductCD=C + tarjeta crédito + email inusual) concentran el riesgo de forma multiplicativa.

5. **Tratar los nulos de identidad como señal:** La ausencia de `DeviceType`, `id_XX` no es un dato faltante — es información: muchos fraudes ocurren sin datos de identidad.

---

## Estructura del Proyecto

```
NUEVO_PROYECTO/
├── notebooks/
│   ├── NB01_auditoria_carga.ipynb        # Auditoría y calidad de datos
│   ├── NB02_fraude_temporal.ipynb        # Análisis temporal
│   ├── NB03_fraude_geo_producto.ipynb    # Geografía, producto y email
│   └── NB04_costo_fraude.ipynb           # Análisis de costo (Senior)
├── data/
│   └── exports/                          # CSVs agregados para dashboard
├── assets/                               # Gráficos PNG exportados
├── dashboard.py                          # Dashboard Streamlit interactivo
├── train_transaction.csv                 # Dataset principal (683 MB)
├── train_identity.csv                    # Dataset de identidad (26 MB)
├── diccionario_variables_fraude_IEEE_CIS.txt
├── README.md                             # Este documento
└── PORTFOLIO_PROJECT.md                  # Versión showcase para portafolio
```

## Cómo Ejecutar

```bash
# Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn plotly streamlit

# Ejecutar notebooks en orden
jupyter notebook notebooks/NB01_auditoria_carga.ipynb

# Lanzar dashboard (después de ejecutar los notebooks)
streamlit run dashboard.py
```

---

## Stack Tecnológico

| Herramienta | Uso |
|---|---|
| Python 3.9+ | Lenguaje principal |
| Pandas | Manipulación de datos |
| NumPy | Cálculos numéricos |
| Matplotlib / Seaborn | Visualizaciones en notebooks |
| Scikit-learn | Modelo de scoring (Random Forest) |
| Plotly | Visualizaciones interactivas en dashboard |
| Streamlit | Dashboard ejecutivo interactivo |
| Jupyter Notebook | Entorno de análisis |

---

*Proyecto de portafolio — Data Analyst Track · IEEE-CIS Fraud Detection Dataset (Vesta Corporation)*
