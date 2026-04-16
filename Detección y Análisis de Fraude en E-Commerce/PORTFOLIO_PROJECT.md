# Fraud Analytics Dashboard
## Detección y Análisis de Fraude en E-Commerce — Nivel Senior

---

## Resumen Ejecutivo

Diseñé y ejecuté un análisis completo de fraude sobre 590,540 transacciones reales de e-commerce (Vesta Corporation), combinando exploración de datos, visualización ejecutiva y análisis financiero de costo. El resultado es un sistema de análisis interactivo que va más allá de métricas técnicas y habla el lenguaje del negocio: **¿cuánto cuesta el fraude y cómo minimizarlo?**

---

## El Problema

Vesta Corporation procesa pagos de e-commerce y necesita detectar el 3.5% de transacciones fraudulentas sin bloquear al 96.5% de clientes legítimos. Este es el problema clásico del desbalance severo en fraude financiero, con una complicación adicional: **los dos tipos de error tienen costos radicalmente distintos.**

Un fraude no detectado puede costar el 100% del monto de la transacción. Bloquear a un cliente legítimo cuesta fricción, pérdida de confianza y riesgo de churn — quizás un 10% del monto. Esta asimetría define toda la estrategia de decisión.

---

## Enfoque y Metodología

Construí el proyecto en 4 capas de análisis progresivas:

**Capa 1 — Auditoría de Datos (NB01)**
Entendí la estructura del dataset antes de tocar números: calidad de datos por grupo de variables, tasas de nulos con implicación de negocio, perfil estadístico de fraude vs. legítima. Establecí que los nulos en variables de identidad no son errores — son señal (solo el 24% de las transacciones tienen datos de dispositivo).

**Capa 2 — ¿Cuándo ocurre el fraude? (NB02)**
Extraje dimensiones temporales desde `TransactionDT` (un timedelta en segundos) para construir un mapa de calor hora × día de la semana. Identifiqué que el horario nocturno (0h–5h) tiene una tasa de fraude significativamente superior — patrón consistente con ataques automatizados mientras el titular de la tarjeta duerme.

**Capa 3 — ¿Dónde y en qué segmento? (NB03)**
Analicé el fraude por tipo de producto, red de tarjeta, tipo de tarjeta, dominio de email y región geográfica. La clave: no solo "qué variable tiene más fraude" sino **el riesgo relativo vs. el promedio global** (risk ratio), que permite priorizar intervenciones de negocio.

**Capa 4 — ¿Cuánto cuesta y cómo optimizarlo? (NB04 — Senior)**
Construí un marco de costos explícito, entrené un modelo de scoring con Random Forest, y calculé la curva de costo total en función del umbral de decisión. El hallazgo central: **el umbral óptimo no es 0.5** — ajustarlo correctamente puede representar un ahorro de varios millones de dólares por período.

---

## Hallazgos Más Relevantes

> "Revisar el 10% de transacciones más sospechosas captura más del 60% del fraude total — un lift de 6x sobre el aleatorio."

| Hallazgo | Impacto |
|---|---|
| Fraude nocturno (0h–5h) × mayor tasa | Habilita alertas automáticas por horario sin impactar experiencia diurna |
| Productos C y R con riesgo 2x | Permite dedicar mayor fricción de verificación a esas categorías |
| Dominios de email inusuales con riesgo 3–5x | Regla de negocio directa: escalar revisión para emails desconocidos |
| Umbral óptimo ≈ 0.15–0.25 (no 0.5) | Ajustar el umbral reduce costos sin entrenar un nuevo modelo |
| Lift de 6–8x en top decil | El equipo de fraude puede ser 6x más eficiente priorizando por score |

---

## Producto Final

Además de los 4 notebooks de análisis, construí un **Dashboard Streamlit interactivo** con:

- Resumen ejecutivo con KPIs en tiempo real
- Análisis temporal con heatmap día × hora interactivo
- Segmentación por producto, tarjeta, email y región con filtros dinámicos
- **Simulador de costo con sliders ajustables** — el usuario puede cambiar los supuestos de costo FN/FP y ver cómo cambia el umbral óptimo en tiempo real

Esto convierte el análisis en una **herramienta operativa** que un equipo de riesgo puede usar directamente.

---

## Habilidades Demostradas

| Área | Habilidades |
|---|---|
| **Datos** | Manejo de datasets de alta dimensionalidad (435 columnas), JOINs, nulos MNAR, desbalance severo |
| **Análisis** | EDA multidimensional, análisis temporal, segmentación, risk ratio, análisis de decil |
| **Modelado** | Random Forest con class_weight, target encoding, threshold optimization |
| **Negocio** | Marco de costos, curva de ganancia, simulación de ahorro, lift analysis |
| **Producto** | Dashboard interactivo con Plotly + Streamlit, exportación modular de datos |
| **Comunicación** | Todos los outputs orientados a stakeholders no técnicos, no a métricas de modelo |

---

## Por Qué Este Proyecto Destaca

La mayoría de proyectos de fraude en portafolio entrenan un modelo, reportan el AUC y terminan. Este proyecto hace algo diferente: **traduce los errores del modelo a dólares perdidos** y demuestra que la decisión de umbral no es técnica — es financiera.

Esto es exactamente lo que hace un analista senior en un equipo de riesgo real:
1. Entiende el negocio antes de tocar datos
2. Cuantifica el impacto, no solo describe patrones
3. Entrega herramientas que el equipo puede operar, no solo reportes estáticos
4. Habla en el lenguaje del CFO y del equipo de operaciones simultáneamente

---

## Próximos Pasos (Posibles Extensiones)

- **SHAP Explainability:** Agregar explicaciones por transacción individual ("¿por qué esta tx fue marcada como fraude?") — esencial en contextos regulatorios
- **Drift Monitoring:** Detectar cuándo el modelo empieza a degradarse al monitorear la distribución de scores semana a semana
- **Reglas de Negocio Automatizadas:** Convertir los hallazgos del EDA en reglas codificadas (ej. email inusual + noche + crédito → threshold más agresivo)
- **API de Scoring en Tiempo Real:** Empaquetar el modelo en una API FastAPI con latencia <100ms

---

## Stack

`Python` · `Pandas` · `Scikit-learn` · `Plotly` · `Streamlit` · `Jupyter` · `Matplotlib` · `Seaborn`

**Dataset:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection)

---

*Data Analyst Portfolio Project · Análisis end-to-end con orientación a negocio*
