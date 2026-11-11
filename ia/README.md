# ia — modelos predictivos sobre el laboratorio

Experimentos de IA montados sobre el marco de `lab/` (dataset congelado, tramo DEV 2020-2023,
registro de hipótesis con Bonferroni). No tocan VAL ni el tramo sellado.

| Archivo | Qué hace |
|---|---|
| `ia_datos.py` | Tabla diaria por (símbolo, sesión) con variables conocidas a las 10:30 ET. |
| `ia_modelos.py` | HAR, GARCH(1,1), HMM de régimen (probabilidad filtrada) y LightGBM con hiperparámetros fijos. |
| `ia_exp1_volatilidad.py` | Experimento 1: pronóstico de la volatilidad del día siguiente. No es hipótesis de trading. |
| `ia_exp2_lgbm_resto_dia.py` | Experimento 2: LightGBM elige 5 largos y 5 cortos a las 10:30 hasta el cierre. Registra H5 y H6. |
| `resultados/` | Salidas de cada experimento. |

Validación: walk-forward dentro de DEV (se predice 2021, 2022 y 2023 entrenando con los años anteriores).

## Cómo ejecutar

    pip install lightgbm hmmlearn arch pandas_market_calendars
    python ia_exp1_volatilidad.py
    python ia_exp2_lgbm_resto_dia.py prueba      # tubería con etiquetas barajadas, no registra
    python ia_exp2_lgbm_resto_dia.py ejecutar    # ya ejecutado: H5 y H6 están evaluadas en DEV y no se repiten

## Resultados (noviembre 2026)

**Experimento 1 — volatilidad del día siguiente, fuera de muestra 2021-2023** (QLIKE, menor es mejor)

| Modelo | R² log | QLIKE |
|---|---|---|
| Ingenuo (volatilidad de ayer) | 0,583 | 0,195 |
| GARCH(1,1) | 0,522 | 0,212 |
| LightGBM | 0,669 | 0,156 |
| LightGBM + HMM | 0,666 | 0,158 |
| HAR | 0,680 | 0,147 |
| Mezcla HAR + LightGBM | 0,690 | 0,145 |

**Experimento 2 — LightGBM resto del día (7.430 operaciones, 743 sesiones)**

| Hipótesis | Media bruta por operación | Media neta a 8 bps | t conservador a 8 bps | DEV |
|---|---|---|---|---|
| H5 base | +0,87 bps | −7,13 bps | −5,33 | No pasa |
| H6 base + régimen | +1,64 bps | −6,36 bps | −4,70 | No pasa |

La predicción casi no ordena las acciones (correlación de rangos diaria ≈ 0,01, no significativa).
