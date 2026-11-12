# adv_trading — redes para decidir compra o venta en SOXL

Esta carpeta necesita el repositorio completo: usa el marco de pruebas de `lab/`.

## Qué hay

| Archivo | Qué hace |
|---|---|
| `preparar_datos_soxl.py` | Descarga, audita, limpia y congela las velas de 5 min de SOXL y su universo (2016 → hoy). |
| `red_politica.py` | La red de política para SOXL: `prueba`, `explorar` y `ejecutar` (hipótesis H11). |
| `config_final.json` | Configuración elegida (estructura, tamaño, regularización, número de semillas). |
| `pol_datos.py` | De velas de 5 min a 12 decisiones por sesión con 40 variables. |
| `pol_modelos.py` | Ocho arquitecturas: lineal, MLP, LSTM, VSN+LSTM, TFT, Transformer, xLSTM, VSN+xLSTM. |
| `pol_entrenar.py` | Entrenamiento por Sharpe ("backtest diferenciable") y conversión de la señal en operaciones. |
| `pol_sintetico.py` | Mundo nulo y señales plantadas para comprobar el método sin mirar resultados reales. |
| `exploracion/` | Comparación de arquitecturas con datos anteriores a 2020 y sus tablas de resultados. |
| `red_secuencias.py` | Experimento anterior (H10: TCN + atención). Ya evaluado. |

## Cómo ejecutar

    pip install torch lightgbm hmmlearn arch pandas_market_calendars requests

    set APCA_API_KEY_ID=...            (en CMD; las mismas claves que para lab/)
    set APCA_API_SECRET_KEY=...
    python adv_trading/preparar_datos_soxl.py      # 23 símbolos desde 2016; reanudable
    python adv_trading/red_politica.py prueba      # mundo nulo -> ~0; señal plantada -> positivo. No registra
    python adv_trading/red_politica.py explorar    # resultado en 2018 y 2019 (fuera de DEV). No registra
    python adv_trading/red_politica.py ejecutar    # registra H11 y la evalúa en DEV. UNA sola vez

Las redes finales tienen entre mil y tres mil parámetros: corren en CPU; no hace falta la tarjeta gráfica.
Los datos descargados no se suben al repositorio (cientos de MB); cada quien los genera con sus claves.

## 1. Qué red tiene la mejor evidencia publicada

La comparación más completa es *Deep Learning for Financial Time Series: A Large-Scale Benchmark of
Risk-Adjusted Performance* (Saly-Kaufmann, Wood, Calliess y Zohren, Oxford, marzo de 2026,
[arXiv 2603.01820](https://arxiv.org/abs/2603.01820), [código](https://github.com/Adir7SK/deep_fin_benchmark)).
Entrena cada arquitectura para maximizar directamente el Sharpe sobre futuros diarios de 2010 a 2025:

| Arquitectura | Sharpe bruto 2010–2025 |
|---|---|
| **VSN + LSTM** (selección de variables + LSTM) | **2,39** |
| LSTM + PatchTST | 2,32 |
| TFT (Temporal Fusion Transformer) | 2,20 |
| VSN + xLSTM | 1,82 |
| xLSTM (la que mejor aguanta costes) | 1,80 |
| PsLSTM | 1,76 |
| LSTM | 1,32 |
| VSN + Mamba2 | 0,97 |
| PatchTST | 0,86 |
| DLinear | 0,77 |
| Mamba2 | 0,62 |
| iTransformer | 0,35 |

Dos conclusiones de ese trabajo marcan el diseño de aquí:

- **No se predice el precio; se decide la posición.** La red emite una posición entre −1 y +1 y se entrena
  para maximizar el Sharpe de esas decisiones ("Deep Momentum Networks"). Para una cuenta pequeña el
  resultado es una función derivable de las posiciones, así que esto es aprendizaje por refuerzo directo
  (Moody y Saffell, 2001), sin la inestabilidad de PPO o DQN.
- **Importa el sesgo inductivo, no el tamaño.** Los Transformers genéricos y Mamba quedan por debajo de
  una LSTM con selección de variables.

Ese resultado es de futuros diarios muy diversificados y sin costes. Aquí el problema es otro: un solo
sector, decisiones intradía y 8 bps de coste. Por eso la base se volvió a medir con nuestros datos.

## 2. Diseño de la red de política

- **Decisiones:** 12 por sesión (10:00, 10:30, … 15:30 ET). En cada una: compra (+1), nada (0) o venta
  en corto (−1) hasta la siguiente; a las 16:00 no queda nada abierto.
- **Variables (40):** hueco de apertura, retorno del día y de los últimos 30/60/120 min, distancia al
  VWAP, posición en el rango, volatilidad y volumen relativos, resúmenes de días previos, reloj, y el
  estado del mercado, de la tecnología y de los semiconductores. Todo en unidades de la volatilidad
  reciente del propio símbolo.
- **Entrenamiento:** Sharpe de las decisiones, sin costes; conjunto de tres redes pequeñas y varias semillas.
- **Costes:** se aplican después, al convertir la señal en operaciones: solo se opera cuando la señal
  supera su tamaño típico.
- **Universo:** se entrena con 23 símbolos (ETF apalancados, semiconductores y referencias de mercado),
  pero solo se opera SOXL.

## 3. Cómo se eligió la configuración (sin tocar el registro)

Dos bancos de pruebas, ambos con datos de **2016–2019**, que quedan fuera de DEV, VAL y SELLADO:

**Señales plantadas.** Se borra toda relación real entre pasado y futuro (signo del retorno al azar por
fecha y tramo) y se suma una regla conocida. Así se sabe cuánto debería ganar quien conoce la regla (el
"techo") y qué parte recupera cada red. Tres reglas: *momento* (el movimiento del día continúa),
*régimen* (continúa solo por la tarde en días volátiles) y *memoria* (lo ocurrido hasta las 10:00 marca
la última hora).

**Datos reales.** Velas públicas de 1 minuto de Alpaca (Hugging Face, `CryptoSpartan/stocks_bars_1m`).
SOXL no está en ese conjunto; se usa como sustituto SMH3X, el triple del movimiento de SMH.

### Arquitecturas con señales plantadas

Porcentaje del Sharpe del techo recuperado en 2019, entrenando con 2016–2018 y la misma receta para todas:

| Estructura | Parámetros | Momento | Régimen | Memoria |
|---|---|---|---|---|
| LSTM | 1.114 | 101 % | 34 % | 4 % |
| xLSTM | 1.325 | 93 % | 27 % | 11 % |
| MLP | 610 | 92 % | 31 % | 8 % |
| Lineal | 82 | 92 % | 25 % | −3 % |
| Transformer | 2.778 | 88 % | 29 % | 13 % |
| VSN + xLSTM | 31.333 | 51 % | 8 % | −8 % |
| VSN + LSTM (la ganadora del benchmark) | 31.746 | 42 % | 10 % | −1 % |
| TFT | 32.658 | 38 % | 13 % | −1 % |

Todas con 8 neuronas. Las tres de selección de variables (VSN) llevan 30 veces más parámetros y son las
peores: con tan pocos datos independientes, la capacidad extra se gasta en memorizar ruido. En las curvas
de aprendizaje (`exploracion/resultados/curvas_de_aprendizaje_h32.txt`) se ve el mecanismo: con 32
neuronas, VSN + LSTM recupera el 75 % en la primera época y cae al 29 % en la 30.

Como conjunto, LSTM + xLSTM + Transformer recupera en el objetivo 112 % / 46 % / 15 %: iguala a la LSTM
en las dos primeras señales y conserva algo de la tercera. Esa es la configuración final.

### Ingredientes (sobre la LSTM pequeña)

Porcentaje del techo recuperado (momento / régimen / memoria). Referencia: 101 % / 34 % / 4 %.

| Cambio respecto a la referencia | Momento | Régimen | Qué enseña |
|---|---|---|---|
| 32 neuronas en vez de 8 | 70 % | 13 % | Más grande es peor |
| 32 neuronas y 60 épocas | 40 % | 7 % | Más grande y más entrenada, mucho peor |
| Dropout 0,1 en vez de 0,5 | 83 % | 23 % | Hace falta mucha regularización |
| Sin apagar variables | 92 % | 23 % | Apagar variables enteras ayuda |
| Sin simetría | 86 % | 21 % | La simetría exacta ayuda |
| Simetría solo por aumento de datos | 96 % | 33 % | Casi igual que la exacta |
| Con el coste de 8 bps dentro del entrenamiento | 73 % | 28 % | Mejor aprender la señal sin costes |
| 5 épocas / 40 épocas | 89 % / 85 % | 32 % / 22 % | El óptimo está cerca de 15 |
| Lotes de 64 en vez de 256 | 91 % | 24 % | Lotes grandes |
| Sin tarea auxiliar de volatilidad | 101 % | 34 % | No aporta: se quitó |
| Sin recortes de valores extremos | 101 % | 35 % | Indiferente |

Entrenar solo con el objetivo, sin el universo, recupera en el objetivo 99 % / 46 % / 28 % frente a
113 % / 47 % / 2 % con universo: el universo ayuda poco, porque casi todos los símbolos se mueven juntos
el mismo día y no son datos independientes.

### Años de historia

Porcentaje del techo recuperado en el objetivo según la historia usada para entrenar:

| Historia | Momento | Régimen | Memoria |
|---|---|---|---|
| 1 año | 81 % | 27 % | 5 % |
| 2 años | 98 % | 28 % | −5 % |
| 3 años | 113 % | 43 % | −1 % |

Más años ayudan de forma sostenida; por eso el dataset de SOXL arranca en 2016. Una señal que dependa de
recordar algo ocurrido horas antes no se detecta con esta cantidad de datos.

### Datos reales, 2018 y 2019

Conjunto de semillas por estructura; Sharpe anualizado en el año de prueba. "Bruto de cartera" es la
señal sin costes sobre los 24 símbolos; las otras columnas son el sustituto de SOXL con 8 bps de coste.

| Estructura | Sharpe bruto de cartera (media) | Sustituto de SOXL, neto 2018 | Neto 2019 | Operaciones por día |
|---|---|---|---|---|
| Transformer | 1,25 | 0,61 | 0,25 | 0,9 |
| xLSTM | 0,73 | 0,05 | −0,37 | 1,0 |
| Lineal | 0,61 | 0,21 | −0,31 | 0,8 |
| MLP | 0,22 | −0,23 | −0,57 | 0,8 |
| LSTM | 0,19 | −0,83 | 0,69 | 0,7 |
| TFT | 0,16 | −2,55 | −1,02 | 2,3 |
| VSN + xLSTM | −0,02 | −2,87 | −0,86 | 2,3 |
| VSN + LSTM | −0,35 | −2,36 | −1,30 | 2,2 |

Con dos años de prueba, el error típico de cada Sharpe es de alrededor de 0,7 a 1: ninguna de las redes
pequeñas se distingue de cero después de costes. Lo único claro es que las tres con VSN pierden.

**El procedimiento final completo** (`red_politica.py explorar` sobre el sustituto de SOXL):

| Año | Operaciones | Por día | Bruto por operación | Neto a 8 bps | Acierto | Sharpe neto |
|---|---|---|---|---|---|---|
| 2018 | 174 | 0,72 | +7,9 bps | −0,2 bps | 43 % | −0,01 |
| 2019 | 220 | 0,94 | +14,6 bps | +6,6 bps | 49 % | 0,59 |

La señal bruta es positiva los dos años, pero después de costes no se distingue de cero. Otra tanda de
semillas del mismo conjunto dio 0,63 y 1,11: esa diferencia es el tamaño del ruido.


## 4. Configuración final (`config_final.json`)

Conjunto de tres redes pequeñas (LSTM, xLSTM y Transformer; 8 neuronas, 5 semillas cada una) con la
receta que salió de las señales plantadas: dropout 0,5, apagado de variables 0,3, simetría exacta, 15
épocas fijas, lotes de 256 trozos y sin costes al entrenar. La señal se convierte en operaciones con un
umbral fijo: 1,0 veces su tamaño típico, con rebaja a 0,5 y a 0 solo si no se alcanza el mínimo de
operaciones.

## 5. Limitaciones

- La comparación con datos reales usa un sustituto de SOXL (3 × SMH) y solo dos años de prueba: sirve
  para descartar, no para confirmar.
- El umbral de Bonferroni cuenta las hipótesis del laboratorio principal más las de este registro, pero
  `lab/` no cuenta las de aquí: si registras algo nuevo en `lab/`, súmalas a mano.
- Coste fijo de 8 bps por operación; el deslizamiento real de SOXL no está medido.
