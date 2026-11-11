# adv_trading — modelos avanzados

Redes profundas para ejecutar en una computadora propia (usan la tarjeta gráfica si PyTorch la detecta).

**Esta carpeta no funciona sola**: necesita el repositorio completo, porque usa el código de `ia/` y el
dataset congelado y el marco de pruebas de `lab/`.

## red_secuencias.py — hipótesis H10 (pendiente de ejecutar)

Red TCN + Transformer con atención sobre las últimas 78 velas de 5 min, más las variables resumidas del
experimento 3. Predice el retorno a 1 hora, 2 horas y al cierre; solo la predicción al cierre se usa
para operar. Universo, horas de decisión, validación y regla de operación son idénticos a H9
(LightGBM), para comparar solo el modelo.

    pip install torch lightgbm hmmlearn arch pandas_market_calendars
    python adv_trading/red_secuencias.py prueba      # unos 13 min en CPU de 4 núcleos; no registra nada
    python adv_trading/red_secuencias.py ejecutar    # registra H10 y la evalúa en DEV, UNA sola vez

Para GPU NVIDIA, instala `torch` con el comando de https://pytorch.org/get-started/locally/.

- La ejecución real tarda entre 3 y 10 horas en CPU (estimado); con GPU, mucho menos.
- No modifiques el archivo antes de `ejecutar`: cualquier cambio es otra hipótesis.
- Al terminar, sube `adv_trading/resultados/exp4_*.csv` y `lab/registro_lab/registro_hipotesis.jsonl`.
