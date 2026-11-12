"""
EXPLORACION - construye el dataset "PRE" (2016-2019) a partir de datos publicos.

Origen: Hugging Face `CryptoSpartan/stocks_bars_1m` (velas de 1 minuto obtenidas de la API de barras
historicas de Alpaca, licencia MIT). Solo se usan velas ANTERIORES a 2020, es decir, fuera de los tres
tramos del laboratorio (DEV 2020-2023, VAL 2024-2025, SELLADO 2026). Sirve para comparar arquitecturas
sin tocar el registro de hipotesis.

Que hace:
  1. Lee las velas de 1 min ya descargadas (carpeta --entrada, un parquet por simbolo).
  2. Se queda con la sesion regular segun el calendario NYSE y agrega a velas de 5 min.
  3. Detecta splits (los datos publicos vienen SIN ajustar) y ajusta el historial anterior.
  4. Crea SMH3X y SOXL3X: sustitutos de SOXL con el triple del movimiento logaritmico de SMH y de
     SOXX (ETF de semiconductores sin apalancar). SOXL no esta en el conjunto publico.
  5. Escribe SIMBOLO_5min.csv en el formato del laboratorio; despues se limpia y congela con las
     herramientas de lab/ (corregir + congelar).

Uso:
    python construir_pre_publico.py --entrada CARPETA_PARQUET_1MIN --salida CARPETA_DATASET
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(RAIZ / "lab"))
import lab_descarga_auditoria_v3 as base  # noqa: E402
import lab_dataset_congelar as cong  # noqa: E402

# Las herramientas de lab/ comparan marcas de tiempo como enteros en nanosegundos, pero segun la version
# de pandas las fechas leidas de un CSV vienen en segundos, micro o nanosegundos. Este envoltorio iguala
# las unidades antes de comparar, sin modificar los archivos de lab/.
_isin_original = base._isin_sorted


def _isin_misma_unidad(a, sorted_b):
    a = np.asarray(a, dtype=np.int64)
    if len(a) and len(sorted_b):
        r = float(np.median(sorted_b)) / max(float(np.median(a)), 1.0)
        for f in (10**3, 10**6, 10**9):
            if 0.5 * f < r < 2 * f:
                a = a * f
    return _isin_original(a, sorted_b)


base._isin_sorted = _isin_misma_unidad

RATIOS = np.array([2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10, 1 / 15, 1 / 20])
COLS = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


def a_5min(path):
    """Velas de 1 min -> 5 min de sesion regular (etiqueta = inicio de la vela), segun calendario NYSE."""
    df = pd.read_parquet(path).rename(columns={"vol_weighted_avg_price": "vwap"})
    ts = pd.DatetimeIndex(df.pop("timestamp")).tz_convert("UTC").as_unit("ns")
    sch = base._calendar_schedule(ts[0].strftime("%Y-%m-%d"), ts[-1].strftime("%Y-%m-%d"))
    ns = ts.asi8
    k = np.searchsorted(sch.open_ns.values, ns, side="right") - 1          # sesion a la que perteneceria
    ok = (k >= 0) & (ns < sch.close_ns.values[np.clip(k, 0, len(sch) - 1)])
    df, ns, k = df[ok].reset_index(drop=True), ns[ok], k[ok]
    barra = sch.open_ns.values[k] + ((ns - sch.open_ns.values[k]) // base.BAR_NS) * base.BAR_NS
    df["volume"], df["trade_count"] = df["volume"].astype("float64"), df["trade_count"].astype("float64")
    df["pv"] = df["vwap"].where(df["vwap"] > 0, df["close"]) * df["volume"]
    g = df.groupby(barra)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(),
                        "trade_count": g["trade_count"].sum(), "pv": g["pv"].sum()})
    out["vwap"] = (out["pv"] / out["volume"].where(out["volume"] > 0)).fillna(out["close"])
    out.index = pd.DatetimeIndex(out.index, tz="UTC")
    return out.drop(columns="pv")


def ajustar_splits(b, sym):
    """Detecta saltos de apertura compatibles con un split y ajusta todo el historial anterior."""
    ny = b.index.tz_convert(base.NY_TZ)
    dia = pd.Series(ny.strftime("%Y-%m-%d"), index=b.index)
    ap, ci = b["open"].groupby(dia.values).first(), b["close"].groupby(dia.values).last()
    razon = (ci.shift(1) / ap).dropna()                    # > 1 si el precio "cae" por un split directo
    eventos = []
    for d, r in razon.items():
        j = np.argmin(np.abs(np.log(r) - np.log(RATIOS)))
        if abs(np.log(r)) > 0.5 and abs(np.log(r) - np.log(RATIOS[j])) < 0.06:
            eventos.append((d, float(RATIOS[j])))
    for d, f in eventos:
        antes = dia.values < d
        b.loc[antes, ["open", "high", "low", "close", "vwap"]] /= f
        b.loc[antes, "volume"] *= f
    if eventos:
        print(f"    {sym}: splits detectados y ajustados -> {eventos}")
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", required=True)
    p.add_argument("--salida", required=True)
    a = p.parse_args()
    entrada, salida = Path(a.entrada), Path(a.salida)
    crudo = salida / "crudo"
    crudo.mkdir(parents=True, exist_ok=True)
    barras = {}
    for f in sorted(entrada.glob("*_1min.parquet")):
        sym = f.name.replace("_1min.parquet", "")
        barras[sym] = ajustar_splits(a_5min(f), sym)
        v = barras[sym]
        print(f"  {sym}: {len(v):,} velas de 5 min | {v.index[0].date()} -> {v.index[-1].date()} | "
              f"volumen mediano por vela {v['volume'].median():,.0f}")
    # Sustitutos de SOXL: triple del movimiento logaritmico de un ETF de semiconductores sin apalancar
    # (aperturas, maximos, minimos y cierres). SMH3X es el principal: SMH es mucho mas liquido que SOXX,
    # asi que su ruido de horquilla, multiplicado por 3, se parece mas al de SOXL real.
    for origen, nombre in (("SMH", "SMH3X"), ("SOXX", "SOXL3X")):
        s = barras[origen].copy()
        p0 = s["open"].iloc[0]
        for c in ("open", "high", "low", "close", "vwap"):
            s[c] = 30.0 * (s[c] / p0) ** 3
        barras[nombre] = s
    for sym, v in barras.items():
        out = v.reset_index(names="timestamp")
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out[COLS].to_csv(crudo / f"{sym}_5min.csv", index=False)
    limpio = salida / "datos_pre_limpio"
    for f in limpio.glob("*"):                              # permitir reconstruir
        os.chmod(f, 0o644)
    base.corregir_dataset(crudo, limpio)
    cong.congelar(limpio, forzar=True)


if __name__ == "__main__":
    main()
