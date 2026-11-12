"""
IA - datos y variables diarias (solo tramo DEV del laboratorio).

Construye, a partir de las velas de 5 min del dataset congelado, una tabla con una fila por
(simbolo, sesion). Todas las variables de una fila se conocen a las 10:30 ET de esa sesion
(o antes); las columnas que empiezan por `y_` son resultados posteriores y NUNCA se usan como
entrada de un modelo.

Se ejecuta desde cualquier carpeta: el modulo se coloca solo en `lab/` porque el marco usa rutas
relativas.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
LAB = RAIZ / "lab"
SALIDA = Path(__file__).resolve().parent / "resultados"
SALIDA.mkdir(exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(LAB))
os.chdir(LAB)

import lab_marco as M  # noqa: E402

CORTE = 10 * 60 + 30          # minuto del dia (ET) de la decision y de la apertura de la vela de entrada
VENTANA, MIN_VENTANA = 20, 15  # sesiones previas para normalizar (y minimo de ellas con dato)
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE"]


def abrir_lab():
    ds = M.Dataset()
    reg = M.Registro(ds.dataset_id)
    ds.attach(reg)
    return ds, reg


def _diario_simbolo(b):
    """Agregados por sesion de un simbolo. `b`: velas de 5 min con indice en hora de Nueva York."""
    b = b.copy()
    b["m"] = b.index.hour * 60 + b.index.minute
    lc = np.log(b["close"])
    ant = lc.groupby(b["date"]).shift(1)
    b["r"] = (lc - ant).fillna(lc - np.log(b["open"]))        # primera vela: apertura -> cierre
    b["r2"] = b["r"] ** 2
    b["pv"] = b["vwap"].fillna((b["high"] + b["low"] + b["close"]) / 3) * b["volume"]
    pre, post = b[b["m"] < CORTE], b[b["m"] >= CORTE]
    g, gp, gq = b.groupby("date"), pre.groupby("date"), post.groupby("date")
    d = pd.DataFrame({
        "m_ini": g["m"].first(), "n_velas": g.size(),
        "open": g["open"].first(), "close": g["close"].last(),
        "rv": np.sqrt(g["r2"].sum()), "vol": g["volume"].sum(),
        "c_pre": gp["close"].last(), "n_pre": gp.size(),
        "hi_pre": gp["high"].max(), "lo_pre": gp["low"].min(),
        "rv_pre": np.sqrt(gp["r2"].sum()), "vol_pre": gp["volume"].sum(), "pv_pre": gp["pv"].sum(),
        "m_ent": gq["m"].first(), "entrada": gq["open"].first(),
        "t_ent": gq.apply(lambda x: x.index[0]), "t_sal": g.apply(lambda x: x.index[-1]),
        "o_ult": g["open"].agg(lambda x: x.iloc[-12] if len(x) >= 12 else np.nan),   # apertura de la ultima hora
    })
    return d


def tabla_diaria(ds, split="dev"):
    """Una fila por (simbolo, sesion) con variables conocidas a las 10:30 y resultados `y_`."""
    cache = SALIDA / f"_tabla_{split}_{ds.dataset_id[:12]}.pkl"      # cache local (no se sube al repositorio)
    if split == "dev" and cache.exists():
        return pd.read_pickle(cache)
    ses = ds.sessions(split)
    pos = {s: i for i, s in enumerate(ses)}
    filas = []
    for sym in ds.symbols:
        d = _diario_simbolo(ds.bars(sym, split))
        d["symbol"] = sym
        d["i"] = [pos[x] for x in d.index]
        # El gap solo existe si el dia anterior es la sesion inmediatamente previa. El retorno cierre a
        # cierre si se calcula a traves de un dia excluido (abarca mas de una sesion): asi un hueco no
        # deja sin variables a las 20 sesiones siguientes. Las medias moviles toleran algun dato ausente.
        consec = d["i"].diff() == 1
        roll = lambda c: d[c].shift(1).rolling(VENTANA, min_periods=MIN_VENTANA).mean()
        d["gap"] = d["open"] / d["close"].shift(1).where(consec) - 1
        d["r_cc"] = d["close"] / d["close"].shift(1) - 1
        d["r_pre"] = d["c_pre"] / d["open"] - 1
        d["rv20"], d["rvpre20"], d["volpre20"] = roll("rv"), roll("rv_pre"), roll("vol_pre")
        d["r_cc_1"] = d["r_cc"].shift(1)
        d["ret5"] = d["r_cc"].shift(1).rolling(5, min_periods=4).sum()
        d["ret20"] = d["r_cc"].shift(1).rolling(VENTANA, min_periods=MIN_VENTANA).sum()
        d["ult_hora_1"] = (d["close"] / d["o_ult"] - 1).shift(1)
        d["rv_1"] = d["rv"].shift(1)
        d["rv_5"] = d["rv"].shift(1).rolling(5, min_periods=4).mean()
        d["rv_22"] = d["rv"].shift(1).rolling(22, min_periods=16).mean()
        d["gap_n"], d["r_pre_n"] = d["gap"] / d["rv20"], d["r_pre"] / d["rv20"]
        d["rv_pre_rel"] = d["rv_pre"] / d["rvpre20"]
        d["vol_pre_rel"] = d["vol_pre"] / d["volpre20"]
        d["dist_vwap_n"] = (d["c_pre"] / (d["pv_pre"] / d["vol_pre"]) - 1) / d["rv20"]
        d["pos_rango"] = (d["c_pre"] - d["lo_pre"]) / (d["hi_pre"] - d["lo_pre"]).replace(0, np.nan)
        # Resultados (no son entradas): retorno desde la apertura de la vela de 10:30 hasta el cierre.
        d["y_resto"] = d["close"] / d["entrada"] - 1
        d["ok"] = (d["m_ini"] == 570) & (d["n_pre"] == 12) & (d["m_ent"] == CORTE)
        filas.append(d.reset_index())
    t = pd.concat(filas, ignore_index=True)
    t["sector"] = t["symbol"].map(lambda s: ds.meta(s)[0])
    t["es_etf"] = t["symbol"].isin(ETFS)
    t["dow"] = pd.to_datetime(t["date"]).dt.dayofweek
    # Contexto de mercado conocido a las 10:30.
    for etf in ("SPY", "QQQ", "IWM"):
        e = t[t["symbol"] == etf].set_index("date")
        for c in ("gap", "r_pre", "rv_pre_rel", "ret5", "rv20"):
            t[f"{etf.lower()}_{c}"] = t["date"].map(e[c])
    t["qqq_iwm_pre"] = t["qqq_r_pre"] - t["iwm_r_pre"]
    t["r_pre_vs_spy"] = t["r_pre"] - t["spy_r_pre"]
    t["gap_vs_spy"] = t["gap"] - t["spy_gap"]
    t = t.sort_values(["date", "symbol"]).reset_index(drop=True)
    if split == "dev":
        t.to_pickle(cache)
    return t


def rangos_transversales(t, cols):
    """Rango percentil de cada variable dentro del dia (solo entre las filas dadas)."""
    out = t.copy()
    for c in cols:
        out[f"rk_{c}"] = out.groupby("date")[c].rank(pct=True)
    return out


def filtrado_hmm(modelo, X):
    """Probabilidades FILTRADAS P(estado_t | datos hasta t): recursion hacia adelante, sin mirar el futuro.

    (hmmlearn.predict_proba devuelve probabilidades suavizadas con toda la serie; no sirven aqui.)
    """
    from scipy.stats import multivariate_normal
    k = modelo.n_components
    B = np.column_stack([multivariate_normal(modelo.means_[j], modelo.covars_[j], allow_singular=True).pdf(X)
                         for j in range(k)])
    a = np.zeros((len(X), k))
    p = modelo.startprob_
    for i in range(len(X)):
        p = (p if i == 0 else p @ modelo.transmat_) * B[i]
        s = p.sum()
        p = p / s if s > 0 else np.full(k, 1.0 / k)
        a[i] = p
    return a
