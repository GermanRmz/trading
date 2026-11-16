"""
ETF DIARIOS - EXPERIMENTO 1: tamano de posicion por volatilidad prevista y tendencia - protocolo dia_v1.

Uso (desde cualquier carpeta):
    python diario/diario_exp1_reglas.py volatilidad   # calidad del pronostico de volatilidad (no es hipotesis)
    python diario/diario_exp1_reglas.py ejecutar      # registra D1, D2 y D3 y las evalua en DEV (una sola vez)

DEFINICIONES FIJAS (no se cambian tras ver resultados; cualquier cambio = otra hipotesis):
  Datos       lab/datos_etf_diario congelado (9 ETF, precios ajustados por splits y dividendos).
  Tramos      los del laboratorio: DEV hasta 2023-12-31, VAL 2024-2025, SELLADO 2026. Aqui solo se lee DEV.
              2016 solo sirve de arranque (hace falta un anio de historia); se evalua 2017-2023.
  Volatilidad varianza diaria = Garman-Klass + hueco de apertura al cuadrado. Pronostico HAR en logaritmos
              (ayer, media de 5 y de 22 dias) de la volatilidad media de los 5 dias siguientes, un modelo
              comun a los 8 ETF con nivel propio por ETF, reajustado cada anio con datos anteriores.
  Calendario  la decision usa datos hasta el cierre del ultimo dia de cada semana, se ejecuta al cierre del
              dia siguiente y se mantiene una semana. Lo no invertido queda en BIL. Sin apalancamiento.
  Costo       2 bps por lado sobre el nocional operado (igual que la linea base).
  Hipotesis   D1  SPY con volatilidad objetivo:      peso = min(1, 12% / vol prevista).   Referencia: SPY.
              D2  8 ETF, cada uno con vol objetivo:   peso_i = (1/8) min(1, 10% / vol_i).  Referencia: igual peso.
              D3  D2 solo en los ETF en tendencia:    retorno de 12 meses (sin el ultimo) > el de BIL.
  Criterio    alfa diaria de la regla frente a su referencia (regresion de excesos sobre BIL), t de
              Newey-West con 6 rezagos >= umbral de Bonferroni sobre TODAS las hipotesis (laboratorio +
              estas), y Sharpe mayor que el de la referencia.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(RAIZ / "lab"))
import lab_etf_diario as E  # noqa: E402

DATOS = RAIZ / "lab" / "datos_etf_diario"
SALIDA = AQUI / "resultados"
REGISTRO = AQUI / "registro_hipotesis_diario.jsonl"
REGISTRO_LAB = RAIZ / "lab" / "registro_lab" / "registro_hipotesis.jsonl"
PROTOCOLO = "dia_v1"
DEV_FIN, EVAL_INI = "2023-12-31", "2017-01-01"
COSTO_BPS, ALFA, REZAGOS = 2.0, 0.05, 6
ACTIVOS = E.RISK_ASSETS
HIPOTESIS = {"D1_spy_vol_objetivo": "B1_SPY", "D2_8etf_vol_objetivo": "B3_igual_peso",
             "D3_8etf_vol_y_tendencia": "B3_igual_peso"}


def cargar_dev():
    ok, prob = E.verify_dataset(DATOS)
    if not ok:
        raise SystemExit("El dataset diario no esta intacto: " + "; ".join(prob))
    d = E.cargar(DATOS)
    campo = lambda c: pd.DataFrame({s: x[c].astype(float) for s, x in d.items()}).dropna()
    o, h, l, c = (campo(k) for k in ("open", "high", "low", "close"))
    idx = c.index[c.index <= DEV_FIN]                      # el resto ni se mantiene en memoria
    return o.loc[idx], h.loc[idx], l.loc[idx], c.loc[idx]


def varianza_diaria(o, h, l, c):
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return (gk.clip(lower=0) + np.log(o / c.shift(1)) ** 2).iloc[1:]


def pronostico_har(v):
    """Volatilidad anualizada prevista para los 5 dias siguientes, con datos hasta cada dia (incluido)."""
    lv = 0.5 * np.log(v[ACTIVOS].clip(lower=1e-10))
    x1, x5, x22 = lv, lv.rolling(5).mean(), lv.rolling(22).mean()
    y = 0.5 * np.log(v[ACTIVOS].rolling(5).mean().shift(-5).clip(lower=1e-10))     # objetivo: 5 dias siguientes
    pred = pd.DataFrame(np.nan, index=v.index, columns=ACTIVOS)
    anios = sorted({d[:4] for d in v.index})
    for a in anios[1:]:
        ent = (v.index < f"{a}-01-01") & (v.index < v.index[np.searchsorted(v.index, f"{a}-01-01") - 6])   # objetivo ya conocido
        filas, obj = [], []
        for i, s in enumerate(ACTIVOS):
            m = ent & x22[s].notna().values & y[s].notna().values
            dummies = np.zeros((m.sum(), len(ACTIVOS)))
            dummies[:, i] = 1
            filas.append(np.column_stack([dummies, x1[s][m], x5[s][m], x22[s][m]]))
            obj.append(y[s][m].values)
        beta = np.linalg.lstsq(np.vstack(filas), np.concatenate(obj), rcond=None)[0]
        test = np.array([d[:4] == a for d in v.index])
        for i, s in enumerate(ACTIVOS):
            pred.loc[test, s] = beta[i] + beta[-3] * x1[s][test] + beta[-2] * x5[s][test] + beta[-1] * x22[s][test]
    return np.exp(pred) * np.sqrt(252), np.exp(y) * np.sqrt(252)


def simular(rets, objetivo):
    """`objetivo`: pesos deseados por fecha (con datos hasta ese cierre). Se decide el ultimo dia de cada
    semana, se ejecuta al cierre del dia siguiente; el resto queda en BIL. Devuelve retornos diarios netos."""
    fechas = list(rets.index)
    sem = pd.to_datetime(pd.Index(fechas)).strftime("%G-%V")
    fin_semana = np.r_[sem[1:] != sem[:-1], False]
    cols = list(objetivo.columns)
    R, rf = rets[cols].to_numpy(), rets[E.CASH].to_numpy()
    k = COSTO_BPS / 1e4
    w = np.zeros(len(cols))
    pendiente, out, rot = None, [], 0.0
    for t in range(len(fechas)):
        r = float(w @ R[t] + (1 - w.sum()) * rf[t])
        w = w * (1 + R[t]) / (1 + r)                        # deriva de los pesos durante el dia
        if pendiente is not None:                           # se ejecuta hoy al cierre lo decidido ayer
            giro = np.abs(pendiente - w).sum()
            r -= k * giro
            rot += giro
            w, pendiente = pendiente.copy(), None
        if fin_semana[t] and np.isfinite(objetivo.iloc[t].to_numpy()).all():
            pendiente = objetivo.iloc[t].to_numpy(float)
        out.append(r)
    return pd.Series(out, index=fechas), rot


def t_newey_west(x, lags=REZAGOS):
    x = np.asarray(x, float)
    xc = x - x.mean()
    lrv = xc @ xc / len(x)
    for j in range(1, lags + 1):
        lrv += 2 * (1 - j / (lags + 1)) * (xc[j:] @ xc[:-j]) / len(x)
    return x.mean() / np.sqrt(lrv / len(x))


def comparar(r, ref, rf):
    """Alfa de la regla frente a su referencia y metricas de ambas (retornos diarios)."""
    e, b = (r - rf).to_numpy(), (ref - rf).to_numpy()
    beta = np.cov(e, b)[0, 1] / b.var(ddof=1)
    resid = e - beta * b                                    # alfa diaria = media del residuo
    sh = lambda z: z.mean() / z.std(ddof=1) * np.sqrt(252)
    dd = lambda z: float(((1 + z).cumprod() / (1 + z).cumprod().cummax() - 1).min())
    return {"sharpe": sh(e), "sharpe_ref": sh(b), "beta": beta, "alfa_anual_pct": 100 * 252 * resid.mean(),
            "t_alfa": t_newey_west(resid), "cagr_pct": 100 * ((1 + r).prod() ** (252 / len(r)) - 1),
            "cagr_ref_pct": 100 * ((1 + ref).prod() ** (252 / len(ref)) - 1),
            "vol_pct": 100 * r.std(ddof=1) * np.sqrt(252), "vol_ref_pct": 100 * ref.std(ddof=1) * np.sqrt(252),
            "max_dd_pct": 100 * dd(r), "max_dd_ref_pct": 100 * dd(ref)}


def n_previas():
    n = 0
    for p in (REGISTRO_LAB, REGISTRO):
        if p.exists():
            n += sum(1 for x in p.read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x).get("tipo") == "hipotesis")
    return n


def anotar(ev):
    with open(REGISTRO, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **ev}, ensure_ascii=False, default=float) + "\n")


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else ""
    if modo not in ("volatilidad", "ejecutar"):
        raise SystemExit(__doc__)
    SALIDA.mkdir(exist_ok=True)
    o, h, l, c = cargar_dev()
    v = varianza_diaria(o, h, l, c)
    vol_prev, vol_real = pronostico_har(v)
    ev = vol_prev.index >= EVAL_INI
    if modo == "volatilidad":
        ewma = np.sqrt(v[ACTIVOS].ewm(alpha=0.06).mean() * 252)
        ayer = np.sqrt(v[ACTIVOS].rolling(5).mean() * 252)
        filas = {}
        for nombre, p in (("HAR", vol_prev), ("EWMA (0,94)", ewma), ("ultimos 5 dias", ayer)):
            m = ev[:, None] & p.notna().values & vol_real.notna().values
            err = (np.log(p.values[m]) - np.log(vol_real.values[m]))
            ratio = (vol_real.values[m] / p.values[m]) ** 2
            y = np.log(vol_real.values[m])
            filas[nombre] = {"R2_log": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                             "QLIKE": (ratio - np.log(ratio) - 1).mean()}
        print("Volatilidad de los 5 dias siguientes, 8 ETF, 2017-2023 (menor QLIKE = mejor):")
        t = pd.DataFrame(filas).T
        print(t.round(3).to_string())
        t.to_csv(SALIDA / "exp1_volatilidad_diaria.csv")
        return
    if REGISTRO.exists() and any(json.loads(x).get("protocolo") == PROTOCOLO and json.loads(x).get("tipo") == "eval_dev"
                                 for x in REGISTRO.read_text(encoding="utf-8").splitlines() if x.strip()):
        raise SystemExit(f"El protocolo {PROTOCOLO} ya se evaluo en DEV. No se repite.")
    rets = c.pct_change().iloc[1:]
    tend = (c[ACTIVOS].shift(21) / c[ACTIVOS].shift(252) > (c[[E.CASH]].shift(21) / c[[E.CASH]].shift(252)).values).reindex(v.index)
    con_hist = c[ACTIVOS].shift(252).reindex(v.index).notna().all(axis=1).values
    esc = (0.10 / vol_prev).clip(upper=1.0)
    objetivos = {
        "D1_spy_vol_objetivo": pd.DataFrame({"SPY": (0.12 / vol_prev["SPY"]).clip(upper=1.0)}),
        "D2_8etf_vol_objetivo": esc / len(ACTIVOS),
        "D3_8etf_vol_y_tendencia": (esc * tend.astype(float)).where(pd.Series(con_hist, index=v.index), axis=0) / len(ACTIVOS),
    }
    refs = {"B1_SPY": pd.DataFrame(1.0, index=v.index, columns=["SPY"]),
            "B3_igual_peso": pd.DataFrame(1 / len(ACTIVOS), index=v.index, columns=ACTIVOS)}
    rets_v = rets.reindex(v.index)
    r_ref = {k: simular(rets_v, w.where(vol_prev.notna().all(axis=1), axis=0))[0] for k, w in refs.items()}
    codigo = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    ids = {}
    for nombre in HIPOTESIS:
        ids[nombre] = hashlib.sha256(f"{PROTOCOLO}|{nombre}".encode()).hexdigest()[:12]
        anotar({"tipo": "hipotesis", "id": ids[nombre], "familia": "etf_diario", "nombre": nombre, "protocolo": PROTOCOLO,
                "descripcion": f"codigo sha256 {codigo}"})
    n_total = n_previas()
    umbral = NormalDist().inv_cdf(1 - ALFA / (2 * n_total))
    rf = rets_v[E.CASH]
    filas, curvas = [], {}
    for nombre, ref in HIPOTESIS.items():
        r, rot = simular(rets_v, objetivos[nombre])
        a, b, f = r[ev], r_ref[ref][ev], rf[ev]
        m = comparar(a, b, f)
        paso = bool(m["t_alfa"] >= umbral and m["sharpe"] > m["sharpe_ref"])
        m.update({"hipotesis": nombre, "referencia": ref, "paso_DEV": paso, "rotacion_anual": rot / (ev.sum() / 252)})
        filas.append(m)
        curvas[nombre], curvas[ref] = (1 + a).cumprod(), (1 + b).cumprod()
        anotar({"tipo": "eval_dev", "id": ids[nombre], "protocolo": PROTOCOLO, "paso": paso, "umbral_t": umbral,
                "n_hipotesis": n_total, "metricas": m})
    t = pd.DataFrame(filas).set_index("hipotesis")
    t.to_csv(SALIDA / "exp1_reglas_dev.csv")
    pd.DataFrame(curvas).to_csv(SALIDA / "exp1_curvas_dev.csv")
    pd.set_option("display.width", 220)
    print(f"DEV 2017-2023 | costo {COSTO_BPS:g} bps por lado | {n_total} hipotesis en total, umbral t = {umbral:.2f}\n")
    cols = ["referencia", "sharpe", "sharpe_ref", "cagr_pct", "cagr_ref_pct", "vol_pct", "vol_ref_pct", "max_dd_pct",
            "max_dd_ref_pct", "beta", "alfa_anual_pct", "t_alfa", "rotacion_anual", "paso_DEV"]
    print(t[cols].round(2).T.to_string())
    anual = pd.DataFrame({k: v_.groupby(v_.index.str[:4]).last() / v_.groupby(v_.index.str[:4]).first() - 1 for k, v_ in curvas.items()})
    print("\nRetorno por anio (%):")
    print((100 * anual).round(1).to_string())


if __name__ == "__main__":
    main()
