"""
LABORATORIO - ETF diarios: descarga, auditoria, congelado y LINEA BASE PASIVA.

Instalacion:  pip install requests pandas_market_calendars
Claves:       variables de entorno APCA_API_KEY_ID y APCA_API_SECRET_KEY

Uso:
    python lab_etf_diario.py descargar          # ETF diarios 2016 -> ayer (precios ajustados por splits Y dividendos)
    python lab_etf_diario.py auditar            # contra el calendario oficial NYSE
    python lab_etf_diario.py congelar           # hashes + manifiesto + solo lectura
    python lab_etf_diario.py verificar
    python lab_etf_diario.py linea_base [--costo-bps 2]

PARA QUE SIRVE: antes de buscar cualquier regla de tendencia hay que saber contra que se compara.
La linea base responde: que habria hecho una exposicion pasiva diversificada y barata en el mismo
periodo, con el mismo costo? Cualquier regla futura debera mejorar eso despues de costos y riesgo.
No es un test de hipotesis ni una recomendacion de inversion: es una referencia descriptiva.

DEFINICIONES FIJAS DE LA LINEA BASE (antes de ver resultados):
  Datos      Retornos cierre a cierre con precios ajustados por splits y dividendos (rendimiento total).
  Efectivo   BIL como aproximacion de la tasa libre de riesgo (para Sharpe en exceso).
  B0         100% BIL (referencia de efectivo).
  B1         100% SPY, comprar y mantener.
  B2         60% SPY / 40% IEF, rebalanceo mensual.
  B3         Igual peso entre SPY, EFA, EEM, IEF, TLT, GLD, VNQ, DBC, rebalanceo mensual.
  Costo      costo_bps por lado sobre el nocional operado (compra inicial y cada rebalanceo).
  Rebalanceo al cierre del ultimo dia de negociacion de cada mes.

LIMITACIONES: ~10 anos (el historico de Alpaca empieza en 2016): poco para juzgar estrategias de
largo plazo. Solo ETF que hoy existen. Impuestos y spread real no modelados.
"""
import argparse
import hashlib
import json
import os
import stat
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"
DATA_DIR = Path("datos_etf_diario")
UNIVERSE = ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "VNQ", "DBC", "BIL"]
RISK_ASSETS = ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "VNQ", "DBC"]
CASH = "BIL"
START_DATE = "2016-01-01"
FEED, ADJUSTMENT, TIMEFRAME = "sip", "all", "1Day"     # all = splits + dividendos
BASE_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
PAUSE_S = 0.35
SIP_DELAY_MIN = 20
MANIFEST = "DATASET_MANIFEST.json"
BASELINES = {
    "B0_efectivo_BIL": {"BIL": 1.0},
    "B1_SPY": {"SPY": 1.0},
    "B2_60SPY_40IEF": {"SPY": 0.6, "IEF": 0.4},
    "B3_igual_peso_8ETF": {s: 1 / 8 for s in RISK_ASSETS},
}
LIMITACIONES = [
    "Historico de ~10 anos (Alpaca desde 2016): poco poder estadistico para reglas de largo plazo.",
    "Solo ETF que existen hoy; sin sesgo de supervivencia grave dentro de este universo, pero tampoco historia previa a 2016.",
    "Precios ajustados por splits y dividendos segun el proveedor; el ajuste de dividendos es retroactivo.",
    "Costos solo como bps sobre nocional: sin spread real, impuestos ni slippage.",
]


# ================================================================ descarga
def _headers():
    kid = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not kid or not sec:
        raise SystemExit("Faltan las variables de entorno APCA_API_KEY_ID y APCA_API_SECRET_KEY.")
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}


def _get_json(session, url, headers, params):
    for attempt in range(7):
        r = session.get(url, headers=headers, params=params, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(60, 2 ** attempt)); continue
        if r.status_code in (401, 403):
            raise SystemExit(f"HTTP {r.status_code}: revisa claves y permisos. {r.text[:200]}")
        if r.status_code in (400, 422):
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Demasiados reintentos")


def _end_stamp(end):
    ts = pd.Timestamp(f"{end}T00:00:00Z")
    limit = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=SIP_DELAY_MIN)
    return min(ts, limit).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_csv(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def descargar(symbols, start, end, out_dir=DATA_DIR, session=None):
    headers = _headers() if session is None else {}
    if session is None:
        import requests
        session = requests.Session()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fallidos, info = [], {}
    for sym in symbols:
        try:
            rows, token = [], None
            while True:
                params = {"timeframe": TIMEFRAME, "start": f"{start}T00:00:00Z", "end": _end_stamp(end),
                          "adjustment": ADJUSTMENT, "feed": FEED, "limit": 10000, "sort": "asc"}
                if token:
                    params["page_token"] = token
                data = _get_json(session, BASE_URL.format(symbol=sym), headers, params)
                rows += data.get("bars") or []
                token = data.get("next_page_token")
                if not token:
                    break
                time.sleep(PAUSE_S)
            if not rows:
                print(f"{sym}: SIN DATOS"); fallidos.append(sym); continue
            df = pd.DataFrame(rows).rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close",
                                                    "v": "volume", "n": "trade_count", "vw": "vwap"})
            for c in ["trade_count", "vwap"]:
                if c not in df.columns:
                    df[c] = np.nan
            df["date"] = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), utc=True)).tz_convert(NY_TZ).strftime("%Y-%m-%d")
            df = df[["date", "open", "high", "low", "close", "volume", "trade_count", "vwap"]]
            df = df.drop_duplicates("date").sort_values("date")
            _atomic_csv(df, out_dir / f"{sym}_1d.csv")
            info[sym] = {"filas": len(df), "primera": df.date.iloc[0], "ultima": df.date.iloc[-1]}
            print(f"{sym}: {len(df):,} dias | {df.date.iloc[0]} -> {df.date.iloc[-1]}")
            time.sleep(PAUSE_S)
        except SystemExit:
            raise
        except Exception as ex:
            print(f"{sym}: ERROR {ex}"); fallidos.append(sym)
    meta = {"feed": FEED, "adjustment": ADJUSTMENT, "timeframe": TIMEFRAME, "inicio": start, "fin": end,
            "descargado": datetime.now().isoformat(timespec="seconds"), "archivos": info, "fallidos": fallidos}
    tmp = out_dir / "descarga_metadata.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, out_dir / "descarga_metadata.json")
    print(f"\nListo. Fallidos: {fallidos or 'ninguno'}")


def cargar(data_dir=DATA_DIR, symbols=UNIVERSE):
    out = {}
    for s in symbols:
        p = Path(data_dir) / f"{s}_1d.csv"
        if not p.exists():
            raise FileNotFoundError(f"Falta {p}")
        out[s] = pd.read_csv(p).set_index("date")
    return out


# ================================================================ auditoria
def _calendar_sessions(start, end):
    try:
        import pandas_market_calendars as mcal
        return list(pd.to_datetime(mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end).index).strftime("%Y-%m-%d"))
    except ImportError:
        raise SystemExit("Instala el calendario oficial:  pip install pandas_market_calendars")


def auditar(data_dir=DATA_DIR):
    data = cargar(data_dir)
    start = min(d.index[0] for d in data.values()); end = max(d.index[-1] for d in data.values())
    sessions = set(_calendar_sessions(start, end))
    rows, problemas = [], []
    for s, d in data.items():
        have = set(d.index)
        first, last = d.index[0], d.index[-1]
        exp = {x for x in sessions if first <= x <= last}
        miss, extra = sorted(exp - have), sorted(have - exp)
        o, h, l, c = (d[k].astype(float) for k in ["open", "high", "low", "close"])
        bad = int(((h < l) | (h < np.maximum(o, c) - 1e-9) | (l > np.minimum(o, c) + 1e-9) | (c <= 0)).sum())
        ret = c.pct_change()
        big = ret[ret.abs() > 0.15]
        rows.append({"symbol": s, "primera": first, "ultima": last, "dias": len(d), "faltantes": len(miss),
                     "fuera_de_calendario": len(extra), "ohlc_invalido": bad, "retornos_gt15pct": len(big),
                     "vol_cero_pct": round(100 * (d["volume"].fillna(0) <= 0).mean(), 2)})
        if miss:
            problemas.append((s, "faltantes", miss[:5]))
        if extra:
            problemas.append((s, "fuera_de_calendario", extra[:5]))
        for dt, r in big.items():
            problemas.append((s, "retorno_extremo", f"{dt}: {r:.1%}"))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))
    same = len({tuple(d.index) for d in data.values()}) == 1
    print(f"\nTodos los ETF comparten exactamente las mismas fechas: {same}")
    print(f"Sesiones del calendario oficial en el rango: {len(sessions)}")
    ok = (res.faltantes.sum() == 0 and res.fuera_de_calendario.sum() == 0 and res.ohlc_invalido.sum() == 0)
    if problemas:
        print("\nPara revisar:")
        for p in problemas[:40]:
            print("  -", p)
    print("\nRESULTADO:", "LIMPIO" if ok else "REVISAR (no congelar hasta entender los hallazgos)")
    res.to_csv("auditoria_etf_diario.csv", index=False)
    return ok


# ================================================================ congelar
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def congelar(data_dir=DATA_DIR, forzar=False):
    d = Path(data_dir); mp = d / MANIFEST
    if mp.exists() and not forzar:
        raise SystemExit("Ya esta congelado. Usa 'verificar' (o --forzar para recongelar: cambia el dataset_id).")
    files = sorted(d.glob("*_1d.csv"))
    if not files:
        raise SystemExit("No hay archivos *_1d.csv")
    info = {f.name: {"sha256": _sha(f), "filas": int(sum(1 for _ in open(f, "rb")) - 1)} for f in files}
    meta = json.loads((d / "descarga_metadata.json").read_text()) if (d / "descarga_metadata.json").exists() else {}
    man = {"creado": datetime.now().isoformat(timespec="seconds"),
           "dataset_id": hashlib.sha256("\n".join(f"{k}:{v['sha256']}" for k, v in sorted(info.items())).encode()).hexdigest(),
           "descarga": {k: meta.get(k) for k in ("feed", "adjustment", "timeframe", "inicio", "fin")},
           "simbolos": [f.name[:-7] for f in files], "archivos": info, "limitaciones": LIMITACIONES}
    if mp.exists():
        os.chmod(mp, stat.S_IWRITE | stat.S_IREAD)
    mp.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    for f in files + [mp]:
        os.chmod(f, stat.S_IREAD)
    print(f"Congelado: {len(files)} ETF | dataset_id {man['dataset_id']}")


def verify_dataset(data_dir=DATA_DIR):
    d = Path(data_dir); mp = d / MANIFEST
    if not mp.exists():
        return False, ["No esta congelado"]
    man = json.loads(mp.read_text(encoding="utf-8"))
    prob = []
    for n, i in man["archivos"].items():
        if not (d / n).exists():
            prob.append(f"Falta {n}")
        elif _sha(d / n) != i["sha256"]:
            prob.append(f"Modificado {n}")
    prob += [f"No registrado {p.name}" for p in d.glob("*_1d.csv") if p.name not in man["archivos"]]
    return not prob, prob


def verificar(data_dir=DATA_DIR):
    ok, prob = verify_dataset(data_dir)
    if ok:
        man = json.loads((Path(data_dir) / MANIFEST).read_text(encoding="utf-8"))
        print(f"DATASET INTACTO. dataset_id={man['dataset_id']}")
    else:
        print("PROBLEMAS:", *prob, sep="\n  - "); sys.exit(1)


# ============================================================ linea base
def simular_cartera(rets, pesos, costo_bps):
    """Curva de valor con rebalanceo mensual al cierre del ultimo dia del mes y costo sobre el nocional operado.
    El primer valor de la curva es el valor tras la compra inicial (ya con su costo)."""
    cols = list(pesos)
    w = np.array([pesos[c] for c in cols], float)
    R = rets[cols].to_numpy()
    dates = list(rets.index)
    mes = pd.Series([d[:7] for d in dates])
    es_ultimo = mes.ne(mes.shift(-1)).to_numpy()
    k = costo_bps / 1e4
    V = 1.0 - k * np.abs(w).sum()
    hold = V * w
    curve = [V]
    for t in range(len(dates)):
        hold = hold * (1 + R[t])
        V = hold.sum()
        if es_ultimo[t] and t < len(dates) - 1:
            turn = np.abs(V * w - hold).sum()
            V -= k * turn
            hold = V * w
        curve.append(V)
    return pd.Series(curve, index=[None] + dates)


def metricas_curva(curve, rf):
    """curve: indice [None, fechas...]; rf: retornos diarios del efectivo alineados a las fechas."""
    v = curve.iloc[1:]
    v0, n = 1.0, len(v)
    r = np.r_[v.iloc[0] / v0 - 1, v.pct_change().dropna().to_numpy()]
    exc = r - rf.to_numpy()
    anios = n / 252
    cagr = v.iloc[-1] ** (1 / anios) - 1
    vol = r.std(ddof=1) * np.sqrt(252)
    peak = np.maximum.accumulate(np.r_[v0, v.to_numpy()])[1:]
    dd = v.to_numpy() / peak - 1
    bajo, mejor_run, run = dd < 0, 0, 0
    for b in bajo:
        run = run + 1 if b else 0
        mejor_run = max(mejor_run, run)
    yr = pd.Series(v.to_numpy(), index=[d[:4] for d in v.index]).groupby(level=0).last()
    prev = pd.Series(np.r_[v0, yr.to_numpy()[:-1]], index=yr.index)
    anual = yr / prev - 1
    return {"CAGR_pct": 100 * cagr, "vol_pct": 100 * vol,
            "Sharpe_exceso_BIL": exc.mean() / exc.std(ddof=1) * np.sqrt(252) if exc.std(ddof=1) > 0 else np.nan,
            "max_drawdown_pct": 100 * dd.min(), "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
            "dias_max_bajo_agua": mejor_run, "peor_anio_pct": 100 * anual.min(), "mejor_anio_pct": 100 * anual.max(),
            "anios_positivos_pct": 100 * (anual > 0).mean(), "valor_final": v.iloc[-1], "dias": n}, anual


def linea_base(data_dir=DATA_DIR, costo_bps=2.0):
    ok, prob = verify_dataset(data_dir)
    if not ok:
        raise SystemExit("El dataset no esta congelado/intacto: " + "; ".join(prob))
    data = cargar(data_dir)
    px = pd.DataFrame({s: d["close"].astype(float) for s, d in data.items()})
    comunes = px.dropna().index
    if len(comunes) < len(px):
        print(f"AVISO: {len(px) - len(comunes)} fechas descartadas por no ser comunes a todos los ETF.")
    px = px.loc[comunes]
    rets = px.pct_change().iloc[1:]
    rf = rets[CASH]
    print(f"Periodo: {rets.index[0]} -> {rets.index[-1]} | {len(rets)} dias | costo {costo_bps:g} bps por lado\n")
    res, anual, curvas = {}, {}, {}
    for nombre, pesos in BASELINES.items():
        cv = simular_cartera(rets, pesos, costo_bps)
        m, a = metricas_curva(cv, rf)
        res[nombre], anual[nombre] = m, a
        curvas[nombre] = cv.iloc[1:].to_numpy()
    tabla = pd.DataFrame(res).T
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(tabla.round(2).to_string())
    ta = pd.DataFrame({k: (v * 100).round(1) for k, v in anual.items()})
    print("\nRetorno por anio calendario (%; el primero y el ultimo pueden ser parciales):"); print(ta.to_string())
    tabla.to_csv("linea_base_resumen.csv"); ta.to_csv("linea_base_anual.csv")
    pd.DataFrame(curvas, index=rets.index).to_csv("linea_base_curvas.csv")
    print("\nReferencia descriptiva, no una recomendacion. Una regla candidata debe mejorar esto DESPUES de costos y riesgo.")
    return tabla


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["descargar", "auditar", "congelar", "verificar", "linea_base"])
    p.add_argument("--inicio", default=START_DATE)
    p.add_argument("--fin", default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--salida", default=str(DATA_DIR))
    p.add_argument("--costo-bps", type=float, default=2.0)
    p.add_argument("--forzar", action="store_true")
    a = p.parse_args()
    {"descargar": lambda: descargar(UNIVERSE, a.inicio, a.fin, a.salida), "auditar": lambda: auditar(a.salida),
     "congelar": lambda: congelar(a.salida, a.forzar), "verificar": lambda: verificar(a.salida),
     "linea_base": lambda: linea_base(a.salida, a.costo_bps)}[a.accion]()


if __name__ == "__main__":
    main()
