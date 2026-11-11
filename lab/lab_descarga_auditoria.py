"""
LABORATORIO COMUN - Fases 1-3: descarga SIP de 5 minutos y auditoria de datos.

Uso (desde la carpeta trading):
    set APCA_API_KEY_ID=...          (Windows)   |  export APCA_API_KEY_ID=...   (Linux/Mac)
    set APCA_API_SECRET_KEY=...                  |  export APCA_API_SECRET_KEY=...

    python lab_descarga_auditoria.py descargar            # descarga el universo (reanudable)
    python lab_descarga_auditoria.py descargar --simbolos AAPL MSFT --inicio 2020-01-01
    python lab_descarga_auditoria.py auditar              # integridad, liquidez, splits
    python lab_descarga_auditoria.py comparar             # compara con tus datos viejos (8 tickers)

Reglas de diseno:
- Solo feed SIP (no se mezcla con IEX). Ajuste "split" (sin dividendos), registrado en metadata.
- Los datos se guardan en una carpeta NUEVA (datos_lab_sip) para no tocar datos_alpaca_sip.
- Solo sesion regular: velas con inicio 09:30-15:55 ET (misma convencion que ORB15).
- La descarga se parte por simbolo y ano; si se corta, se reanuda sin repetir lo ya bajado
  (el ano en curso siempre se vuelve a bajar).
- Las claves van en variables de entorno, nunca dentro del codigo.
- Las listas de simbolos son un universo PROPUESTO: la auditoria dice que simbolos tienen
  historia utilizable; no se asume que todos la tengan desde 2020.
- SESGO DE SUPERVIVENCIA: este universo son empresas que existen hoy. No incluye
  deslistadas ni adquiridas. Debe documentarse como limitacion en cualquier resultado.
- Los ETF apalancados (SOXL, NUGT, etc.) quedan FUERA del universo principal.
"""
import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"
OUT_DIR = Path("datos_lab_sip")
OLD_DIR = Path("datos_alpaca_sip")
FEED = "sip"
ADJUSTMENT = "split"
TIMEFRAME = "5Min"
START_DATE = "2020-01-01"
BASE_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
PAUSE_S = 0.35  # ~170 peticiones/min, por debajo del limite habitual de 200/min

UNIVERSE = {
    "tecnologia": ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "AVGO"],
    "finanzas": ["JPM", "BAC", "GS", "MS", "V", "MA", "BRK.B"],
    "consumo_energia": ["XOM", "CVX", "WMT", "COST", "KO", "PG", "HD"],
    "salud": ["UNH", "JNJ", "LLY", "ABBV", "MRK"],
    "etf_mercado": ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE"],
}
SYMBOLS = [s for grp in UNIVERSE.values() for s in grp]
SECTOR_OF = {s: g for g, lst in UNIVERSE.items() for s in lst}

# ---- Criterios de calidad pre-registrados para la auditoria ----
MIN_DOLLAR_VOL_M = 50.0      # mediana de volumen diario en dolares (millones)
MIN_COMPLETE_PCT = 97.0      # % de dias con 78 velas (o cierre temprano valido)
MAX_ZERO_VOL_PCT = 2.0       # % de velas con volumen cero/nulo
EARLY_CLOSE_LAST_BAR = "12:55"


# ================================================================= DESCARGA
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
            wait = min(60, 2 ** attempt)
            print(f"    HTTP {r.status_code}, reintento en {wait}s")
            time.sleep(wait)
            continue
        if r.status_code in (401, 403):
            raise SystemExit(f"HTTP {r.status_code}: revisa claves y permisos de datos SIP. {r.text[:200]}")
        if r.status_code == 422 or r.status_code == 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Demasiados reintentos")


def fetch_range(session, headers, symbol, start, end):
    url = BASE_URL.format(symbol=symbol)
    rows, token = [], None
    while True:
        params = {"timeframe": TIMEFRAME, "start": f"{start}T00:00:00Z", "end": f"{end}T00:00:00Z",
                  "adjustment": ADJUSTMENT, "feed": FEED, "limit": 10000, "sort": "asc"}
        if token:
            params["page_token"] = token
        data = _get_json(session, url, headers, params)
        rows.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        if not token:
            return rows
        time.sleep(PAUSE_S)


def _to_frame(rows):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low",
                                           "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap"})
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols].copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    ny = ts.dt.tz_convert(NY_TZ)
    keep = (ny.dt.hour * 60 + ny.dt.minute >= 9 * 60 + 30) & (ny.dt.hour * 60 + ny.dt.minute <= 15 * 60 + 55)
    df = df[keep.values].copy()
    df["timestamp"] = ts[keep.values].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df


def descargar(symbols, start, end, out_dir=OUT_DIR, session=None):
    headers = _headers() if session is None else {}
    if session is None:
        try:
            import requests
        except ImportError:
            raise SystemExit("Instala requests:  pip install requests")
        session = requests.Session()
    out_dir = Path(out_dir)
    parts_dir = out_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    today = date.today()
    failures = []

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}")
        try:
            part_files = []
            for y in range(start_d.year, end_d.year + 1):
                part = parts_dir / f"{sym}_{y}.csv"
                part_files.append(part)
                if part.exists() and y < today.year:
                    continue
                s = max(start_d, date(y, 1, 1))
                e = min(end_d + timedelta(days=1), date(y + 1, 1, 1))
                rows = fetch_range(session, headers, sym, s.isoformat(), e.isoformat())
                _to_frame(rows).to_csv(part, index=False)
                print(f"    {y}: {len(rows):,} velas descargadas")
                time.sleep(PAUSE_S)
            frames = [pd.read_csv(p) for p in part_files if p.exists() and p.stat().st_size > 0]
            frames = [f for f in frames if len(f)]
            if not frames:
                print("    SIN DATOS")
                failures.append(sym)
                continue
            df = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")
            df.to_csv(out_dir / f"{sym}_5min.csv", index=False)
            print(f"    total: {len(df):,} velas | {df.timestamp.iloc[0][:10]} -> {df.timestamp.iloc[-1][:10]}")
        except SystemExit:
            raise
        except Exception as ex:
            print(f"    ERROR: {ex}")
            failures.append(sym)

    meta = {"feed": FEED, "adjustment": ADJUSTMENT, "timeframe": TIMEFRAME, "start": start, "end": end,
            "sesion": "regular 09:30-15:55 ET (inicio de vela)", "descargado": datetime.now().isoformat(timespec="seconds"),
            "simbolos": symbols, "fallidos": failures,
            "nota_supervivencia": "Universo actual; no incluye deslistadas ni adquiridas."}
    (out_dir / "descarga_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nListo. Fallidos/sin datos: {failures or 'ninguno'}")
    if "BRK.B" in failures:
        print("Pista: si BRK.B falla, revisa el formato de simbolo que acepta tu API (BRK.B / BRK-B).")


# ================================================================ AUDITORIA
def load_bars(path):
    raw = pd.read_csv(path)
    tc = next((c for c in ["timestamp", "Timestamp", "time", "datetime", "date"] if c in raw.columns), raw.columns[0])
    ts = pd.to_datetime(raw.pop(tc), utc=True, errors="coerce")
    raw.index = pd.DatetimeIndex(ts).tz_convert(NY_TZ)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    return raw.sort_index()


SPLIT_RATIOS = np.array([1/2, 1/3, 1/4, 1/5, 1/8, 1/10, 1/15, 1/20, 2, 3, 4, 5, 8, 10, 15, 20])


def audit_symbol(sym, df, ref_dates):
    dup_ts = int(df.index.duplicated().sum())
    df = df.loc[~df.index.duplicated(keep="first")]
    df = df.between_time("09:30", "15:55")
    date_s = pd.Series(df.index.strftime("%Y-%m-%d"), index=df.index)
    o, h, l, c = (pd.to_numeric(df[k], errors="coerce") for k in ["open", "high", "low", "close"])
    v = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df else pd.Series(np.nan, index=df.index)

    eps = 1e-9
    bad_ohlc = ((h < l) | (h < np.maximum(o, c) - eps) | (l > np.minimum(o, c) + eps)
                | (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | o.isna() | c.isna())
    zero_vol = (v.fillna(0) <= 0)

    g = df.groupby(date_s.values)
    n_bars = g.size()
    last_bar = pd.Series(df.index.strftime("%H:%M"), index=df.index).groupby(date_s.values).last()
    full = n_bars == 78
    early = (last_bar == EARLY_CLOSE_LAST_BAR) & (n_bars >= 40)
    complete = full | early

    d_open = o.groupby(date_s.values).first()
    d_close = c.groupby(date_s.values).last()
    gap = d_open / d_close.shift(1)
    big_gap = (gap - 1).abs() > 0.25
    nearest = np.abs(np.log(gap.values[:, None]) - np.log(SPLIT_RATIOS[None, :])).min(axis=1)
    split_suspect = pd.Series(nearest < 0.04, index=gap.index) & ((gap - 1).abs() > 0.30)

    intrabar = ((h - l) / c) > 0.15
    dollar_vol = (c * v).groupby(date_s.values).sum()

    dates = set(n_bars.index)
    first, last = n_bars.index[0], n_bars.index[-1]
    expected = {d for d in ref_dates if first <= d <= last}
    missing_days = sorted(expected - dates)

    s = {
        "symbol": sym, "sector": SECTOR_OF.get(sym, "otro"),
        "first_date": first, "last_date": last, "n_days": len(n_bars), "n_bars": len(df),
        "median_bars_per_day": float(n_bars.median()),
        "complete_days_pct": 100 * complete.mean(),
        "early_close_days": int(early.sum()),
        "missing_days_vs_ref": len(missing_days),
        "zero_volume_pct": 100 * zero_vol.mean(),
        "bad_ohlc_bars": int(bad_ohlc.sum()), "duplicate_timestamps": dup_ts,
        "big_gap_days_gt25pct": int(big_gap.sum()), "split_suspect_days": int(split_suspect.sum()),
        "extreme_intrabar_gt15pct": int(intrabar.sum()),
        "median_daily_dollar_vol_M": float(dollar_vol.median() / 1e6),
    }
    flags = []
    if s["median_daily_dollar_vol_M"] < MIN_DOLLAR_VOL_M: flags.append("liquidez")
    if s["complete_days_pct"] < MIN_COMPLETE_PCT: flags.append("dias_incompletos")
    if s["zero_volume_pct"] > MAX_ZERO_VOL_PCT: flags.append("vol_cero")
    if s["bad_ohlc_bars"] > 0 or s["duplicate_timestamps"] > 0: flags.append("ohlc_invalido")
    if s["split_suspect_days"] > 0: flags.append("posible_split_sin_ajustar")
    s["status"] = "OK" if not flags else "REVISAR: " + ",".join(flags)

    day_flags = pd.DataFrame({"symbol": sym, "date": n_bars.index, "n_bars": n_bars.values,
                              "last_bar": last_bar.values, "complete": complete.values,
                              "split_suspect": split_suspect.reindex(n_bars.index).values,
                              "big_gap": big_gap.reindex(n_bars.index).values})
    day_flags = day_flags[(~day_flags.complete) | day_flags.split_suspect | day_flags.big_gap]
    return s, day_flags, missing_days


def auditar(data_dir=OUT_DIR):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*_5min.csv"))
    if not files:
        raise SystemExit(f"No hay archivos *_5min.csv en {data_dir}")
    frames = {f.name.replace("_5min.csv", ""): load_bars(f) for f in files}
    # Calendario de referencia: SPY si existe; si no, fechas presentes en >=70% de simbolos
    if "SPY" in frames:
        ref = set(frames["SPY"].between_time("09:30", "15:55").index.strftime("%Y-%m-%d"))
    else:
        cnt = pd.Series([d for f in frames.values() for d in set(f.index.strftime("%Y-%m-%d"))]).value_counts()
        ref = set(cnt[cnt >= 0.7 * len(frames)].index)

    rows, flags, missing = [], [], []
    for sym, df in frames.items():
        s, dfl, miss = audit_symbol(sym, df, ref)
        rows.append(s); flags.append(dfl)
        missing += [{"symbol": sym, "date": d} for d in miss]
    res = pd.DataFrame(rows)
    res.to_csv("auditoria_datos_resumen.csv", index=False)
    pd.concat(flags).to_csv("auditoria_datos_dias_marcados.csv", index=False)
    pd.DataFrame(missing).to_csv("auditoria_datos_dias_faltantes.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    show = ["symbol", "sector", "first_date", "last_date", "n_days", "complete_days_pct",
            "missing_days_vs_ref", "zero_volume_pct", "split_suspect_days", "big_gap_days_gt25pct",
            "median_daily_dollar_vol_M", "status"]
    print("\n=== AUDITORIA DE DATOS ===")
    print(res[show].round(2).to_string(index=False))
    ok = res[res.status == "OK"]
    print(f"\nSimbolos OK: {len(ok)}/{len(res)}")
    print(f"Fecha de inicio comun a TODOS los OK: {ok.first_date.max() if len(ok) else 'n/d'}")
    fd = res.first_date.sort_values().reset_index(drop=True)
    print(f"Fecha de inicio mediana: {fd.iloc[len(fd) // 2] if len(fd) else 'n/d'}")
    print("Archivos: auditoria_datos_resumen.csv, auditoria_datos_dias_marcados.csv, auditoria_datos_dias_faltantes.csv")
    print("\nLIMITACION A DOCUMENTAR: universo actual, sin deslistadas ni adquiridas (sesgo de supervivencia).")
    print("Revisa a mano 'posible_split_sin_ajustar' y 'big_gap': pueden ser gaps reales (earnings).")


def comparar(new_dir=OUT_DIR, old_dir=OLD_DIR):
    new_dir, old_dir = Path(new_dir), Path(old_dir)
    print("=== COMPARACION datos viejos vs nuevos (timestamps en comun) ===")
    found = False
    for f in sorted(old_dir.glob("*_5min.csv")):
        sym = f.name.replace("_5min.csv", "")
        nf = new_dir / f.name
        if not nf.exists():
            continue
        found = True
        a = load_bars(f).between_time("09:30", "15:55"); b = load_bars(nf).between_time("09:30", "15:55")
        a = a.loc[~a.index.duplicated()]; b = b.loc[~b.index.duplicated()]
        j = a[["open", "high", "low", "close", "volume"]].join(
            b[["open", "high", "low", "close", "volume"]], lsuffix="_old", rsuffix="_new", how="inner")
        if j.empty:
            print(f"{sym}: sin timestamps en comun"); continue
        same_px = (np.abs(j.close_old / j.close_new - 1) < 1e-4).mean() * 100
        same_vol = (np.abs(j.volume_old - j.volume_new) <= 0.5).mean() * 100
        print(f"{sym}: velas comunes={len(j):,} | cierre igual (1e-4)={same_px:.2f}% | volumen igual={same_vol:.2f}%")
    if not found:
        print("No hay simbolos en comun entre ambas carpetas.")
    print("Si el % de cierre igual es <99%, los datos viejos y nuevos NO usan el mismo ajuste: no los mezcles.")


# ====================================================================== CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["descargar", "auditar", "comparar"])
    p.add_argument("--simbolos", nargs="+", default=None)
    p.add_argument("--inicio", default=START_DATE)
    p.add_argument("--fin", default=date.today().isoformat())
    p.add_argument("--salida", default=str(OUT_DIR))
    a = p.parse_args()
    if a.accion == "descargar":
        descargar(a.simbolos or SYMBOLS, a.inicio, a.fin, a.salida)
    elif a.accion == "auditar":
        auditar(a.salida)
    else:
        comparar(a.salida, OLD_DIR)


if __name__ == "__main__":
    main()
