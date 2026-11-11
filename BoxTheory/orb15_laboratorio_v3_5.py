"""
LABORATORIO COMUN - Fases 1-3 (v2): descarga SIP de 5 minutos y auditoria de datos.

Instalacion:   pip install requests pandas_market_calendars
Claves:        variables de entorno APCA_API_KEY_ID y APCA_API_SECRET_KEY (nunca en el codigo)

Uso (desde la carpeta trading):
    python lab_descarga_auditoria.py descargar            # universo completo, reanudable
    python lab_descarga_auditoria.py descargar --simbolos AAPL MSFT --inicio 2020-01-01
    python lab_descarga_auditoria.py auditar              # contra el calendario oficial NYSE
    python lab_descarga_auditoria.py comparar             # contra tus datos viejos

Cambios de v2 (tras revision externa):
 - Descarga robusta: cada ano se escribe en un archivo temporal y se renombra solo si la
   descarga termino; un manifiesto registra rango, feed, ajuste y numero de velas. Un ano solo
   se omite si el manifiesto coincide EXACTAMENTE con lo pedido y el archivo tiene el numero
   de filas registrado. Cambiar --fin, el feed o el ajuste fuerza una nueva descarga.
 - Auditoria contra el calendario oficial (incluye cierres anticipados): se comparan los
   timestamps reales con los esperados, sin depender de SPY ni de heuristicas.
 - Se distinguen dias sin datos, dias incompletos y velas inesperadas (fuera de sesion).
 - Los gaps compatibles con splits pasan a ser ALERTAS para revision manual, no un criterio
   automatico de descarte.
 - comparar() revisa OHLCV, timestamps comunes y exclusivos de cada archivo.
 - Liquidez con VWAP de la vela x volumen (si el VWAP es valido).

Reglas de diseno que se mantienen: solo SIP (no se mezcla con IEX), ajuste "split",
sesion regular (inicio de vela 09:30-15:55 ET), carpeta de salida nueva (datos_lab_sip).
LIMITACION A DOCUMENTAR EN CUALQUIER RESULTADO: universo actual, sin deslistadas ni
adquiridas (sesgo de supervivencia). Ver notas en el informe de laboratorio.
Los ETF apalancados quedan fuera del universo principal.
"""
import argparse
import json
import os
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
PAUSE_S = 0.35
BAR_NS = 5 * 60 * 10**9
MANIFEST = "manifest.json"

UNIVERSE = {
    "tecnologia": ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "AVGO"],
    "finanzas": ["JPM", "BAC", "GS", "MS", "V", "MA", "BRK.B"],
    "consumo_energia": ["XOM", "CVX", "WMT", "COST", "KO", "PG", "HD"],
    "salud": ["UNH", "JNJ", "LLY", "ABBV", "MRK"],
    "etf_mercado": ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE"],
}
SYMBOLS = [s for grp in UNIVERSE.values() for s in grp]
SECTOR_OF = {s: g for g, lst in UNIVERSE.items() for s in lst}

# ---- Criterios de calidad pre-registrados (fallar uno => status REVISAR) ----
MIN_DOLLAR_VOL_M = 50.0        # mediana de volumen diario en dolares (millones)
MAX_MISSING_BARS_PCT = 1.0     # % de velas esperadas que faltan (dentro del rango del simbolo)
INCOMPLETE_DAY_FRAC = 0.05     # un dia es "incompleto" si le falta mas de 5% de las velas
MAX_INCOMPLETE_DAYS_PCT = 3.0  # % de dias incompletos tolerado
MAX_NODATA_DAYS = 2            # sesiones sin ninguna vela dentro del rango del simbolo
MAX_ZERO_VOL_PCT = 2.0
# ---- Alertas (se reportan, NO descartan) ----
LATE_START_SESSIONS = 20       # el simbolo empieza >20 sesiones despues del inicio del calendario
SPLIT_RATIOS = np.array([1/2, 1/3, 1/4, 1/5, 1/8, 1/10, 1/15, 1/20, 2, 3, 4, 5, 8, 10, 15, 20])


# ============================================================ utilidades E/S
def _atomic_write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_csv(df, path):
    tmp = path.with_name(path.name + ".tmp")  # *.csv.tmp no coincide con *.csv
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _count_rows(path):
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


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
            if "recent SIP" in r.text:
                raise SystemExit("HTTP 403: tu plan no permite SIP reciente. Usa --fin con una fecha de hace 1-2 dias "
                                 f"(el script ya limita el extremo a ahora-{SIP_DELAY_MIN} min). Detalle: {r.text[:200]}")
            raise SystemExit(f"HTTP {r.status_code}: revisa claves y permisos SIP. {r.text[:200]}")
        if r.status_code in (400, 422):
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Demasiados reintentos")


SIP_DELAY_MIN = 20  # los planes sin SIP en tiempo real no permiten consultar los ultimos 15 min


def _end_stamp(end):
    ts = pd.Timestamp(f"{end}T00:00:00Z")
    limit = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=SIP_DELAY_MIN)
    return min(ts, limit).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_range(session, headers, symbol, start, end):
    url = BASE_URL.format(symbol=symbol)
    rows, token = [], None
    while True:
        params = {"timeframe": TIMEFRAME, "start": f"{start}T00:00:00Z", "end": _end_stamp(end),
                  "adjustment": ADJUSTMENT, "feed": FEED, "limit": 10000, "sort": "asc"}
        if token:
            params["page_token"] = token
        data = _get_json(session, url, headers, params)
        rows.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        if not token:
            return rows
        time.sleep(PAUSE_S)


COLS = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


def _to_frame(rows):
    if not rows:
        return pd.DataFrame(columns=COLS), 0
    df = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low",
                                           "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap"})
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLS].copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    ny = ts.dt.tz_convert(NY_TZ)
    mins = ny.dt.hour * 60 + ny.dt.minute
    keep = ((mins >= 9 * 60 + 30) & (mins <= 15 * 60 + 55)).values
    df = df[keep].copy()
    df["timestamp"] = ts[keep].dt.strftime("%Y-%m-%dT%H:%M:%SZ").values
    return df, len(rows)


def _entry_valid(entry, part, s, e, today):
    """El archivo anual solo se reutiliza si el manifiesto coincide exactamente con lo pedido."""
    return bool(entry
                and entry.get("start") == s.isoformat() and entry.get("end") == e.isoformat()
                and entry.get("feed") == FEED and entry.get("adjustment") == ADJUSTMENT
                and entry.get("timeframe") == TIMEFRAME
                and e <= today                       # un rango que incluye hoy siempre se vuelve a bajar
                and part.exists() and _count_rows(part) == entry.get("n_rows"))


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
    manifest = _load_json(parts_dir / MANIFEST)
    start_d, end_d, today = date.fromisoformat(start), date.fromisoformat(end), date.today()
    failures, summary = [], {}

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}")
        try:
            part_files = []
            for y in range(start_d.year, end_d.year + 1):
                part = parts_dir / f"{sym}_{y}.csv"
                s = max(start_d, date(y, 1, 1))
                e = min(end_d + timedelta(days=1), date(y + 1, 1, 1))   # fin exclusivo
                key = f"{sym}|{y}"
                part_files.append(part)
                if _entry_valid(manifest.get(key), part, s, e, today):
                    continue
                rows = fetch_range(session, headers, sym, s.isoformat(), e.isoformat())
                df, n_raw = _to_frame(rows)
                _atomic_write_csv(df, part)  # solo se renombra si la escritura termino
                manifest[key] = {"start": s.isoformat(), "end": e.isoformat(), "feed": FEED,
                                 "adjustment": ADJUSTMENT, "timeframe": TIMEFRAME,
                                 "n_raw": n_raw, "n_rows": len(df),
                                 "first_ts": df.timestamp.iloc[0] if len(df) else None,
                                 "last_ts": df.timestamp.iloc[-1] if len(df) else None,
                                 "completed_at": datetime.now().isoformat(timespec="seconds")}
                _atomic_write_text(parts_dir / MANIFEST, json.dumps(manifest, indent=1))
                print(f"    {y}: {len(df):,} velas regulares ({n_raw:,} totales)")
                time.sleep(PAUSE_S)
            frames = [pd.read_csv(p) for p in part_files if p.exists() and p.stat().st_size > 0]
            frames = [f for f in frames if len(f)]
            if not frames:
                print("    SIN DATOS")
                failures.append(sym)
                continue
            df = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")
            _atomic_write_csv(df, out_dir / f"{sym}_5min.csv")
            summary[sym] = {"n_rows": len(df), "first": df.timestamp.iloc[0], "last": df.timestamp.iloc[-1]}
            print(f"    total: {len(df):,} velas | {df.timestamp.iloc[0][:10]} -> {df.timestamp.iloc[-1][:10]}")
        except SystemExit:
            raise
        except Exception as ex:
            print(f"    ERROR (se reintentara en la proxima ejecucion): {ex}")
            failures.append(sym)

    meta_path = out_dir / "descarga_metadata.json"
    meta = _load_json(meta_path)
    meta.update({"feed": FEED, "adjustment": ADJUSTMENT, "timeframe": TIMEFRAME,
                 "sesion": "regular 09:30-15:55 ET (inicio de vela)",
                 "ultima_descarga": datetime.now().isoformat(timespec="seconds"),
                 "nota_supervivencia": "Universo actual; no incluye deslistadas ni adquiridas."})
    meta.setdefault("archivos", {}).update(summary)
    meta["fallidos_ultima_ejecucion"] = failures
    _atomic_write_text(meta_path, json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nListo. Fallidos/sin datos: {failures or 'ninguno'}")
    if "BRK.B" in failures:
        print("Pista: si BRK.B falla, revisa el formato de simbolo aceptado por la API (BRK.B / BRK-B).")


# ============================================================== CALENDARIO
def _calendar_schedule(start, end):
    """Sesiones oficiales NYSE (incluye cierres anticipados). Indice 'YYYY-MM-DD';
    columnas open_ns / close_ns (UTC, nanosegundos). Se sustituye en las pruebas."""
    sch = None
    try:
        import pandas_market_calendars as mcal
        s = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
        o, c = s["market_open"], s["market_close"]
    except ImportError:
        try:
            import exchange_calendars as xcals
            s = xcals.get_calendar("XNYS").schedule.loc[start:end]
            o, c = s["open"], s["close"]
        except ImportError:
            raise SystemExit("La auditoria necesita un calendario oficial:  pip install pandas_market_calendars")
    idx = pd.to_datetime(s.index).strftime("%Y-%m-%d")
    return pd.DataFrame({"open_ns": pd.DatetimeIndex(pd.to_datetime(o, utc=True)).as_unit("ns").asi8,
                         "close_ns": pd.DatetimeIndex(pd.to_datetime(c, utc=True)).as_unit("ns").asi8}, index=idx)


def expected_bars(sch):
    """Timestamps de inicio de vela esperados por sesion (cierres anticipados incluidos)."""
    ts, sid = [], []
    for k, (o, c) in enumerate(zip(sch.open_ns.values, sch.close_ns.values)):
        arr = np.arange(o, c, BAR_NS, dtype=np.int64)
        ts.append(arr); sid.append(np.full(len(arr), k, dtype=np.int32))
    return np.concatenate(ts), np.concatenate(sid)


def _isin_sorted(a, sorted_b):
    if len(sorted_b) == 0:
        return np.zeros(len(a), bool)
    pos = np.clip(np.searchsorted(sorted_b, a), 0, len(sorted_b) - 1)
    return sorted_b[pos] == a


# ================================================================ AUDITORIA
def load_bars(path):
    raw = pd.read_csv(path)
    tc = next((c for c in ["timestamp", "Timestamp", "time", "datetime", "date"] if c in raw.columns), raw.columns[0])
    ts = pd.to_datetime(raw.pop(tc), utc=True, errors="coerce")
    raw.index = pd.DatetimeIndex(ts).tz_convert(NY_TZ)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    return raw.sort_index()


def audit_symbol(sym, df, sessions, exp_ts, exp_sid):
    dup_ts = int(df.index.duplicated().sum())
    df = df.loc[~df.index.duplicated(keep="first")].between_time("09:30", "15:55")
    date_s = pd.Series(df.index.strftime("%Y-%m-%d"), index=df.index)
    act = np.unique(df.index.as_unit("ns").asi8)  # siempre nanosegundos (pandas 3 usa otras unidades)

    # ---- Cobertura contra el calendario oficial
    first_d, last_d = date_s.iloc[0], date_s.iloc[-1]
    lo = int(np.searchsorted(sessions, first_d, side="left"))
    hi = int(np.searchsorted(sessions, last_d, side="right"))
    m = (exp_sid >= lo) & (exp_sid < hi)
    sub_ts, sub_sid = exp_ts[m], exp_sid[m] - lo
    found = _isin_sorted(sub_ts, act)
    n_sess = hi - lo
    exp_n = np.bincount(sub_sid, minlength=n_sess)
    found_n = np.bincount(sub_sid[found], minlength=n_sess)
    miss_n = exp_n - found_n
    sess_dates = np.array(sessions[lo:hi])
    no_data = found_n == 0
    incomplete = (found_n > 0) & (miss_n / np.maximum(exp_n, 1) > INCOMPLETE_DAY_FRAC)
    unexpected_mask = ~_isin_sorted(act, exp_ts)
    unexpected_dates = pd.Series(pd.DatetimeIndex(act[unexpected_mask], tz="UTC").tz_convert(NY_TZ)
                                 .strftime("%Y-%m-%d")).value_counts()

    # ---- Integridad de precios y volumen
    o, h, l, c = (pd.to_numeric(df[k], errors="coerce") for k in ["open", "high", "low", "close"])
    v = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df else pd.Series(np.nan, index=df.index)
    eps = 1e-9
    bad_ohlc = ((h < l) | (h < np.maximum(o, c) - eps) | (l > np.minimum(o, c) + eps)
                | (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | o.isna() | c.isna())
    zero_vol = v.fillna(0) <= 0
    intrabar = ((h - l) / c) > 0.15

    # ---- Liquidez con VWAP de la vela (si es valido; si no, cierre)
    vw = pd.to_numeric(df["vwap"], errors="coerce") if "vwap" in df else pd.Series(np.nan, index=df.index)
    px = vw.where(vw > 0, c)
    dollar_vol = (px * v).groupby(date_s.values).sum()

    # ---- Alertas de gaps (no descartan)
    d_open = o.groupby(date_s.values).first()
    d_close = c.groupby(date_s.values).last()
    gap = d_open / d_close.shift(1)
    big_gap = (gap - 1).abs() > 0.25
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.nan_to_num(np.log(gap.values.astype(float)), nan=50.0, posinf=50.0, neginf=50.0)
    nearest = np.abs(lg[:, None] - np.log(SPLIT_RATIOS[None, :])).min(axis=1)
    split_like = pd.Series(nearest < 0.04, index=gap.index) & ((gap - 1).abs() > 0.30)

    s = {
        "symbol": sym, "sector": SECTOR_OF.get(sym, "otro"),
        "first_date": first_d, "last_date": last_d, "n_days": int(date_s.nunique()), "n_bars": len(df),
        "start_gap_sessions": lo, "end_gap_sessions": len(sessions) - hi,
        "expected_bars": int(exp_n.sum()), "missing_bars_pct": 100 * miss_n.sum() / max(exp_n.sum(), 1),
        "no_data_days": int(no_data.sum()), "incomplete_days": int(incomplete.sum()),
        "incomplete_days_pct": 100 * incomplete.mean() if n_sess else np.nan,
        "unexpected_bars": int(unexpected_mask.sum()),
        "zero_volume_pct": 100 * zero_vol.mean(), "bad_ohlc_bars": int(bad_ohlc.sum()),
        "duplicate_timestamps": dup_ts, "extreme_intrabar_gt15pct": int(intrabar.sum()),
        "gap_gt25pct_days": int(big_gap.sum()), "gap_compatible_con_split_days": int(split_like.sum()),
        "median_daily_dollar_vol_M": float(dollar_vol.median() / 1e6),
    }
    fails, alerts = [], []
    if s["median_daily_dollar_vol_M"] < MIN_DOLLAR_VOL_M: fails.append("liquidez")
    if s["missing_bars_pct"] > MAX_MISSING_BARS_PCT: fails.append("velas_faltantes")
    if s["incomplete_days_pct"] > MAX_INCOMPLETE_DAYS_PCT: fails.append("dias_incompletos")
    if s["no_data_days"] > MAX_NODATA_DAYS: fails.append("dias_sin_datos")
    if s["unexpected_bars"] > 0: fails.append("velas_fuera_de_calendario")
    if s["zero_volume_pct"] > MAX_ZERO_VOL_PCT: fails.append("vol_cero")
    if s["bad_ohlc_bars"] > 0 or s["duplicate_timestamps"] > 0: fails.append("ohlc_invalido")
    if s["gap_compatible_con_split_days"] > 0: alerts.append(f"gap_compatible_con_split({s['gap_compatible_con_split_days']})")
    if s["gap_gt25pct_days"] > 0: alerts.append(f"gap>25%({s['gap_gt25pct_days']})")
    if lo > LATE_START_SESSIONS: alerts.append(f"inicio_tardio({lo}_sesiones)")
    if s["end_gap_sessions"] > 3: alerts.append(f"termina_antes({s['end_gap_sessions']}_sesiones)")
    s["alertas"] = " ".join(alerts)
    s["status"] = "OK" if not fails else "REVISAR: " + ",".join(fails)

    parts = []
    if no_data.any():
        parts.append(pd.DataFrame({"symbol": sym, "date": sess_dates[no_data], "tipo": "sin_datos",
                                   "esperadas": exp_n[no_data], "encontradas": 0, "detalle": ""}))
    if incomplete.any():
        parts.append(pd.DataFrame({"symbol": sym, "date": sess_dates[incomplete], "tipo": "incompleto",
                                   "esperadas": exp_n[incomplete], "encontradas": found_n[incomplete], "detalle": ""}))
    if len(unexpected_dates):
        parts.append(pd.DataFrame({"symbol": sym, "date": unexpected_dates.index, "tipo": "velas_fuera_de_calendario",
                                   "esperadas": 0, "encontradas": unexpected_dates.values, "detalle": ""}))
    sl = split_like[split_like].index
    if len(sl):
        parts.append(pd.DataFrame({"symbol": sym, "date": sl, "tipo": "gap_compatible_con_split",
                                   "esperadas": np.nan, "encontradas": np.nan,
                                   "detalle": [f"gap={gap[d]:.4f} (verificar contra eventos corporativos)" for d in sl]}))
    return s, (pd.concat(parts) if parts else pd.DataFrame())


def auditar(data_dir=OUT_DIR):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*_5min.csv"))
    if not files:
        raise SystemExit(f"No hay archivos *_5min.csv en {data_dir}")
    frames = {f.name.replace("_5min.csv", ""): load_bars(f) for f in files}
    start = min(f.index[0] for f in frames.values()).strftime("%Y-%m-%d")
    end = max(f.index[-1] for f in frames.values()).strftime("%Y-%m-%d")
    sch = _calendar_schedule(start, end)
    if sch.empty:
        raise SystemExit("El calendario no devolvio sesiones para el rango de los datos.")
    sessions = list(sch.index)
    exp_ts, exp_sid = expected_bars(sch)
    print(f"Calendario oficial: {len(sessions)} sesiones ({start} -> {end}), {len(exp_ts):,} velas esperadas por simbolo completo")

    meta = _load_json(data_dir / "descarga_metadata.json").get("archivos", {})
    rows, day_flags = [], []
    for sym, df in frames.items():
        s, dfl = audit_symbol(sym, df, sessions, exp_ts, exp_sid)
        if sym in meta and meta[sym].get("n_rows") != len(df):
            s["alertas"] = (s["alertas"] + " archivo_distinto_a_metadata").strip()
        rows.append(s)
        if len(dfl):
            day_flags.append(dfl)
    res = pd.DataFrame(rows)
    res.to_csv("auditoria_datos_resumen.csv", index=False)
    (pd.concat(day_flags) if day_flags else pd.DataFrame()).to_csv("auditoria_datos_dias_marcados.csv", index=False)

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    show = ["symbol", "sector", "first_date", "last_date", "start_gap_sessions", "n_days", "missing_bars_pct",
            "no_data_days", "incomplete_days", "unexpected_bars", "zero_volume_pct",
            "median_daily_dollar_vol_M", "alertas", "status"]
    print("\n=== AUDITORIA DE DATOS (contra calendario oficial) ===")
    print(res[show].round(2).to_string(index=False))
    ok = res[res.status == "OK"]
    print(f"\nSimbolos OK: {len(ok)}/{len(res)}")
    if len(ok):
        print(f"Inicio comun a TODOS los OK: {ok.first_date.max()} | inicio mas temprano: {ok.first_date.min()}")
    print("Archivos: auditoria_datos_resumen.csv, auditoria_datos_dias_marcados.csv")
    print("\nLas ALERTAS (gap compatible con split, gap>25%, inicio tardio) no descartan un simbolo:")
    print("verifica las fechas en auditoria_datos_dias_marcados.csv contra eventos corporativos conocidos.")
    print("LIMITACION: universo actual, sin deslistadas ni adquiridas. Resultados = exploratorios.")
    print("Spread y liquidez en el momento de entrada NO se miden con barras; se calibran con cotizaciones")
    print("de muestra o, mejor, con paper trading.")


def comparar(new_dir=OUT_DIR, old_dir=OLD_DIR):
    new_dir, old_dir = Path(new_dir), Path(old_dir)
    cols = ["open", "high", "low", "close", "volume"]
    print("=== COMPARACION datos viejos vs nuevos (solo periodo solapado) ===")
    found = False
    for f in sorted(old_dir.glob("*_5min.csv")):
        sym = f.name.replace("_5min.csv", "")
        nf = new_dir / f.name
        if not nf.exists():
            continue
        found = True
        a = load_bars(f).between_time("09:30", "15:55"); b = load_bars(nf).between_time("09:30", "15:55")
        a = a.loc[~a.index.duplicated()]; b = b.loc[~b.index.duplicated()]
        lo, hi = max(a.index[0], b.index[0]), min(a.index[-1], b.index[-1])
        if lo > hi:
            print(f"{sym}: sin periodo solapado"); continue
        a, b = a.loc[lo:hi], b.loc[lo:hi]
        common = a.index.intersection(b.index)
        only_old, only_new = len(a.index.difference(b.index)), len(b.index.difference(a.index))
        cov = 100 * len(common) / max(len(a), 1)
        ja, jb = a.loc[common, cols].astype(float), b.loc[common, cols].astype(float)
        match = {k: 100 * (np.abs(ja[k] / jb[k] - 1) < 1e-4).mean() for k in ["open", "high", "low", "close"]}
        match["volume"] = 100 * (np.abs(ja["volume"] - jb["volume"]) <= 0.5).mean()
        ratio = (ja["close"] / jb["close"]).median()
        compat = min(match.values()) >= 99.9 and cov >= 99.0 and only_new <= 0.01 * len(b)
        print(f"{sym}: comunes={len(common):,} solo_viejo={only_old:,} solo_nuevo={only_new:,} cobertura={cov:.2f}% | "
              + " ".join(f"{k}={v:.2f}%" for k, v in match.items())
              + f" | razon_cierre_mediana={ratio:.4f} -> {'COMPATIBLE' if compat else 'NO COMPATIBLE'}")
    if not found:
        print("No hay simbolos en comun entre ambas carpetas.")
    print("No mezcles datos viejos y nuevos salvo que todos digan COMPATIBLE. Una razon de cierre distinta de 1")
    print("sugiere ajuste distinto (raw vs split); eso no significa que la descarga nueva este mal.")


# ====================================================================== CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["descargar", "auditar", "comparar"])
    p.add_argument("--simbolos", nargs="+", default=None)
    p.add_argument("--inicio", default=START_DATE)
    p.add_argument("--fin", default=(date.today() - timedelta(days=1)).isoformat())
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