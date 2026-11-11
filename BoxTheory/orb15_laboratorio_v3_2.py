"""
ORB15 v3.1 - Diagnostico MFE/MAE con correccion robusta de timestamps.
Ejecutar desde la carpeta trading:
    python orb15_laboratorio_v3.py
Requiere orb15_v2_trades.csv y datos_alpaca_sip/*.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(".")
TRADES_FILE = BASE / "orb15_v2_trades.csv"
DATA_DIR = BASE / "datos_alpaca_sip"
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "SOXL", "NUGT"]
NY_TZ = "America/New_York"
BASE_VARIANT = "ORB15_T1.5_none_all_day"

def parse_market_timestamps(series):
    """Aware timestamps are converted to NY; naive timestamps are treated as NY local time."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"Hay {bad} timestamps inválidos en el CSV de velas.")
    # Pandas returns a DatetimeIndex/Series with tz info for consistently aware values.
    try:
        tz = parsed.dt.tz
    except AttributeError:
        parsed = pd.Series(parsed)
        tz = parsed.dt.tz
    if tz is None:
        return pd.DatetimeIndex(parsed.dt.tz_localize(NY_TZ, ambiguous="infer", nonexistent="shift_forward"))
    return pd.DatetimeIndex(parsed.dt.tz_convert(NY_TZ))

def load_bars(ticker):
    path = DATA_DIR / f"{ticker}_5min.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path}")
    raw = pd.read_csv(path)
    candidates = ["timestamp", "Timestamp", "time", "Time", "datetime", "Datetime", "date", "Date"]
    tc = next((c for c in candidates if c in raw.columns), raw.columns[0])
    idx = parse_market_timestamps(raw.pop(tc))
    raw.index = idx
    raw = raw.sort_index()
    raw = raw.loc[~raw.index.duplicated(keep="first")].copy()
    raw.columns = [str(c).strip().capitalize() for c in raw.columns]
    for c in ["Open", "High", "Low", "Close"]:
        if c not in raw.columns:
            raise ValueError(f"{path}: falta columna {c}. Columnas: {list(raw.columns)}")
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if "Volume" not in raw:
        raw["Volume"] = np.nan
    raw["Volume"] = pd.to_numeric(raw["Volume"], errors="coerce")
    return raw.dropna(subset=["Open", "High", "Low", "Close"])

def parse_trade_times(trades):
    for c in ["signal_time", "entry_time", "exit_time"]:
        # Trade CSV timestamps include offsets; UTC conversion normalizes them safely.
        trades[c] = pd.to_datetime(trades[c], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    if trades[["signal_time", "entry_time", "exit_time"]].isna().any().any():
        raise ValueError("Hay timestamps inválidos en orb15_v2_trades.csv")
    return trades

def summarize(g):
    wins = g["net_R_4bps"] > 0
    gp = g.loc[wins, "net_R_4bps"].sum()
    gl = -g.loc[g["net_R_4bps"] < 0, "net_R_4bps"].sum()
    return pd.Series({
        "n": len(g), "mean_net_R": g["net_R_4bps"].mean(),
        "profit_factor": gp/gl if gl > 0 else np.nan,
        "win_rate_pct": 100*wins.mean(),
        "mean_MFE_R": g["MFE_R"].mean(), "median_MFE_R": g["MFE_R"].median(),
        "mean_MAE_R": g["MAE_R"].mean(), "median_MAE_R": g["MAE_R"].median(),
        "mean_bars_held": g["bars_held"].mean()
    })

def main():
    if not TRADES_FILE.exists():
        raise FileNotFoundError("Copia orb15_v2_trades.csv junto a este script.")
    if not DATA_DIR.exists():
        raise FileNotFoundError("No encuentro datos_alpaca_sip.")
    print("Leyendo operaciones...")
    trades = pd.read_csv(TRADES_FILE)
    trades = trades[(trades["variant"] == BASE_VARIANT) & (trades["cost_bps"] == 4)].copy()
    trades = parse_trade_times(trades)
    trades["date"] = trades["date"].astype(str).str[:10]
    print(f"Operaciones base únicas: {len(trades):,}")

    enriched = []
    counts = {"no_session_date": 0, "entry_after_data": 0, "exit_before_entry": 0,
              "bad_risk": 0, "bad_signal_index": 0, "matched": 0}

    for ticker in TICKERS:
        part = trades[trades.ticker == ticker].copy()
        if part.empty:
            continue
        print(f"Procesando {ticker}: {len(part):,} operaciones")
        bars = load_bars(ticker)
        print(f"  Velas NY: {bars.index.min()} -> {bars.index.max()}")
        # Group by date in trade file; use NY date from the market data.
        bars_by_date = {d: g for d, g in bars.groupby(bars.index.strftime("%Y-%m-%d"))}
        for date, daytrades in part.groupby("date", sort=False):
            day = bars_by_date.get(date)
            if day is None or day.empty:
                counts["no_session_date"] += len(daytrades)
                continue
            # Usar búsqueda de pandas con Timestamp timezone-aware.
            # Evita comparar enteros nanosegundo procedentes de objetos con
            # distintas representaciones internas de zona horaria.
            hi, lo = day["High"].to_numpy(float), day["Low"].to_numpy(float)
            op, cl = day["Open"].to_numpy(float), day["Close"].to_numpy(float)
            vol = day["Volume"].to_numpy(float)

            for r in daytrades.itertuples(index=False):
                # Find first bar at/after entry and last bar at/before exit.
                entry_ts = pd.Timestamp(r.entry_time).tz_convert(NY_TZ)
                exit_ts = pd.Timestamp(r.exit_time).tz_convert(NY_TZ)
                a = int(day.index.searchsorted(entry_ts, side="left"))
                b_right = int(day.index.searchsorted(exit_ts, side="right"))
                if a >= len(day):
                    counts["entry_after_data"] += 1
                    continue
                if b_right <= a:
                    counts["exit_before_entry"] += 1
                    continue
                risk = float(r.risk_per_share)
                if not np.isfinite(risk) or risk <= 0:
                    counts["bad_risk"] += 1
                    continue
                last = min(b_right, len(day)) - 1
                sl = slice(a, last + 1)
                if r.direction == "long":
                    mfe = (np.max(hi[sl]) - float(r.entry)) / risk
                    mae = (float(r.entry) - np.min(lo[sl])) / risk
                else:
                    mfe = (float(r.entry) - np.min(lo[sl])) / risk
                    mae = (np.max(hi[sl]) - float(r.entry)) / risk

                signal_ts = pd.Timestamp(r.signal_time).tz_convert(NY_TZ)
                sig_idx = int(day.index.searchsorted(signal_ts, side="left"))
                if sig_idx >= len(day):
                    counts["bad_signal_index"] += 1
                    continue
                rng = max(hi[sig_idx] - lo[sig_idx], 1e-12)
                body_frac = abs(cl[sig_idx] - op[sig_idx]) / rng
                close_loc = (cl[sig_idx] - lo[sig_idx]) / rng
                or_width = float(r.opening_range_high - r.opening_range_low)
                breakout_ext = (cl[sig_idx] - float(r.opening_range_high)) if r.direction == "long" else (float(r.opening_range_low) - cl[sig_idx])
                signal_vol = vol[sig_idx]
                prior = vol[max(0, sig_idx-20):sig_idx]
                prior = prior[np.isfinite(prior) & (prior > 0)]
                prior_vol = float(np.median(prior)) if len(prior) else np.nan
                signal_rvol = signal_vol/prior_vol if np.isfinite(signal_vol) and np.isfinite(prior_vol) and prior_vol > 0 else np.nan

                enriched.append({
                    "ticker": ticker, "date": date, "year": r.year, "month": r.month,
                    "direction": r.direction, "signal_time": r.signal_time,
                    "entry_time": r.entry_time, "exit_time": r.exit_time,
                    "entry": r.entry, "exit": r.exit, "stop": r.stop, "target": r.target,
                    "risk_per_share": risk, "exit_reason": r.exit_reason,
                    "gross_R": r.gross_R, "net_R_4bps": r.net_R,
                    "MFE_R": mfe, "MAE_R": mae, "MFE_minus_MAE_R": mfe-mae,
                    "bars_held": last-a+1, "opening_range_width": or_width,
                    "opening_range_pct_entry": 100*or_width/float(r.entry),
                    "signal_body_fraction": body_frac, "signal_close_location": close_loc,
                    "breakout_extension_R": breakout_ext/risk,
                    "signal_volume": signal_vol, "signal_RVOL_vs_prior_20_bars": signal_rvol
                })
                counts["matched"] += 1

    print("\n=== AUDITORÍA DE EMPAREJAMIENTO ===")
    for k, v in counts.items():
        print(f"{k}: {v:,}")
    out = pd.DataFrame(enriched)
    if out.empty:
        raise RuntimeError("No se emparejó ninguna operación. Revisa la auditoría anterior y comparte su salida.")
    out.to_csv("orb15_v3_operaciones_diagnostico.csv", index=False)

    for keys, filename in [
        (["year"], "orb15_v3_por_anio.csv"),
        (["direction"], "orb15_v3_por_direccion.csv"),
        (["ticker"], "orb15_v3_por_simbolo.csv"),
        (["exit_reason"], "orb15_v3_por_salida.csv"),
        (["month"], "orb15_v3_por_mes.csv"),
    ]:
        out.groupby(keys, dropna=False).apply(summarize).reset_index().to_csv(filename, index=False)

    out["MFE_bucket"] = pd.cut(out["MFE_R"], [-np.inf, .5, 1, 1.5, 2, np.inf],
                               labels=["<0.5R", "0.5-1R", "1-1.5R", "1.5-2R", ">=2R"])
    out["MAE_bucket"] = pd.cut(out["MAE_R"], [-np.inf, .25, .5, .75, 1, np.inf],
                               labels=["<0.25R", "0.25-0.5R", "0.5-0.75R", "0.75-1R", ">=1R"])
    out.groupby("MFE_bucket", observed=False).apply(summarize).reset_index().to_csv("orb15_v3_mfe_buckets.csv", index=False)
    out.groupby("MAE_bucket", observed=False).apply(summarize).reset_index().to_csv("orb15_v3_mae_buckets.csv", index=False)

    print("\n=== RESUMEN GENERAL ORB15 BASE (4 bps) ===")
    print(summarize(out).to_string())
    print("\nArchivos generados: orb15_v3_operaciones_diagnostico.csv y 7 resúmenes.")
    print("Nota: MFE/MAE usa High/Low de velas de 5 min; no revela el orden intravela.")

if __name__ == "__main__":
    main()
