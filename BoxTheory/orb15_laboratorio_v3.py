"""
ORB15 v3 - Diagnostico de operaciones (MFE/MAE y calidad del breakout)
Coloca este archivo en la carpeta donde están:
  - orb15_v2_trades.csv
  - datos_alpaca_sip/<TICKER>_5min.csv

Ejecuta:
  python orb15_laboratorio_v3.py

No descarga datos ni necesita API keys.
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

def load_bars(ticker):
    path = DATA_DIR / f"{ticker}_5min.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path}")
    raw = pd.read_csv(path)
    candidates = ["timestamp", "Timestamp", "time", "Time", "datetime", "Datetime", "date", "Date"]
    tc = next((c for c in candidates if c in raw.columns), raw.columns[0])
    idx = pd.to_datetime(raw.pop(tc), utc=True, errors="coerce")
    if idx.isna().any():
        raise ValueError(f"Hay timestamps inválidos en {path}")
    raw.index = pd.DatetimeIndex(idx).tz_convert(NY_TZ)
    raw = raw.sort_index()
    raw = raw.loc[~raw.index.duplicated(keep="first")].copy()
    raw.columns = [str(c).strip().capitalize() for c in raw.columns]
    for c in ["Open", "High", "Low", "Close"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if "Volume" not in raw:
        raw["Volume"] = np.nan
    raw["Volume"] = pd.to_numeric(raw["Volume"], errors="coerce")
    return raw.dropna(subset=["Open", "High", "Low", "Close"])

def main():
    if not TRADES_FILE.exists():
        raise FileNotFoundError("Copia orb15_v2_trades.csv junto a este script.")
    if not DATA_DIR.exists():
        raise FileNotFoundError("No encuentro datos_alpaca_sip. Copia esa carpeta junto al script.")
    print("Leyendo operaciones...")
    trades = pd.read_csv(TRADES_FILE)
    # La tabla v2 repite la misma operación para 4, 6 y 8 bps.
    # Conservamos una sola fila por operación real, usando 4 bps.
    trades = trades[(trades["variant"] == BASE_VARIANT) & (trades["cost_bps"] == 4)].copy()
    trades["date"] = trades["date"].astype(str)
    for c in ["signal_time", "entry_time", "exit_time"]:
        trades[c] = pd.to_datetime(trades[c], utc=True).dt.tz_convert(NY_TZ)
    print(f"Operaciones base únicas: {len(trades):,}")

    enriched = []
    summaries = []
    for ticker in TICKERS:
        part = trades[trades.ticker == ticker].copy()
        if part.empty:
            continue
        print(f"Procesando {ticker}: {len(part):,} operaciones")
        bars = load_bars(ticker)
        # Procesa por fecha para limitar búsquedas a la sesión correspondiente.
        for date, daytrades in part.groupby("date", sort=False):
            day = bars[bars.index.strftime("%Y-%m-%d") == date]
            if day.empty:
                continue
            idx_ns = day.index.asi8
            hi = day["High"].to_numpy(float)
            lo = day["Low"].to_numpy(float)
            op = day["Open"].to_numpy(float)
            cl = day["Close"].to_numpy(float)
            vol = day["Volume"].to_numpy(float)
            for r in daytrades.itertuples(index=False):
                # Las marcas de tiempo de Alpaca representan apertura de vela.
                a = int(np.searchsorted(idx_ns, r.entry_time.value, side="left"))
                b = int(np.searchsorted(idx_ns, r.exit_time.value, side="right"))
                if a >= len(day) or b <= a:
                    continue
                # Incluir las velas desde entrada hasta salida, ambas inclusive.
                sl = slice(a, min(b, len(day)))
                risk = float(r.risk_per_share)
                if not np.isfinite(risk) or risk <= 0:
                    continue
                if r.direction == "long":
                    mfe = (np.max(hi[sl]) - float(r.entry)) / risk
                    mae = (float(r.entry) - np.min(lo[sl])) / risk
                else:
                    mfe = (float(r.entry) - np.min(lo[sl])) / risk
                    mae = (np.max(hi[sl]) - float(r.entry)) / risk

                # Características de la vela que confirmó el breakout.
                sig_idx = int(np.searchsorted(idx_ns, r.signal_time.value, side="left"))
                if sig_idx >= len(day):
                    continue
                rng = max(hi[sig_idx] - lo[sig_idx], 1e-12)
                body_frac = abs(cl[sig_idx] - op[sig_idx]) / rng
                close_loc = (cl[sig_idx] - lo[sig_idx]) / rng
                or_width = float(r.opening_range_high - r.opening_range_low)
                breakout_ext = ((cl[sig_idx] - float(r.opening_range_high)) if r.direction == "long"
                                else (float(r.opening_range_low) - cl[sig_idx]))
                breakout_ext_r = breakout_ext / risk
                signal_vol = vol[sig_idx]
                prior_vol = np.nan
                if sig_idx >= 1:
                    prior = vol[max(0, sig_idx-20):sig_idx]
                    prior = prior[np.isfinite(prior) & (prior > 0)]
                    if len(prior):
                        prior_vol = float(np.median(prior))
                signal_rvol = signal_vol / prior_vol if np.isfinite(signal_vol) and np.isfinite(prior_vol) and prior_vol > 0 else np.nan

                enriched.append({
                    "ticker": ticker, "date": date, "year": r.year, "month": r.month,
                    "direction": r.direction, "signal_time": r.signal_time,
                    "entry_time": r.entry_time, "exit_time": r.exit_time,
                    "entry": r.entry, "exit": r.exit, "stop": r.stop, "target": r.target,
                    "risk_per_share": risk, "exit_reason": r.exit_reason,
                    "gross_R": r.gross_R, "net_R_4bps": r.net_R,
                    "MFE_R": mfe, "MAE_R": mae, "MFE_minus_MAE_R": mfe-mae,
                    "bars_held": b-a, "opening_range_width": or_width,
                    "opening_range_pct_entry": 100*or_width/float(r.entry),
                    "signal_body_fraction": body_frac, "signal_close_location": close_loc,
                    "breakout_extension_R": breakout_ext_r,
                    "signal_volume": signal_vol, "signal_RVOL_vs_prior_20_bars": signal_rvol
                })

    out = pd.DataFrame(enriched)
    if out.empty:
        raise RuntimeError("No se pudieron reconstruir operaciones. Revisa timestamps y datos locales.")
    out.to_csv("orb15_v3_operaciones_diagnostico.csv", index=False)

    # Resúmenes: no mezclar costos ni variantes.
    def summarize(g):
        wins = g["net_R_4bps"] > 0
        gp = g.loc[g["net_R_4bps"] > 0, "net_R_4bps"].sum()
        gl = -g.loc[g["net_R_4bps"] < 0, "net_R_4bps"].sum()
        return pd.Series({
            "n": len(g), "mean_net_R": g["net_R_4bps"].mean(),
            "profit_factor": gp/gl if gl > 0 else np.nan,
            "win_rate_pct": 100*wins.mean(),
            "mean_MFE_R": g["MFE_R"].mean(), "median_MFE_R": g["MFE_R"].median(),
            "mean_MAE_R": g["MAE_R"].mean(), "median_MAE_R": g["MAE_R"].median(),
            "mean_bars_held": g["bars_held"].mean()
        })

    for keys, filename in [
        (["year"], "orb15_v3_por_anio.csv"),
        (["direction"], "orb15_v3_por_direccion.csv"),
        (["ticker"], "orb15_v3_por_simbolo.csv"),
        (["exit_reason"], "orb15_v3_por_salida.csv"),
        (["month"], "orb15_v3_por_mes.csv"),
    ]:
        out.groupby(keys, dropna=False).apply(summarize).reset_index().to_csv(filename, index=False)

    # Diagnóstico por rangos de MFE/MAE y calidad de señal.
    out["MFE_bucket"] = pd.cut(out["MFE_R"], [-np.inf, 0.5, 1, 1.5, 2, np.inf],
                               labels=["<0.5R", "0.5-1R", "1-1.5R", "1.5-2R", ">=2R"])
    out["MAE_bucket"] = pd.cut(out["MAE_R"], [-np.inf, 0.25, 0.5, 0.75, 1, np.inf],
                               labels=["<0.25R", "0.25-0.5R", "0.5-0.75R", "0.75-1R", ">=1R"])
    out.groupby("MFE_bucket", observed=False).apply(summarize).reset_index().to_csv("orb15_v3_mfe_buckets.csv", index=False)
    out.groupby("MAE_bucket", observed=False).apply(summarize).reset_index().to_csv("orb15_v3_mae_buckets.csv", index=False)

    print("\n=== RESUMEN GENERAL ORB15 BASE (4 bps) ===")
    print(summarize(out).to_string())
    print("\nArchivos generados:")
    for f in ["orb15_v3_operaciones_diagnostico.csv", "orb15_v3_por_anio.csv",
              "orb15_v3_por_direccion.csv", "orb15_v3_por_simbolo.csv",
              "orb15_v3_por_salida.csv", "orb15_v3_por_mes.csv",
              "orb15_v3_mfe_buckets.csv", "orb15_v3_mae_buckets.csv"]:
        print(" -", f)
    print("\nNota: MFE/MAE usan High/Low de velas de 5 min; no reconstruyen el orden intravela.")

if __name__ == "__main__":
    main()
