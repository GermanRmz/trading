"""
ORB15 v3.4 - Filtros de ENTRADA y salida por tiempo sobre ORB15 (sin gestion BE/trailing,
que en v3.3 no aporto nada).

Usa orb15_v2_trades.csv y datos_alpaca_sip/<TICKER>_5min.csv.
Ejecutar desde la carpeta trading:
    python orb15_laboratorio_v3_4.py

DISENO ANTI-AUTOENGANO
- Los filtros y umbrales estan fijados ABAJO, antes de ver resultados. No los ajustes
  despues de mirar la salida; si lo haces, aumenta el numero de pruebas y vuelve a
  correr (el umbral de significancia se recalcula solo).
- Toda variable de filtro usa solo informacion disponible ANTES de la entrada
  (ATR, SMA, volumen relativo y VWAP se calculan sin mirar al futuro).
- Los filtros se evaluan por separado en DEV (2024-2025) y OOS (2026).
  Elige candidatos SOLO con DEV; mira OOS una vez.
- Estadistico principal: t-stat con retornos agregados POR DIA (las operaciones del
  mismo dia estan correlacionadas, el t por operacion es demasiado optimista).
- Umbral de significancia con Bonferroni segun el numero total de pruebas.

Convenciones de simulacion (iguales a v3.3 para el baseline):
- Stop y objetivo en la misma vela: stop primero.
- Gap adverso: si la vela abre mas alla del stop, se sale a la apertura.
- Costo: COST_BPS sobre el precio de entrada, expresado en R.
"""
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd

DATA_DIR = Path("datos_alpaca_sip")
TRADES_FILE = Path("orb15_v2_trades.csv")
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "SOXL", "NUGT"]
NY_TZ = "America/New_York"
VARIANT = "ORB15_T1.5_none_all_day"
COST_BPS = 4.0
DEV_YEARS = [2024, 2025]
OOS_YEARS = [2026]
MIN_N_DEV = 300  # minimo de operaciones en DEV para considerar un filtro

# ---- Umbrales pre-registrados (NO tocar tras ver resultados) ----
RISK_ATR_MIN, RISK_ATR_MAX = 0.15, 0.50   # stop / ATR14 diario
MIN_RISK_BPS = 15.0                        # stop minimo en bps del precio
RELVOL_MIN = 1.5                           # volumen vela senal / media 20 dias misma hora
EARLY_CUTOFF_MIN = 11 * 60                 # entrada antes de 11:00

# Variantes de salida: None = hasta stop/target/cierre; minutos = salida por tiempo
EXITS = {"hold_to_close": None, "time_13:00": 13 * 60}

FILTERS = {
    "00_all (control)":        lambda d: np.ones(len(d), bool),
    "01_vwap_aligned":         lambda d: d.vwap_aligned.values,
    "02_gap_aligned":          lambda d: d.gap_aligned.values,
    "03_trend_aligned":        lambda d: d.trend_aligned.values,
    "04_vwap+trend":           lambda d: d.vwap_aligned.values & d.trend_aligned.values,
    "05_gap+vwap":             lambda d: d.gap_aligned.values & d.vwap_aligned.values,
    "06_risk_atr_normal":      lambda d: ((d.risk_atr >= RISK_ATR_MIN) & (d.risk_atr <= RISK_ATR_MAX)).values,
    "07_min_risk_bps":         lambda d: (d.risk_bps >= MIN_RISK_BPS).values,
    "08_relvol_high":          lambda d: (d.relvol >= RELVOL_MIN).values,
    "09_entry_before_11":      lambda d: (d.entry_min < EARLY_CUTOFF_MIN).values,
    "10_vwap+trend+early":     lambda d: d.vwap_aligned.values & d.trend_aligned.values & (d.entry_min < EARLY_CUTOFF_MIN).values,
    "11_long_only":            lambda d: (d.direction == "long").values,
    "12_short_only":           lambda d: (d.direction == "short").values,
    "13_vwap+relvol":          lambda d: d.vwap_aligned.values & (d.relvol >= RELVOL_MIN).values,
}


def load_bars(ticker):
    path = DATA_DIR / f"{ticker}_5min.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path}")
    raw = pd.read_csv(path)
    tc = next((c for c in ["timestamp", "Timestamp", "time", "Time", "datetime", "Datetime", "date", "Date"]
               if c in raw.columns), raw.columns[0])
    ts = pd.to_datetime(raw.pop(tc), utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Timestamps invalidos en {path}")
    raw.index = pd.DatetimeIndex(ts).tz_convert(NY_TZ)
    raw = raw.sort_index()
    raw = raw.loc[~raw.index.duplicated(keep="first")].copy()
    raw.columns = [str(c).strip().capitalize() for c in raw.columns]
    for c in ["Open", "High", "Low", "Close"]:
        if c not in raw.columns:
            raise ValueError(f"Falta columna {c} en {path}")
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["Volume"] = pd.to_numeric(raw["Volume"], errors="coerce") if "Volume" in raw else np.nan
    raw = raw.dropna(subset=["Open", "High", "Low", "Close"])
    return raw.between_time("09:30", "15:55").copy()


def prepare_ticker(ticker):
    """Devuelve (dict fecha->DataFrame de barras con indicadores, DataFrame diario)."""
    b = load_bars(ticker)
    b["date"] = b.index.strftime("%Y-%m-%d")
    b["tod"] = b.index.strftime("%H:%M")
    vol = b["Volume"]
    # Volumen relativo: misma hora, 20 dias previos (shift(1) -> sin lookahead)
    avg_vol = vol.groupby(b["tod"]).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    b["relvol"] = vol / avg_vol
    tp = (b.High + b.Low + b.Close) / 3
    b["_pv"] = (tp * vol).groupby(b["date"]).cumsum()
    b["_cv"] = vol.groupby(b["date"]).cumsum()
    b["vwap"] = b["_pv"] / b["_cv"].replace(0, np.nan)

    daily = b.groupby("date").agg(d_open=("Open", "first"), d_high=("High", "max"),
                                  d_low=("Low", "min"), d_close=("Close", "last"))
    daily["prev_close"] = daily.d_close.shift(1)
    daily["atr14"] = (daily.d_high - daily.d_low).rolling(14, min_periods=10).mean().shift(1)
    daily["sma20"] = daily.d_close.rolling(20, min_periods=15).mean().shift(1)
    daily["gap"] = daily.d_open / daily.prev_close - 1.0
    days = {d: g for d, g in b.groupby("date")}
    return days, daily


def simulate(day, ix, direction, entry, stop, target, risk, time_limit):
    o = day["Open"].values; h = day["High"].values
    l = day["Low"].values; c = day["Close"].values
    mins = day.index.hour * 60 + day.index.minute
    exit_price, reason = float(c[-1]), "session_close"
    mfe = mae = 0.0
    for k in range(ix, len(day)):
        if time_limit is not None and k > ix and mins[k] >= time_limit:
            exit_price, reason = float(o[k]), "time_exit"
            break
        if direction == "long":
            mfe = max(mfe, (h[k] - entry) / risk); mae = max(mae, (entry - l[k]) / risk)
            s_hit, t_hit = l[k] <= stop, h[k] >= target
        else:
            mfe = max(mfe, (entry - l[k]) / risk); mae = max(mae, (h[k] - entry) / risk)
            s_hit, t_hit = h[k] >= stop, l[k] <= target
        if s_hit and t_hit:
            exit_price, reason = stop, "stop_same_bar"; break
        if s_hit:
            gap = (direction == "long" and o[k] < stop) or (direction == "short" and o[k] > stop)
            exit_price, reason = (float(o[k]) if gap else stop), "stop"; break
        if t_hit:
            exit_price, reason = target, "target"; break
    pnl = (exit_price - entry) if direction == "long" else (entry - exit_price)
    gross_r = pnl / risk
    cost_r = (entry * COST_BPS / 10000.0) / risk
    return gross_r - cost_r, reason, mfe, mae


def profit_factor(x):
    x = pd.Series(x, dtype=float)
    g, ls = x[x > 0].sum(), -x[x < 0].sum()
    return g / ls if ls > 0 else (np.inf if g > 0 else np.nan)


def t_daily(df, col="net_R"):
    """t-stat sobre retorno medio por dia (agrupa operaciones del mismo dia)."""
    s = df.groupby("date")[col].mean()
    if len(s) < 5 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def build_trade_table():
    trades = pd.read_csv(TRADES_FILE)
    base = trades[(trades.variant == VARIANT) & (trades.cost_bps == COST_BPS)].copy()
    for c in ["signal_time", "entry_time", "exit_time"]:
        base[c] = pd.to_datetime(base[c], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    base["date"] = base["date"].astype(str).str[:10]
    print(f"Operaciones base: {len(base):,}")

    rows, missing = [], 0
    for ticker in TICKERS:
        z = base[base.ticker == ticker]
        if z.empty:
            continue
        days, daily = prepare_ticker(ticker)
        print(f"{ticker}: {len(z)} operaciones")
        for r in z.itertuples(index=False):
            day = days.get(r.date)
            if day is None or day.empty:
                missing += 1; continue
            ix = int(day.index.searchsorted(pd.Timestamp(r.entry_time), side="left"))
            if ix >= len(day) or ix == 0:
                missing += 1; continue
            entry, stop, target, risk = float(r.entry), float(r.stop), float(r.target), float(r.risk_per_share)
            sig = day.iloc[ix - 1]  # vela de senal (cerrada antes de la entrada)
            dd = daily.loc[r.date]
            long_ = r.direction == "long"
            vwap_al = (entry > sig.vwap) if long_ else (entry < sig.vwap)
            gap_al = (dd.gap > 0) if long_ else (dd.gap < 0)
            trend_al = ((dd.prev_close > dd.sma20) if long_ else (dd.prev_close < dd.sma20))
            em = day.index[ix].hour * 60 + day.index[ix].minute
            feat = {
                "ticker": ticker, "date": r.date, "year": int(r.year), "month": r.month,
                "direction": r.direction, "entry_min": em,
                "risk_bps": risk / entry * 1e4,
                "risk_atr": risk / dd.atr14 if pd.notna(dd.atr14) and dd.atr14 > 0 else np.nan,
                "relvol": sig.relvol,
                "vwap_aligned": bool(vwap_al) if pd.notna(sig.vwap) else False,
                "gap_aligned": bool(gap_al) if pd.notna(dd.gap) else False,
                "trend_aligned": bool(trend_al) if pd.notna(dd.sma20) else False,
                "historical_net_R": r.net_R,
            }
            for name, tl in EXITS.items():
                net, reason, mfe, mae = simulate(day, ix, r.direction, entry, stop, target, risk, tl)
                rows.append({**feat, "exit_variant": name, "net_R": net, "exit_reason": reason,
                             "MFE_R": mfe, "MAE_R": mae})
    print(f"Operaciones omitidas (dia/entrada no encontrados): {missing}")
    return pd.DataFrame(rows)


def evaluate(out):
    n_tests = len(FILTERS) * len(EXITS)
    z_thr = NormalDist().inv_cdf(1 - 0.05 / (2 * n_tests))  # Bonferroni bilateral
    print(f"\nPruebas totales: {n_tests} -> |t| necesario (Bonferroni 5%): {z_thr:.2f}")

    res = []
    for ev in EXITS:
        sub = out[out.exit_variant == ev]
        total_all = len(sub)
        for fname, fn in FILTERS.items():
            g = sub[fn(sub)]
            dev = g[g.year.isin(DEV_YEARS)]
            oos = g[g.year.isin(OOS_YEARS)]
            by_year = dev.groupby("year").net_R.mean() if len(dev) else pd.Series(dtype=float)
            by_tk = dev.groupby("ticker").net_R.mean() if len(dev) else pd.Series(dtype=float)
            res.append({
                "filter": fname, "exit": ev,
                "pct_trades_kept": 100 * len(g) / total_all,
                "n_dev": len(dev), "mean_R_dev": dev.net_R.mean(), "t_daily_dev": t_daily(dev),
                "PF_dev": profit_factor(dev.net_R), "win_pct_dev": 100 * (dev.net_R > 0).mean(),
                "total_R_dev": dev.net_R.sum(),
                "years_pos_dev": f"{int((by_year > 0).sum())}/{len(by_year)}",
                "tickers_pos_dev": f"{int((by_tk > 0).sum())}/{len(by_tk)}",
                "n_oos": len(oos), "mean_R_oos": oos.net_R.mean(),
                "PF_oos": profit_factor(oos.net_R), "t_daily_oos": t_daily(oos),
            })
    res = pd.DataFrame(res)
    res["pasa_bonferroni_dev"] = (res.t_daily_dev > z_thr) & (res.n_dev >= MIN_N_DEV)
    return res, z_thr


def main():
    if not TRADES_FILE.exists():
        raise FileNotFoundError("No encuentro orb15_v2_trades.csv en esta carpeta.")
    if not DATA_DIR.exists():
        raise FileNotFoundError("No encuentro la carpeta datos_alpaca_sip.")

    out = build_trade_table()
    if out.empty:
        raise RuntimeError("No se reconstruyeron operaciones.")
    out.to_csv("orb15_v3_4_operaciones_features.csv", index=False)

    # Control: el baseline re-simulado debe parecerse al historico en trades que no llegan a cierre
    b = out[out.exit_variant == "hold_to_close"]
    closed = b[b.exit_reason.isin(["stop", "stop_same_bar", "target"])]
    if len(closed):
        diff = (closed.net_R - closed.historical_net_R).abs()
        print(f"\n[Control] En trades con stop/target, |net_R sim - historico|: "
              f"mediana={diff.median():.4f}, %<0.02R = {100*(diff < 0.02).mean():.1f}%")

    res, z_thr = evaluate(out)
    res.to_csv("orb15_v3_4_resultados_filtros.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

    cols = ["filter", "exit", "pct_trades_kept", "n_dev", "mean_R_dev", "t_daily_dev", "PF_dev",
            "years_pos_dev", "tickers_pos_dev", "n_oos", "mean_R_oos", "PF_oos", "pasa_bonferroni_dev"]
    print("\n=== TODOS LOS FILTROS (ordenados por t diario en DEV; OOS solo informativo) ===")
    print(res.sort_values("t_daily_dev", ascending=False)[cols].round(4).to_string(index=False))

    winners = res[res.pasa_bonferroni_dev]
    print("\n=== CONCLUSION AUTOMATICA ===")
    if winners.empty:
        print(f"Ningun filtro supera |t| > {z_thr:.2f} en DEV tras corregir por {len(res)} pruebas.")
        print("=> No hay evidencia de edge. Lo mas probable es que las mejoras aparentes sean azar.")
    else:
        print("Candidatos que superan Bonferroni en DEV (confirmar en OOS UNA vez):")
        print(winners[cols].round(4).to_string(index=False))

    # Desglose por simbolo y anio para los 3 mejores en DEV (diagnostico, no para elegir)
    top = res.sort_values("t_daily_dev", ascending=False).head(3)
    print("\n=== DESGLOSE DE LOS 3 MEJORES EN DEV (diagnostico) ===")
    for _, t in top.iterrows():
        sub = out[out.exit_variant == t["exit"]]
        g = sub[FILTERS[t["filter"]](sub)]
        print(f"\n{t['filter']} | {t['exit']}")
        print(g.groupby("year").net_R.agg(n="count", mean="mean", total="sum").round(4).to_string())
        print(g.groupby("ticker").net_R.agg(n="count", mean="mean", total="sum").round(4).to_string())
        print(g.groupby("direction").net_R.agg(n="count", mean="mean", total="sum").round(4).to_string())

    print("\nArchivos: orb15_v3_4_operaciones_features.csv, orb15_v3_4_resultados_filtros.csv")
    print("RECUERDA: no ajustes umbrales mirando OOS 2026. Cada filtro nuevo sube el numero de pruebas.")


if __name__ == "__main__":
    main()