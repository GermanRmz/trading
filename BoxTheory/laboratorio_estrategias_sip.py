"""
Laboratorio de cinco estrategias intradía, 5 minutos, datos locales Alpaca SIP.

Reglas predefinidas:
1. ORB15: ruptura del máximo/mínimo de las primeras 3 velas; stop al otro lado del rango; objetivo 1.5R.
2. VWAP_Reversion: cierre >2 desviaciones típicas sobre/bajo VWAP acumulado; entrar contra el movimiento; objetivo VWAP; stop a 1.5 desviaciones del cierre de señal.
3. Trend_Pullback: EMA20/EMA50 alineadas; retroceso a EMA20 y cierre de nuevo a favor; stop al extremo de las últimas 3 velas; objetivo 1.5R.
4. Gap_Reversal: gap >=0.5% vs cierre anterior; señal contraria al gap por ruptura del máximo/mínimo de vela anterior; objetivo cierre anterior.
5. RVOL_Momentum: ORB15 más ruptura con volumen >=1.5x la mediana del volumen de esa misma franja horaria en hasta 20 sesiones completas anteriores.

Convenciones:
- CSV: ./datos_alpaca_sip/SYMBOL_5min.csv
- Solo sesión regular NY 09:30-15:55; 78 barras esperadas.
- Sesiones incompletas excluidas de operación y referencias.
- Entrada en apertura de la siguiente vela; no se entra desde la última vela.
- Una operación por símbolo/estrategia/día.
- Stop primero si stop y objetivo se tocan en la misma vela.
- Salida al cierre si no se alcanza stop/objetivo.
- Desarrollo 2024-2025; prueba OOS 2026 (hasta la última fecha disponible).
- Coste estimado 4 bps ida y vuelta sobre el precio de entrada.
No descarga datos, no requiere API keys. Investigación experimental, no asesoría financiera.
"""

from pathlib import Path
import math
import sys
import numpy as np
import pandas as pd

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "SOXL", "NUGT"]
DATA_DIR = Path("datos_alpaca_sip")
NY_TZ = "America/New_York"
EXPECTED_BARS = 78
COST_BPS = 4.0
TARGET_R = 1.5
RVOL_LOOKBACK = 20
RVOL_MULTIPLIER = 1.5
STRATEGIES = ["ORB15", "VWAP_Reversion", "Trend_Pullback", "Gap_Reversal", "RVOL_Momentum"]

TRADES_OUT = "laboratorio_trades_sip.csv"
SESSION_AUDIT_OUT = "laboratorio_auditoria_sesiones.csv"
FILE_AUDIT_OUT = "laboratorio_auditoria_archivos.csv"
SUMMARY_OUT = "laboratorio_metricas.csv"


def load_data(ticker):
    path = DATA_DIR / f"{ticker}_5min.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    raw = pd.read_csv(path)
    candidates = ["timestamp", "Timestamp", "time", "Time", "datetime", "Datetime", "date", "Date"]
    tc = next((c for c in candidates if c in raw.columns), None)
    if tc is None:
        tc = raw.columns[0]
    timestamps = pd.to_datetime(raw[tc], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Timestamp inválido en {path}, columna {tc}")
    raw = raw.drop(columns=[tc])
    raw.index = pd.DatetimeIndex(timestamps).tz_convert(NY_TZ)
    raw = raw.sort_index()
    dupes = int(raw.index.duplicated(keep="first").sum())
    raw = raw.loc[~raw.index.duplicated(keep="first")].copy()
    raw.columns = [str(c).strip().capitalize() for c in raw.columns]
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas {missing} en {path}; disponibles: {list(raw.columns)}")
    for c in required:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    bad = int(raw[required].isna().any(axis=1).sum())
    raw = raw.dropna(subset=required).copy()
    if "Volume" not in raw.columns:
        raw["Volume"] = np.nan
    else:
        raw["Volume"] = pd.to_numeric(raw["Volume"], errors="coerce")
    regular = raw.between_time("09:30", "15:55").copy()
    regular["session"] = regular.index.date
    audit = {
        "ticker": ticker, "source_file": str(path), "rows_in_file": len(pd.read_csv(path)),
        "regular_rows_after_cleaning": len(regular), "duplicate_timestamps_removed": dupes,
        "invalid_ohlc_rows_removed": bad,
        "first_regular_timestamp": str(regular.index.min()) if len(regular) else "",
        "last_regular_timestamp": str(regular.index.max()) if len(regular) else "",
        "volume_available": bool(regular["Volume"].notna().any()),
    }
    return regular, audit


def build_sessions(df):
    sessions = []
    for date, day in df.groupby("session", sort=True):
        day = day.sort_index().copy()
        expected = pd.date_range(
            start=pd.Timestamp(str(date), tz=NY_TZ) + pd.Timedelta(hours=9, minutes=30),
            periods=EXPECTED_BARS, freq="5min"
        )
        complete = len(day) == EXPECTED_BARS and day.index.equals(expected)
        day["EMA20"] = day["Close"].ewm(span=20, adjust=False).mean()
        day["EMA50"] = day["Close"].ewm(span=50, adjust=False).mean()
        typical = (day["High"] + day["Low"] + day["Close"]) / 3
        vol = day["Volume"].clip(lower=0)
        if vol.notna().any() and vol.fillna(0).sum() > 0:
            day["VWAP"] = (typical * vol.fillna(0)).cumsum() / vol.fillna(0).cumsum().replace(0, np.nan)
        else:
            day["VWAP"] = np.nan
        day["TypicalStd"] = typical.expanding(min_periods=3).std(ddof=0)
        sessions.append({
            "date": date, "data": day, "bars": len(day), "complete": bool(complete),
            "first_bar": str(day.index[0]) if len(day) else "",
            "last_bar": str(day.index[-1]) if len(day) else ""
        })
    return sessions


def previous_close(sessions, idx):
    for j in range(idx - 1, -1, -1):
        if sessions[j]["complete"]:
            return float(sessions[j]["data"].iloc[-1]["Close"])
    return None


def prior_slot_median_volume(sessions, idx, slot):
    # Solo sesiones completas estrictamente anteriores al día de la señal.
    prior = [s for s in sessions[:idx] if s["complete"]][-RVOL_LOOKBACK:]
    vals = []
    for s in prior:
        v = float(s["data"].iloc[slot]["Volume"])
        if np.isfinite(v) and v >= 0:
            vals.append(v)
    if len(vals) < 10:
        return np.nan
    return float(np.median(vals))


def find_signal(strategy, day, sessions, idx):
    if len(day) != EXPECTED_BARS:
        return None

    if strategy in ("ORB15", "RVOL_Momentum"):
        orb = day.iloc[:3]
        rh, rl = float(orb["High"].max()), float(orb["Low"].min())
        for pos in range(3, len(day) - 1):
            bar = day.iloc[pos]
            if strategy == "RVOL_Momentum":
                ref = prior_slot_median_volume(sessions, idx, pos)
                volume = float(bar["Volume"])
                if not np.isfinite(ref) or ref <= 0 or not np.isfinite(volume) or volume < RVOL_MULTIPLIER * ref:
                    continue
            close = float(bar["Close"])
            if close > rh:
                return pos, "long", rl, None
            if close < rl:
                return pos, "short", rh, None
        return None

    if strategy == "VWAP_Reversion":
        for pos in range(2, len(day) - 1):
            b = day.iloc[pos]
            vwap, std, close = float(b["VWAP"]), float(b["TypicalStd"]), float(b["Close"])
            if not np.isfinite(vwap) or not np.isfinite(std) or std <= 0:
                continue
            entry = float(day.iloc[pos + 1]["Open"])
            if close > vwap + 2 * std:
                stop, target = close + 1.5 * std, vwap
                if stop > entry > target:
                    return pos, "short", stop, target
            elif close < vwap - 2 * std:
                stop, target = close - 1.5 * std, vwap
                if stop < entry < target:
                    return pos, "long", stop, target
        return None

    if strategy == "Trend_Pullback":
        for pos in range(50, len(day) - 1):
            b, prev = day.iloc[pos], day.iloc[pos - 1]
            e20, e50 = float(b["EMA20"]), float(b["EMA50"])
            recent = day.iloc[pos - 2:pos + 1]
            close = float(b["Close"])
            if e20 > e50 and float(b["Low"]) <= e20 and close > e20 and float(prev["Close"]) > float(prev["EMA20"]):
                direction, stop = "long", float(recent["Low"].min())
            elif e20 < e50 and float(b["High"]) >= e20 and close < e20 and float(prev["Close"]) < float(prev["EMA20"]):
                direction, stop = "short", float(recent["High"].max())
            else:
                continue
            entry = float(day.iloc[pos + 1]["Open"])
            if (direction == "long" and stop >= entry) or (direction == "short" and stop <= entry):
                continue
            return pos, direction, stop, None
        return None

    if strategy == "Gap_Reversal":
        prev_close = previous_close(sessions, idx)
        if prev_close is None or prev_close <= 0:
            return None
        gap = (float(day.iloc[0]["Open"]) / prev_close - 1) * 100
        if abs(gap) < 0.5:
            return None
        for pos in range(1, len(day) - 1):
            b, prev = day.iloc[pos], day.iloc[pos - 1]
            entry = float(day.iloc[pos + 1]["Open"])
            if gap > 0 and float(b["Close"]) < float(prev["Low"]):
                stop, target = float(b["High"]), prev_close
                if stop > entry > target:
                    return pos, "short", stop, target
            if gap < 0 and float(b["Close"]) > float(prev["High"]):
                stop, target = float(b["Low"]), prev_close
                if stop < entry < target:
                    return pos, "long", stop, target
        return None

    return None


def make_trade(ticker, strategy, session, day, signal):
    pos, direction, stop, fixed_target = signal
    entry_pos = pos + 1
    if entry_pos >= len(day):
        return None
    entry = float(day.iloc[entry_pos]["Open"])
    if (direction == "long" and stop >= entry) or (direction == "short" and stop <= entry):
        return None
    risk = abs(entry - stop)
    if not np.isfinite(risk) or risk <= 0:
        return None
    target = fixed_target
    if target is None:
        target = entry + TARGET_R * risk if direction == "long" else entry - TARGET_R * risk
    # Require target to be on the profit side.
    if (direction == "long" and target <= entry) or (direction == "short" and target >= entry):
        return None

    exit_price = float(day.iloc[-1]["Close"])
    exit_time = day.index[-1]
    reason = "session_close"
    for j in range(entry_pos, len(day)):
        b = day.iloc[j]
        hi, lo = float(b["High"]), float(b["Low"])
        if direction == "long":
            hit_stop, hit_target = lo <= stop, hi >= target
        else:
            hit_stop, hit_target = hi >= stop, lo <= target
        if hit_stop and hit_target:
            exit_price, exit_time, reason = stop, day.index[j], "stop_same_bar"
            break
        if hit_stop:
            exit_price, exit_time, reason = stop, day.index[j], "stop"
            break
        if hit_target:
            exit_price, exit_time, reason = target, day.index[j], "target"
            break

    pnl = exit_price - entry if direction == "long" else entry - exit_price
    gross_r = pnl / risk
    cost_per_share = entry * COST_BPS / 10000
    cost_r = cost_per_share / risk
    return {
        "ticker": ticker, "strategy": strategy, "date": str(session),
        "year": pd.Timestamp(str(session)).year, "direction": direction,
        "signal_time": str(day.index[pos]), "entry_time": str(day.index[entry_pos]),
        "entry": entry, "stop": float(stop), "target": float(target),
        "exit_time": str(exit_time), "exit": float(exit_price), "exit_reason": reason,
        "risk_per_share": risk, "gross_R": gross_r, "estimated_cost_R": cost_r,
        "net_R": gross_r - cost_r, "cost_bps_round_trip": COST_BPS
    }


def pf(values):
    s = pd.Series(values, dtype=float)
    wins, losses = s[s > 0].sum(), abs(s[s < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else np.nan
    return float(wins / losses)


def metric_row(period, strategy, ticker, g):
    g = g.sort_values(["date", "entry_time", "ticker"])
    net = g["net_R"].astype(float)
    equity = net.cumsum()
    dd = equity - equity.cummax()
    return {
        "period": period, "strategy": strategy, "ticker": ticker, "trades": len(g),
        "win_rate_net": float((net > 0).mean()), "mean_gross_R": float(g["gross_R"].mean()),
        "mean_net_R": float(net.mean()), "median_net_R": float(net.median()),
        "profit_factor_net": pf(net), "max_drawdown_R": float(dd.min()),
        "total_net_R": float(net.sum())
    }


def make_summary(trades):
    rows = []
    periods = [
        ("Development_2024_2025", trades["year"].isin([2024, 2025])),
        ("OOS_2026", trades["year"].eq(2026)),
        ("All", pd.Series(True, index=trades.index))
    ]
    for pname, mask in periods:
        sub = trades.loc[mask]
        for (strategy, ticker), g in sub.groupby(["strategy", "ticker"]):
            rows.append(metric_row(pname, strategy, ticker, g))
        for strategy, g in sub.groupby("strategy"):
            rows.append(metric_row(pname, strategy, "ALL_SYMBOLS", g))
    return pd.DataFrame(rows)


def main():
    trades_all, audits, file_audits = [], [], []
    for ticker in TICKERS:
        try:
            df, file_audit = load_data(ticker)
            file_audits.append(file_audit)
            sessions = build_sessions(df)
            complete_count = sum(s["complete"] for s in sessions)
            print(f"{ticker}: {complete_count}/{len(sessions)} sesiones completas")
            for idx, s in enumerate(sessions):
                prior_ok = any(x["complete"] for x in sessions[:idx])
                audits.append({
                    "ticker": ticker, "date": str(s["date"]),
                    "year": pd.Timestamp(str(s["date"])).year, "bars": s["bars"],
                    "complete": s["complete"], "first_bar": s["first_bar"],
                    "last_bar": s["last_bar"],
                    "volume_present": bool(s["data"]["Volume"].notna().any()),
                    "eligible": bool(s["complete"] and prior_ok),
                    "excluded_reason": "" if s["complete"] and prior_ok else
                        ("incomplete_session" if not s["complete"] else "no_prior_complete_session")
                })
                if not s["complete"] or not prior_ok:
                    continue
                for strategy in STRATEGIES:
                    signal = find_signal(strategy, s["data"], sessions, idx)
                    if signal is None:
                        continue
                    tr = make_trade(ticker, strategy, s["date"], s["data"], signal)
                    if tr:
                        trades_all.append(tr)
        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}")

    if file_audits:
        pd.DataFrame(file_audits).to_csv(FILE_AUDIT_OUT, index=False)
    if audits:
        pd.DataFrame(audits).to_csv(SESSION_AUDIT_OUT, index=False)
    if not trades_all:
        print("No se generaron operaciones. Revisa archivos y errores.")
        sys.exit(1)

    trades = pd.DataFrame(trades_all).sort_values(
        ["date", "ticker", "strategy", "entry_time"]
    ).reset_index(drop=True)
    trades.to_csv(TRADES_OUT, index=False)
    summary = make_summary(trades)
    summary.to_csv(SUMMARY_OUT, index=False)

    pooled = summary[summary["ticker"] == "ALL_SYMBOLS"]
    print("\n=== COMPARACIÓN POR ESTRATEGIA ===")
    for period in ["Development_2024_2025", "OOS_2026", "All"]:
        part = pooled[pooled["period"] == period].sort_values("strategy")
        if part.empty:
            continue
        print(f"\n--- {period} ---")
        for _, r in part.iterrows():
            p = r["profit_factor_net"]
            ptxt = "inf" if pd.notna(p) and math.isinf(p) else (f"{p:.3f}" if pd.notna(p) else "NA")
            print(f"{r['strategy']:18s} n={int(r['trades']):4d} | bruto={r['mean_gross_R']:+.3f}R | neto={r['mean_net_R']:+.3f}R | PF={ptxt} | DD={r['max_drawdown_R']:.1f}R")

    print("\nCSV generados:")
    print(f"  {TRADES_OUT}")
    print(f"  {SESSION_AUDIT_OUT}")
    print(f"  {FILE_AUDIT_OUT}")
    print(f"  {SUMMARY_OUT}")
    print(f"Coste supuesto: {COST_BPS:.1f} bps ida y vuelta.")
    print("2026 es una prueba temporal fuera de muestra y no es un año completo.")
    print("No ajustes reglas usando 2026 y luego sigas llamándolo OOS.")


if __name__ == "__main__":
    main()
