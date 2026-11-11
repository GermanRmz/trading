"""
Backtest exploratorio: ruptura del máximo/mínimo de la sesión anterior (PDH/PDL).
Datos: Yahoo Finance vía yfinance; velas de 5 minutos; solo sesión regular NYSE/Nasdaq.
IMPORTANTE: el histórico intradía de Yahoo suele limitarse a los últimos 60 días.
No es asesoría financiera. El backtest es exploratorio y los fills son aproximaciones.
"""

from datetime import datetime, timedelta
import sys
import pandas as pd
import yfinance as yf

# Universo inicial fijo; evita elegir acciones después de ver cuáles ganaron.
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"]
INTERVAL = "5m"
PERIOD = "60d"
TARGET_R = 1.5
ROUND_TRIP_COST_BPS = 4.0  # supuesto: spread/comisiones/deslizamiento combinados
NY_TZ = "America/New_York"


def regular_session(df):
    """Filtra velas de 9:30 a 16:00 Nueva York y normaliza el índice a NY."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(NY_TZ)
    out = df.copy()
    out.index = idx
    out = out[~out.index.duplicated(keep="first")]
    out = out.between_time("09:30", "15:55")
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def backtest_ticker(ticker):
    raw = yf.download(
        ticker, period=PERIOD, interval=INTERVAL,
        auto_adjust=False, prepost=False, progress=False, threads=False
    )
    if raw.empty:
        print(f"[AVISO] Sin datos para {ticker}")
        return []

    # Algunas versiones de yfinance devuelven columnas MultiIndex incluso para un ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(-1):
            raw = raw.xs(ticker, axis=1, level=-1)
        else:
            raw.columns = raw.columns.get_level_values(0)

    df = regular_session(raw)
    if df.empty:
        print(f"[AVISO] Sin velas regulares para {ticker}")
        return []

    df["session"] = df.index.date
    sessions = list(df.groupby("session", sort=True))
    trades = []

    for i in range(1, len(sessions)):
        prev_date, prev = sessions[i - 1]
        today_date, today = sessions[i]
        if len(prev) < 30 or len(today) < 30:
            continue

        pdh = float(prev["High"].max())
        pdl = float(prev["Low"].min())
        day_open = float(today.iloc[0]["Open"])

        # Excluir gaps que ya abren fuera de la caja.
        if not (pdl <= day_open <= pdh):
            continue

        # Señal al cierre de una vela; entrada en la apertura de la siguiente.
        signal = None
        signal_pos = None
        for pos in range(0, len(today) - 1):
            bar = today.iloc[pos]
            if float(bar["Close"]) > pdh:
                signal, signal_pos = "long", pos
                break
            if float(bar["Close"]) < pdl:
                signal, signal_pos = "short", pos
                break

        if signal is None or signal_pos is None:
            continue

        sig_bar = today.iloc[signal_pos]
        entry_bar = today.iloc[signal_pos + 1]
        entry = float(entry_bar["Open"])
        stop = float(sig_bar["Low"] if signal == "long" else sig_bar["High"])

        # Si la entrada siguiente ya invalida el stop, descartar la operación.
        if (signal == "long" and stop >= entry) or (signal == "short" and stop <= entry):
            continue

        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            continue

        target = entry + TARGET_R * risk_per_share if signal == "long" else entry - TARGET_R * risk_per_share
        exit_price = float(today.iloc[-1]["Close"])
        exit_time = today.index[-1]
        exit_reason = "session_close"

        # Empezar desde la vela de entrada. Si stop y target ocurren en la misma vela,
        # se asume stop primero (conservador). La entrada se aproxima al Open de esa vela.
        for j in range(signal_pos + 1, len(today)):
            bar = today.iloc[j]
            hi, lo = float(bar["High"]), float(bar["Low"])
            if signal == "long":
                hit_stop = lo <= stop
                hit_target = hi >= target
            else:
                hit_stop = hi >= stop
                hit_target = lo <= target

            if hit_stop and hit_target:
                exit_price, exit_time, exit_reason = stop, today.index[j], "stop_same_bar"
                break
            if hit_stop:
                exit_price, exit_time, exit_reason = stop, today.index[j], "stop"
                break
            if hit_target:
                exit_price, exit_time, exit_reason = target, today.index[j], "target"
                break

        gross_pnl_per_share = (exit_price - entry) if signal == "long" else (entry - exit_price)
        gross_r = gross_pnl_per_share / risk_per_share
        cost_per_share = entry * (ROUND_TRIP_COST_BPS / 10000.0)
        net_r = (gross_pnl_per_share - cost_per_share) / risk_per_share

        trades.append({
            "ticker": ticker,
            "date": str(today_date),
            "direction": signal,
            "previous_high": pdh,
            "previous_low": pdl,
            "signal_time": str(today.index[signal_pos]),
            "entry_time": str(today.index[signal_pos + 1]),
            "entry": entry,
            "stop": stop,
            "target": target,
            "exit_time": str(exit_time),
            "exit": exit_price,
            "exit_reason": exit_reason,
            "risk_per_share": risk_per_share,
            "gross_R": gross_r,
            "estimated_cost_R": cost_per_share / risk_per_share,
            "net_R": net_r,
        })

    print(f"{ticker}: {len(trades)} operaciones")
    return trades


def main():
    all_trades = []
    for ticker in TICKERS:
        try:
            all_trades.extend(backtest_ticker(ticker))
        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}")

    if not all_trades:
        print("\nNo se generaron operaciones. Revisa conexión, datos o criterios.")
        sys.exit(0)

    trades = pd.DataFrame(all_trades).sort_values(["date", "ticker"])
    trades.to_csv("caja_trades.csv", index=False)

    wins = (trades["net_R"] > 0).sum()
    losses = (trades["net_R"] < 0).sum()
    total = len(trades)
    avg_net = trades["net_R"].mean()
    gross_avg = trades["gross_R"].mean()
    profit_factor = (
        trades.loc[trades.net_R > 0, "net_R"].sum()
        / abs(trades.loc[trades.net_R < 0, "net_R"].sum())
        if losses else float("inf")
    )
    equity = trades["net_R"].cumsum()
    drawdown_R = (equity - equity.cummax()).min()

    print("\n=== RESULTADOS EXPLORATORIOS ===")
    print(f"Operaciones: {total}")
    print(f"Acierto neto: {wins / total:.1%}")
    print(f"Ganadoras / perdedoras / neutras: {wins} / {losses} / {(trades.net_R == 0).sum()}")
    print(f"R medio bruto por operación: {gross_avg:.3f}R")
    print(f"R medio neto estimado por operación: {avg_net:.3f}R")
    print(f"Profit factor neto: {profit_factor:.3f}")
    print(f"Máximo drawdown acumulado (R, orden cronológico de la tabla): {drawdown_R:.2f}R")
    print(f"Coste estimado: {ROUND_TRIP_COST_BPS:.1f} puntos básicos ida y vuelta")
    print("\nGuardado: caja_trades.csv")
    print("Revisa el CSV y no uses estos resultados como prueba definitiva: la muestra es corta,")
    print("los costes son supuestos y los fills intrabar no se conocen exactamente.")


if __name__ == "__main__":
    main()
