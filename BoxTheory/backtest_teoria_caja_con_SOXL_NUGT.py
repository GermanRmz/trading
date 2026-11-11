
"""
Backtest: ruptura del máximo/mínimo de la sesión regular anterior (PDH/PDL).

Fuente: CSV históricos de Alpaca SIP.
Temporalidad: 5 minutos.
Sesión regular: 09:30-16:00, hora de Nueva York.
Caja: máximo y mínimo de la sesión regular anterior completa.
Entrada: apertura de la vela posterior a la primera vela que cierra
         fuera de la caja.
Stop: mínimo/máximo de la vela de señal.
Objetivo: 1.5R.
Máximo: una operación por símbolo por sesión.
Si stop y objetivo se alcanzan en la misma vela, se asume primero el stop.

Las sesiones con menos o más de 78 velas regulares se excluyen.
No es asesoría financiera. Los resultados son una simulación.
"""

from pathlib import Path
import sys
import pandas as pd

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "SOXL", "NUGT"]

DATA_DIR = Path("datos_alpaca_sip")
OUTPUT_TRADES = "caja_trades_sip.csv"
OUTPUT_SESSIONS = "caja_sesiones_sip.csv"

TARGET_R = 1.5
ROUND_TRIP_COST_BPS = 4.0
NY_TZ = "America/New_York"

EXPECTED_BARS = 78  # 09:30, 09:35, ..., 15:55


def load_data(ticker):
    """Lee CSV de Alpaca y convierte las fechas a Nueva York."""
    path = DATA_DIR / f"{ticker}_5min.csv"

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    df = pd.read_csv(path)

    # Admite nombres comunes para la columna temporal.
    possible_time_columns = [
        "timestamp", "Timestamp", "time", "Time", "datetime", "Datetime"
    ]
    time_column = next(
        (col for col in possible_time_columns if col in df.columns),
        None
    )

    if time_column is not None:
        df[time_column] = pd.to_datetime(df[time_column], utc=True)
        df = df.set_index(time_column)
    else:
        # Algunos CSV exportan el timestamp como primera columna sin nombre.
        first_col = df.columns[0]
        parsed = pd.to_datetime(df[first_col], utc=True, errors="coerce")
        if parsed.notna().all():
            df = df.drop(columns=[first_col])
            df.index = parsed
        else:
            raise ValueError(
                f"No se encontró una columna de fecha reconocible en {path}. "
                f"Columnas: {list(df.columns)}"
            )

    df.index = pd.DatetimeIndex(df.index).tz_convert(NY_TZ)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Normaliza nombres de OHLC a mayúsculas.
    rename = {col: str(col).strip().capitalize() for col in df.columns}
    df = df.rename(columns=rename)

    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} en {path}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    # Solo velas de la sesión regular.
    df = df.between_time("09:30", "15:55").copy()
    df["session"] = df.index.date

    return df


def get_sessions(df):
    """
    Devuelve sesiones con su número de velas y etiqueta de integridad.
    Se requiere exactamente una vela por cada intervalo regular de 5 min.
    """
    sessions = []

    for session_date, day in df.groupby("session", sort=True):
        day = day.sort_index()

        # Verifica cantidad y que los timestamps coincidan con la rejilla
        # esperada de 5 minutos entre 09:30 y 15:55.
        expected_index = pd.date_range(
            start=pd.Timestamp(str(session_date), tz=NY_TZ)
                  + pd.Timedelta(hours=9, minutes=30),
            periods=EXPECTED_BARS,
            freq="5min",
        )
        complete = (
            len(day) == EXPECTED_BARS
            and day.index.equals(expected_index)
        )

        sessions.append({
            "date": session_date,
            "data": day,
            "bars": len(day),
            "complete": complete,
        })

    return sessions


def backtest_ticker(ticker):
    df = load_data(ticker)
    sessions = get_sessions(df)

    if not sessions:
        print(f"[AVISO] Sin sesiones para {ticker}")
        return [], []

    trades = []
    session_audit = []

    for i, current in enumerate(sessions):
        previous = sessions[i - 1] if i > 0 else None

        session_audit.append({
            "ticker": ticker,
            "date": str(current["date"]),
            "bars": current["bars"],
            "complete": current["complete"],
            "previous_date": (
                str(previous["date"]) if previous is not None else ""
            ),
            "previous_complete": (
                previous["complete"] if previous is not None else False
            ),
            "eligible_for_backtest": bool(
                previous is not None
                and current["complete"]
                and previous["complete"]
            ),
        })

        # No operar si el día actual o la sesión inmediatamente anterior
        # no están completos.
        if previous is None or not current["complete"] or not previous["complete"]:
            continue

        prev = previous["data"]
        today = current["data"]

        pdh = float(prev["High"].max())
        pdl = float(prev["Low"].min())
        day_open = float(today.iloc[0]["Open"])

        # Excluir gaps que abren fuera de la caja anterior.
        if not (pdl <= day_open <= pdh):
            continue

        signal = None
        signal_pos = None

        # La última vela no puede generar una entrada porque no habría
        # una vela posterior dentro de la sesión para ejecutar.
        for pos in range(len(today) - 1):
            close = float(today.iloc[pos]["Close"])

            if close > pdh:
                signal = "long"
                signal_pos = pos
                break
            if close < pdl:
                signal = "short"
                signal_pos = pos
                break

        if signal is None:
            continue

        sig_bar = today.iloc[signal_pos]
        entry_bar = today.iloc[signal_pos + 1]

        entry = float(entry_bar["Open"])
        stop = float(
            sig_bar["Low"] if signal == "long" else sig_bar["High"]
        )

        # Rechazar una entrada que ya está más allá del stop.
        if (signal == "long" and stop >= entry) or (
            signal == "short" and stop <= entry
        ):
            continue

        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            continue

        target = (
            entry + TARGET_R * risk_per_share
            if signal == "long"
            else entry - TARGET_R * risk_per_share
        )

        exit_price = float(today.iloc[-1]["Close"])
        exit_time = today.index[-1]
        exit_reason = "session_close"

        # Examinar desde la vela de entrada inclusive.
        for j in range(signal_pos + 1, len(today)):
            bar = today.iloc[j]
            hi = float(bar["High"])
            lo = float(bar["Low"])

            if signal == "long":
                hit_stop = lo <= stop
                hit_target = hi >= target
            else:
                hit_stop = hi >= stop
                hit_target = lo <= target

            if hit_stop and hit_target:
                exit_price = stop
                exit_time = today.index[j]
                exit_reason = "stop_same_bar"
                break
            if hit_stop:
                exit_price = stop
                exit_time = today.index[j]
                exit_reason = "stop"
                break
            if hit_target:
                exit_price = target
                exit_time = today.index[j]
                exit_reason = "target"
                break

        gross_pnl_per_share = (
            exit_price - entry
            if signal == "long"
            else entry - exit_price
        )
        gross_r = gross_pnl_per_share / risk_per_share

        # Aproximación de coste proporcional al precio de entrada.
        cost_per_share = entry * ROUND_TRIP_COST_BPS / 10000.0
        cost_r = cost_per_share / risk_per_share
        net_r = gross_r - cost_r

        trades.append({
            "ticker": ticker,
            "date": str(current["date"]),
            "previous_session": str(previous["date"]),
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
            "estimated_cost_R": cost_r,
            "net_R": net_r,
        })

    print(
        f"{ticker}: {len(trades)} operaciones | "
        f"{sum(s['complete'] for s in sessions)}/{len(sessions)} "
        "sesiones completas"
    )
    return trades, session_audit


def profit_factor(series):
    gains = series[series > 0].sum()
    losses = abs(series[series < 0].sum())
    return gains / losses if losses > 0 else float("inf")


def main():
    all_trades = []
    all_sessions = []

    for ticker in TICKERS:
        try:
            trades, audit = backtest_ticker(ticker)
            all_trades.extend(trades)
            all_sessions.extend(audit)
        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}")

    if all_sessions:
        pd.DataFrame(all_sessions).to_csv(OUTPUT_SESSIONS, index=False)
        print(f"\nAuditoría de sesiones guardada en: {OUTPUT_SESSIONS}")

    if not all_trades:
        print("\nNo se generaron operaciones.")
        print("Revisa los errores anteriores y el formato de los CSV.")
        sys.exit(0)

    trades = pd.DataFrame(all_trades).sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)
    trades.to_csv(OUTPUT_TRADES, index=False)

    wins = int((trades["net_R"] > 0).sum())
    losses = int((trades["net_R"] < 0).sum())
    total = len(trades)

    equity = trades["net_R"].cumsum()
    drawdown = (equity - equity.cummax()).min()

    print("\n=== RESULTADOS EXPLORATORIOS SIP ===")
    print(f"Operaciones: {total}")
    print(f"Acierto neto: {wins / total:.1%}")
    print(
        f"Ganadoras / perdedoras / neutras: "
        f"{wins} / {losses} / {(trades['net_R'] == 0).sum()}"
    )
    print(f"R medio bruto: {trades['gross_R'].mean():.3f}R")
    print(f"R medio neto: {trades['net_R'].mean():.3f}R")
    print(f"Profit factor neto: {profit_factor(trades['net_R']):.3f}")
    print(f"Drawdown acumulado de operaciones: {drawdown:.2f}R")
    print(f"Coste supuesto: {ROUND_TRIP_COST_BPS:.1f} bps ida y vuelta")

    print("\n--- Por símbolo ---")
    for ticker, group in trades.groupby("ticker"):
        print(
            f"{ticker:5s} | n={len(group):4d} | "
            f"media neta={group['net_R'].mean():+.3f}R | "
            f"PF={profit_factor(group['net_R']):.3f}"
        )

    print(f"\nOperaciones guardadas en: {OUTPUT_TRADES}")
    print(
        "Nota: los resultados son una simulación. Los costes, "
        "deslizamiento y ejecución intrabar son aproximaciones."
    )


if __name__ == "__main__":
    main()
