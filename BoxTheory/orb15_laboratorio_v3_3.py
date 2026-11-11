"""
ORB15 v3.3 - Simulacion de gestion de riesgo:
- Baseline: stop original + target 1.5R.
- Break-even al tocar +0.5R.
- Break-even al tocar +0.75R.
- Trailing stop 0.5R despues de tocar +1R.

Usa orb15_v2_trades.csv y datos_alpaca_sip/<TICKER>_5min.csv.
Ejecutar desde la carpeta trading:
    python orb15_laboratorio_v3_3.py

Convencion conservadora:
- Si stop vigente y objetivo se tocan en la misma vela, stop primero.
- Un stop nuevo activado por el maximo/minimo de una vela solo entra en vigor
  desde la vela siguiente (no se inventa el orden intravela).
- MFE/MAE se calculan desde la entrada hasta la salida simulada.
- Las operaciones se vuelven a simular hasta el cierre de la sesion, no se
  recortan en la salida historica original.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("datos_alpaca_sip")
TRADES_FILE = Path("orb15_v2_trades.csv")
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "SOXL", "NUGT"]
NY_TZ = "America/New_York"
VARIANT = "ORB15_T1.5_none_all_day"
COST_BPS = 4.0

POLICIES = [
    {"name": "baseline", "trigger_r": None, "mode": "none", "trail_r": None},
    {"name": "BE_0.50R", "trigger_r": 0.50, "mode": "be", "trail_r": None},
    {"name": "BE_0.75R", "trigger_r": 0.75, "mode": "be", "trail_r": None},
    {"name": "TRAIL_0.50R_after_1R", "trigger_r": 1.00, "mode": "trail", "trail_r": 0.50},
]

def load_bars(ticker):
    path = DATA_DIR / f"{ticker}_5min.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path}")
    raw = pd.read_csv(path)
    tc = next((c for c in ["timestamp","Timestamp","time","Time","datetime","Datetime","date","Date"] if c in raw.columns), raw.columns[0])
    ts = pd.to_datetime(raw.pop(tc), utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Timestamps invalidos en {path}")
    raw.index = pd.DatetimeIndex(ts).tz_convert(NY_TZ)
    raw = raw.sort_index()
    raw = raw.loc[~raw.index.duplicated(keep="first")].copy()
    raw.columns = [str(c).strip().capitalize() for c in raw.columns]
    for c in ["Open","High","Low","Close"]:
        if c not in raw.columns:
            raise ValueError(f"Falta columna {c} en {path}")
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if "Volume" not in raw:
        raw["Volume"] = np.nan
    raw = raw.dropna(subset=["Open","High","Low","Close"])
    raw = raw.between_time("09:30","15:55").copy()
    return raw

def pf(x):
    x = pd.Series(x, dtype=float)
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    return gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan)

def simulate(day, entry_time, direction, entry, initial_stop, target, risk, policy):
    entry_ts = pd.Timestamp(entry_time).tz_convert(NY_TZ)
    start = int(day.index.searchsorted(entry_ts, side="left"))
    if start >= len(day):
        return None
    bars = day.iloc[start:]
    stop = float(initial_stop)
    best_fav = 0.0
    mfe = 0.0
    mae = 0.0
    triggered = False
    ambiguous_trigger_bar = 0
    exit_price = float(bars.iloc[-1]["Close"])
    exit_time = bars.index[-1]
    reason = "session_close"
    bars_held = 0

    for j, (ts, bar) in enumerate(bars.iterrows()):
        o, h, l, c = map(float, (bar["Open"],bar["High"],bar["Low"],bar["Close"]))
        bars_held += 1
        if direction == "long":
            fav = (h-entry)/risk
            adv = (entry-l)/risk
            stop_hit = l <= stop
            target_hit = h >= target
        else:
            fav = (entry-l)/risk
            adv = (h-entry)/risk
            stop_hit = h >= stop
            target_hit = l <= target
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # Stop/target existentes se evalúan antes de activar cualquier nuevo stop.
        if stop_hit and target_hit:
            exit_price, exit_time, reason = stop, ts, "stop_same_bar"
            break
        if stop_hit:
            # Gap adverso: usar apertura si ya abrió más allá del stop.
            if direction == "long" and o < stop:
                exit_price = o
            elif direction == "short" and o > stop:
                exit_price = o
            else:
                exit_price = stop
            exit_time, reason = ts, ("break_even" if abs(stop-entry) < 1e-9 else ("trailing_stop" if triggered and policy["mode"]=="trail" else "stop"))
            break
        if target_hit:
            exit_price, exit_time, reason = float(target), ts, "target"
            break

        # Activación al cierre de esta vela; stop nuevo aplica desde la siguiente.
        if policy["mode"] == "be" and not triggered and fav >= policy["trigger_r"]:
            triggered = True
            stop = float(entry)
            # Si la vela también atravesó el nivel de BE, no inferimos orden intravela.
            if (direction == "long" and l <= entry) or (direction == "short" and h >= entry):
                ambiguous_trigger_bar += 1
        elif policy["mode"] == "trail":
            best_fav = max(best_fav, fav)
            if best_fav >= policy["trigger_r"]:
                triggered = True
                new_stop = (entry + (best_fav-policy["trail_r"])*risk if direction=="long"
                            else entry - (best_fav-policy["trail_r"])*risk)
                stop = max(stop, new_stop) if direction=="long" else min(stop, new_stop)

    pnl = (exit_price-entry) if direction=="long" else (entry-exit_price)
    gross_r = pnl/risk
    cost_r = (entry*COST_BPS/10000.0)/risk
    return {
        "policy":policy["name"], "exit_time_sim":exit_time, "exit_price_sim":exit_price,
        "exit_reason_sim":reason, "gross_R_sim":gross_r, "cost_R":cost_r,
        "net_R_sim":gross_r-cost_r, "MFE_R_sim":mfe, "MAE_R_sim":mae,
        "bars_held_sim":bars_held, "management_triggered":triggered,
        "ambiguous_trigger_bar":ambiguous_trigger_bar
    }

def main():
    if not TRADES_FILE.exists():
        raise FileNotFoundError("No encuentro orb15_v2_trades.csv en esta carpeta.")
    if not DATA_DIR.exists():
        raise FileNotFoundError("No encuentro la carpeta datos_alpaca_sip.")
    trades = pd.read_csv(TRADES_FILE)
    base = trades[(trades.variant == VARIANT) & (trades.cost_bps == COST_BPS)].copy()
    for c in ["signal_time","entry_time","exit_time"]:
        base[c] = pd.to_datetime(base[c], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    base["date"] = base["date"].astype(str).str[:10]
    print(f"Operaciones base únicas: {len(base):,}")

    all_rows = []
    audit = []
    for ticker in TICKERS:
        z = base[base.ticker == ticker]
        if z.empty:
            continue
        bars = load_bars(ticker)
        days = {d:g for d,g in bars.groupby(bars.index.strftime("%Y-%m-%d"))}
        print(f"{ticker}: {len(z)} operaciones")
        for r in z.itertuples(index=False):
            day = days.get(r.date)
            if day is None or day.empty:
                audit.append({"ticker":ticker,"date":r.date,"status":"missing_day"})
                continue
            # Comprobar que entry sea una apertura de vela disponible.
            et = pd.Timestamp(r.entry_time).tz_convert(NY_TZ)
            ix = int(day.index.searchsorted(et, side="left"))
            if ix >= len(day):
                audit.append({"ticker":ticker,"date":r.date,"status":"entry_after_session"})
                continue
            for pol in POLICIES:
                result = simulate(day, r.entry_time, r.direction, float(r.entry),
                                  float(r.stop), float(r.target), float(r.risk_per_share), pol)
                if result is None:
                    continue
                all_rows.append({
                    "ticker":ticker,"date":r.date,"year":int(r.year),"month":r.month,
                    "direction":r.direction,"entry_time":r.entry_time,"entry":r.entry,
                    "initial_stop":r.stop,"target":r.target,"risk_per_share":r.risk_per_share,
                    "historical_exit_reason":r.exit_reason,"historical_net_R":r.net_R,
                    **result
                })
            audit.append({"ticker":ticker,"date":r.date,"status":"matched"})
    out = pd.DataFrame(all_rows)
    if out.empty:
        raise RuntimeError("No se reconstruyeron operaciones. Comparte la auditoría.")
    out.to_csv("orb15_v3_3_operaciones_gestion.csv", index=False)
    pd.DataFrame(audit).to_csv("orb15_v3_3_auditoria.csv", index=False)

    def summarize(g):
        net = g.net_R_sim.astype(float)
        return pd.Series({
            "n":len(g),"mean_net_R":net.mean(),"median_net_R":net.median(),
            "profit_factor":pf(net),"win_rate_pct":100*(net>0).mean(),
            "total_net_R":net.sum(),"mean_MFE_R":g.MFE_R_sim.mean(),
            "mean_MAE_R":g.MAE_R_sim.mean(),
            "stop_pct":100*g.exit_reason_sim.isin(["stop","stop_same_bar"]).mean(),
            "BE_pct":100*(g.exit_reason_sim=="break_even").mean(),
            "trailing_pct":100*(g.exit_reason_sim=="trailing_stop").mean(),
            "target_pct":100*(g.exit_reason_sim=="target").mean(),
            "session_close_pct":100*(g.exit_reason_sim=="session_close").mean(),
            "ambiguous_trigger_bars":int(g.ambiguous_trigger_bar.sum())
        })
    for keys, fn in [
        (["policy"],"orb15_v3_3_resumen.csv"),
        (["policy","year"],"orb15_v3_3_por_anio.csv"),
        (["policy","ticker"],"orb15_v3_3_por_simbolo.csv"),
        (["policy","direction"],"orb15_v3_3_por_direccion.csv"),
        (["policy","month"],"orb15_v3_3_por_mes.csv"),
    ]:
        out.groupby(keys, dropna=False).apply(summarize).reset_index().to_csv(fn,index=False)

    print("\n=== RESULTADOS DE GESTIÓN (4 bps) ===")
    print(out.groupby("policy").apply(summarize).to_string())
    print("\n=== DESARROLLO 2024-2025 vs OOS 2026 ===")
    for period, mask in [("DEV_2024_2025",out.year.isin([2024,2025])),("OOS_2026",out.year.eq(2026))]:
        print(f"\n{period}")
        print(out[mask].groupby("policy").apply(summarize)[["n","mean_net_R","profit_factor","total_net_R","win_rate_pct"]].to_string())
    print("\nArchivos generados:")
    for f in ["orb15_v3_3_operaciones_gestion.csv","orb15_v3_3_resumen.csv",
              "orb15_v3_3_por_anio.csv","orb15_v3_3_por_simbolo.csv",
              "orb15_v3_3_por_direccion.csv","orb15_v3_3_por_mes.csv","orb15_v3_3_auditoria.csv"]:
        print(" -",f)
    print("\nIMPORTANTE: resultados experimentales; no seleccionar reglas usando OOS 2026.")
    print("La gestion nueva se activa al final de la vela que alcanza el umbral y aplica desde la siguiente vela.")

if __name__ == "__main__":
    main()
