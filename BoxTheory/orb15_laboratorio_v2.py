"""
Laboratorio ORB15 v2: sensibilidad de reglas, filtros y costes.
Usa CSV locales de Alpaca SIP; no descarga datos ni necesita API keys.

Método:
- Universo: SPY, QQQ, AAPL, MSFT, NVDA, TSLA, SOXL, NUGT.
- Barras de 5 min, sesión regular NY 09:30-15:55; sesiones incompletas excluidas.
- Opening ranges: 15, 30 y 45 minutos (3, 6, 9 velas).
- Señal: cierre de vela fuera del rango; entrada en apertura de la siguiente vela.
- Stop: borde opuesto del opening range.
- Objetivos: 1R, 1.5R y 2R.
- Filtros: ninguno, VWAP alineado, RVOL >= 1.2x mediana del mismo slot en hasta 20 sesiones previas,
  o ambos.
- Ventana de entrada: señal antes o a las 11:30 ET, además de baseline sin corte horario.
- Una operación por símbolo/variante/día; salida stop, target o cierre de sesión.
- Si stop y target se tocan en la misma vela, stop primero.
- Costes comparados: 4, 6 y 8 bps ida y vuelta sobre precio de entrada.
- Selección de variantes: solo con 2024-2025. 2026 se reporta como chequeo temporal posterior,
  no debe usarse para seleccionar parámetros.
"""
from pathlib import Path
import itertools
import math
import sys
import numpy as np
import pandas as pd

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "SOXL", "NUGT"]
DATA_DIR = Path("datos_alpaca_sip")
NY_TZ = "America/New_York"
EXPECTED_BARS = 78
COSTS_BPS = [4.0, 6.0, 8.0]
ORB_BARS_SET = [3, 6, 9]       # 15, 30, 45 minutos
TARGET_R_SET = [1.0, 1.5, 2.0]
FILTER_SET = ["none", "vwap", "rvol", "vwap_rvol"]
CUTOFF_SET = ["all_day", "1130"]  # hora de la vela de señal
RVOL_LOOKBACK = 20
RVOL_MULTIPLIER = 1.2
TRADES_OUT = "orb15_v2_trades.csv"
METRICS_OUT = "orb15_v2_metricas.csv"
MONTHLY_OUT = "orb15_v2_mensual.csv"
AUDIT_OUT = "orb15_v2_auditoria.csv"
VARIANTS_OUT = "orb15_v2_variantes_desarrollo.csv"

def load_data(ticker):
    path = DATA_DIR / f"{ticker}_5min.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    raw = pd.read_csv(path)
    candidates = ["timestamp", "Timestamp", "time", "Time", "datetime", "Datetime", "date", "Date"]
    tc = next((c for c in candidates if c in raw.columns), raw.columns[0])
    ts = pd.to_datetime(raw[tc], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Timestamp inválido en {path}")
    raw = raw.drop(columns=[tc])
    raw.index = pd.DatetimeIndex(ts).tz_convert(NY_TZ)
    raw = raw.sort_index()
    raw = raw.loc[~raw.index.duplicated(keep="first")].copy()
    raw.columns = [str(c).strip().capitalize() for c in raw.columns]
    for c in ["Open", "High", "Low", "Close"]:
        if c not in raw:
            raise ValueError(f"Falta columna {c} en {path}")
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in raw:
        raw["Volume"] = np.nan
    raw["Volume"] = pd.to_numeric(raw["Volume"], errors="coerce")
    raw = raw.between_time("09:30", "15:55").copy()
    raw["session"] = raw.index.date
    return raw

def build_sessions(df):
    out = []
    for date, day in df.groupby("session", sort=True):
        day = day.sort_index().copy()
        expected = pd.date_range(pd.Timestamp(str(date), tz=NY_TZ) + pd.Timedelta(hours=9, minutes=30),
                                 periods=EXPECTED_BARS, freq="5min")
        complete = len(day) == EXPECTED_BARS and day.index.equals(expected)
        typical = (day["High"] + day["Low"] + day["Close"]) / 3.0
        vol = day["Volume"].clip(lower=0)
        if vol.notna().any() and vol.fillna(0).sum() > 0:
            day["VWAP"] = (typical * vol.fillna(0)).cumsum() / vol.fillna(0).cumsum().replace(0, np.nan)
        else:
            day["VWAP"] = np.nan
        out.append({"date": date, "data": day, "complete": bool(complete), "bars": len(day)})
    return out

def prior_slot_median(sessions, idx, slot):
    prior = [s for s in sessions[:idx] if s["complete"]][-RVOL_LOOKBACK:]
    vals = []
    for s in prior:
        v = s["data"].iloc[slot]["Volume"]
        if pd.notna(v) and float(v) > 0:
            vals.append(float(v))
    return float(np.median(vals)) if len(vals) >= 10 else np.nan

def make_variants():
    return [
        {"variant": f"ORB{n*5}_T{target:g}_{flt}_{cut}",
         "orb_bars": n, "target_r": target, "filter": flt, "cutoff": cut}
        for n, target, flt, cut in itertools.product(ORB_BARS_SET, TARGET_R_SET, FILTER_SET, CUTOFF_SET)
    ]

def find_orb_signal(sessions, idx, v):
    s = sessions[idx]
    if not s["complete"]:
        return None
    day = s["data"]
    n = v["orb_bars"]
    rh = float(day.iloc[:n]["High"].max())
    rl = float(day.iloc[:n]["Low"].min())
    if not np.isfinite(rh + rl) or rh <= rl:
        return None
    # slot de entrada empieza después de completarse el rango
    for pos in range(n, len(day) - 1):
        bar = day.iloc[pos]
        if v["cutoff"] == "1130" and day.index[pos].time() > pd.Timestamp("11:30").time():
            break
        close = float(bar["Close"])
        direction = "long" if close > rh else ("short" if close < rl else None)
        if direction is None:
            continue
        if v["filter"] in ("vwap", "vwap_rvol"):
            vw = bar["VWAP"]
            if pd.isna(vw) or (direction == "long" and close <= float(vw)) or (direction == "short" and close >= float(vw)):
                continue
        if v["filter"] in ("rvol", "vwap_rvol"):
            ref = prior_slot_median(sessions, idx, pos)
            vol = bar["Volume"]
            if pd.isna(ref) or ref <= 0 or pd.isna(vol) or float(vol) < RVOL_MULTIPLIER * ref:
                continue
        return pos, direction, (rl if direction == "long" else rh), rh, rl
    return None

def simulate_trade(ticker, s, day, v, sig, cost_bps):
    pos, direction, stop, rh, rl = sig
    ep = pos + 1
    entry = float(day.iloc[ep]["Open"])
    if (direction == "long" and stop >= entry) or (direction == "short" and stop <= entry):
        return None
    risk = abs(entry - stop)
    if not np.isfinite(risk) or risk <= 0:
        return None
    target_r = float(v["target_r"])
    target = entry + target_r*risk if direction == "long" else entry-target_r*risk
    exit_price = float(day.iloc[-1]["Close"])
    exit_time = day.index[-1]
    reason = "session_close"
    for j in range(ep, len(day)):
        b = day.iloc[j]
        hi, lo = float(b["High"]), float(b["Low"])
        hs = lo <= stop if direction == "long" else hi >= stop
        ht = hi >= target if direction == "long" else lo <= target
        if hs and ht:
            exit_price, exit_time, reason = stop, day.index[j], "stop_same_bar"
            break
        if hs:
            exit_price, exit_time, reason = stop, day.index[j], "stop"
            break
        if ht:
            exit_price, exit_time, reason = target, day.index[j], "target"
            break
    pnl = exit_price-entry if direction == "long" else entry-exit_price
    gross_r = pnl/risk
    cost_r = (entry*cost_bps/10000)/risk
    return {
        "ticker":ticker, "variant":v["variant"], "orb_minutes":v["orb_bars"]*5,
        "target_r":target_r, "filter":v["filter"], "cutoff":v["cutoff"],
        "date":str(s["date"]), "year":pd.Timestamp(str(s["date"])).year,
        "month":str(s["date"])[:7], "direction":direction,
        "signal_time":str(day.index[pos]), "entry_time":str(day.index[ep]),
        "entry":entry, "opening_range_high":rh, "opening_range_low":rl,
        "stop":stop, "target":target, "exit_time":str(exit_time), "exit":exit_price,
        "exit_reason":reason, "risk_per_share":risk, "gross_R":gross_r,
        "cost_bps":cost_bps, "estimated_cost_R":cost_r, "net_R":gross_r-cost_r
    }

def profit_factor(x):
    x = pd.Series(x, dtype=float)
    wins = x[x>0].sum()
    losses = abs(x[x<0].sum())
    return float(wins/losses) if losses else (float("inf") if wins>0 else np.nan)

def metrics(g, group_cols):
    rows=[]
    for keys, z in g.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple): keys=(keys,)
        net=z["net_R"].astype(float)
        eq=net.cumsum()
        dd=(eq-np.maximum.accumulate(np.r_[0.0, eq.values])[1:]).min() if len(eq) else 0.0
        row=dict(zip(group_cols,keys))
        row.update({"trades":len(z), "win_rate":float((net>0).mean()),
                    "mean_gross_R":float(z["gross_R"].mean()), "mean_net_R":float(net.mean()),
                    "median_net_R":float(net.median()), "profit_factor":profit_factor(net),
                    "total_net_R":float(net.sum()), "max_drawdown_R":float(dd)})
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    variants=make_variants()
    all_trades=[]
    audit=[]
    for ticker in TICKERS:
        try:
            df=load_data(ticker)
            sessions=build_sessions(df)
            complete=sum(x["complete"] for x in sessions)
            print(f"{ticker}: {complete}/{len(sessions)} sesiones completas")
            for idx,s in enumerate(sessions):
                audit.append({"ticker":ticker,"date":str(s["date"]),"bars":s["bars"],"complete":s["complete"]})
                if not s["complete"]:
                    continue
                for v in variants:
                    sig=find_orb_signal(sessions,idx,v)
                    if sig is None:
                        continue
                    for cost in COSTS_BPS:
                        tr=simulate_trade(ticker,s,s["data"],v,sig,cost)
                        if tr: all_trades.append(tr)
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
    if not all_trades:
        print("No hubo operaciones; revisa los CSV y sus columnas.")
        sys.exit(1)
    trades=pd.DataFrame(all_trades)
    trades.to_csv(TRADES_OUT,index=False)
    pd.DataFrame(audit).to_csv(AUDIT_OUT,index=False)

    # Comparaciones agregadas; la sensibilidad de costes se mantiene explícita.
    key=["variant","orb_minutes","target_r","filter","cutoff","cost_bps"]
    summary=metrics(trades[trades["year"].isin([2024,2025])],key)
    summary["period"]="Development_2024_2025"
    oos=metrics(trades[trades["year"].eq(2026)],key)
    oos["period"]="OOS_2026"
    allm=metrics(trades,key)
    allm["period"]="All"
    pd.concat([summary,oos,allm],ignore_index=True).to_csv(METRICS_OUT,index=False)

    # Resultados por símbolo, dirección y mes para el baseline ORB15 1.5R sin filtros.
    base=trades[(trades["orb_minutes"]==15)&(trades["target_r"]==1.5)&
                (trades["filter"]=="none")&(trades["cutoff"]=="all_day")&(trades["cost_bps"]==4)]
    detail=metrics(base,["ticker","direction","year"])
    detail.to_csv("orb15_v2_simbolo_direccion_anual.csv",index=False)
    monthly=metrics(base,["ticker","month"])
    monthly.to_csv(MONTHLY_OUT,index=False)

    # Ranking basado exclusivamente en desarrollo, con coste de 6 bps como criterio prudente.
    dev=trades[(trades["year"].isin([2024,2025]))&(trades["cost_bps"]==6)]
    rank=metrics(dev,key[:-1])
    rank=rank.sort_values(["mean_net_R","profit_factor"],ascending=[False,False])
    rank.to_csv(VARIANTS_OUT,index=False)

    print("\n=== BASELINE ORB15 / target 1.5R / sin filtros / coste 4 bps ===")
    for year,label in [(2024,"2024"),(2025,"2025"),(2026,"OOS 2026")]:
        z=base[base["year"]==year]
        if len(z):
            print(f"{label}: n={len(z)} neto medio={z['net_R'].mean():+.3f}R PF={profit_factor(z['net_R']):.3f} total={z['net_R'].sum():+.1f}R")
    print("\n=== TOP 10 VARIANTES EN DESARROLLO (2024-2025, coste 6 bps) ===")
    for _,r in rank.head(10).iterrows():
        print(f"{r['variant']:34s} n={int(r['trades']):4d} mean={r['mean_net_R']:+.3f}R PF={r['profit_factor']:.3f} total={r['total_net_R']:+.1f}R")
    print("\nArchivos generados:")
    for f in [TRADES_OUT,METRICS_OUT,MONTHLY_OUT,"orb15_v2_simbolo_direccion_anual.csv",AUDIT_OUT,VARIANTS_OUT]:
        print(" ",f)
    print("\nIMPORTANTE: el ranking usa solo 2024-2025. No elijas parámetros por el resultado OOS 2026.")
    print("Costes estimados, no fills reales; el stop se asume primero si stop y target se tocan en la misma vela.")

if __name__=="__main__":
    main()
