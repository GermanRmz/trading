"""
Análisis avanzado del CSV producido por backtest_teoria_caja_con_SOXL_NUGT.py.

Incluye:
- Resumen por símbolo y grupo (SOXL/NUGT frente al resto).
- Sensibilidad a costes de 0, 1, 2 y 4 bps ida y vuelta.
- Bootstrap por día de mercado para considerar dependencia entre operaciones del mismo día.
- Drawdown ordenado por fecha/hora de entrada (medida ilustrativa en R).

Ejecutar en la misma carpeta que caja_trades.csv:
    py -m pip install numpy pandas
    py analizar_caja_avanzado.py
"""
import numpy as np
import pandas as pd

INPUT = "caja_trades.csv"
BASE_COST_BPS = 4.0
COST_SCENARIOS_BPS = [0, 1, 2, 4]
BOOTSTRAPS = 10000
SEED = 20261028
LEVERAGED = {"SOXL", "NUGT"}

df = pd.read_csv(INPUT)
required = {"ticker", "date", "gross_R", "estimated_cost_R"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Faltan columnas en {INPUT}: {sorted(missing)}")

for col in ["gross_R", "estimated_cost_R"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["ticker", "date", "gross_R", "estimated_cost_R"]).copy()

if "entry_time" in df.columns:
    df = df.sort_values(["date", "entry_time", "ticker"])
else:
    df = df.sort_values(["date", "ticker"])

def metrics(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, win_rate=np.nan, mean_R=np.nan, median_R=np.nan,
                    profit_factor=np.nan, max_drawdown_R=np.nan)
    positive = x[x > 0].sum()
    negative = abs(x[x < 0].sum())
    pf = positive / negative if negative else np.inf
    equity = np.cumsum(x)
    peak = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    dd = equity - peak
    return dict(
        n=len(x),
        win_rate=float((x > 0).mean()),
        mean_R=float(x.mean()),
        median_R=float(np.median(x)),
        profit_factor=float(pf),
        max_drawdown_R=float(dd.min())
    )

# Cost sensitivity using the original per-trade estimated cost scaled from 4 bps.
print("\n=== SENSIBILIDAD A COSTES: UNIVERSO COMPLETO ===")
sens_rows = []
for bps in COST_SCENARIOS_BPS:
    net = df["gross_R"] - df["estimated_cost_R"] * (bps / BASE_COST_BPS)
    m = metrics(net)
    m["cost_bps_round_trip"] = bps
    sens_rows.append(m)
sens = pd.DataFrame(sens_rows)[[
    "cost_bps_round_trip", "n", "win_rate", "mean_R",
    "profit_factor", "max_drawdown_R"
]]
print(sens.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Per-symbol and group results at the original 4 bps cost assumption.
df["net_R_4bps"] = df["gross_R"] - df["estimated_cost_R"]
print("\n=== RESULTADOS POR SÍMBOLO (4 bps) ===")
ticker_rows = []
for ticker, g in df.groupby("ticker"):
    m = metrics(g["net_R_4bps"])
    m["ticker"] = ticker
    ticker_rows.append(m)
ticker_table = pd.DataFrame(ticker_rows)[[
    "ticker", "n", "win_rate", "mean_R", "median_R",
    "profit_factor", "max_drawdown_R"
]]
print(ticker_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n=== COMPARACIÓN DE GRUPOS (4 bps) ===")
df["group"] = np.where(df["ticker"].isin(LEVERAGED), "SOXL_NUGT", "RESTO")
group_rows = []
for group, g in df.groupby("group"):
    m = metrics(g["net_R_4bps"])
    m["group"] = group
    group_rows.append(m)
group_table = pd.DataFrame(group_rows)[[
    "group", "n", "win_rate", "mean_R", "profit_factor", "max_drawdown_R"
]]
print(group_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Cluster bootstrap by date: resample entire market dates, keeping all trades on each sampled date.
print("\n=== BOOTSTRAP POR DÍA DE MERCADO (4 bps) ===")
rng = np.random.default_rng(SEED)
def cluster_bootstrap(group_df, label):
    dates = group_df["date"].drop_duplicates().tolist()
    if len(dates) < 2:
        print(f"{label}: datos insuficientes para bootstrap por fecha.")
        return None
    per_day = {
        d: group_df.loc[group_df["date"] == d, "net_R_4bps"].to_numpy(dtype=float)
        for d in dates
    }
    boot_means = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        sampled_values = np.concatenate([per_day[d] for d in sampled_dates])
        boot_means[i] = sampled_values.mean()
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    result = {
        "group": label,
        "trading_days": len(dates),
        "trades": len(group_df),
        "observed_mean_R": group_df["net_R_4bps"].mean(),
        "ci95_low_R": lo,
        "ci95_high_R": hi,
    }
    print(
        f"{label}: {len(group_df)} operaciones en {len(dates)} fechas | "
        f"media={result['observed_mean_R']:.3f}R | "
        f"IC bootstrap 95%=[{lo:.3f}, {hi:.3f}]R"
    )
    return result

boot_rows = []
for label, g in [
    ("Universo completo", df),
    ("SOXL + NUGT", df[df["ticker"].isin(LEVERAGED)]),
    ("Resto", df[~df["ticker"].isin(LEVERAGED)]),
    ("SOXL", df[df["ticker"] == "SOXL"]),
    ("NUGT", df[df["ticker"] == "NUGT"]),
]:
    result = cluster_bootstrap(g, label)
    if result:
        boot_rows.append(result)

sens.to_csv("caja_sensibilidad_avanzada.csv", index=False)
ticker_table.to_csv("caja_resultados_por_simbolo.csv", index=False)
group_table.to_csv("caja_resultados_por_grupo.csv", index=False)
if boot_rows:
    pd.DataFrame(boot_rows).to_csv("caja_bootstrap_por_dia.csv", index=False)

print("\nArchivos guardados:")
print("- caja_sensibilidad_avanzada.csv")
print("- caja_resultados_por_simbolo.csv")
print("- caja_resultados_por_grupo.csv")
print("- caja_bootstrap_por_dia.csv (si había suficientes fechas)")
print("\nNota: el bootstrap por fecha mejora el tratamiento de correlación diaria, pero no corrige")
print("sesgos de datos, ejecución, selección de símbolos ni el limitado histórico intradía.")
