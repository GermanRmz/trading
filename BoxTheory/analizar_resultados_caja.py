"""
Analiza los resultados de caja_trades.csv:
- Sensibilidad a costes de 0, 1, 2 y 4 puntos básicos ida y vuelta.
- Resumen por símbolo.
- Intervalo de confianza bootstrap para la expectativa neta (descriptivo; operaciones no necesariamente independientes).
- No descarga datos ni vuelve a ejecutar el backtest.
"""
import numpy as np
import pandas as pd

INPUT = "caja_trades.csv"
COST_SCENARIOS_BPS = [0, 1, 2, 4]
BASE_COST_BPS = 4.0
BOOTSTRAPS = 10000
SEED = 42

df = pd.read_csv(INPUT)
required = {"ticker", "gross_R", "estimated_cost_R"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Faltan columnas en {INPUT}: {sorted(missing)}")

df["gross_R"] = pd.to_numeric(df["gross_R"], errors="coerce")
df["estimated_cost_R"] = pd.to_numeric(df["estimated_cost_R"], errors="coerce")
df = df.dropna(subset=["ticker", "gross_R", "estimated_cost_R"]).copy()

def summarize(x):
    x = pd.Series(x).dropna().astype(float)
    wins = (x > 0).sum()
    losses = (x < 0).sum()
    gross_wins = x[x > 0].sum()
    gross_losses = abs(x[x < 0].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else np.inf
    equity = x.cumsum()
    dd = (equity - equity.cummax()).min() if len(x) else np.nan
    return {
        "n": len(x),
        "win_rate": (x > 0).mean() if len(x) else np.nan,
        "mean_R": x.mean() if len(x) else np.nan,
        "median_R": x.median() if len(x) else np.nan,
        "profit_factor": pf,
        "max_drawdown_R_ordered": dd,
        "wins": int(wins),
        "losses": int(losses),
    }

print("\n=== SENSIBILIDAD A COSTES ===")
rows = []
for bps in COST_SCENARIOS_BPS:
    # estimated_cost_R in the original file is the cost at BASE_COST_BPS.
    net = df["gross_R"] - df["estimated_cost_R"] * (bps / BASE_COST_BPS)
    s = summarize(net)
    s["cost_bps_round_trip"] = bps
    rows.append(s)
sens = pd.DataFrame(rows)[[
    "cost_bps_round_trip", "n", "win_rate", "mean_R", "median_R",
    "profit_factor", "max_drawdown_R_ordered", "wins", "losses"
]]
print(sens.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n=== RESULTADOS POR SÍMBOLO (coste base de 4 bps) ===")
by_ticker = []
for ticker, group in df.groupby("ticker"):
    net = group["gross_R"] - group["estimated_cost_R"]
    s = summarize(net)
    s["ticker"] = ticker
    by_ticker.append(s)
by = pd.DataFrame(by_ticker)[[
    "ticker", "n", "win_rate", "mean_R", "median_R",
    "profit_factor", "max_drawdown_R_ordered"
]]
print(by.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n=== BOOTSTRAP DE EXPECTATIVA NETA (4 bps) ===")
net = (df["gross_R"] - df["estimated_cost_R"]).to_numpy()
rng = np.random.default_rng(SEED)
means = np.empty(BOOTSTRAPS)
for i in range(BOOTSTRAPS):
    sample = rng.choice(net, size=len(net), replace=True)
    means[i] = sample.mean()
lo, hi = np.quantile(means, [0.025, 0.975])
print(f"Expectativa observada: {net.mean():.3f} R por operación")
print(f"IC bootstrap 95%: [{lo:.3f} R, {hi:.3f} R]")
print("Advertencia: este intervalo trata las operaciones como si fueran independientes.")
print("Como varias acciones se operan en fechas comunes, puede subestimar la incertidumbre.")
print("Para una inferencia mejor, conviene hacer bootstrap por día de mercado y probar un periodo fuera de muestra.")

sens.to_csv("sensibilidad_costes.csv", index=False)
by.to_csv("resultados_por_simbolo.csv", index=False)
print("\nArchivos guardados: sensibilidad_costes.csv y resultados_por_simbolo.csv")
