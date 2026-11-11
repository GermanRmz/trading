
import pandas as pd

df = pd.read_csv("caja_trades_sip.csv")

for bps in [0, 1, 2, 4]:
    # El coste guardado corresponde al supuesto original de 4 bps.
    if bps == 0:
        net = df["gross_R"]
    else:
        net = df["gross_R"] - df["estimated_cost_R"] * (bps / 4.0)

    gains = net[net > 0].sum()
    losses = abs(net[net < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    print(f"\n=== Coste: {bps} bps ida y vuelta ===")
    print(f"Operaciones: {len(net)}")
    print(f"Acierto: {(net > 0).mean():.1%}")
    print(f"R medio: {net.mean():+.3f}R")
    print(f"Profit factor: {pf:.3f}")

print("\n=== Expectativa por símbolo con 0 bps ===")
for ticker, group in df.groupby("ticker"):
    print(
        f"{ticker:5s} | n={len(group):4d} | "
        f"R bruto medio={group['gross_R'].mean():+.3f}R"
    )
