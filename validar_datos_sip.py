
from pathlib import Path
import pandas as pd

DATA_DIR = Path("datos_alpaca_sip")
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "SOXL", "NUGT"]

print("VALIDACION DE DATOS SIP")
print("=" * 75)

for symbol in SYMBOLS:
    file = DATA_DIR / f"{symbol}_5min.csv"

    if not file.exists():
        print(f"\n{symbol}: NO SE ENCONTRO EL ARCHIVO")
        continue

    df = pd.read_csv(file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    # Validaciones generales
    duplicates = int(df["timestamp"].duplicated().sum())
    missing = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())

    incoherent = int((
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1)) |
        (df["high"] < df["low"])
    ).sum())

    # Convertir a hora local de Nueva York
    local = df["timestamp"].dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute

    # Velas de la sesión regular: 09:30 a 15:55
    regular_mask = (
        (local.dt.day_of_week < 5) &
        (minutes >= 570) &
        (minutes < 960)
    )

    regular = df.loc[regular_mask].copy()
    regular["session"] = local.loc[regular_mask].dt.strftime("%Y-%m-%d").values

    counts = regular.groupby("session").size()

    # Una sesión completa normalmente tiene 78 velas de 5 minutos
    complete = int((counts == 78).sum())
    short = int((counts < 78).sum())
    over = int((counts > 78).sum())

    print(f"\n{symbol}")
    print(f"  Filas totales:              {len(df):,}")
    print(f"  Primera fecha UTC:          {df['timestamp'].min()}")
    print(f"  Ultima fecha UTC:           {df['timestamp'].max()}")
    print(f"  Timestamps duplicados:      {duplicates}")
    print(f"  Filas con precios vacios:   {missing}")
    print(f"  Velas OHLC incoherentes:    {incoherent}")
    print(f"  Velas en horario regular:   {len(regular):,}")
    print(f"  Sesiones con 78 velas:      {complete}")
    print(f"  Sesiones con menos de 78:   {short}")
    print(f"  Sesiones con mas de 78:     {over}")

    if not counts.empty:
        print(f"  Velas por sesion (min/mediana/max): "
              f"{counts.min()}/{counts.median():.0f}/{counts.max()}")

print("\nValidacion terminada.")
