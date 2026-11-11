import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed


SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT",
    "NVDA", "TSLA", "SOXL", "NUGT"
]

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 10, 2, tzinfo=timezone.utc)

OUTPUT_DIR = Path("datos_alpaca_sip")
OUTPUT_DIR.mkdir(exist_ok=True)

api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")

if not api_key or not secret_key:
    raise RuntimeError(
        "Configura APCA_API_KEY_ID y APCA_API_SECRET_KEY "
        "en PowerShell antes de ejecutar el programa."
    )

client = StockHistoricalDataClient(api_key, secret_key)

for symbol in SYMBOLS:
    print(f"\nDescargando {symbol}...")

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=START,
        end=END,
        feed=DataFeed.SIP,
        adjustment=Adjustment.SPLIT,
    )

    try:
        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            print(f"AVISO: Alpaca no devolvió datos para {symbol}.")
            continue

        # La respuesta suele incluir un índice de símbolo y timestamp.
        df = df.reset_index()

        if "timestamp" not in df.columns:
            raise ValueError(
                f"No se encontró la columna timestamp para {symbol}."
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True
        )

        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"])

        output_file = OUTPUT_DIR / f"{symbol}_5min.csv"
        df.to_csv(output_file, index=False)

        print(f"Filas: {len(df):,}")
        print(f"Primera vela UTC: {df['timestamp'].min()}")
        print(f"Última vela UTC:  {df['timestamp'].max()}")
        print(f"Archivo: {output_file}")

    except Exception as error:
        print(f"ERROR en {symbol}: {error}")

print("\nDescarga finalizada. Revisa los archivos y los errores.")