import os
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from datetime import datetime, timezone

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

key = os.getenv("APCA_API_KEY_ID")
secret = os.getenv("APCA_API_SECRET_KEY")

if not key or not secret:
    raise SystemExit("Faltan las variables de entorno.")

client = StockHistoricalDataClient(key, secret)

request = StockBarsRequest(
    symbol_or_symbols="SPY",
    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
    start=datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc),
    end=datetime(2024, 1, 3, 21, 0, tzinfo=timezone.utc),
    feed=DataFeed.SIP,
)

try:
    result = client.get_stock_bars(request)
    bars = result.data.get("SPY", [])
    print("Acceso SIP correcto.")
    print("Velas recibidas:", len(bars))
    if bars:
        print("Primera vela:", bars[0].timestamp)
except Exception as exc:
    print("No se pudo consultar SIP.")
    print(type(exc).__name__, str(exc)[:500])
