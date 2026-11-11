import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta, timezone

key = os.getenv("APCA_API_KEY_ID")
secret = os.getenv("APCA_API_SECRET_KEY")

if not key or not secret:
    raise SystemExit("Faltan las variables de entorno.")

try:
    client = StockHistoricalDataClient(key, secret)

    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=2)

    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )

    result = client.get_stock_bars(request)
    print("Conexión correcta.")
    print("Velas recibidas:", len(result.data.get("SPY", [])))

except Exception as e:
    print("La conexión falló.")
    print(type(e).__name__, str(e)[:500])