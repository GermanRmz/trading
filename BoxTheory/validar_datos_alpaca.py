
from pathlib import Path
import pandas as pd

DATA_DIR = Path("datos_alpaca")
FILES = sorted(DATA_DIR.glob("*_5min.csv"))

REQUIRED = {
    "timestamp", "open", "high", "low", "close", "volume"
}

print("=" * 75)
print("VALIDACIÓN DE DATOS HISTÓRICOS DE ALPACA")
print("=" * 75)

if len(FILES) != 8:
    print(f"AVISO: se encontraron {len(FILES)} archivos; se esperaban 8.")

total_rows = 0

for file in FILES:
    symbol = file.name.replace("_5min.csv", "")
    print(f"\n--- {symbol} ---")

    try:
        df = pd.read_csv(file)
        missing_cols = REQUIRED - set(df.columns)

        if missing_cols:
            print("ERROR: faltan columnas:", sorted(missing_cols))
            print("Columnas disponibles:", list(df.columns))
            continue

        total_rows += len(df)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce"
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        invalid_time = df["timestamp"].isna().sum()
        duplicate_time = df["timestamp"].duplicated().sum()
        invalid_price = df[
            ["open", "high", "low", "close"]
        ].isna().any(axis=1).sum()

        bad_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        ).sum()

        print(f"Filas: {len(df):,}")
        print(f"Primera vela UTC: {df['timestamp'].min()}")
        print(f"Última vela UTC:  {df['timestamp'].max()}")
        print(f"Timestamps inválidos: {invalid_time}")
        print(f"Timestamps duplicados: {duplicate_time}")
        print(f"Filas con precios inválidos: {invalid_price}")
        print(f"Filas con OHLC incoherente: {bad_ohlc}")

        # Convertir a la zona horaria de la bolsa.
        local = df["timestamp"].dt.tz_convert(
            "America/New_York"
        )

        # Sesión regular: 9:30 inclusive a 16:00 exclusive.
        minutes = local.dt.hour * 60 + local.dt.minute
        regular = (
            (local.dt.dayofweek < 5)
            & (minutes >= 9 * 60 + 30)
            & (minutes < 16 * 60)
        )

        session = df.loc[regular].copy()
        session["date_ny"] = local.loc[regular].dt.date

        daily = session.groupby("date_ny").size()

        print(f"Velas dentro del horario regular: {len(session):,}")
        print(f"Sesiones con al menos una vela: {len(daily):,}")

        if not daily.empty:
            print(f"Velas por sesión (mín./mediana/máx.): "
                  f"{daily.min()} / {daily.median():.0f} / {daily.max()}")
            print(f"Sesiones con menos de 60 velas: "
                  f"{(daily < 60).sum()}")

        # Reportar, no eliminar, registros sospechosos.
        if invalid_time or duplicate_time or invalid_price or bad_ohlc:
            print("RESULTADO: REVISAR registros sospechosos.")
        else:
            print("RESULTADO: controles básicos superados.")

    except Exception as exc:
        print(f"ERROR al leer o validar: {type(exc).__name__}: {exc}")

print("\n" + "=" * 75)
print(f"TOTAL DE FILAS LEÍDAS: {total_rows:,}")
print("Nota: IEX puede tener menos velas que el feed consolidado SIP.")
print("Los días con pocas velas requieren revisión; no son prueba")
print("automática de datos corruptos.")
