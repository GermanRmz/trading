"""
DESCARGA DE NOTICIAS (Alpaca News API, fuente Benzinga) con hora de publicacion.

Instalacion:   pip install requests pandas
Claves:        variables de entorno APCA_API_KEY_ID y APCA_API_SECRET_KEY (nunca en el codigo).
               En CMD:   set APCA_API_KEY_ID=tu_clave      y      set APCA_API_SECRET_KEY=tu_secreto

Uso (desde cualquier carpeta):
    python ia/descargar_noticias_alpaca.py              # descarga todo (se puede interrumpir y reanudar)
    python ia/descargar_noticias_alpaca.py resumen      # solo recalcula el resumen con lo ya descargado

Que hace:
  * Baja titulares y resumenes (no el cuerpo completo) de cada simbolo, desde 2020-01-01 hasta hoy.
  * Guarda un archivo por simbolo y anio en lab/datos_noticias_alpaca/_parts/ y, al final, un CSV
    por simbolo en lab/datos_noticias_alpaca/SIMBOLO_noticias.csv (sin duplicados, orden temporal).
  * Si se corta, al relanzarlo continua: los anios ya completos no se vuelven a pedir.
  * Escribe resumen_noticias.csv: noticias por simbolo y anio, y porcentaje de sesiones con al menos
    una noticia. Es un diagnostico de COBERTURA: no mira precios ni retornos.

Columnas: id, created_at (UTC, hora de publicacion), updated_at, headline, summary, source, author,
          url, symbols (todos los simbolos etiquetados en la noticia, separados por ';').

Nota: una noticia que menciona varios simbolos aparece en el archivo de cada uno. `created_at` es la
hora que usa el modelo para decidir si la noticia ya existia; `updated_at` puede ser posterior.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "lab" / "datos_noticias_alpaca"
PARTES = SALIDA / "_parts"
URL = "https://data.alpaca.markets/v1beta1/news"
INICIO = 2020
LIMITE, PAUSA_S, REINTENTOS = 50, 0.35, 8          # 50 es el maximo por pagina de la API
COLS = ["id", "created_at", "updated_at", "headline", "summary", "source", "author", "url", "symbols"]
EXTRA = ["SOXL"]                                    # simbolos fuera del dataset del laboratorio


def simbolos():
    man = RAIZ / "lab" / "datos_lab_sip_limpio" / "DATASET_MANIFEST.json"
    base = json.loads(man.read_text(encoding="utf-8"))["simbolos"] if man.exists() else []
    return list(dict.fromkeys(base + EXTRA))


def _headers():
    kid = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not kid or not sec:
        raise SystemExit("Faltan las variables de entorno APCA_API_KEY_ID y APCA_API_SECRET_KEY.")
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}


def _get_json(session, headers, params):
    for intento in range(REINTENTOS):
        r = session.get(URL, headers=headers, params=params, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504):
            espera = min(60, 2 ** intento)
            print(f"    HTTP {r.status_code}; reintento en {espera} s", flush=True)
            time.sleep(espera)
            continue
        if r.status_code in (400, 401, 403, 422):
            raise SystemExit(f"HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        return r.json()
    raise SystemExit("Demasiados reintentos; vuelve a ejecutar mas tarde (se reanuda donde quedo).")


def bajar_anio(session, headers, simbolo, anio, fin):
    filas, token = [], None
    while True:
        params = {"symbols": simbolo, "start": f"{anio}-01-01T00:00:00Z", "end": fin, "limit": LIMITE,
                  "sort": "asc", "include_content": "false", "exclude_contentless": "false"}
        if token:
            params["page_token"] = token
        data = _get_json(session, headers, params)
        for n in data.get("news") or []:
            filas.append({"id": n.get("id"), "created_at": n.get("created_at"), "updated_at": n.get("updated_at"),
                          "headline": n.get("headline"), "summary": n.get("summary"), "source": n.get("source"),
                          "author": n.get("author"), "url": n.get("url"),
                          "symbols": ";".join(n.get("symbols") or [])})
        token = data.get("next_page_token")
        if not token:
            return pd.DataFrame(filas, columns=COLS)
        time.sleep(PAUSA_S)


def descargar():
    try:
        import requests
    except ImportError:
        raise SystemExit("Instala requests:  pip install requests")
    headers = _headers()
    PARTES.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    ahora = datetime.now(timezone.utc)
    lista = simbolos()
    for i, sym in enumerate(lista, 1):
        total = 0
        for anio in range(INICIO, ahora.year + 1):
            parte = PARTES / f"{sym}_{anio}.csv"
            completo = anio < ahora.year                      # el anio en curso siempre se vuelve a pedir
            if parte.exists() and completo:
                total += max(sum(1 for _ in open(parte, encoding="utf-8")) - 1, 0)
                continue
            fin = f"{anio}-12-31T23:59:59Z" if completo else ahora.strftime("%Y-%m-%dT%H:%M:%SZ")
            df = bajar_anio(session, headers, sym, anio, fin)
            tmp = parte.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(parte)                                # solo queda el archivo si el anio termino entero
            total += len(df)
            print(f"  {sym} {anio}: {len(df)} noticias", flush=True)
        unir(sym)
        print(f"[{i}/{len(lista)}] {sym}: {total} noticias", flush=True)


def unir(sym):
    partes = sorted(PARTES.glob(f"{sym}_*.csv"))
    if not partes:
        return None
    df = pd.concat([pd.read_csv(p, dtype=str) for p in partes], ignore_index=True)
    df = df.drop_duplicates("id").sort_values(["created_at", "id"]).reset_index(drop=True)
    df.to_csv(SALIDA / f"{sym}_noticias.csv", index=False)
    return df


def resumen():
    """Cobertura: noticias por simbolo y anio y % de dias habiles (lunes a viernes) con alguna noticia."""
    filas = []
    for sym in simbolos():
        f = SALIDA / f"{sym}_noticias.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype=str)
        if df.empty:
            filas.append({"symbol": sym, "anio": "total", "noticias": 0})
            continue
        t = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert("America/New_York")
        dia, anio = t.dt.strftime("%Y-%m-%d"), t.dt.year
        for a in sorted(anio.unique()):
            m = anio == a
            ultimo = min(pd.Timestamp(f"{a}-12-31"), pd.Timestamp.now().normalize())
            habiles = len(pd.bdate_range(f"{a}-01-01", ultimo))
            con = pd.to_datetime(dia[m].unique())
            filas.append({"symbol": sym, "anio": int(a), "noticias": int(m.sum()),
                          "por_dia_habil": round(m.sum() / max(habiles, 1), 2),
                          "pct_dias_habiles_con_noticia": round(100 * (con.dayofweek < 5).sum() / max(habiles, 1), 1),
                          "pct_en_horario_mercado": round(100 * ((t[m].dt.hour * 60 + t[m].dt.minute).between(570, 959)
                                                                 & (t[m].dt.dayofweek < 5)).mean(), 1)})
    if not filas:
        raise SystemExit(f"No hay noticias descargadas en {SALIDA}.")
    r = pd.DataFrame(filas)
    r.to_csv(SALIDA / "resumen_noticias.csv", index=False)
    tabla = r.pivot(index="symbol", columns="anio", values="noticias").fillna(0).astype(int)
    print("\nNoticias por simbolo y anio:")
    print(tabla.to_string())
    cob = r.pivot(index="symbol", columns="anio", values="pct_dias_habiles_con_noticia")
    print("\n% de dias habiles con al menos una noticia:")
    print(cob.round(0).to_string())
    print(f"\nGuardado: {SALIDA / 'resumen_noticias.csv'}")


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "") != "resumen":
        descargar()
    resumen()
