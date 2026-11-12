"""
PREPARAR EL DATASET DE SOXL: descarga, auditoria, limpieza y congelado (con las herramientas de lab/).

Claves:  variables de entorno APCA_API_KEY_ID y APCA_API_SECRET_KEY (igual que para lab/).
Uso (desde la raiz del repositorio):
    python adv_trading/preparar_datos_soxl.py            # hace los cuatro pasos; la descarga es reanudable
    python adv_trading/preparar_datos_soxl.py --solo-congelar

Que baja: velas SIP de 5 minutos, ajustadas por splits, desde 2016-01-01 (inicio del historico de Alpaca).
  objetivo      SOXL
  apalancados   TQQQ, SQQQ, SPXL, UPRO          (misma mecanica de ETF apalancado)
  semis         SOXX, SMH, NVDA, AMD, AVGO, TSM, MU, QCOM, INTC, TXN, AMAT, LRCX, KLAC
  mercado       SPY, QQQ, XLK, AAPL, MSFT

Por que tantos simbolos si solo se opera SOXL: una red entrenada solo con SOXL tendria unas 2.500
sesiones; con el universo aprende de unas 55.000 y despues se especializa. Los años 2016-2019 quedan
fuera de DEV/VAL/SELLADO: sirven para entrenar y para explorar arquitecturas sin gastar hipotesis.

Resultado: adv_trading/datos_soxl_sip_limpio/ (dataset congelado con su manifiesto y mascara de dias).
La carpeta de datos NO se sube al repositorio (pesa cientos de MB); cada quien la genera con sus claves.
"""
import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(RAIZ / "lab"))
import lab_descarga_auditoria_v3 as base  # noqa: E402
import lab_dataset_congelar as cong  # noqa: E402

UNIVERSO = {
    "objetivo": ["SOXL"],
    "apalancados": ["TQQQ", "SQQQ", "SPXL", "UPRO"],
    "semis": ["SOXX", "SMH", "NVDA", "AMD", "AVGO", "TSM", "MU", "QCOM", "INTC", "TXN", "AMAT", "LRCX", "KLAC"],
    "mercado": ["SPY", "QQQ", "XLK", "AAPL", "MSFT"],
}
SIMBOLOS = [s for g in UNIVERSO.values() for s in g]
OBJETIVO = "SOXL"
CONTEXTO = {"mercado": "SPY", "tec": "QQQ", "semis": "SOXX"}
INICIO = "2016-01-01"
CRUDO, LIMPIO = AQUI / "datos_soxl_sip", AQUI / "datos_soxl_sip_limpio"
EXCLUSIONES = AQUI / "exclusiones_manuales_soxl.csv"

# Las herramientas de lab/ comparan marcas de tiempo como enteros en nanosegundos, pero segun la version
# de pandas las fechas leidas vienen en segundos, micro o nanosegundos. Este envoltorio iguala las
# unidades antes de comparar, sin modificar los archivos de lab/.
_isin_original = base._isin_sorted


def _isin_misma_unidad(a, sorted_b):
    a = np.asarray(a, dtype=np.int64)
    if len(a) and len(sorted_b):
        r = float(np.median(sorted_b)) / max(float(np.median(a)), 1.0)
        for f in (10**3, 10**6, 10**9):
            if 0.5 * f < r < 2 * f:
                a = a * f
    return _isin_original(a, sorted_b)


base._isin_sorted = _isin_misma_unidad
base.SECTOR_OF.update({s: g for g, lista in UNIVERSO.items() for s in lista})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fin", default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--solo-congelar", action="store_true", help="omite descarga, auditoria y limpieza")
    p.add_argument("--forzar", action="store_true", help="recongelar (cambia el dataset_id)")
    a = p.parse_args()
    if not a.solo_congelar:
        print(f"1/4 Descarga de {len(SIMBOLOS)} simbolos, {INICIO} -> {a.fin}")
        base.descargar(SIMBOLOS, INICIO, a.fin, CRUDO)
        faltan = [s for s in SIMBOLOS if not (CRUDO / f"{s}_5min.csv").exists()]
        if faltan:
            raise SystemExit(f"Faltan simbolos tras la descarga: {faltan}. Vuelve a ejecutar (continua donde quedo).")
        print("\n2/4 Auditoria contra el calendario NYSE")
        cwd = os.getcwd()
        os.chdir(AQUI)                                   # los CSV de auditoria quedan en adv_trading/
        try:
            base.auditar(CRUDO)
        finally:
            os.chdir(cwd)
        print("\n3/4 Limpieza por calendario (no modifica la descarga)")
        for f in LIMPIO.glob("*"):
            os.chmod(f, 0o644)
        base.corregir_dataset(CRUDO, LIMPIO)
    print("\n4/4 Congelado")
    cong.congelar(LIMPIO, a.forzar or not a.solo_congelar)
    if not EXCLUSIONES.exists():
        EXCLUSIONES.write_text(
            "symbol,date,decision,motivo,fuente,verificacion\n"
            'TSM,2023-01-24,excluir,"Fallo de apertura de NYSE del 2023-01-24 (TSM cotiza en NYSE); por analogia '
            'con las exclusiones del laboratorio",lab/exclusiones_manuales.csv,por_analogia_revisar\n',
            encoding="utf-8", newline="\n")
        print(f"Creado {EXCLUSIONES.name}: revisa integridad_revision.csv y anade ahi lo que debas excluir.")
    print("\nListo. Siguiente paso:  python adv_trading/red_politica.py prueba")


if __name__ == "__main__":
    main()
