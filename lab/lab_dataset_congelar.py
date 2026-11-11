"""
LABORATORIO COMUN - Congelar y verificar el dataset base.

Uso (desde la carpeta trading, con lab_descarga_auditoria_v3.py en la misma carpeta):
    python lab_dataset_congelar.py congelar  --entrada datos_lab_sip_limpio
    python lab_dataset_congelar.py verificar --entrada datos_lab_sip_limpio

QUE HACE `congelar` (no modifica ningun dato; solo anade archivos de control):
  1. Comprueba que el dataset este limpio: sin velas fuera del calendario NYSE ni duplicados.
  2. Calcula SHA-256, filas y rango de fechas de cada archivo.
  3. Genera dias_excluidos.csv: pares (simbolo, dia) con cobertura < 95% de las velas esperadas.
     El marco de pruebas debe NO operar esos pares: con 1 o 15 velas el VWAP y los
     indicadores intradia no tienen sentido.
  4. Genera integridad_revision.csv: gaps de apertura > 20% y velas con rango > 10% del precio,
     para revisarlos A MANO contra eventos conocidos (no se descarta nada automaticamente).
  5. Escribe DATASET_MANIFEST.json con el dataset_id, parametros de descarga y limitaciones,
     y marca los archivos como solo lectura.

`verificar` recalcula los hashes y avisa si algo cambio. El marco de pruebas debe llamar a
verify_dataset() al arrancar para garantizar que todos los experimentos usan los mismos datos.
"""
import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lab_descarga_auditoria_v3 as base
except ImportError:
    raise SystemExit("Coloca lab_descarga_auditoria_v3.py en esta carpeta (se usa su calendario y cargador).")

MANIFEST_NAME = "DATASET_MANIFEST.json"
EXCLUDED_NAME = "dias_excluidos.csv"
REVIEW_NAME = "integridad_revision.csv"
MIN_COVERAGE = 0.95        # cobertura minima de velas esperadas por (simbolo, dia)
GAP_REVIEW = 0.20          # gap de apertura a revisar (20%)
BAR_RANGE_REVIEW = 0.10    # rango de una vela de 5 min a revisar (10% del cierre)
LIMITACIONES = [
    "Universo actual: no incluye empresas deslistadas ni adquiridas (sesgo de supervivencia; "
    "favorece a estrategias que compran caidas).",
    "Ajuste solo por splits; los dividendos NO estan ajustados.",
    "Velas fuera de la sesion oficial NYSE retiradas (after-hours en cierres anticipados).",
    "Dias con cobertura < 95% listados en dias_excluidos.csv; deben excluirse en los backtests.",
    "Sin datos de cotizaciones: el spread real y el slippage no estan medidos.",
]


# ------------------------------------------------------------------ utilidades
def _ns(index):
    """Nanosegundos UTC, independiente de la unidad interna de pandas."""
    return index.as_unit("ns").asi8


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _make_writable(path):
    if Path(path).exists():
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def _make_readonly(path):
    os.chmod(path, stat.S_IREAD)


def _dataset_id(file_hashes):
    return hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(file_hashes.items())).encode()).hexdigest()


# -------------------------------------------------------------------- congelar
def congelar(entrada, forzar=False):
    entrada = Path(entrada)
    files = sorted(entrada.glob("*_5min.csv"))
    if not files:
        raise SystemExit(f"No hay archivos *_5min.csv en {entrada}")
    manifest_path = entrada / MANIFEST_NAME
    if manifest_path.exists() and not forzar:
        raise SystemExit(f"{MANIFEST_NAME} ya existe: el dataset esta congelado. Usa 'verificar', "
                         "o --forzar si de verdad quieres recongelar (cambiara el dataset_id).")

    frames = {f.name.replace("_5min.csv", ""): base.load_bars(f) for f in files}
    start = min(df.index[0] for df in frames.values()).strftime("%Y-%m-%d")
    end = max(df.index[-1] for df in frames.values()).strftime("%Y-%m-%d")
    sch = base._calendar_schedule(start, end)
    sessions = list(sch.index)
    exp_ts, exp_sid = base.expected_bars(sch)
    print(f"Calendario NYSE: {len(sessions)} sesiones ({start} -> {end})")

    excluded, review, problems = [], [], []
    for sym, df in frames.items():
        act_raw = _ns(df.index)
        if len(act_raw) != len(np.unique(act_raw)):
            problems.append(f"{sym}: timestamps duplicados")
        act = np.unique(act_raw)
        unexpected = int((~base._isin_sorted(act, exp_ts)).sum())
        if unexpected:
            problems.append(f"{sym}: {unexpected} velas fuera del calendario (ejecuta 'corregir' primero)")
            continue

        date_s = df.index.strftime("%Y-%m-%d")
        first_d, last_d = date_s[0], date_s[-1]
        lo = int(np.searchsorted(sessions, first_d, side="left"))
        hi = int(np.searchsorted(sessions, last_d, side="right"))
        m = (exp_sid >= lo) & (exp_sid < hi)
        sub_ts, sub_sid = exp_ts[m], exp_sid[m] - lo
        found = base._isin_sorted(sub_ts, act)
        n = hi - lo
        exp_n = np.bincount(sub_sid, minlength=n)
        found_n = np.bincount(sub_sid[found], minlength=n)
        bad = found_n / np.maximum(exp_n, 1) < MIN_COVERAGE
        for k in np.flatnonzero(bad):
            excluded.append({"symbol": sym, "date": sessions[lo + k], "esperadas": int(exp_n[k]),
                             "encontradas": int(found_n[k]),
                             "motivo": "sin_datos" if found_n[k] == 0 else "cobertura_menor_95pct"})

        # gaps de apertura y velas extremas (solo para revision manual)
        o = pd.to_numeric(df["open"], errors="coerce"); h = pd.to_numeric(df["high"], errors="coerce")
        l = pd.to_numeric(df["low"], errors="coerce"); c = pd.to_numeric(df["close"], errors="coerce")
        d_open = o.groupby(date_s).first(); d_close = c.groupby(date_s).last()
        gap = d_open / d_close.shift(1) - 1
        for d in gap.index[gap.abs() > GAP_REVIEW]:
            review.append({"symbol": sym, "date": d, "tipo": "gap_apertura_grande",
                           "valor": round(float(gap[d]), 4),
                           "detalle": f"cierre_previo={d_close.shift(1)[d]:.4f} apertura={d_open[d]:.4f}"})
        rng = (h - l) / c
        for t in df.index[rng.values > BAR_RANGE_REVIEW]:
            review.append({"symbol": sym, "date": t.strftime("%Y-%m-%d"), "tipo": "vela_rango_extremo",
                           "valor": round(float(rng[t]), 4), "detalle": t.strftime("%H:%M")})

    if problems:
        print("\nEL DATASET NO ESTA LIMPIO; no se congela:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)

    # Archivos auxiliares (se escriben antes de calcular el manifiesto)
    aux = {EXCLUDED_NAME: pd.DataFrame(excluded, columns=["symbol", "date", "esperadas", "encontradas", "motivo"]),
           REVIEW_NAME: pd.DataFrame(review, columns=["symbol", "date", "tipo", "valor", "detalle"])}
    for name, frame in aux.items():
        _make_writable(entrada / name)
        frame.sort_values(["symbol", "date"]).to_csv(entrada / name, index=False)

    file_info = {}
    for f in files:
        sym = f.name.replace("_5min.csv", "")
        df = frames[sym]
        file_info[f.name] = {"sha256": _sha256(f), "bytes": f.stat().st_size, "filas": len(df),
                             "primera_vela": df.index[0].tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "ultima_vela": df.index[-1].tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}
    aux_info = {name: {"sha256": _sha256(entrada / name)} for name in aux}

    meta_src = {}
    if (entrada / "descarga_metadata.json").exists():
        raw = json.loads((entrada / "descarga_metadata.json").read_text(encoding="utf-8"))
        meta_src = {k: raw[k] for k in ("feed", "adjustment", "timeframe", "sesion") if k in raw}
    try:
        import pandas_market_calendars as mcal
        cal_ver = f"pandas_market_calendars {mcal.__version__}"
    except Exception:
        cal_ver = "desconocida"

    manifest = {
        "creado": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": _dataset_id({k: v["sha256"] for k, v in file_info.items()}),
        "simbolos": sorted(frames), "n_simbolos": len(frames),
        "sesiones": len(sessions), "rango": {"inicio": start, "fin": end},
        "descarga": meta_src, "calendario": {"nombre": "NYSE", "libreria": cal_ver},
        "pandas": pd.__version__,
        "pares_simbolo_dia_excluidos": len(excluded), "cobertura_minima": MIN_COVERAGE,
        "items_para_revision_manual": len(review),
        "limitaciones": LIMITACIONES, "archivos": file_info, "auxiliares": aux_info,
    }
    _make_writable(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for f in files:
        _make_readonly(f)
    for name in list(aux) + [MANIFEST_NAME]:
        _make_readonly(entrada / name)

    total = len(frames) * len(sessions)
    print(f"\nDataset congelado: {len(frames)} simbolos, {sum(v['filas'] for v in file_info.values()):,} velas")
    print(f"dataset_id: {manifest['dataset_id']}")
    print(f"Pares (simbolo, dia) excluidos: {len(excluded)} de {total:,} ({100*len(excluded)/total:.3f}%)")
    if excluded:
        ex = pd.DataFrame(excluded)
        print(ex.groupby("date").agg(simbolos=("symbol", "nunique"), min_velas=("encontradas", "min"),
                                     max_velas=("encontradas", "max")).to_string())
    print(f"\nPara revision manual ({len(review)}): ver {entrada / REVIEW_NAME}")
    if review:
        print(pd.DataFrame(review).to_string(index=False))
    print(f"\nArchivos de control: {MANIFEST_NAME}, {EXCLUDED_NAME}, {REVIEW_NAME}")


# ------------------------------------------------------------------- verificar
def verify_dataset(entrada):
    """Devuelve (ok, lista_de_problemas). Pensada para llamarse al arrancar el marco de pruebas."""
    entrada = Path(entrada)
    mp = entrada / MANIFEST_NAME
    if not mp.exists():
        return False, [f"No existe {MANIFEST_NAME}: el dataset no esta congelado."]
    man = json.loads(mp.read_text(encoding="utf-8"))
    problems = []
    for name, info in man["archivos"].items():
        p = entrada / name
        if not p.exists():
            problems.append(f"Falta {name}")
        elif _sha256(p) != info["sha256"]:
            problems.append(f"Modificado: {name}")
    for name, info in man.get("auxiliares", {}).items():
        p = entrada / name
        if not p.exists():
            problems.append(f"Falta {name}")
        elif _sha256(p) != info["sha256"]:
            problems.append(f"Modificado: {name}")
    extra = sorted(set(f.name for f in entrada.glob("*_5min.csv")) - set(man["archivos"]))
    problems += [f"Archivo no registrado en el manifiesto: {n}" for n in extra]
    return (not problems), problems


def verificar(entrada):
    ok, problems = verify_dataset(entrada)
    if ok:
        man = json.loads((Path(entrada) / MANIFEST_NAME).read_text(encoding="utf-8"))
        print(f"DATASET INTACTO. dataset_id={man['dataset_id']}")
        print(f"{man['n_simbolos']} simbolos | {man['rango']['inicio']} -> {man['rango']['fin']} | creado {man['creado']}")
        return
    print("EL DATASET HA CAMBIADO O ESTA INCOMPLETO:")
    for p in problems:
        print("  -", p)
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["congelar", "verificar"])
    p.add_argument("--entrada", default="datos_lab_sip_limpio")
    p.add_argument("--forzar", action="store_true", help="recongelar un dataset ya congelado (cambia el dataset_id)")
    a = p.parse_args()
    congelar(a.entrada, a.forzar) if a.accion == "congelar" else verificar(a.entrada)


if __name__ == "__main__":
    main()
