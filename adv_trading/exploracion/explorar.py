"""
EXPLORACION de arquitecturas con datos ANTERIORES a 2020 (fuera de DEV, VAL y SELLADO: no toca el registro).

Etapas:
  plantadas     Senales "plantadas" sobre datos reales: se borra toda relacion real entre pasado y futuro
                (signo del retorno al azar por fecha y tramo) y se anade una senal conocida. Se entrena
                hasta 2018 y se mide en 2019 que parte del Sharpe del "techo" (quien conoce la regla)
                recupera cada estructura, promediando 30 sorteos del mundo nulo para quitar el azar.
  mejoras       Lo mismo, variando un ingrediente cada vez sobre una red de referencia.
  reales        Las estructuras con datos reales: entrenar hasta 2017 -> probar 2018, y entrenar hasta
                2018 -> probar 2019; conjunto de varias semillas.
  analizar      Tablas a partir de las ejecuciones guardadas.

Cada ejecucion se guarda en un archivo; si se interrumpe, al relanzar continua donde quedo.

Uso:
    python explorar.py plantadas --tabla pasos_pre.pkl --salida corridas
    python explorar.py mejoras   --tabla pasos_pre.pkl --salida corridas
    python explorar.py reales    --tabla pasos_pre.pkl --salida corridas
    python explorar.py analizar  --tabla pasos_pre.pkl --salida corridas
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pol_datos as PD  # noqa: E402
import pol_entrenar as PE  # noqa: E402
import pol_modelos as PM  # noqa: E402

# Receta comun, calibrada con senales plantadas: la senal se entrena SIN costes, con simetria exacta,
# apagado de variables, redes pequenas, mucha regularizacion y pocas epocas fijas.
RECETA = dict(h=8, dropout=0.5, decaimiento=1e-2, lr=1e-3, lote=256, epocas=15, paciencia=0, costo_bps=0.0,
              espejo=False, antisimetrica=True, dropout_var=0.3, recorte_x=3.0, recorte_y=4.0)
SENALES = {"momento": 0.06, "regimen": 0.12, "memoria": 0.12}
PLIEGUE_PLANT = ("2018-12-31", ("2019",))
PLIEGUES = {"A": ("2017-12-31", ("2018",)), "B": ("2018-12-31", ("2019",))}
VARIANTES = {
    "referencia": {},
    "h=32": {"h": 32}, "h=32, 60 epocas": {"h": 32, "epocas": 60},
    "dropout=0.1": {"dropout": 0.1},
    "sin apagar variables": {"dropout_var": 0.0},
    "sin simetria": {"antisimetrica": False},
    "simetria por aumento": {"antisimetrica": False, "espejo": True},
    "sin recortes": {"recorte_x": 5.0, "recorte_y": 0.0},
    "5 epocas": {"epocas": 5}, "40 epocas": {"epocas": 40},
    "lr=3e-4": {"lr": 3e-4}, "lote=64": {"lote": 64},
    "con coste de 8 bps al entrenar": {"costo_bps": 8.0},
    "sin tarea de volatilidad": {"peso_vol": 0.0},
    "contexto de 2 dias": {"dias_trozo": 3},
    "solo el objetivo (sin universo)": {"universo": False},
}
_T = None


def _tabla(ruta):
    global _T
    if _T is None:
        _T = pickle.load(open(ruta, "rb"))
    return _T


def senal(T, tipo):
    """Regla conocida g (por muestra y tramo) que se planta en el retorno."""
    v = T["variables"].index
    X = T["X"]
    if tipo == "momento":            # el movimiento del dia continua (relacion suave, visible en una variable)
        return np.tanh(X[:, :, v("r_dia")])
    if tipo == "regimen":            # continua solo por la tarde y en dias de volatilidad alta (interaccion)
        return (np.sign(X[:, :, v("r_cc")]) * (X[:, :, v("rv_rel")] > 0) * (X[:, :, v("paso")] >= 0.6)).astype(np.float32)
    if tipo == "memoria":            # lo ocurrido hasta las 10:00 marca la ultima hora (hay que recordarlo)
        g = np.zeros(X.shape[:2], np.float32)
        g[:, -2:] = np.tanh(X[:, :1, v("r_cc")])
        return g
    raise ValueError(tipo)


def plantar(T, tipo, beta, semilla=123):
    """Copia de la tabla con retornos sinteticos: mundo nulo realista + beta x senal conocida.

    Mundo nulo: el retorno real con signo al azar por (fecha, tramo), comun a todos los simbolos. Conserva
    volatilidad, horario y correlacion entre simbolos, pero rompe toda relacion con el pasado.
    """
    rng = np.random.default_rng(semilla)
    _, inv = np.unique(T["fecha"], return_inverse=True)
    signo = rng.choice([-1.0, 1.0], size=(inv.max() + 1, PD.J)).astype(np.float32)[inv]
    g = senal(T, tipo)
    S = dict(T)
    S["Y"] = np.clip(T["Y"] * signo + beta * T["Y"].std(axis=0, keepdims=True) * g, -5, 5).astype(np.float32)
    S["RB"] = (1e4 * S["Y"] * T["sig"][:, None]).astype(np.float32)
    return S


def sharpe_esperado(T, tipo, beta, P, filas, sorteos=30, semilla=7):
    """Sharpe bruto medio de las posiciones P frente a muchos sorteos del mundo nulo (mas la senal)."""
    rng = np.random.default_rng(semilla)
    f = T["fecha"][filas]
    _, inv = np.unique(f, return_inverse=True)
    yr, plant = T["Y"][filas], beta * T["Y"].std(axis=0, keepdims=True) * senal(T, tipo)[filas]
    out = []
    for _ in range(sorteos):
        sg = rng.choice([-1.0, 1.0], size=(inv.max() + 1, PD.J))[inv]
        out.append(PE.sharpe_cartera((P * (yr * sg + plant)).sum(1), f))
    return float(np.mean(out))


def _mascaras(conj, hasta, anios, frac):
    antes = conj.fecha <= hasta
    f = np.unique(conj.fecha[antes])
    corte = f[min(int((1 - frac) * len(f)), len(f) - 1)]
    return antes & (conj.fecha < corte), antes & (conj.fecha >= corte), np.isin(conj.anio, anios)


def _id(trabajo):
    return hashlib.sha1(json.dumps(trabajo, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _correr(arg):
    ruta_tabla, salida, trabajo = arg
    import torch
    torch.set_num_threads(1)
    destino = Path(salida) / f"{trabajo['etapa']}_{_id(trabajo)}.pkl"
    if destino.exists():
        return destino.name, 0.0
    T0 = _tabla(ruta_tabla)
    plantada = trabajo.get("senal")
    T = plantar(T0, *plantada) if plantada else T0
    cfg = PE.Config(**trabajo["cfg"])
    conj = PE.Conjunto(T)
    hasta, anios = trabajo["pliegue_def"]
    ent, val, pru = _mascaras(conj, hasta, anios, 0.0 if plantada else cfg.frac_validacion)
    if plantada:
        ent = ent | val
    conj.escalar(ent, cfg.recorte_x)
    conj.fijar_costo(trabajo["objetivo"], ent, cfg.costo_comun)
    t0 = time.time()
    red, hist = PE.entrenar(cfg, conj, ent, val, trabajo["semilla"], objetivo=trabajo["objetivo"])
    marca = pru if plantada else (val | pru)
    P, V = PE.predecir(red, conj, marca, cfg)
    filas = np.flatnonzero(marca & np.isfinite(P[:, 0]))
    out = {"trabajo": trabajo, "filas": filas, "P": P[filas], "V": V[filas], "es_val": val[filas],
           "segundos": time.time() - t0, "parametros": PM.n_parametros(red)}
    tmp = destino.with_suffix(".tmp")
    pickle.dump(out, open(tmp, "wb"))
    tmp.replace(destino)
    return destino.name, out["segundos"]


def _lanzar(trabajos, a):
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    args = [(a.tabla, a.salida, t) for t in trabajos]
    t0 = time.time()
    with Pool(a.trabajadores) as pool:
        for k, (nombre, seg) in enumerate(pool.imap_unordered(_correr, args), 1):
            print(f"[{k}/{len(args)}] {nombre} {seg:.0f}s | total {time.time() - t0:.0f}s", flush=True)


def _trabajo(etapa, nombre, cambios, pliegue, pliegue_def, semilla, objetivo, plantada=None):
    cfg = PE.Config(**{**RECETA, **cambios})
    return {"etapa": etapa, "nombre": nombre, "cfg": asdict(cfg), "pliegue": pliegue, "pliegue_def": pliegue_def,
            "semilla": semilla, "objetivo": objetivo, "senal": plantada}


def trabajos(a):
    if a.etapa == "plantadas":
        return [_trabajo("plantadas", est, {"estructura": est}, "P", PLIEGUE_PLANT, s, a.objetivo, (tipo, beta))
                for tipo, beta in SENALES.items() for est in a.estructuras for s in range(a.semillas)]
    if a.etapa == "mejoras":
        return [_trabajo("mejoras", nombre, {"estructura": a.referencia, **cambios}, "P", PLIEGUE_PLANT, s, a.objetivo,
                         (tipo, beta))
                for tipo, beta in SENALES.items() for nombre, cambios in VARIANTES.items() for s in range(a.semillas)]
    return [_trabajo("reales", est, {"estructura": est}, p, d, s, a.objetivo)
            for est in a.estructuras for p, d in PLIEGUES.items() for s in range(a.semillas)]


# ------------------------------------------------------------------ analisis
def _cargar(a, etapa):
    return [pickle.load(open(f, "rb")) for f in sorted(Path(a.salida).glob(f"{etapa}_*.pkl"))]


def _tabla_plantadas(T0, corridas, objetivo):
    obj = T0["simbolos"].index(objetivo)
    techos = {}
    for c in corridas:
        tipo, beta = c["trabajo"]["senal"]
        if tipo not in techos:
            f = c["filas"]
            m = T0["sym"][f] == obj
            g = np.sign(senal(T0, tipo))
            techos[tipo] = (sharpe_esperado(T0, tipo, beta, g[f], f), sharpe_esperado(T0, tipo, beta, g[f[m]], f[m]))
    filas = []
    for nombre in dict.fromkeys(c["trabajo"]["nombre"] for c in corridas):
        fila = {"config": nombre}
        for tipo, beta in SENALES.items():
            sel = [c for c in corridas if c["trabajo"]["nombre"] == nombre and c["trabajo"]["senal"][0] == tipo]
            if not sel:
                continue
            f = sel[0]["filas"]
            m = T0["sym"][f] == obj
            conj = np.mean([c["P"] for c in sel], axis=0)
            fila[tipo] = np.mean([100 * sharpe_esperado(T0, tipo, beta, c["P"], f) / techos[tipo][0] for c in sel])
            fila[f"{tipo}_conjunto"] = 100 * sharpe_esperado(T0, tipo, beta, conj, f) / techos[tipo][0]
            fila[f"{tipo}_objetivo"] = 100 * sharpe_esperado(T0, tipo, beta, conj[m], f[m]) / techos[tipo][1]
            fila["segundos"] = np.mean([c["segundos"] for c in sel])
            fila["parametros"] = sel[0]["parametros"]
        filas.append(fila)
    return pd.DataFrame(filas).set_index("config"), techos


def _suave(conj, c):
    P = np.full((conj.n, PD.J), np.nan, np.float32)
    P[c["filas"]] = c["P"]
    pru = np.zeros(conj.n, bool)
    pru[c["filas"][~c["es_val"]]] = True
    i, g = PE.ganancia_suave(conj, P, pru, 0.0)
    return PE.sharpe_cartera(g, conj.fecha[i])


def _eval_real(T, conj, sel, objetivo, costo=8.0):
    """Conjunto de semillas (mismo pliegue): metricas en el anio de prueba."""
    filas, es_val = sel[0]["filas"], sel[0]["es_val"]
    P = np.full((conj.n, PD.J), np.nan, np.float32)
    P[filas] = np.mean([c["P"] for c in sel], axis=0)
    val, pru = np.zeros(conj.n, bool), np.zeros(conj.n, bool)
    val[filas[es_val]], pru[filas[~es_val]] = True, True
    obj = T["simbolos"].index(objetivo)
    r = {}
    i, g = PE.ganancia_suave(conj, P, pru, 0.0)
    r["cartera_bruto"] = PE.sharpe_cartera(g, conj.fecha[i])
    r["cartera_bruto_semilla"] = float(np.mean([_suave(conj, c) for c in sel]))
    u, q = PE.elegir_umbral(conj, P, val, costo, obj)
    D = PE.discretizar(P, u, q)
    ip = np.flatnonzero(pru)
    r["umbral"] = u
    r["cartera_neto"] = PE.sharpe_cartera(PE.ganancia_dura_norm(conj, D[ip], ip, costo), conj.fecha[ip])
    for nombre, lista in (("objetivo", [objetivo]), ("apalancados", ["TQQQ", "SPXL", "UPRO"])):
        ks = [T["simbolos"].index(s) for s in lista if s in T["simbolos"]]
        m = ip[np.isin(T["sym"][ip], ks)]
        if not len(m):
            continue
        neto, n_ops, rot = PE.ganancia_dura(T["RB"][m], D[m], costo)
        bruto = neto + (costo / 2) * rot
        r[f"{nombre}_sharpe_neto"] = PE.sharpe_cartera(neto, conj.fecha[m])
        r[f"{nombre}_bps_op_bruto"] = float(bruto.sum() / max(n_ops.sum(), 1))
        r[f"{nombre}_ops_dia"] = float(n_ops.mean())
    V = np.mean([c["V"] for c in sel], axis=0)[~es_val]
    vt = T["VT"][ip]
    r["r2_vol"] = float(1 - ((vt - V) ** 2).mean() / ((vt - vt.mean(0)) ** 2).mean())
    r["segundos"] = float(np.mean([c["segundos"] for c in sel]))
    return r


def analizar(a):
    T0 = _tabla(a.tabla)
    pd.set_option("display.width", 250)
    out = Path(a.salida)
    for etapa, titulo in (("plantadas", "ESTRUCTURAS con senales plantadas"), ("mejoras", "INGREDIENTES (red de referencia)")):
        cs = _cargar(a, etapa)
        if not cs:
            continue
        df, techos = _tabla_plantadas(T0, cs, a.objetivo)
        df.to_csv(out / f"tabla_{etapa}.csv")
        print(f"\n=== {titulo}: % del Sharpe del techo recuperado en 2019 (media de semillas | conjunto | solo objetivo) ===")
        print(df.round(0).to_string())
        print("Techo (Sharpe bruto esperado de quien conoce la regla; cartera, objetivo):",
              {k: (round(v[0], 2), round(v[1], 2)) for k, v in techos.items()})
    cs = _cargar(a, "reales")
    if cs:
        conj = PE.Conjunto(T0)
        filas = []
        for est in dict.fromkeys(c["trabajo"]["nombre"] for c in cs):
            for p, (hasta, anios) in PLIEGUES.items():
                sel = [c for c in cs if c["trabajo"]["nombre"] == est and c["trabajo"]["pliegue"] == p]
                if not sel:
                    continue
                cfg = PE.Config(**sel[0]["trabajo"]["cfg"])
                ent, _, _ = _mascaras(conj, hasta, anios, cfg.frac_validacion)
                conj.fijar_costo(a.objetivo, ent, cfg.costo_comun)
                filas.append({"estructura": est, "prueba": anios[0], "semillas": len(sel), **_eval_real(T0, conj, sel, a.objetivo)})
        df = pd.DataFrame(filas)
        df.to_csv(out / "tabla_reales_por_anio.csv", index=False)
        m = df.drop(columns=["prueba", "semillas", "umbral"]).groupby("estructura").mean()
        m = m.reindex([e for e in PM.ESTRUCTURAS if e in m.index])
        m.to_csv(out / "tabla_reales.csv")
        print("\n=== DATOS REALES 2016-2019: Sharpe anualizado en el anio de prueba (media de 2018 y 2019) ===")
        print(m.round(2).to_string())
        print("\nPor anio de prueba:")
        cols = ["cartera_bruto", "cartera_neto", "objetivo_sharpe_neto", "objetivo_bps_op_bruto", "objetivo_ops_dia", "umbral"]
        print(df.pivot(index="estructura", columns="prueba", values=cols).round(2).reindex(m.index).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("etapa", choices=["plantadas", "mejoras", "reales", "analizar"])
    p.add_argument("--tabla", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--trabajadores", type=int, default=4)
    p.add_argument("--semillas", type=int, default=3)
    p.add_argument("--objetivo", default="SMH3X")
    p.add_argument("--referencia", default="lstm")
    p.add_argument("--estructuras", nargs="+", default=list(PM.ESTRUCTURAS))
    a = p.parse_args()
    if a.etapa == "analizar":
        return analizar(a)
    lista = trabajos(a)
    print(f"{len(lista)} ejecuciones", flush=True)
    _lanzar(lista, a)


if __name__ == "__main__":
    main()
