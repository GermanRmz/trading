"""
EXPLORACION de arquitecturas con datos ANTERIORES a 2020 (fuera de DEV, VAL y SELLADO: no toca el registro).

Etapas:
  sintetico     Senales "plantadas" sobre datos reales: se borra cualquier relacion real entre pasado y
                futuro (signo del retorno al azar por fecha y tramo) y se anade una senal conocida. Mide
                cuanta de esa senal recupera cada estructura, comparada con el techo (quien conoce la regla).
  estructuras   Las ocho estructuras con datos reales: dos pliegues (entrenar hasta 2017 -> probar 2018;
                entrenar hasta 2018 -> probar 2019), varias semillas.
  mejoras       Variantes de la base (tamano, regularizacion, contexto, espejo, tarea auxiliar, universo...).
  analizar      Tablas a partir de las ejecuciones guardadas.

Cada ejecucion se guarda en un archivo; si se interrumpe, al relanzar continua donde quedo.

Uso:
    python explorar.py sintetico   --tabla pasos_pre.pkl --salida corridas --trabajadores 4
    python explorar.py estructuras --tabla pasos_pre.pkl --salida corridas
    python explorar.py mejoras     --tabla pasos_pre.pkl --salida corridas
    python explorar.py analizar    --tabla pasos_pre.pkl --salida corridas
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pol_datos as PD  # noqa: E402
import pol_entrenar as PE  # noqa: E402
import pol_modelos as PM  # noqa: E402

PLIEGUES = {"A": ("2017-12-31", ("2018",)), "B": ("2018-12-31", ("2019",))}
PLIEGUE_SINT = {"S": ("2017-12-31", ("2018", "2019"))}
SENALES = ("momento", "regimen", "memoria")
_T = None


def _tabla(ruta):
    global _T
    if _T is None:
        _T = pickle.load(open(ruta, "rb"))
    return _T


def plantar(T, tipo, beta, semilla=123):
    """Devuelve copia de la tabla con retornos sinteticos: nulo realista + beta x senal conocida.

    Nulo: el retorno real con signo al azar por (fecha, tramo), comun a todos los simbolos; conserva
    volatilidad, horario y correlacion entre simbolos, pero rompe toda relacion con el pasado.
    """
    rng = np.random.default_rng(semilla)
    v = T["variables"].index
    X, Y = T["X"], T["Y"]
    fechas, inv = np.unique(T["fecha"], return_inverse=True)
    signo = rng.choice([-1.0, 1.0], size=(len(fechas), PD.J)).astype(np.float32)[inv]
    nulo = Y * signo
    escala = Y.std(axis=0, keepdims=True)                          # desviacion tipica por tramo
    if tipo == "momento":            # el movimiento del dia continua (relacion suave, visible en una variable)
        g = np.tanh(X[:, :, v("r_dia")])
    elif tipo == "regimen":          # continua solo por la tarde y en dias de volatilidad alta (interaccion)
        g = np.sign(X[:, :, v("r_cc")]) * (X[:, :, v("rv_rel")] > 0) * (X[:, :, v("paso")] >= 0.6)
    elif tipo == "memoria":          # lo ocurrido hasta las 10:00 marca la ultima hora (hay que recordarlo)
        g = np.zeros_like(Y)
        g[:, -2:] = np.tanh(X[:, :1, v("r_cc")])
    else:
        raise ValueError(tipo)
    S = dict(T)
    S["Y"] = np.clip(nulo + beta * escala * g, -5, 5).astype(np.float32)
    S["RB"] = (1e4 * S["Y"] * T["sig"][:, None]).astype(np.float32)
    S["_g"] = g.astype(np.float32)
    return S


def _mascaras(conj, hasta, anios, frac):
    antes = conj.fecha <= hasta
    f = np.unique(conj.fecha[antes])
    corte = f[int((1 - frac) * len(f))]
    return antes & (conj.fecha < corte), antes & (conj.fecha >= corte), np.isin(conj.anio, anios)


def _id(trabajo):
    crudo = json.dumps(trabajo, sort_keys=True, default=str)
    return hashlib.sha1(crudo.encode()).hexdigest()[:12]


def _correr(arg):
    ruta_tabla, salida, trabajo = arg
    import torch
    torch.set_num_threads(1)
    destino = Path(salida) / f"{trabajo['etapa']}_{_id(trabajo)}.pkl"
    if destino.exists():
        return destino.name, 0.0
    T = _tabla(ruta_tabla)
    if trabajo.get("senal"):
        T = plantar(T, *trabajo["senal"])
    cfg = PE.Config(**trabajo["cfg"])
    conj = PE.Conjunto(T)
    hasta, anios = trabajo["pliegue_def"]
    ent, val, pru = _mascaras(conj, hasta, anios, cfg.frac_validacion)
    conj.escalar(ent)
    conj.fijar_costo(trabajo["objetivo"], ent, cfg.costo_comun)
    t0 = time.time()
    red, hist = PE.entrenar(cfg, conj, ent, val, trabajo["semilla"], objetivo=trabajo["objetivo"])
    P, V = PE.predecir(red, conj, val | pru, cfg)
    filas = np.flatnonzero(val | pru)
    out = {"trabajo": trabajo, "filas": filas, "P": P[filas].astype(np.float32), "V": V[filas].astype(np.float32),
           "es_val": val[filas], "hist": hist, "segundos": time.time() - t0, "parametros": PM.n_parametros(red)}
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


def _trabajo(etapa, cfg, pliegue, pliegue_def, semilla, objetivo, senal=None):
    return {"etapa": etapa, "cfg": asdict(cfg), "pliegue": pliegue, "pliegue_def": pliegue_def, "semilla": semilla,
            "objetivo": objetivo, "senal": senal}


def trabajos_sintetico(a):
    out = []
    for tipo in SENALES:
        for beta in (0.03, 0.06):
            for est in PM.ESTRUCTURAS:
                for s in range(a.semillas):
                    cfg = PE.Config(estructura=est, espejo=False)
                    out.append(_trabajo("sintetico", cfg, "S", PLIEGUE_SINT["S"], s, a.objetivo, (tipo, beta)))
    return out


def trabajos_estructuras(a):
    return [_trabajo("estructuras", PE.Config(estructura=est), p, d, s, a.objetivo)
            for est in PM.ESTRUCTURAS for p, d in PLIEGUES.items() for s in range(a.semillas)]


VARIANTES = {
    "base": {},
    "h=16": {"h": 16}, "h=64": {"h": 64},
    "dropout=0.1": {"dropout": 0.1}, "dropout=0.5": {"dropout": 0.5},
    "contexto corto (3 dias)": {"dias_trozo": 3}, "contexto largo (11 dias)": {"dias_trozo": 11},
    "sin espejo": {"espejo": False},
    "sin tarea de volatilidad": {"peso_vol": 0.0},
    "solo el objetivo": {"universo": False},
    "sin coste al entrenar": {"costo_bps": 0.0},
    "coste doble al entrenar": {"costo_bps": 16.0},
    "decaimiento=1e-2": {"decaimiento": 1e-2},
    "lr=3e-4": {"lr": 3e-4, "epocas": 50},
}


def trabajos_mejoras(a):
    base = PE.Config(estructura=a.estructura)
    return [_trabajo("mejoras", replace(base, **cambios), p, d, s, a.objetivo)
            for cambios in VARIANTES.values() for p, d in PLIEGUES.items() for s in range(a.semillas)]


# ------------------------------------------------------------------ analisis
def _cargar(a, etapa):
    filas = []
    for f in sorted(Path(a.salida).glob(f"{etapa}_*.pkl")):
        filas.append(pickle.load(open(f, "rb")))
    return filas


def _evaluar(T, conj, corridas, costo=8.0, objetivo=None):
    """Conjunto de semillas de una misma configuracion y pliegue -> metricas del promedio de posiciones."""
    filas = corridas[0]["filas"]
    es_val = corridas[0]["es_val"]
    P = np.full((len(T["sym"]), PD.J), np.nan, np.float32)
    P[filas] = np.mean([c["P"] for c in corridas], axis=0)
    val, pru = np.zeros(len(P), bool), np.zeros(len(P), bool)
    val[filas[es_val]], pru[filas[~es_val]] = True, True
    r = {}
    i, g = PE.ganancia_suave(conj, P, pru, costo)
    r["sharpe_suave"] = PE.sharpe_cartera(g, conj.fecha[i])
    i0, g0 = PE.ganancia_suave(conj, P, pru, 0.0)
    r["sharpe_suave_bruto"] = PE.sharpe_cartera(g0, conj.fecha[i0])
    obj = T["simbolos"].index(objetivo) if objetivo is not None else None
    u, q = PE.elegir_umbral(conj, P, val, costo, obj)
    D = PE.discretizar(P, u, q)
    ip = np.flatnonzero(pru)
    neto, n_ops, rot = PE.ganancia_dura(T["RB"][ip], D[ip], costo)
    r["umbral"] = u
    r["sharpe_duro"] = PE.sharpe_cartera(PE.ganancia_dura_norm(conj, D[ip], ip, costo), conj.fecha[ip])
    r["ops_dia"] = float(n_ops.mean())
    r["bps_dia"] = float(neto.mean())
    if objetivo is not None:
        m = T["sym"][ip] == T["simbolos"].index(objetivo)
        if m.sum() > 20 and neto[m].std() > 0:
            r["obj_sharpe"] = float(neto[m].mean() / neto[m].std() * np.sqrt(252))
            r["obj_bps_dia"], r["obj_ops_dia"] = float(neto[m].mean()), float(n_ops[m].mean())
            bruto = neto[m] + 4.0 * rot[m]
            r["obj_bps_op_bruto"] = float(bruto.sum() / max(n_ops[m].sum(), 1))
    # tarea auxiliar: cuanto explica el pronostico de volatilidad (R2 frente a la media por tramo)
    V = np.mean([c["V"] for c in corridas], axis=0)[~es_val]
    vt = T["VT"][ip]
    r["r2_vol"] = float(1 - ((vt - V) ** 2).mean() / ((vt - vt.mean(0)) ** 2).mean())
    r["sharpe_val"] = float(np.mean([c["hist"]["mejor_sharpe_val"] for c in corridas]))
    r["epocas"] = float(np.mean([c["hist"]["epocas"] for c in corridas]))
    r["segundos"] = float(np.mean([c["segundos"] for c in corridas]))
    r["semillas"] = len(corridas)
    r["sharpe_semillas_sd"] = float(np.nanstd([_solo_suave(T, conj, c, costo) for c in corridas]))
    return r


def _solo_suave(T, conj, c, costo):
    P = np.full((len(T["sym"]), PD.J), np.nan, np.float32)
    P[c["filas"]] = c["P"]
    pru = np.zeros(len(P), bool)
    pru[c["filas"][~c["es_val"]]] = True
    i, g = PE.ganancia_suave(conj, P, pru, costo)
    return PE.sharpe_cartera(g, conj.fecha[i])


def analizar(a):
    T0 = _tabla(a.tabla)
    pd.set_option("display.width", 250)
    out = Path(a.salida)
    # ---- sintetico
    cs = _cargar(a, "sintetico")
    if cs:
        filas = []
        for (tipo, beta) in sorted({tuple(c["trabajo"]["senal"]) for c in cs}):
            T = plantar(T0, tipo, beta)
            conj = PE.Conjunto(T)
            pru = np.isin(conj.anio, PLIEGUE_SINT["S"][1])
            Po = np.where(pru[:, None], np.sign(T["_g"]), np.nan).astype(np.float32)
            i, g = PE.ganancia_suave(conj, Po, pru, 8.0)
            techo = PE.sharpe_cartera(g, conj.fecha[i])
            for est in PM.ESTRUCTURAS:
                sel = [c for c in cs if tuple(c["trabajo"]["senal"]) == (tipo, beta) and c["trabajo"]["cfg"]["estructura"] == est]
                if not sel:
                    continue
                r = _evaluar(T, conj, sel)
                filas.append({"senal": tipo, "beta": beta, "estructura": est, "techo": techo, **r})
        df = pd.DataFrame(filas)
        df["recuperado_pct"] = 100 * df["sharpe_suave"] / df["techo"]
        df.to_csv(out / "tabla_sintetico.csv", index=False)
        print("\n=== SENALES PLANTADAS: % del Sharpe del techo que recupera cada estructura (neto de 8 bps) ===")
        print(df.pivot_table(index="estructura", columns=["senal", "beta"], values="recuperado_pct").round(0)
              .reindex(PM.ESTRUCTURAS).to_string())
        print("Techo (Sharpe de quien conoce la regla):")
        print(df.groupby(["senal", "beta"])["techo"].first().round(2).to_string())
    # ---- datos reales
    conj = PE.Conjunto(T0)
    for etapa, clave in (("estructuras", lambda c: c["trabajo"]["cfg"]["estructura"]),
                         ("mejoras", lambda c: PE.Config(**c["trabajo"]["cfg"]).clave())):
        cs = _cargar(a, etapa)
        if not cs:
            continue
        filas = []
        for k in sorted({clave(c) for c in cs}):
            for p in PLIEGUES:
                sel = [c for c in cs if clave(c) == k and c["trabajo"]["pliegue"] == p]
                if sel:
                    filas.append({"config": k, "pliegue": p, **_evaluar(T0, conj, sel, objetivo=a.objetivo)})
        df = pd.DataFrame(filas)
        df.to_csv(out / f"tabla_{etapa}_por_pliegue.csv", index=False)
        cols = ["sharpe_suave", "sharpe_suave_bruto", "sharpe_duro", "obj_sharpe", "obj_bps_dia", "obj_ops_dia",
                "obj_bps_op_bruto", "ops_dia", "r2_vol", "sharpe_val", "sharpe_semillas_sd", "epocas", "segundos"]
        m = df.groupby("config")[cols].mean().sort_values("sharpe_suave", ascending=False)
        m.to_csv(out / f"tabla_{etapa}.csv")
        print(f"\n=== {etapa.upper()} con datos reales 2016-2019 (media de los pliegues 2018 y 2019; neto de 8 bps) ===")
        print(m.round(2).to_string())
        print("Por pliegue (Sharpe suave de cartera):")
        print(df.pivot(index="config", columns="pliegue", values="sharpe_suave").round(2).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("etapa", choices=["sintetico", "estructuras", "mejoras", "analizar"])
    p.add_argument("--tabla", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--trabajadores", type=int, default=4)
    p.add_argument("--semillas", type=int, default=3)
    p.add_argument("--objetivo", default="SMH3X")
    p.add_argument("--estructura", default="vlstm")
    a = p.parse_args()
    if a.etapa == "analizar":
        return analizar(a)
    trabajos = {"sintetico": trabajos_sintetico, "estructuras": trabajos_estructuras, "mejoras": trabajos_mejoras}[a.etapa](a)
    print(f"{len(trabajos)} ejecuciones", flush=True)
    _lanzar(trabajos, a)


if __name__ == "__main__":
    main()
