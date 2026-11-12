"""
RED DE POLITICA PARA SOXL - protocolo pol_v1 (hipotesis H11).

Antes hay que preparar los datos:   python adv_trading/preparar_datos_soxl.py

Uso (desde la raiz del repositorio):
    python adv_trading/red_politica.py prueba      # comprueba la tuberia; NO registra ni mira resultados reales
    python adv_trading/red_politica.py explorar    # resultado en 2018 y 2019 (fuera de DEV): gratis, no registra
    python adv_trading/red_politica.py ejecutar    # registra H11 y la evalua en DEV (una sola vez)

DEFINICIONES FIJAS (no se cambian tras ver resultados; cualquier cambio = otra hipotesis):
  Objetivo    SOXL, solo intradia. 12 decisiones por sesion: 10:00, 10:30, ... 15:30 ET. En cada una la
              red elige comprar (+1), nada (0) o vender en corto (-1) hasta la decision siguiente; a las
              16:00 no queda posicion abierta. Nocional fijo por operacion.
  Operacion   una operacion = racha de decisiones seguidas con el mismo signo. Entrada a la APERTURA de la
              vela de la primera decision de la racha; salida a la apertura de la vela de la decision en
              que cambia, o al cierre de la sesion. Sin stops.
  Datos       dataset congelado adv_trading/datos_soxl_sip_limpio (velas SIP de 5 min desde 2016).
              2016-2019 solo entrena. DEV = 2020-2023, evaluado con ventana expansiva: cada anio se opera
              con redes entrenadas unicamente con anios anteriores.
  Red         la de adv_trading/config_final.json: conjunto de tres redes pequenas (LSTM, xLSTM y
              Transformer, 8 neuronas), elegido con datos anteriores a 2020 y con senales plantadas (ver
              README). Se entrena la SENAL sin costes, maximizando el Sharpe, con simetria exacta (no
              puede aprender un sesgo fijo de compra).
  Conjunto    cada estructura aporta el promedio de sus semillas; se igualan las escalas y se promedian.
  Costes      se aplican al convertir la senal en operaciones: solo se opera cuando la senal supera su
              tamano tipico (umbral 1,0); si asi hubiera menos de 0,55 operaciones por sesion en la
              validacion interna del pliegue, se baja a 0,5 y luego a 0. Regla fija, sin optimizar.
              Evaluacion del laboratorio a 4, 8 y 12 bps; el criterio es a 8.
  Umbral t    Bonferroni sobre TODAS las hipotesis: las del registro de lab/ mas las de este registro.
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(AQUI))
import pol_datos as PD  # noqa: E402
import pol_entrenar as PE  # noqa: E402
import pol_sintetico as PS  # noqa: E402
import preparar_datos_soxl as PREP  # noqa: E402

M = PD.M
PROTOCOLO, FAMILIA, NOMBRE = "pol_v1", "ia_red_politica", "H11_politica_soxl"
ANIOS_DEV = ("2020", "2021", "2022", "2023")
ANIOS_PRE = ("2018", "2019")          # fuera de DEV/VAL/SELLADO: se pueden mirar sin gastar hipotesis
CONFIG = AQUI / "config_final.json"
SALIDA = AQUI / "resultados"
REGISTRO = AQUI / "registro" / "registro_hipotesis.jsonl"
REGISTRO_LAB = RAIZ / "lab" / "registro_lab" / "registro_hipotesis.jsonl"


class RegistroGlobal(M.Registro):
    """Registro propio de este dataset que CUENTA tambien las hipotesis del laboratorio principal:
    el umbral de Bonferroni se calcula sobre el total, venga de la familia que venga."""

    def n_previas(self):
        if not REGISTRO_LAB.exists():
            return 0
        return sum(1 for l in REGISTRO_LAB.read_text(encoding="utf-8").splitlines()
                   if l.strip() and json.loads(l).get("tipo") == "hipotesis")

    def n_hipotesis(self):
        return super().n_hipotesis() + self.n_previas()


def leer_config():
    """Devuelve (lista de (Config, semillas) -una por estructura del conjunto-, reglas de operacion)."""
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    redes = [(PE.Config(**{**c["receta"], "estructura": r["estructura"]}), int(r["semillas"])) for r in c["redes"]]
    return redes, c["operacion"]


def _sha_codigo():
    h = hashlib.sha256()
    for f in ("red_politica.py", "pol_datos.py", "pol_modelos.py", "pol_entrenar.py", "config_final.json"):
        h.update((AQUI / f).read_bytes())
    return h.hexdigest()[:16]


def abrir():
    if not (PREP.LIMPIO / "DATASET_MANIFEST.json").exists():
        raise SystemExit("Falta el dataset de SOXL. Ejecuta primero:  python adv_trading/preparar_datos_soxl.py")
    ds = PD.abrir(PREP.LIMPIO, PREP.EXCLUSIONES)
    falta = [s for s in PREP.SIMBOLOS if s not in ds.symbols]
    if falta:
        raise SystemExit(f"El dataset no contiene {falta}. Vuelve a ejecutar preparar_datos_soxl.py")
    return ds


def tabla(ds):
    """Tabla de pasos de decision hasta el final de DEV (en cache local; no se sube al repositorio)."""
    SALIDA.mkdir(exist_ok=True)
    cache = SALIDA / f"_pasos_{ds.dataset_id[:12]}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    print("Construyendo la tabla de pasos (solo la primera vez)...", flush=True)
    T = PD.construir(ds, PREP.SIMBOLOS, PREP.CONTEXTO, M.SPLITS["dev"][1])
    pickle.dump(T, open(cache, "wb"))
    return T


def decidir(T, redes, operacion, anios, verbose=True):
    """Ventana expansiva: para cada anio, entrena el conjunto con anios anteriores y decide el anio.

    El conjunto promedia varias estructuras pequenas; cada una aporta el promedio de sus semillas,
    llevado antes a la misma escala. Devuelve D [n, 12] con +1/0/-1 (NaN fuera de los anios pedidos)
    y un diario por pliegue.
    """
    # Redes de unos mil parametros: en CPU tardan poco y se evita depender de la tarjeta grafica.
    conj = PE.Conjunto(T, "cpu")
    cfg0 = redes[0][0]
    if verbose:
        print("Conjunto: " + " + ".join(f"{c.estructura} x{n}" for c, n in redes) + f" (h={cfg0.h}) | CPU", flush=True)
    obj = T["simbolos"].index(PREP.OBJETIVO)
    D = np.full((conj.n, PD.J), np.nan, np.float32)
    diario = []
    for anio in anios:
        t0 = time.time()
        antes = conj.anio < anio
        fechas = np.unique(conj.fecha[antes])
        corte = fechas[int((1 - cfg0.frac_validacion) * len(fechas))]
        ent, val, pru = antes & (conj.fecha < corte), antes & (conj.fecha >= corte), conj.anio == anio
        conj.escalar(ent, cfg0.recorte_x)
        conj.fijar_costo(PREP.OBJETIVO, ent, cfg0.costo_comun)
        P = np.zeros((conj.n, PD.J), np.float32)
        for k, (cfg, n_sem) in enumerate(redes):
            Pe = np.zeros((conj.n, PD.J), np.float32)
            for s in range(n_sem):
                red, _ = PE.entrenar(cfg, conj, ent, val, 1000 * int(anio) + 100 * k + s, objetivo=PREP.OBJETIVO)
                p, _ = PE.predecir(red, conj, val | pru, cfg)
                Pe += np.nan_to_num(p) / n_sem
            P += Pe / max(PE.escala(Pe[val]), 1e-12) / len(redes)       # misma escala antes de promediar
            if verbose:
                print(f"    {anio} {cfg.estructura}: {n_sem} semillas, {time.time() - t0:.0f} s acumulados", flush=True)
        P[~(val | pru)] = np.nan
        u, q = PE.elegir_umbral(conj, P, val, obj, operacion["min_ops_dia"], tuple(operacion["umbrales"]))
        D[pru] = PE.discretizar(P[pru], u, q)
        m = pru & (conj.sym == obj)
        diario.append({"anio": anio, "entrena_hasta": corte, "sesiones_entren": int(ent.sum()), "umbral": u,
                       "ops_dia_objetivo": float(PE.ganancia_dura(T["RB"][m], D[m], 0.0)[1].mean()) if m.any() else 0.0,
                       "segundos": round(time.time() - t0)})
        if verbose:
            print(f"  {anio}: entrenado hasta {corte} | umbral {u} | {diario[-1]['ops_dia_objetivo']:.2f} operaciones/dia "
                  f"en {PREP.OBJETIVO} | {diario[-1]['segundos']} s", flush=True)
    return D, diario


def operaciones(T, D):
    """Posiciones discretas del objetivo -> operaciones con el contrato del laboratorio."""
    obj = T["simbolos"].index(PREP.OBJETIVO)
    filas = []
    for i in np.flatnonzero((T["sym"] == obj) & np.isfinite(D[:, 0])):
        d, j = D[i], 0
        while j < PD.J:
            if d[j] == 0:
                j += 1
                continue
            k = j
            while k + 1 < PD.J and d[k + 1] == d[j]:
                k += 1
            ini, fin = PD.horas_tramo(T["fecha"][i], j)[0], PD.horas_tramo(T["fecha"][i], k)[1]
            filas.append({"symbol": PREP.OBJETIVO, "date": T["fecha"][i], "entry_time": str(ini), "exit_time": str(fin),
                          "ret_bps": float(d[j] * T["RB"][i, j:k + 1].sum()),
                          "direction": "compra" if d[j] > 0 else "venta", "hora_decision": ini.strftime("%H:%M"),
                          "tramos": k - j + 1})
            j = k + 1
    cols = ["symbol", "date", "entry_time", "exit_time", "ret_bps", "direction", "hora_decision", "tramos"]
    return pd.DataFrame(filas, columns=cols)


def resumen_libre(ops, sesiones_por_anio, costo=8.0):
    """Tabla por anio para la exploracion (no usa el marco: son anios fuera de DEV)."""
    filas = []
    for anio, n_ses in sesiones_por_anio.items():
        o = ops[ops["date"].str[:4] == anio]
        diario = (o["ret_bps"] - costo).groupby(o["date"]).sum()
        d = np.r_[diario.values, np.zeros(max(n_ses - len(diario), 0))]          # dias sin operar = 0
        filas.append({"anio": anio, "operaciones": len(o), "ops_por_dia": len(o) / max(n_ses, 1),
                      "bruto_bps_por_op": o["ret_bps"].mean() if len(o) else np.nan,
                      "neto_8_bps_por_op": (o["ret_bps"] - costo).mean() if len(o) else np.nan,
                      "acierto_pct": 100 * (o["ret_bps"] > 0).mean() if len(o) else np.nan,
                      "sharpe_neto": d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan,
                      "t_neto": d.mean() / (d.std() / np.sqrt(len(d))) if d.std() > 0 else np.nan})
    return pd.DataFrame(filas)


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else ""
    if modo not in ("prueba", "explorar", "ejecutar"):
        raise SystemExit(__doc__)
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("Falta PyTorch (ver adv_trading/README.md).")
    redes, operacion = leer_config()
    ds = abrir()
    T = tabla(ds)
    n_obj = int((T["sym"] == T["simbolos"].index(PREP.OBJETIVO)).sum())
    print(f"Tabla: {len(T['sym'])} sesiones de {len(T['simbolos'])} simbolos ({T['fecha'].min()} -> {T['fecha'].max()}); "
          f"{PREP.OBJETIVO}: {n_obj}", flush=True)
    if modo == "prueba":
        # Dos mundos sinteticos (no se mira ningun retorno real de DEV):
        #   nulo     -> el resultado bruto debe ser compatible con 0
        #   plantado -> con una senal fuerte conocida, el resultado bruto debe salir claramente positivo
        rapido = [(c, 2) for c, _ in redes]
        for nombre, S in (("mundo nulo (debe dar ~0)", PS.plantar(T)),
                          ("senal plantada fuerte (debe dar positivo)", PS.plantar(T, "momento", PS.SENALES["momento"]))):
            D, _ = decidir(S, rapido, operacion, ANIOS_DEV[-2:], verbose=False)
            ops = operaciones(S, D)
            t = ops["ret_bps"].mean() / (ops["ret_bps"].std() / np.sqrt(max(len(ops), 2))) if len(ops) > 2 else float("nan")
            print(f"PRUEBA {nombre}: {len(ops)} operaciones en {ANIOS_DEV[-2]}-{ANIOS_DEV[-1]}, "
                  f"media bruta {ops['ret_bps'].mean():.2f} bps por operacion, t = {t:.2f}", flush=True)
        print("La tuberia funciona de principio a fin. No se ha registrado nada.")
        return
    if modo == "explorar":
        # Mismo procedimiento que `ejecutar`, pero sobre 2018 y 2019: anios anteriores a DEV que se pueden
        # mirar libremente. Sirve para decidir si merece la pena gastar la hipotesis en DEV.
        D, _ = decidir(T, redes, operacion, ANIOS_PRE)
        ops = operaciones(T, D)
        obj = T["simbolos"].index(PREP.OBJETIVO)
        n_ses = {a: int(((T["sym"] == obj) & (np.char.startswith(T["fecha"].astype(str), a))).sum()) for a in ANIOS_PRE}
        tab = resumen_libre(ops, n_ses)
        SALIDA.mkdir(exist_ok=True)
        tab.to_csv(SALIDA / "pol_exploracion_2018_2019.csv", index=False)
        print(f"\nEXPLORACION en {PREP.OBJETIVO}, {ANIOS_PRE[0]} y {ANIOS_PRE[1]} (fuera de DEV; no se registra nada):")
        print(tab.round(2).to_string(index=False))
        print("Orientacion: con dos anios, un Sharpe neto por debajo de ~1,4 no se distingue del azar.")
        return
    reg = RegistroGlobal(ds.dataset_id, path=REGISTRO, manual_path=PREP.EXCLUSIONES)
    ds.attach(reg)
    hid = reg.registrar(FAMILIA, NOMBRE, {"protocolo": PROTOCOLO, "objetivo": PREP.OBJETIVO,
                                          "config": json.loads(CONFIG.read_text(encoding="utf-8")),
                                          "hipotesis_previas_lab": reg.n_previas()},
                        f"Red de politica {PROTOCOLO}; codigo sha256 {_sha_codigo()}")
    if reg.estado(hid)["dev_evaluada"]:
        raise SystemExit(f"{NOMBRE} ({hid}) ya se evaluo en DEV. No se repite.")
    SALIDA.mkdir(exist_ok=True)
    D, diario = decidir(T, redes, operacion, ANIOS_DEV)
    ops = operaciones(T, D)
    ops.to_csv(SALIDA / "pol_operaciones_dev.csv", index=False)           # se guarda antes de evaluar
    pd.DataFrame(diario).to_csv(SALIDA / "pol_pliegues_dev.csv", index=False)
    paso, res = M.evaluar_dev(reg, ds, hid, ops)
    ses = ds.sessions("dev")
    bruto = M.metricas(M.con_costo(ops, 0.0), ses)
    print(f"Bruto por operacion {bruto['media_por_op']:.2f} bps (t {bruto['t_conservador']:.2f}) | "
          f"acierto {100 * (ops['ret_bps'] > 0).mean():.1f}% | compras {100 * (ops['direction'] == 'compra').mean():.0f}%")
    o8 = M.con_costo(ops, 8.0).assign(anio=ops["date"].str[:4])
    for clave in ("anio", "direction", "hora_decision"):
        tab = o8.groupby(clave).agg(n=("ret_bps", "size"), bruto=("ret_bps", "mean"), neto_8=("net_bps", "mean")).round(2)
        tab.to_csv(SALIDA / f"pol_desglose_{clave}.csv")
        print(tab.to_string())
    m8 = res[8.0]
    pd.DataFrame([{"hipotesis": NOMBRE, "id": hid, "paso_DEV": paso, "n_ops": m8["n_ops"],
                   "media_op_bruta": bruto["media_por_op"], "t_conserv_bruto": bruto["t_conservador"],
                   "media_op_8": m8["media_por_op"], "t_conserv_8": m8["t_conservador"], "PF_8": m8["PF"],
                   "umbral_t": reg.umbral_t(), "n_hipotesis_total": reg.n_hipotesis()}]).to_csv(
        SALIDA / "pol_resumen_dev.csv", index=False)
    print(f"\nHipotesis en total (laboratorio + este registro): {reg.n_hipotesis()} | umbral t: {reg.umbral_t():.2f}")
    print("Sube al repositorio: adv_trading/resultados/pol_*.csv y adv_trading/registro/registro_hipotesis.jsonl")


if __name__ == "__main__":
    main()
