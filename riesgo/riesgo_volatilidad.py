"""
CONTROL DE RIESGO POR VOLATILIDAD - cuanto se puede mover SOXL y que tamano de posicion llevar.

No dice si comprar o vender (eso no resulto predecible). Dice, para cada hora del dia y cada plazo,
cuanto es un movimiento normal, y de ahi salen la distancia del stop y el numero de acciones para que
cada operacion arriesgue lo mismo.

Uso (desde la raiz del repositorio):
    python riesgo/riesgo_volatilidad.py evaluar                # mide la calidad del pronostico en DEV y guarda el modelo
    python riesgo/riesgo_volatilidad.py actualizar             # baja las velas recientes de SOXL (claves de Alpaca)
    python riesgo/riesgo_volatilidad.py calcular --capital 10000 --riesgo 0.5 --stop 2

  evaluar     Usa el dataset congelado de adv_trading (hasta 2023). Ventana expansiva: cada anio se pronostica
              con un modelo ajustado solo con anios anteriores. No es una hipotesis de trading: no hay
              operaciones ni ganancias, solo se compara volatilidad prevista con la que ocurrio.
  calcular    Tabla para la PROXIMA sesion con las velas recientes: movimiento tipico (1 sigma) por hora y
              plazo, distancia del stop y acciones. --riesgo es el % del capital que aceptas perder si salta
              el stop; --stop es su distancia en sigmas.

Modelo (todo en logaritmos, minimos cuadrados):
  volatilidad del dia   HAR: la de ayer, la media de 5 dias y la de 22 dias.
  volatilidad del tramo nivel propio de cada hora + volatilidad prevista del dia
                        (+ en `evaluar`, lo ocurrido hoy hasta esa hora, para medir cuanto anade).
"""
import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(RAIZ / "adv_trading"))
import pol_datos as PD  # noqa: E402
import preparar_datos_soxl as PREP  # noqa: E402  (tambien iguala las unidades de tiempo de las herramientas de lab/)

M = PD.M
N, K_DEC = PD.N_VELAS, PD.K_DEC
PLAZOS = {"30 min": 6, "1 hora": 12, "2 horas": 24, "cierre": None}
MODELO = AQUI / "modelo_volatilidad.json"
RECIENTES = AQUI / "datos_recientes"
SALIDA = AQUI / "resultados"
HORA = lambda k: f"{(570 + 5 * k) // 60:02d}:{(570 + 5 * k) % 60:02d}"


def sesiones(barras):
    """Retornos por vela, aperturas y cierres de las sesiones completas. `barras`: velas con indice NY y `date`."""
    fechas, presentes, C, O, H, L, V, PV = PD._matrices(barras)
    ok = presentes >= PD.MIN_VELAS_DIA
    fechas, C, O = fechas[ok], C[ok], O[ok]
    r = np.diff(np.log(C), axis=1, prepend=np.log(O[:, :1]))
    return fechas, r, O, C


def tabla_dia(fechas, r, O, C):
    """Variables diarias: volatilidad realizada y sus medias previas (lo conocido antes de abrir)."""
    rv = np.sqrt((r ** 2).sum(1))
    s = pd.Series(np.log(rv))
    d = pd.DataFrame({"fecha": fechas, "lrv": s, "l1": s.shift(1), "l5": np.log(pd.Series(rv).shift(1).rolling(5).mean()),
                      "l22": np.log(pd.Series(rv).shift(1).rolling(22).mean()),
                      "gap": np.abs(np.r_[np.nan, np.log(O[1:, 0] / C[:-1, -1])])})
    return d


def ajustar_har(d, ent):
    m = ent & d[["l1", "l5", "l22", "lrv"]].notna().all(axis=1).values
    A = np.column_stack([np.ones(m.sum()), d.loc[m, ["l1", "l5", "l22"]].values])
    return np.linalg.lstsq(A, d.loc[m, "lrv"].values, rcond=None)[0]


def tabla_tramos(d, r, O, C, lhar):
    """Una fila por (sesion, hora de decision): variables conocidas a esa hora y lo que ocurrio despues."""
    r2 = r ** 2
    cum = np.cumsum(r2, axis=1)
    sig20 = np.exp(d["l22"].values)                                   # media de 22 dias: escala del simbolo
    filas = []
    for j, k in enumerate(K_DEC):
        f = pd.DataFrame({"i": np.arange(len(d)), "fecha": d["fecha"].values, "j": j, "lhar": lhar,
                          "lsig": d["l22"].values,
                          "lhoy": np.log(np.sqrt(cum[:, k - 1]) / (sig20 * np.sqrt(k / N))),
                          "l30": np.log(np.maximum(np.sqrt(cum[:, k - 1] - (cum[:, k - 7] if k >= 7 else 0)),
                                                   0.05 * sig20 * np.sqrt(6 / N)) / (sig20 * np.sqrt(6 / N))),
                          "gap": d["gap"].values / sig20})
        for nombre, h in PLAZOS.items():
            fin = N if h is None else k + h
            if fin > N:
                f[f"y_{nombre}"], f[f"ret_{nombre}"] = np.nan, np.nan
                continue
            f[f"y_{nombre}"] = 0.5 * np.log(np.maximum(r2[:, k:fin].sum(1), 1e-12))
            f[f"ret_{nombre}"] = np.log(C[:, fin - 1] / O[:, k])
        filas.append(f)
    return pd.concat(filas, ignore_index=True)


VARS = {"solo la hora": ["lsig"], "antes de abrir": ["lhar"], "con lo ocurrido hoy": ["lhar", "lhoy", "l30", "gap"]}


def _X(t, cols):
    dum = np.zeros((len(t), len(K_DEC)))
    dum[np.arange(len(t)), t["j"].values] = 1.0
    return np.column_stack([dum, t[cols].values])


def ajustar_tramo(t, ent, cols, nombre):
    m = ent & t[cols + [f"y_{nombre}"]].notna().all(axis=1).values
    X, y = _X(t[m], cols), t.loc[m, f"y_{nombre}"].values
    usar = X.any(axis=0)                                               # horas sin ese plazo (p. ej. 2 horas a las 15:00)
    beta = np.zeros(X.shape[1])
    beta[usar] = np.linalg.lstsq(X[:, usar], y, rcond=None)[0]
    pred = X @ beta
    c2 = float(np.mean(t.loc[m, f"ret_{nombre}"].values ** 2 / np.exp(2 * pred)))   # calibra: E[ret^2 / sigma^2] = 1
    return beta, c2


def evaluar(a):
    ds = PD.abrir(a.datos, a.exclusiones)
    fechas, r, O, C = sesiones(PD._velas(ds, a.simbolo, M.SPLITS["dev"][1]))
    d = tabla_dia(fechas, r, O, C)
    anio = np.array([f[:4] for f in fechas])
    anios = [y for y in sorted(set(anio)) if (anio < y).sum() >= 200]
    print(f"{a.simbolo}: {len(fechas)} sesiones completas ({fechas[0]} -> {fechas[-1]}); anios evaluados: {anios[0]}-{anios[-1]}")
    partes = []
    for y in anios:
        ent_d = anio < y
        b = ajustar_har(d, ent_d)
        lhar = b[0] + d[["l1", "l5", "l22"]].values @ b[1:]
        t = tabla_tramos(d, r, O, C, lhar)
        ent, pru = ent_d[t["i"].values], (anio == y)[t["i"].values]
        o = t.loc[pru, ["fecha", "j"] + [c for c in t.columns if c.startswith(("y_", "ret_"))]].copy()
        for modelo, cols in VARS.items():
            for nombre in PLAZOS:
                beta, c2 = ajustar_tramo(t, ent, cols, nombre)
                ok = t.loc[pru, cols].notna().all(axis=1).values
                o[f"s_{modelo}_{nombre}"] = np.where(ok, np.exp(_X(t[pru], cols) @ beta) * np.sqrt(c2), np.nan)
        # tamano fijo: una sola desviacion tipica por hora y plazo, medida en el entrenamiento
        for nombre in PLAZOS:
            fijo = t[ent].groupby("j")[f"ret_{nombre}"].std()
            o[f"s_fijo_{nombre}"] = t.loc[pru, "j"].map(fijo).values
        partes.append(o)
    P = pd.concat(partes, ignore_index=True)
    SALIDA.mkdir(exist_ok=True)
    filas = []
    for nombre in PLAZOS:
        for modelo in list(VARS) + ["fijo"]:
            s, y, ret = P[f"s_{modelo}_{nombre}"], P[f"y_{nombre}"], P[f"ret_{nombre}"]
            m = s.notna() & y.notna()
            z = (ret[m] / s[m]).values
            e = y[m].values - np.log(s[m].values)
            e = e - e.mean()                                           # el nivel ya lo fija la calibracion
            filas.append({"plazo": nombre, "modelo": modelo, "R2_log": 1 - (e ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum(),
                          "dentro_1_sigma_pct": 100 * (np.abs(z) <= 1).mean(), "dentro_2_sigma_pct": 100 * (np.abs(z) <= 2).mean(),
                          "mas_de_3_sigma_pct": 100 * (np.abs(z) > 3).mean(), "peor_caso_en_sigmas": np.abs(z).max(), "n": int(m.sum())})
    T = pd.DataFrame(filas)
    T.to_csv(SALIDA / f"evaluacion_{a.simbolo}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nCalidad del pronostico fuera de muestra (en una distribucion normal: 68 % dentro de 1 sigma, 95 % dentro de 2, 0,3 % mas alla de 3):")
    print(T.round(2).to_string(index=False))
    # Estabilidad del riesgo: con tamano fijo el riesgo real sube en dias volatiles; con tamano por volatilidad, no.
    nombre = "1 hora"
    m = P[f"s_antes de abrir_{nombre}"].notna() & P[f"ret_{nombre}"].notna()
    q = pd.qcut(P.loc[m, f"s_antes de abrir_{nombre}"], 3, labels=["dias tranquilos", "dias normales", "dias agitados"])
    est = pd.DataFrame({"tamano fijo": (P.loc[m, f"ret_{nombre}"] / P.loc[m, f"s_fijo_{nombre}"]).groupby(q, observed=True).std(),
                        "tamano por volatilidad": (P.loc[m, f"ret_{nombre}"] / P.loc[m, f"s_antes de abrir_{nombre}"]).groupby(q, observed=True).std()})
    est.to_csv(SALIDA / f"estabilidad_{a.simbolo}.csv")
    print("\nRiesgo real de una operacion de 1 hora (1,00 = el riesgo planeado):")
    print(est.round(2).to_string())
    # Modelo para `calcular`: ajustado con todo lo disponible hasta el final de DEV.
    todo = np.ones(len(d), bool)
    b = ajustar_har(d, todo)
    t = tabla_tramos(d, r, O, C, b[0] + d[["l1", "l5", "l22"]].values @ b[1:])
    modelo = {"simbolo": a.simbolo, "ajustado_hasta": str(fechas[-1]), "har": list(map(float, b)), "plazos": {}}
    for nombre in PLAZOS:
        beta, c2 = ajustar_tramo(t, np.ones(len(t), bool), VARS["antes de abrir"], nombre)
        modelo["plazos"][nombre] = {"por_hora": list(map(float, beta[:len(K_DEC)])), "pendiente": float(beta[-1]), "c2": c2}
    MODELO.write_text(json.dumps(modelo, indent=1), encoding="utf-8")
    print(f"\nModelo guardado en {MODELO.relative_to(RAIZ)} (ajustado hasta {fechas[-1]}).")


def actualizar(a):
    fin = (date.today() - timedelta(days=1)).isoformat()
    ini = (date.today() - timedelta(days=150)).isoformat()
    PREP.base.descargar([a.simbolo], ini, fin, RECIENTES)
    print(f"Velas recientes en {RECIENTES.relative_to(RAIZ)}. Ahora:  python riesgo/riesgo_volatilidad.py calcular")


def calcular(a):
    if not MODELO.exists():
        raise SystemExit("Falta el modelo. Ejecuta primero:  python riesgo/riesgo_volatilidad.py evaluar")
    mod = json.loads(MODELO.read_text(encoding="utf-8"))
    f = Path(a.recientes) / f"{mod['simbolo']}_5min.csv"
    if not f.exists():
        raise SystemExit(f"Faltan las velas recientes ({f}). Ejecuta:  python riesgo/riesgo_volatilidad.py actualizar")
    b = PREP.base.load_bars(f)
    b["date"] = b.index.strftime("%Y-%m-%d")
    fechas, r, O, C = sesiones(b)
    if len(fechas) < 25:
        raise SystemExit("Hacen falta al menos 25 sesiones completas recientes.")
    rv = np.sqrt((r ** 2).sum(1))
    h = mod["har"]
    lhar = h[0] + h[1] * np.log(rv[-1]) + h[2] * np.log(rv[-5:].mean()) + h[3] * np.log(rv[-22:].mean())
    precio = a.precio or float(C[-1, -1])
    riesgo_usd = a.capital * a.riesgo / 100
    print(f"{mod['simbolo']} | ultima sesion usada: {fechas[-1]} | precio de referencia {precio:.2f} | "
          f"capital {a.capital:,.0f} | riesgo por operacion {a.riesgo:g} % = {riesgo_usd:,.0f}")
    print(f"Volatilidad prevista para la proxima sesion: {100 * np.exp(lhar):.2f} % (media de 22 dias: {100 * rv[-22:].mean():.2f} %)\n")
    filas = []
    for j, k in enumerate(K_DEC):
        fila = {"hora ET": HORA(k)}
        for nombre, p in mod["plazos"].items():
            if p["por_hora"][j] == 0.0:
                fila[f"{nombre}: 1 sigma %"], fila[f"{nombre}: acciones"] = np.nan, np.nan
                continue
            s = np.exp(p["por_hora"][j] + p["pendiente"] * lhar) * np.sqrt(p["c2"])
            fila[f"{nombre}: 1 sigma %"] = 100 * s
            fila[f"{nombre}: acciones"] = np.floor(min(riesgo_usd / (precio * a.stop * s), a.capital / precio))   # sin apalancamiento
        filas.append(fila)
    T = pd.DataFrame(filas)
    pd.set_option("display.width", 220)
    print(T.round(2).to_string(index=False))
    print(f"\nStop a {a.stop:g} sigmas del precio de entrada. Las acciones estan calculadas para perder {riesgo_usd:,.0f} si salta,")
    print("con tope en lo que el capital permite comprar sin apalancamiento. Un stop puede saltar con hueco: alrededor")
    print("del 1 % de las veces el movimiento supera 3 sigmas.")
    print("Es una medida de riesgo, no una senal: no indica direccion.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["evaluar", "actualizar", "calcular"])
    p.add_argument("--simbolo", default=PREP.OBJETIVO)
    p.add_argument("--datos", default=str(PREP.LIMPIO))
    p.add_argument("--exclusiones", default=str(PREP.EXCLUSIONES))
    p.add_argument("--recientes", default=str(RECIENTES))
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--riesgo", type=float, default=0.5, help="%% del capital que se pierde si salta el stop")
    p.add_argument("--stop", type=float, default=2.0, help="distancia del stop en sigmas")
    p.add_argument("--precio", type=float, default=None, help="precio de entrada (por defecto, el ultimo cierre)")
    a = p.parse_args()
    {"evaluar": evaluar, "actualizar": actualizar, "calcular": calcular}[a.accion](a)


if __name__ == "__main__":
    main()
