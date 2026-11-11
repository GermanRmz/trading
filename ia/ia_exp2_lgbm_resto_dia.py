"""
IA - EXPERIMENTO 2: LightGBM, seleccion de acciones para el resto del dia - protocolo ia_v1.

Uso (desde cualquier carpeta):
    python ia_exp2_lgbm_resto_dia.py prueba     # comprueba la tuberia con etiquetas barajadas; NO registra
    python ia_exp2_lgbm_resto_dia.py ejecutar   # registra H5 y H6 y las evalua en DEV (una sola vez)

DEFINICIONES FIJAS (no se cambian tras ver resultados; cualquier cambio = otra hipotesis):
  Universo   las 27 acciones del dataset (los 7 ETF solo aportan contexto, no se operan).
  Decision   10:30 ET, con la informacion de las 12 velas de 09:30 a 10:25.
  Objetivo   retorno de cada accion desde la APERTURA de la vela de 10:30 hasta el cierre de la
             sesion, menos la media de las acciones ese dia (retorno relativo).
  Modelo     LightGBM de regresion, un modelo conjunto para todas las acciones, hiperparametros
             fijos (ia_modelos.LGBM_PARAMS), promedio de 5 semillas. Sin busqueda de hiperparametros.
  Validacion walk-forward con ventana expansiva dentro de DEV: se predice 2021 entrenando con 2020,
             2022 con 2020-2021 y 2023 con 2020-2022. 2020 no tiene operaciones (cuenta como ceros).
  Cartera    cada sesion con al menos 15 acciones validas: largo en las 5 de mayor prediccion y
             corto en las 5 de menor. Nocional unitario por operacion. Entrada a la apertura de la
             vela de 10:30, salida al cierre de la ultima vela de la sesion. Sin stops.
  Hipotesis  H5 variables base (apertura, primera hora, dias previos y contexto de mercado).
             H6 base + regimen: volatilidad esperada (HAR) y probabilidad filtrada de regimen de
                alta volatilidad (HMM sobre SPY), ajustados solo con el entrenamiento de cada pliegue.
  Costos     4, 8 y 12 bps ida y vuelta por operacion; el criterio se evalua a 8.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ia_datos as D
import ia_modelos as MOD

M = D.M
PROTOCOLO = "ia_v1"
FAMILIA = "ia_lgbm_resto_dia"
N_LADO, MIN_ACCIONES = 5, 15
RANGOS = ["gap_n", "r_pre_n", "vol_pre_rel", "rv_pre_rel", "ret5"]
BASE = ["gap_n", "r_pre_n", "rv_pre_rel", "vol_pre_rel", "dist_vwap_n", "pos_rango", "r_cc_1", "ret5", "ret20",
        "rv20", "ult_hora_1", "r_pre_vs_spy", "gap_vs_spy", "spy_gap", "spy_r_pre", "spy_rv_pre_rel",
        "qqq_iwm_pre", "dow"] + [f"rk_{c}" for c in RANGOS]
REGIMEN = ["vol_esp_rel", "spy_vol_esp_rel", "sorpresa_vol_pre", "p_hmm"]
HIPOTESIS = [("H5_lgbm_base", "base", BASE), ("H6_lgbm_regimen", "base+regimen", BASE + REGIMEN)]


def _code_sha():
    h = hashlib.sha256()
    for f in ("ia_exp2_lgbm_resto_dia.py", "ia_modelos.py", "ia_datos.py"):
        h.update((Path(__file__).resolve().parent / f).read_bytes())
    return h.hexdigest()[:16]


def predecir(t, cols, barajar=False):
    """Predicciones fuera de muestra (walk-forward) para las acciones. Devuelve filas de prueba con `pred`."""
    rng = np.random.default_rng(0)
    salida, importancias = [], []
    for anio, tr, te in MOD.pliegues(t):
        x = t.copy()
        if any(c in cols for c in REGIMEN):
            lhar = MOD.har(x, tr)
            x["vol_esp_rel"] = lhar - np.log(x["rv20"])
            x["spy_vol_esp_rel"] = x["date"].map(x[x["symbol"] == "SPY"].set_index("date")["vol_esp_rel"])
            x["sorpresa_vol_pre"] = np.log(x["rv_pre"]) - lhar
            x["p_hmm"] = x["date"].map(MOD.hmm_regimen(x, tr))
        crudas = [c for c in cols if not c.startswith("rk_")]      # los rangos se calculan despues de filtrar
        ok = (x["ok"] & ~x["es_etf"] & x[crudas + ["y_resto"]].notna().all(axis=1)).values
        x = D.rangos_transversales(x[ok], RANGOS)
        tr, te = tr[ok], te[ok]
        x["y_rel"] = x["y_resto"] - x.groupby("date")["y_resto"].transform("mean")
        ytr = x.loc[tr, "y_rel"].values
        if barajar:
            ytr = rng.permutation(ytr)
        p, imp = MOD.lgbm(x.loc[tr, cols].values, ytr, x.loc[te, cols].values)
        o = x.loc[te, ["symbol", "date", "t_ent", "t_sal", "y_resto", "y_rel"]].copy()
        o["pred"] = p
        salida.append(o)
        importancias.append(pd.Series(imp, index=cols, name=anio))
    return pd.concat(salida, ignore_index=True), pd.concat(importancias, axis=1).mean(axis=1)


def operaciones(P):
    """Largo en las N de mayor prediccion y corto en las N de menor, por sesion."""
    P = P[P.groupby("date")["pred"].transform("size") >= MIN_ACCIONES].copy()
    P["r"] = P.groupby("date")["pred"].rank(method="first")
    n = P.groupby("date")["pred"].transform("size")
    P["direction"] = np.where(P["r"] > n - N_LADO, "largo", np.where(P["r"] <= N_LADO, "corto", ""))
    T = P[P["direction"] != ""].copy()
    T["ret_bps"] = 1e4 * T["y_resto"] * np.where(T["direction"] == "largo", 1.0, -1.0)
    T["entry_time"], T["exit_time"] = T["t_ent"].astype(str), T["t_sal"].astype(str)
    return T[["symbol", "date", "entry_time", "exit_time", "ret_bps", "direction"]].reset_index(drop=True)


def ic_diario(P):
    """Correlacion de rangos diaria entre prediccion y retorno relativo (calidad pura de la prediccion)."""
    ic = P.groupby("date").apply(lambda g: g["pred"].rank().corr(g["y_rel"].rank()), include_groups=False)
    return ic.mean(), ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))), ic


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else ""
    if modo not in ("prueba", "ejecutar"):
        raise SystemExit(__doc__)
    ds, reg = D.abrir_lab()
    t = D.tabla_diaria(ds)
    if modo == "prueba":
        P, _ = predecir(t, BASE + REGIMEN, barajar=True)
        T = operaciones(P)
        m = M.metricas(M.con_costo(T, 0.0), ds.sessions("dev"))
        print(f"PRUEBA con etiquetas barajadas (debe dar ~0 bruto): {len(T)} operaciones, "
              f"{T['date'].nunique()} sesiones, media bruta {m['media_por_op']:.2f} bps, t {m['t_conservador']:.2f}, "
              f"IC {ic_diario(P)[0]:.4f}")
        print(T.groupby("direction").size().to_string())
        return
    sha = _code_sha()
    filas = []
    for nombre, etiqueta, cols in HIPOTESIS:
        hid = reg.registrar(FAMILIA, nombre, {"protocolo": PROTOCOLO, "variables": etiqueta, "n_por_lado": N_LADO},
                            f"LightGBM resto del dia {PROTOCOLO}; codigo sha256 {sha}")
        if reg.estado(hid)["dev_evaluada"]:
            raise SystemExit(f"{nombre} ({hid}) ya se evaluo en DEV. No se repite.")
        P, imp = predecir(t, cols)
        T = operaciones(P)
        paso, res = M.evaluar_dev(reg, ds, hid, T)
        ic, t_ic, serie_ic = ic_diario(P)
        print(f"IC diario medio {ic:.4f} (t {t_ic:.2f}) | importancia: "
              + ", ".join(f"{k} {v:.0f}%" for k, v in (100 * imp / imp.sum()).nlargest(6).items()))
        T.to_csv(D.SALIDA / f"exp2_trades_{nombre}_dev.csv", index=False)
        for k, v in M.desgloses(ds, M.con_costo(T, 8.0), ds.sessions("dev")).items():
            if isinstance(v, pd.DataFrame):
                v.to_csv(D.SALIDA / f"exp2_desglose_{nombre}_{k}.csv")
                if k in ("por_anio", "por_direccion"):
                    print(v.to_string())
        m4, m8 = res[4.0], res[8.0]
        filas.append({"hipotesis": nombre, "id": hid, "paso_DEV": paso, "n_ops": m8["n_ops"],
                      "media_op_bruta": m4["media_por_op"] + 4.0, "media_op_8": m8["media_por_op"],
                      "t_conserv_8": m8["t_conservador"], "t_conserv_bruto": np.nan, "PF_8": m8["PF"],
                      "max_dd_8": m8["max_drawdown"], "IC_diario": ic, "t_IC": t_ic})
        filas[-1]["t_conserv_bruto"] = M.metricas(M.con_costo(T, 0.0), ds.sessions("dev"))["t_conservador"]
    pd.DataFrame(filas).to_csv(D.SALIDA / "exp2_resumen_dev.csv", index=False)
    print(f"\nHipotesis en el registro: {reg.n_hipotesis()} | umbral t: {reg.umbral_t():.2f}")


if __name__ == "__main__":
    main()
