"""
IA - EXPERIMENTO 1: pronostico de la volatilidad del dia siguiente (solo DEV).

Pregunta: con datos hasta el cierre de ayer, quien pronostica mejor la volatilidad realizada de
hoy (raiz de la suma de retornos de 5 min al cuadrado, sesion regular)?

  ingenuo   la volatilidad de ayer
  HAR       regresion con la volatilidad de ayer, la media de 5 dias y la de 22 dias (por simbolo)
  GARCH     GARCH(1,1) sobre retornos diarios cierre a cierre, recalibrado a la escala intradia
  LightGBM  modelo conjunto de todos los simbolos con esas mismas piezas y contexto de SPY
  +HMM      el mismo LightGBM anadiendo la probabilidad filtrada de regimen de alta volatilidad
  mezcla    promedio de HAR y LightGBM

Validacion: walk-forward con ventana expansiva (se prueba 2021, 2022 y 2023 entrenando con los
anios anteriores). No es una hipotesis de trading: no genera operaciones ni toca el registro.

Uso:  python ia_exp1_volatilidad.py
"""
import numpy as np
import pandas as pd

import ia_datos as D
import ia_modelos as MOD

# El objetivo es log rv menos su media de 22 dias, asi que las entradas van en la misma escala relativa
# (un arbol no puede restar dos columnas por su cuenta).
BASE = ["d1", "d5", "l22", "r_cc_1", "abs_r1", "ret5", "gap_abs_1", "vol_rel_1", "spy_d1", "spy_d5", "spy_l22",
        "dow", "dgarch"]


def preparar(t):
    t = t.copy()
    t["y"] = np.log(t["rv"])
    t["l1"], t["l5"], t["l22"] = np.log(t["rv_1"]), np.log(t["rv_5"]), np.log(t["rv_22"])
    g = t.groupby("symbol")
    t["abs_r1"] = t["r_cc_1"].abs()
    t["gap_abs_1"] = g["gap"].shift(1).abs()
    t["vol_rel_1"] = g["vol"].shift(1) / g["vol"].transform(lambda x: x.shift(1).rolling(20, min_periods=15).mean())
    t["d1"], t["d5"] = t["l1"] - t["l22"], t["l5"] - t["l22"]
    spy = t[t["symbol"] == "SPY"].set_index("date")
    for c in ("d1", "d5", "l22"):
        t[f"spy_{c}"] = t["date"].map(spy[c])
    return t


def metricas(y, f):
    """y, f en log de volatilidad. QLIKE sobre varianza; R2 respecto a la media de la muestra evaluada."""
    e = y - f
    ratio = np.exp(2 * e)
    return {"n": len(y), "R2_log": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "RMSE_log": np.sqrt((e ** 2).mean()), "QLIKE": (ratio - np.log(ratio) - 1).mean()}


def main():
    ds, _ = D.abrir_lab()
    t = preparar(D.tabla_diaria(ds))
    preds = []
    importancias = []
    for anio, tr, te in MOD.pliegues(t):
        t["lhar"] = MOD.har(t, tr)
        t["lgarch_raw"] = MOD.garch(t, tr)
        # GARCH pronostica volatilidad cierre a cierre; se lleva a la escala intradia con el entrenamiento.
        f = tr & t[["lgarch_raw", "y"]].notna().all(axis=1).values
        b = np.polyfit(t.loc[f, "lgarch_raw"], t.loc[f, "y"], 1)
        t["lgarch"] = np.polyval(b, t["lgarch_raw"])
        t["dgarch"] = t["lgarch"] - t["l22"]
        t["p_hmm"] = t["date"].map(MOD.hmm_regimen(t, tr))
        cols_ok = t[BASE + ["p_hmm", "lhar", "y"]].notna().all(axis=1).values
        a, b_ = tr & cols_ok, te & cols_ok
        out = t.loc[b_, ["symbol", "date", "y"]].copy()
        out["ingenuo"], out["HAR"], out["GARCH"] = t.loc[b_, "l1"], t.loc[b_, "lhar"], t.loc[b_, "lgarch"]
        # El objetivo del LightGBM se normaliza con la media de 22 dias para poder juntar simbolos.
        ytr = (t.loc[a, "y"] - t.loc[a, "l22"]).values
        p, imp = MOD.lgbm(t.loc[a, BASE].values, ytr, t.loc[b_, BASE].values)
        out["LightGBM"] = p + t.loc[b_, "l22"].values
        p2, _ = MOD.lgbm(t.loc[a, BASE + ["p_hmm"]].values, ytr, t.loc[b_, BASE + ["p_hmm"]].values)
        out["LightGBM+HMM"] = p2 + t.loc[b_, "l22"].values
        out["mezcla HAR+LightGBM"] = (out["HAR"] + out["LightGBM"]) / 2
        preds.append(out)
        importancias.append(pd.Series(imp, index=BASE, name=anio))
        print(f"{anio}: entrenamiento {a.sum()} filas, prueba {b_.sum()} filas")
    P = pd.concat(preds, ignore_index=True)
    modelos = [c for c in P.columns if c not in ("symbol", "date", "y")]
    total = pd.DataFrame({m: metricas(P["y"], P[m]) for m in modelos}).T
    por_anio = pd.DataFrame({(m, a): metricas(g["y"], g[m]) for m in modelos
                             for a, g in P.groupby(P["date"].str[:4])}).T["QLIKE"].unstack()
    total = total.join(por_anio.add_prefix("QLIKE_"))
    print("\n=== Volatilidad del dia siguiente, fuera de muestra 2021-2023 (menor QLIKE = mejor) ===")
    print(total.round(4).to_string())
    imp = pd.concat(importancias, axis=1).mean(axis=1)
    print("\nImportancia en LightGBM (% de ganancia):")
    print((100 * imp / imp.sum()).sort_values(ascending=False).round(1).to_string())
    total.to_csv(D.SALIDA / "exp1_volatilidad_resumen.csv")
    P.to_csv(D.SALIDA / "exp1_volatilidad_predicciones.csv", index=False)


if __name__ == "__main__":
    main()
