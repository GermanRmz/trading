"""
IA - EXPERIMENTO 3: LightGBM, prediccion horaria de direccion (1 hora, 2 horas y cierre) - protocolo ia_v2.

Uso (desde cualquier carpeta):
    python ia_exp3_lgbm_horario.py prueba     # comprueba la tuberia con etiquetas barajadas; NO registra
    python ia_exp3_lgbm_horario.py ejecutar   # registra H7, H8 y H9 y las evalua en DEV (una sola vez)

DEFINICIONES FIJAS (no se cambian tras ver resultados; cualquier cambio = otra hipotesis):
  Universo    las 27 acciones del dataset (los ETF solo aportan contexto).
  Decisiones  09:45, 10:45, 11:45, 12:45, 13:45 y 14:45 ET (08:45 ... 13:45 hora del centro de Mexico),
              cada una con la informacion de las velas de 5 min ya cerradas hasta ese momento.
  Objetivo    retorno de la accion desde la APERTURA de la vela de la hora de decision hasta
                H7: 60 minutos despues | H8: 120 minutos despues | H9: el cierre de la sesion,
              dividido por la volatilidad diaria media de 20 sesiones (para juntar acciones).
              H8 no decide a las 14:45 (no caben 2 horas). Solo sesiones completas de 78 velas
              (se excluyen cierres anticipados y dias con velas ausentes al inicio).
  Modelo      LightGBM de regresion, un modelo por horizonte para todas las acciones y horas,
              hiperparametros fijos (ia_modelos.LGBM_PARAMS), promedio de 5 semillas.
  Validacion  walk-forward con ventana expansiva dentro de DEV (2021, 2022 y 2023 fuera de muestra).
  Operaciones en cada hora de decision con al menos 15 acciones validas: las 5 con mayor retorno
              esperado en valor absoluto (prediccion x volatilidad). Prediccion positiva = COMPRA,
              negativa = VENTA en corto; las 5 pueden ir todas en el mismo sentido. Nocional unitario,
              sin stops; las operaciones de horas distintas pueden solaparse.
  Costos      4, 8 y 12 bps ida y vuelta por operacion; el criterio se evalua a 8.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ia_datos as D
import ia_modelos as MOD

M = D.M
PROTOCOLO = "ia_v2"
FAMILIA = "ia_lgbm_horario"
N_OPS, MIN_ACCIONES, N_VELAS = 5, 15, 78
DECISIONES = (3, 15, 27, 39, 51, 63)            # indice de la vela de entrada: 09:45, 10:45, ... 14:45
HIPOTESIS = [("H7_lgbm_1h", "1h", 12), ("H8_lgbm_2h", "2h", 24), ("H9_lgbm_cierre", "cierre", None)]
DIARIAS = ["gap", "r_cc_1", "ret5", "ret20", "rv20", "rv_1"]
VARS = ["r_ap_n", "r_15_n", "r_60_n", "gap_n", "dist_vwap_n", "pos_rango", "rv_rel", "vol_rel", "r_cc_1_n",
        "ret5_n", "ret20_n", "rv20", "rv_1_rel", "spy_r_ap", "spy_r_15", "spy_r_60", "spy_rv_rel", "spy_dist_vwap",
        "spy_gap", "r_ap_vs_spy_n", "hora", "dow"]


def _code_sha():
    h = hashlib.sha256()
    for f in ("ia_exp3_lgbm_horario.py", "ia_modelos.py", "ia_datos.py"):
        h.update((Path(__file__).resolve().parent / f).read_bytes())
    return h.hexdigest()[:16]


def _matrices(b):
    """Velas de un simbolo como matrices sesion x 78. Huecos: cierre arrastrado, volumen 0."""
    b = b.copy()
    b["k"] = ((b.index.hour * 60 + b.index.minute) - 570) // 5
    b = b[(b["k"] >= 0) & (b["k"] < N_VELAS)]
    piv = lambda c: b.pivot(index="date", columns="k", values=c).reindex(columns=range(N_VELAS))
    C, O, H, L, V, W = (piv(c) for c in ("close", "open", "high", "low", "volume", "vwap"))
    presente = C.notna().values
    Cf = C.ffill(axis=1)
    prevC = Cf.shift(1, axis=1)
    O = O.fillna(prevC)
    H, L = H.fillna(prevC), L.fillna(prevC)
    V = V.fillna(0.0)
    W = W.fillna((H + L + Cf) / 3)
    return C.index.values, presente, Cf.values, O.values, H.values, L.values, V.values, (W * V).values


def tabla_horaria(ds):
    """Una fila por (simbolo, sesion, hora de decision): variables conocidas a esa hora y resultados y_."""
    cache = D.SALIDA / f"_tabla_horaria_{ds.dataset_id[:12]}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    diaria = D.tabla_diaria(ds).set_index(["symbol", "date"])
    filas = []
    for sym in ds.symbols:
        fechas, pres, C, O, H, L, V, PV = _matrices(ds.bars(sym, "dev"))
        completa = pres.sum(axis=1) == N_VELAS
        lc = np.log(C)
        r = np.diff(lc, axis=1, prepend=np.log(O[:, :1]))
        for k in DECISIONES:
            c, ap = C[:, k - 1], O[:, 0]
            hi, lo = np.nanmax(H[:, :k], axis=1), np.nanmin(L[:, :k], axis=1)
            vol = V[:, :k].sum(axis=1)
            d = pd.DataFrame({
                "symbol": sym, "date": fechas, "k": k, "hora": k / 12.0, "ok": completa,
                "r_ap": c / ap - 1,
                "r_15": c / (C[:, k - 4] if k >= 4 else ap) - 1,
                "r_60": c / (C[:, k - 13] if k >= 13 else ap) - 1,
                "dist_vwap": c / (PV[:, :k].sum(axis=1) / np.where(vol > 0, vol, np.nan)) - 1,
                "pos_rango": (c - lo) / np.where(hi > lo, hi - lo, np.nan),
                "rv_k": np.sqrt((r[:, :k] ** 2).sum(axis=1)), "vol_k": vol,
                "entrada": O[:, k],
                "y_1h": C[:, k + 11] / O[:, k] - 1 if k + 11 < N_VELAS else np.nan,
                "y_2h": C[:, k + 23] / O[:, k] - 1 if k + 23 < N_VELAS else np.nan,
                "y_cierre": C[:, N_VELAS - 1] / O[:, k] - 1,
            })
            # Normalizadores a la MISMA hora en sesiones anteriores (sin la actual).
            for a, b_ in (("rv_k", "rv_rel"), ("vol_k", "vol_rel")):
                d[b_] = d[a] / d[a].shift(1).rolling(D.VENTANA, min_periods=D.MIN_VENTANA).mean()
            filas.append(d)
    t = pd.concat(filas, ignore_index=True)
    t = t.join(diaria[DIARIAS], on=["symbol", "date"])
    t["es_etf"] = t["symbol"].isin(D.ETFS)
    t["dow"] = pd.to_datetime(t["date"]).dt.dayofweek
    for c in ("r_ap", "r_15", "r_60", "gap", "dist_vwap", "r_cc_1", "ret5", "ret20"):
        t[f"{c}_n"] = t[c] / t["rv20"]
    t["rv_1_rel"] = t["rv_1"] / t["rv20"]
    spy = t[t["symbol"] == "SPY"].set_index(["date", "k"])
    for c, n in (("r_ap", "spy_r_ap"), ("r_15", "spy_r_15"), ("r_60", "spy_r_60"), ("rv_rel", "spy_rv_rel"),
                 ("dist_vwap", "spy_dist_vwap"), ("gap", "spy_gap")):
        t[n] = pd.MultiIndex.from_frame(t[["date", "k"]]).map(spy[c])
    t["r_ap_vs_spy_n"] = (t["r_ap"] - t["spy_r_ap"]) / t["rv20"]
    t = t.sort_values(["date", "k", "symbol"]).reset_index(drop=True)
    t.to_pickle(cache)
    return t


def predecir(t, ycol, barajar=False):
    rng = np.random.default_rng(0)
    ok = (t["ok"] & ~t["es_etf"] & t[VARS + [ycol]].notna().all(axis=1)).values
    x = t[ok].reset_index(drop=True)
    x["y_n"] = x[ycol] / x["rv20"]
    salida, importancias = [], []
    for anio, tr, te in MOD.pliegues(x):
        ytr = x.loc[tr, "y_n"].values
        if barajar:
            ytr = rng.permutation(ytr)
        p, imp = MOD.lgbm(x.loc[tr, VARS].values, ytr, x.loc[te, VARS].values)
        o = x.loc[te, ["symbol", "date", "k", "rv20", ycol, "y_n"]].copy()
        o["pred_n"] = p
        salida.append(o)
        importancias.append(pd.Series(imp, index=VARS, name=anio))
    return pd.concat(salida, ignore_index=True), pd.concat(importancias, axis=1).mean(axis=1)


def operaciones(P, ycol, velas):
    P = P.copy()
    P["pred_bps"] = 1e4 * P["pred_n"] * P["rv20"]
    g = P.groupby(["date", "k"])["pred_bps"]
    P = P[g.transform("size") >= MIN_ACCIONES]
    T = P[P["pred_bps"].abs().groupby([P["date"], P["k"]]).rank(method="first", ascending=False) <= N_OPS].copy()
    T["direction"] = np.where(T["pred_bps"] > 0, "compra", "venta")
    T["ret_bps"] = 1e4 * T[ycol] * np.where(T["pred_bps"] > 0, 1.0, -1.0)
    ini = pd.to_datetime(T["date"]).dt.tz_localize(M.NY_TZ) + pd.to_timedelta(570 + 5 * T["k"], unit="m")
    k_fin = (T["k"] + velas) if velas else N_VELAS
    fin = pd.to_datetime(T["date"]).dt.tz_localize(M.NY_TZ) + pd.to_timedelta(570 + 5 * k_fin, unit="m")
    T["entry_time"], T["exit_time"] = ini.astype(str), fin.astype(str)
    T["hora_decision"] = ini.dt.strftime("%H:%M")
    return T[["symbol", "date", "entry_time", "exit_time", "ret_bps", "direction", "hora_decision",
              "pred_bps"]].reset_index(drop=True)


def ic(P):
    """Correlacion de rangos entre prediccion y resultado, por hora de decision (media y t entre sesiones)."""
    s = P.groupby(["date", "k"]).apply(lambda g: g["pred_n"].rank().corr(g["y_n"].rank()), include_groups=False)
    d = s.groupby("date").mean()
    return d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


def acierto_direccion(P, ycol):
    return 100 * (np.sign(P["pred_n"]) == np.sign(P[ycol])).mean()


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else ""
    if modo not in ("prueba", "ejecutar"):
        raise SystemExit(__doc__)
    ds, reg = D.abrir_lab()
    t = tabla_horaria(ds)
    ses = ds.sessions("dev")
    if modo == "prueba":
        for nombre, etiqueta, velas in HIPOTESIS:
            ycol = f"y_{etiqueta}"
            P, _ = predecir(t, ycol, barajar=True)
            T = operaciones(P, ycol, velas)
            m = M.metricas(M.con_costo(T, 0.0), ses)
            print(f"PRUEBA {nombre} (etiquetas barajadas, debe dar ~0 bruto): {len(T)} ops, {T['date'].nunique()} sesiones, "
                  f"media bruta {m['media_por_op']:.2f} bps, t {m['t_conservador']:.2f}, "
                  f"compras {100 * (T['direction'] == 'compra').mean():.0f}%, horas {sorted(T['hora_decision'].unique())}")
        return
    sha = _code_sha()
    filas = []
    for nombre, etiqueta, velas in HIPOTESIS:
        ycol = f"y_{etiqueta}"
        hid = reg.registrar(FAMILIA, nombre, {"protocolo": PROTOCOLO, "horizonte": etiqueta, "n_por_decision": N_OPS},
                            f"LightGBM horario {PROTOCOLO}; codigo sha256 {sha}")
        if reg.estado(hid)["dev_evaluada"]:
            raise SystemExit(f"{nombre} ({hid}) ya se evaluo en DEV. No se repite.")
        P, imp = predecir(t, ycol)
        T = operaciones(P, ycol, velas)
        paso, res = M.evaluar_dev(reg, ds, hid, T)
        ic_m, ic_t = ic(P)
        ac = acierto_direccion(P, ycol)
        bruto = M.metricas(M.con_costo(T, 0.0), ses)
        print(f"IC medio {ic_m:.4f} (t {ic_t:.2f}) | acierto de direccion en todas las predicciones {ac:.1f}% | "
              f"bruto por op {bruto['media_por_op']:.2f} bps (t {bruto['t_conservador']:.2f})")
        print("importancia: " + ", ".join(f"{k} {v:.0f}%" for k, v in (100 * imp / imp.sum()).nlargest(6).items()))
        T8 = M.con_costo(T, 8.0)
        for clave in ("hora_decision", "direction"):
            tab = T8.groupby(clave).agg(n=("ret_bps", "size"), bruto=("ret_bps", "mean"), neto_8=("net_bps", "mean")).round(2)
            tab.to_csv(D.SALIDA / f"exp3_desglose_{nombre}_{clave}.csv")
            print(tab.to_string())
        T8.assign(anio=T8["date"].str[:4]).groupby("anio").agg(
            n=("ret_bps", "size"), bruto=("ret_bps", "mean"), neto_8=("net_bps", "mean")).round(2).to_csv(
            D.SALIDA / f"exp3_desglose_{nombre}_anio.csv")
        T.to_csv(D.SALIDA / f"exp3_trades_{nombre}_dev.csv", index=False)
        m8 = res[8.0]
        filas.append({"hipotesis": nombre, "id": hid, "paso_DEV": paso, "n_ops": m8["n_ops"],
                      "media_op_bruta": bruto["media_por_op"], "t_conserv_bruto": bruto["t_conservador"],
                      "media_op_8": m8["media_por_op"], "t_conserv_8": m8["t_conservador"], "PF_8": m8["PF"],
                      "acierto_ops_pct": 100 * (T["ret_bps"] > 0).mean(), "acierto_direccion_pct": ac,
                      "IC": ic_m, "t_IC": ic_t, "max_simultaneas": m8["max_simultaneas"]})
    pd.DataFrame(filas).to_csv(D.SALIDA / "exp3_resumen_dev.csv", index=False)
    print(f"\nHipotesis en el registro: {reg.n_hipotesis()} | umbral t: {reg.umbral_t():.2f}")


if __name__ == "__main__":
    main()
