"""
IA - piezas de modelado compartidas: HAR (volatilidad), GARCH, HMM de regimen y LightGBM.

Todas las funciones reciben que sesiones son de entrenamiento y solo ajustan parametros con ellas.
Los hiperparametros estan FIJOS a proposito: ajustarlos mirando resultados serian mas hipotesis.
"""
import warnings

import numpy as np
import pandas as pd

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, verbose=-1)
SEMILLAS = (11, 22, 33, 44, 55)
ANIOS_PRUEBA = ("2021", "2022", "2023")      # walk-forward dentro de DEV: se entrena con los anios anteriores


def pliegues(t):
    """Devuelve [(anio, mascara_entrenamiento, mascara_prueba)] con ventana expansiva."""
    a = t["date"].str[:4]
    return [(y, (a < y).values, (a == y).values) for y in ANIOS_PRUEBA]


def har(t, tr):
    """HAR-RV por simbolo en logaritmos: log rv_d ~ log rv_{d-1} + log media5 + log media22."""
    out = pd.Series(np.nan, index=t.index)
    X = np.log(t[["rv_1", "rv_5", "rv_22"]])
    y = np.log(t["rv"])
    ok = X.notna().all(axis=1) & y.notna()
    for sym, idx in t.groupby("symbol").groups.items():
        m = pd.Series(False, index=t.index)
        m.loc[idx] = True
        f = (m & ok & tr).values
        if f.sum() < 60:
            continue
        A = np.column_stack([np.ones(f.sum()), X[f].values])
        beta = np.linalg.lstsq(A, y[f].values, rcond=None)[0]
        p = (m & X.notna().all(axis=1)).values
        out[p] = np.column_stack([np.ones(p.sum()), X[p].values]) @ beta
    return out          # pronostico de log rv


def garch(t, tr):
    """GARCH(1,1) por simbolo sobre retornos cierre a cierre. Parametros estimados solo con
    entrenamiento; despues se filtra hacia adelante con parametros fijos. Devuelve log sigma
    pronosticada para cada sesion con datos hasta la anterior."""
    from arch import arch_model
    out = pd.Series(np.nan, index=t.index)
    trs = pd.Series(tr, index=t.index)
    for sym, g in t.groupby("symbol"):
        g = g[g["r_cc"].notna()]
        n_tr = int(trs[g.index].sum())
        if n_tr < 150:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(100 * g["r_cc"].values, mean="Zero", vol="GARCH", p=1, q=1).fit(
                last_obs=n_tr, disp="off")
            h = res.forecast(start=0, horizon=1, reindex=True).variance.values[:, 0]
        sig = np.r_[np.nan, np.sqrt(h[:-1])] / 100.0       # el pronostico hecho en d-1 vale para d
        out[g.index] = np.log(sig)
    return out


def hmm_regimen(t, tr, n_estados=2, semilla=7):
    """HMM gaussiano sobre SPY diario [retorno cierre a cierre, log rv]. Ajustado solo con
    entrenamiento; probabilidad FILTRADA del estado de mayor volatilidad, retrasada una sesion.
    Devuelve una Serie indexada por fecha."""
    from hmmlearn.hmm import GaussianHMM
    from ia_datos import filtrado_hmm
    s = t[t["symbol"] == "SPY"].copy()
    s["tr"] = np.asarray(tr)[s.index]
    s = s[s["r_cc"].notna()].reset_index(drop=True)
    X = np.column_stack([s["r_cc"].values, np.log(s["rv"].values)])
    mu, sd = X[s["tr"].values].mean(0), X[s["tr"].values].std(0)
    Z = (X - mu) / sd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = GaussianHMM(n_components=n_estados, covariance_type="full", n_iter=200, random_state=semilla)
        m.fit(Z[s["tr"].values])
    alto = int(np.argmax(m.means_[:, 1]))                  # estado con mayor volatilidad media
    p = filtrado_hmm(m, Z)[:, alto]
    return pd.Series(p, index=s["date"]).shift(1)          # lo conocido antes de abrir la sesion


def lgbm(Xtr, ytr, Xte):
    """Promedio de varias semillas con hiperparametros fijos."""
    import lightgbm as lgb
    pred = np.zeros(len(Xte))
    imp = np.zeros(Xtr.shape[1])
    for s in SEMILLAS:
        m = lgb.LGBMRegressor(random_state=s, **LGBM_PARAMS).fit(Xtr, ytr)
        pred += m.predict(Xte) / len(SEMILLAS)
        imp += m.booster_.feature_importance("gain") / len(SEMILLAS)
    return pred, imp
