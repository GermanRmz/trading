"""
RED DE POLITICA - datos: de velas de 5 min a "pasos de decision" cada 30 minutos.

Una MUESTRA es un (simbolo, sesion). Dentro de la sesion hay 12 decisiones: 10:00, 10:30, ... 15:30 ET.
En cada decision la red ve lo ocurrido hasta la vela anterior y elige una posicion (compra, nada o
venta) que se mantiene hasta la siguiente decision; la ultima se cierra al cierre de la sesion.

Para cada decision se calculan 40 variables, todas en unidades de la volatilidad diaria reciente del
propio simbolo (media de 20 sesiones), de modo que SOXL, NVDA o QQQ sean comparables:
  propias   hueco de apertura, retorno desde la apertura y desde el cierre previo, ultimos 30/60/120 min,
            distancia al VWAP, posicion en el rango del dia, distancia al maximo y al minimo, tendencia
            frente al ruido, volatilidad y volumen relativos (del dia y de los ultimos 30 min), y
            resumenes de los dias previos
  reloj     paso del dia y dia de la semana
  contexto  hueco, retorno del dia, ultimos 30 min y volatilidad relativa de tres referencias:
            mercado (SPY o VOO), tecnologia (QQQ) y semiconductores (SOXX)

Nada de esto mira el futuro: las variables de una decision usan velas ya cerradas y sesiones previas.
Los objetivos (retorno de cada tramo) empiezan en la APERTURA de la vela de la decision.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
if str(RAIZ / "lab") not in sys.path:
    sys.path.insert(0, str(RAIZ / "lab"))
import lab_marco as M  # noqa: E402

N_VELAS = 78
K_DEC = tuple(range(6, 73, 6))                 # vela de entrada de cada decision: 10:00, 10:30, ... 15:30
J = len(K_DEC)
VENTANA, MIN_VENTANA = 20, 15
MIN_VELAS_DIA = 75                             # 95% de 78, igual que la mascara del laboratorio
ROLES = ("mercado", "tec", "semis")

PROPIAS_SIGNO = ["r_gap", "r_dia", "r_cc", "r_30", "r_60", "r_120", "dvwap", "pos", "z_dia",
                 "r1", "ret5", "ret20", "r_ult1", "r_intra1"]
PROPIAS_EXTREMO = ["d_alto", "d_bajo"]
PROPIAS_SIN_SIGNO = ["rv_rel", "rv30_rel", "vol_rel", "vol30_rel", "rv1_rel", "lsig"]
RELOJ = ["paso"] + [f"dow{i}" for i in range(5)]
CONTEXTO = [f"{r}_{v}" for r in ROLES for v in ("r_gap", "r_dia", "r_30", "rv_rel")]
VARIABLES = PROPIAS_SIGNO + PROPIAS_EXTREMO + PROPIAS_SIN_SIGNO + RELOJ + CONTEXTO
F = len(VARIABLES)
# Vector de signos para el "espejo" (mundo invertido: lo que subio, bajo). Las de volatilidad, volumen y
# reloj no cambian; d_alto y d_bajo ademas se intercambian.
SIGNO = np.array([-1.0 if (v in PROPIAS_SIGNO or v in PROPIAS_EXTREMO or v.endswith(("_r_gap", "_r_dia", "_r_30")))
                  else 1.0 for v in VARIABLES], dtype=np.float32)
I_ALTO, I_BAJO = VARIABLES.index("d_alto"), VARIABLES.index("d_bajo")
CON_MEDIA_CERO = SIGNO < 0                     # variables con signo: no se centran al normalizar


def abrir(data_dir, manual=None):
    """Abre un dataset congelado con el marco del laboratorio (verifica los hashes)."""
    return M.Dataset(data_dir=Path(data_dir), manual_path=manual)


def _velas(ds, sym, hasta):
    """Velas del simbolo hasta la fecha `hasta` incluida, sin los dias excluidos. Nunca toca el tramo sellado."""
    if hasta > M.SPLITS["val"][1]:
        raise PermissionError("Este modulo no lee el tramo sellado.")
    b = ds._load(sym, False)
    b = b[b["date"] <= hasta]
    ex = ds.excluded.get(sym)
    return b[~b["date"].isin(ex)] if ex else b


def _matrices(b):
    """Matrices sesion x 78 velas. Huecos: precio arrastrado y volumen 0."""
    k = ((b.index.hour * 60 + b.index.minute) - 570) // 5
    b = b.assign(k=k)
    b = b[(b["k"] >= 0) & (b["k"] < N_VELAS)]
    piv = lambda c: b.pivot(index="date", columns="k", values=c).reindex(columns=range(N_VELAS))
    C, O, H, L, V, W = (piv(c) for c in ("close", "open", "high", "low", "volume", "vwap"))
    presentes = C.notna().sum(axis=1).values
    primera = O.bfill(axis=1).iloc[:, 0]                   # primer precio conocido del dia
    Cf = C.ffill(axis=1)
    Cf = Cf.apply(lambda col: col.fillna(primera))
    prev = Cf.shift(1, axis=1)
    prev.iloc[:, 0] = primera
    O, H, L = O.fillna(prev), H.fillna(prev), L.fillna(prev)
    V = V.fillna(0.0)
    W = W.where(W > 0).fillna((H + L + Cf) / 3)
    return C.index.values.astype(str), presentes, Cf.values, O.values, H.values, L.values, V.values, (W * V).values


def _media_previa(a):
    """Media de las VENTANA filas anteriores (sin la actual), por columna."""
    return pd.DataFrame(a).shift(1).rolling(VENTANA, min_periods=MIN_VENTANA).mean().values


def _simbolo(ds, sym, hasta, pos_sesion):
    """Variables propias, objetivos y metadatos de un simbolo. Devuelve un dict de arrays por sesion valida."""
    fechas, presentes, C, O, H, L, V, PV = _matrices(_velas(ds, sym, hasta))
    ok = presentes >= MIN_VELAS_DIA
    fechas, C, O, H, L, V, PV = fechas[ok], C[ok], O[ok], H[ok], L[ok], V[ok], PV[ok]
    n = len(fechas)
    if n < VENTANA + 5:
        return None
    ses = np.array([pos_sesion[f] for f in fechas])
    consec = np.r_[False, np.diff(ses) == 1]
    lc = np.log(C)
    r = np.diff(lc, axis=1, prepend=np.log(O[:, :1]))             # retorno de cada vela (la primera: apertura -> cierre)
    r2 = r ** 2
    rv_d = np.sqrt(r2.sum(axis=1))
    sig = _media_previa(rv_d[:, None])[:, 0]                       # volatilidad diaria tipica, conocida antes de abrir
    c_prev = np.r_[np.nan, C[:-1, -1]]
    cc = np.log(C[:, -1] / c_prev)                                 # cierre a cierre
    cc_s = pd.Series(cc)
    cumr2, cumv, cumpv = np.cumsum(r2, axis=1), np.cumsum(V, axis=1), np.cumsum(PV, axis=1)
    hmax, lmin = np.maximum.accumulate(H, axis=1), np.minimum.accumulate(L, axis=1)
    raiz = lambda nb: np.sqrt(nb / N_VELAS)

    X = np.full((n, J, len(PROPIAS_SIGNO) + 2 + len(PROPIAS_SIN_SIGNO)), np.nan)
    rv_cum = np.column_stack([np.sqrt(cumr2[:, k - 1]) for k in K_DEC])
    v_cum = np.column_stack([cumv[:, k - 1] for k in K_DEC])
    v_30 = np.column_stack([cumv[:, k - 1] - (cumv[:, k - 7] if k >= 7 else 0.0) for k in K_DEC])
    rv_esp, v_esp, v30_esp = _media_previa(rv_cum), _media_previa(v_cum), _media_previa(v_30)
    eps = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.log(O[:, 0] / c_prev)
        r1 = cc_s.shift(1).values / sig
        ret5 = cc_s.shift(1).rolling(5, min_periods=4).sum().values / (sig * np.sqrt(5))
        ret20 = cc_s.shift(1).rolling(VENTANA, min_periods=MIN_VENTANA).sum().values / (sig * np.sqrt(VENTANA))
        rv1_rel = np.log(np.r_[np.nan, rv_d[:-1]] / sig)
        r_ult1 = np.r_[np.nan, np.log(C[:-1, -1] / O[:-1, 72])] / (sig * raiz(6))
        r_intra1 = np.r_[np.nan, np.log(C[:-1, -1] / O[:-1, 0])] / sig
        for j, k in enumerate(K_DEC):
            p = C[:, k - 1]
            ref = lambda nb: (C[:, k - 1 - nb] if k - 1 - nb >= 0 else O[:, 0], min(nb, k))
            cols = {}
            cols["r_gap"] = gap / sig
            cols["r_dia"] = np.log(p / O[:, 0]) / sig
            cols["r_cc"] = cols["r_gap"] + cols["r_dia"]
            for nombre, nb in (("r_30", 6), ("r_60", 12), ("r_120", 24)):
                base_p, nn = ref(nb)
                cols[nombre] = np.log(p / base_p) / (sig * raiz(nn))
            cols["dvwap"] = np.log(p / (cumpv[:, k - 1] / np.maximum(cumv[:, k - 1], eps))) / sig
            rango = hmax[:, k - 1] - lmin[:, k - 1]
            cols["pos"] = np.where(rango > 0, 2 * (p - lmin[:, k - 1]) / np.where(rango > 0, rango, 1.0) - 1, 0.0)
            cols["z_dia"] = np.log(p / O[:, 0]) / np.maximum(rv_cum[:, j], eps)
            cols["r1"], cols["ret5"], cols["ret20"] = r1, ret5, ret20
            cols["r_ult1"], cols["r_intra1"] = r_ult1, r_intra1
            cols["d_alto"] = np.log(p / hmax[:, k - 1]) / sig
            cols["d_bajo"] = np.log(p / lmin[:, k - 1]) / sig
            cols["rv_rel"] = np.log(np.maximum(rv_cum[:, j], eps) / rv_esp[:, j])
            rv30 = np.sqrt(cumr2[:, k - 1] - (cumr2[:, k - 7] if k >= 7 else 0.0))
            cols["rv30_rel"] = np.log(np.maximum(rv30, 0.05 * sig * raiz(6)) / (sig * raiz(6)))
            cols["vol_rel"] = np.log(np.maximum(v_cum[:, j], 1.0) / v_esp[:, j])
            cols["vol30_rel"] = np.log(np.maximum(v_30[:, j], 1.0) / np.maximum(v30_esp[:, j], 1.0))
            cols["rv1_rel"], cols["lsig"] = rv1_rel, np.log(sig)
            X[:, j, :] = np.column_stack([cols[v] for v in PROPIAS_SIGNO + PROPIAS_EXTREMO + PROPIAS_SIN_SIGNO])
        # Objetivos: retorno de cada tramo desde la APERTURA de la vela de decision.
        ent = np.column_stack([O[:, k] for k in K_DEC])
        sal = np.column_stack([O[:, k] for k in K_DEC[1:]] + [C[:, -1]])
        seg = np.log(sal / ent)
        lim = list(K_DEC) + [N_VELAS]
        rv_seg = np.column_stack([np.sqrt(r2[:, lim[j]:lim[j + 1]].sum(axis=1)) for j in range(J)])
        nb_seg = np.array([lim[j + 1] - lim[j] for j in range(J)])
        vt = np.log(np.maximum(rv_seg, 0.05 * sig[:, None] * raiz(nb_seg)) / (sig[:, None] * raiz(nb_seg)))
    valida = consec & np.isfinite(X).all(axis=(1, 2)) & np.isfinite(seg).all(axis=1) & np.isfinite(sig) & (sig > 0)
    ctx = np.stack([X[:, :, VARIABLES.index(v)] for v in ("r_gap", "r_dia", "r_30", "rv_rel")], axis=2)
    return {"fechas": fechas, "X": X, "seg": seg, "vt": vt, "sig": sig, "valida": valida, "ctx": ctx}


def construir(ds, simbolos, contexto, hasta, verbose=True):
    """Tabla de pasos para `simbolos`. `contexto`: {"mercado": "SPY", "tec": "QQQ", "semis": "SOXX"}.

    Devuelve un dict con X [N, 12, 40], Y (retorno normalizado de cada tramo), RB (retorno en bps),
    VT (objetivo de volatilidad), sig, sym (indice en `simbolos`), fecha y prev (muestra anterior del simbolo).
    """
    todas = sorted({d for s in set(simbolos) | set(contexto.values())
                    for d in np.unique(_velas(ds, s, hasta)["date"].values)})
    pos_sesion = {d: i for i, d in enumerate(todas)}
    crudo = {s: _simbolo(ds, s, hasta, pos_sesion) for s in sorted(set(simbolos) | set(contexto.values()))}
    partes = []
    for si, sym in enumerate(simbolos):
        d = crudo[sym]
        if d is None:
            continue
        n = len(d["fechas"])
        valida = d["valida"].copy()
        bloques = []
        for rol in ROLES:
            c = crudo[contexto[rol]]
            idx = pd.Series(np.arange(len(c["fechas"])), index=c["fechas"]).reindex(d["fechas"]).values
            falta = np.isnan(idx)
            idx = np.where(falta, 0, idx).astype(int)
            valida &= ~falta & c["valida"][idx]
            bloques.append(c["ctx"][idx])
        dow = pd.to_datetime(d["fechas"]).dayofweek.values
        reloj = np.zeros((n, J, len(RELOJ)))
        reloj[:, :, 0] = np.linspace(0.0, 1.0, J)[None, :]
        reloj[np.arange(n), :, 1 + np.clip(dow, 0, 4)] = 1.0
        X = np.concatenate([d["X"], reloj] + bloques, axis=2)
        valida &= np.isfinite(X).all(axis=(1, 2))
        i = np.flatnonzero(valida)
        partes.append({"X": X[i], "seg": d["seg"][i], "vt": d["vt"][i], "sig": d["sig"][i],
                       "sym": np.full(len(i), si), "fecha": d["fechas"][i]})
        if verbose:
            print(f"  {sym}: {len(i)} sesiones validas ({d['fechas'][i][0]} -> {d['fechas'][i][-1]})", flush=True)
    cat = lambda k: np.concatenate([p[k] for p in partes])
    X, seg, sig = cat("X"), cat("seg"), cat("sig")
    sym, fecha = cat("sym"), cat("fecha")
    prev = np.full(len(sym), -1)
    mismo = np.r_[False, sym[1:] == sym[:-1]]
    prev[mismo] = np.flatnonzero(mismo) - 1
    return {"X": np.clip(X, -8, 8).astype(np.float32),
            "Y": np.clip(seg / sig[:, None], -5, 5).astype(np.float32),
            "RB": (1e4 * seg).astype(np.float32), "VT": np.clip(cat("vt"), -4, 4).astype(np.float32),
            "sig": sig.astype(np.float32), "sym": sym.astype(np.int64), "fecha": fecha, "prev": prev,
            "simbolos": list(simbolos), "variables": list(VARIABLES)}


def espejo(X, Y):
    """Mundo invertido: cambia el signo de las variables direccionales y de los objetivos."""
    Xe = X * SIGNO
    Xe[..., I_ALTO], Xe[..., I_BAJO] = -X[..., I_BAJO], -X[..., I_ALTO]
    return Xe, -Y


def horas_tramo(fecha, j):
    """Marcas de entrada y salida (hora de Nueva York) del tramo j de una sesion."""
    base_ts = pd.Timestamp(fecha, tz=M.NY_TZ) + pd.Timedelta(minutes=570)
    lim = list(K_DEC) + [N_VELAS]
    return base_ts + pd.Timedelta(minutes=5 * lim[j]), base_ts + pd.Timedelta(minutes=5 * lim[j + 1])
