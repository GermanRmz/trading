"""
IA - EXPERIMENTO 4: red profunda sobre secuencias de velas (TCN + atencion) - protocolo ia_v3.

Pensado para ejecutarse en tu computadora (usa la tarjeta grafica si PyTorch la detecta; si no, CPU).

    pip install torch            # para GPU NVIDIA, usa el comando que da https://pytorch.org/get-started/locally/
    python ia_exp4_red_secuencias.py prueba     # tuberia con etiquetas barajadas, corta; NO registra
    python ia_exp4_red_secuencias.py ejecutar   # registra H10 y la evalua en DEV (una sola vez)

DEFINICIONES FIJAS (no se cambian tras ver resultados; cualquier cambio = otra hipotesis):
  Universo, decisiones, validacion, regla de operacion y costos: IDENTICOS a H9 del experimento 3
              (27 acciones; 09:45 ... 14:45 ET; walk-forward 2021-2023; en cada decision las 5 acciones
              con mayor retorno esperado en valor absoluto, compra si es positivo y venta si es negativo;
              salida al cierre). Lo unico que cambia es el modelo, para compararlo con LightGBM.
  Entrada     (a) secuencia de las ultimas 78 velas de 5 min ya cerradas (lo que va de hoy + el final de
              ayer), con 9 canales por vela: retorno, gap de apertura, rango, volumen relativo, desvio
              del VWAP, hora, marca de "hoy", y retorno y rango de SPY en la misma vela;
              (b) las 22 variables resumidas del experimento 3.
              Requiere que la sesion de hoy y la anterior esten completas (78 velas) y sean consecutivas.
  Modelo      5 bloques convolucionales temporales (dilataciones 1,2,4,8,16; 96 canales) + 2 capas
              Transformer (4 cabezas) + agregacion por atencion; se une con las variables resumidas y
              salen 3 predicciones: retorno a 1 hora, a 2 horas y al cierre (aprendizaje multitarea).
              Solo la prediccion AL CIERRE se usa para operar.
  Entrenam.   AdamW (lr 1e-3, decaimiento 1e-4), perdida Huber, lotes de 512, hasta 20 epocas con parada
              temprana (paciencia 3) usando como validacion interna el ultimo 15% de las fechas de
              entrenamiento de cada pliegue. Promedio de 3 semillas.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ia_datos as D
import ia_exp3_lgbm_horario as E
import ia_modelos as MOD

M = D.M
PROTOCOLO = "ia_v3"
FAMILIA = "ia_red_secuencias"
NOMBRE = "H10_tcn_atencion_cierre"
L, CANALES = E.N_VELAS, 9
OBJETIVOS = ["y_1h", "y_2h", "y_cierre"]
CFG = dict(canales=96, dilataciones=(1, 2, 4, 8, 16), capas_transformer=2, cabezas=4, dropout=0.2,
           lr=1e-3, decaimiento=1e-4, lote=512, epocas=20, paciencia=3, frac_validacion=0.15, semillas=(1, 2, 3))


def _code_sha():
    h = hashlib.sha256()
    for f in ("ia_exp4_red_secuencias.py", "ia_exp3_lgbm_horario.py", "ia_modelos.py", "ia_datos.py"):
        h.update((Path(__file__).resolve().parent / f).read_bytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ secuencias
def _dias(ds, sym, rv20, pos):
    """Canales propios por sesion de un simbolo: array [sesiones, 78, 5] y validez de cada sesion."""
    fechas, pres, C, O, H, Lo, V, PV = E._matrices(ds.bars(sym, "dev"))
    n = len(fechas)
    completa = pres.sum(axis=1) == L
    i = np.array([pos[f] for f in fechas])
    consec = np.r_[False, np.diff(i) == 1]
    sig = pd.Series(fechas).map(rv20).values / np.sqrt(L)             # volatilidad tipica de una vela
    prevC = np.r_[np.nan, C[:-1, -1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(C), axis=1, prepend=np.log(O[:, :1])) / sig[:, None]
        gap = np.zeros((n, L))
        gap[:, 0] = np.where(consec, np.log(O[:, 0] / prevC), 0.0) / sig
        rango = np.log(H / Lo) / sig[:, None]
        vol_d = pd.Series(V.sum(axis=1)).shift(1).rolling(D.VENTANA, min_periods=D.MIN_VENTANA).mean().values
        vol = np.log1p(V / (vol_d[:, None] / L))
        vwap = np.where(V > 0, PV / np.where(V > 0, V, 1.0), C)
        dv = (vwap / C - 1) / sig[:, None]
    X = np.stack([r, gap, rango, vol, dv], axis=2)
    valida = completa & np.isfinite(X).all(axis=(1, 2))
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -20, 20).astype(np.float32)
    return fechas, X, valida, consec


def construir(ds, x):
    """Devuelve (S [n,78,9], indices de las filas de `x` que tienen secuencia valida)."""
    diaria = D.tabla_diaria(ds)
    pos = {s: i for i, s in enumerate(ds.sessions("dev"))}
    rv = {s: g.set_index("date")["rv20"] for s, g in diaria.groupby("symbol")}
    f_spy, X_spy, v_spy, _ = _dias(ds, "SPY", rv["SPY"], pos)
    spy = {f: (X_spy[j][:, [0, 2]], v_spy[j]) for j, f in enumerate(f_spy)}
    hora = (np.arange(L, dtype=np.float32) / L)[None, :, None]
    partes, idx = [], []
    for sym, g in x.groupby("symbol"):
        fechas, X, valida, consec = _dias(ds, sym, rv[sym], pos)
        S = np.stack([spy[f][0] if f in spy else np.zeros((L, 2), np.float32) for f in fechas])
        valida = valida & np.array([f in spy and spy[f][1] for f in fechas])
        X = np.concatenate([X, np.broadcast_to(hora, (len(fechas), L, 1)), np.zeros((len(fechas), L, 1), np.float32), S],
                           axis=2).astype(np.float32)               # canales: 0-4 propios, 5 hora, 6 hoy, 7-8 SPY
        d = pd.Series(np.arange(len(fechas)), index=fechas).reindex(g["date"]).values
        k = g["k"].values
        ok = ~np.isnan(d)
        d = np.where(ok, d, 1).astype(int)
        ok &= (d >= 1) & valida[d] & valida[np.maximum(d - 1, 0)] & consec[d]
        d, k, filas = d[ok], k[ok], g.index.values[ok]
        dos = np.concatenate([X[d - 1], X[d]], axis=1)              # ayer + hoy: 156 velas
        dos[:, L:, 6] = 1.0
        ventana = dos[np.arange(len(d))[:, None], k[:, None] + np.arange(L)[None, :]]   # termina en la vela k-1 de hoy
        partes.append(ventana)
        idx.append(filas)
    return np.concatenate(partes), np.concatenate(idx)


# ---------------------------------------------------------------------- modelo
def crear_modelo(n_est):
    import torch
    from torch import nn

    class Bloque(nn.Module):
        def __init__(self, c, dil, p):
            super().__init__()
            self.c1 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil)
            self.c2 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil)
            self.n1, self.n2, self.dp = nn.BatchNorm1d(c), nn.BatchNorm1d(c), nn.Dropout(p)

        def forward(self, x):
            h = self.dp(torch.relu(self.n1(self.c1(x))))
            return torch.relu(x + self.dp(self.n2(self.c2(h))))

    class Red(nn.Module):
        def __init__(self):
            super().__init__()
            c, p = CFG["canales"], CFG["dropout"]
            self.entrada = nn.Conv1d(CANALES, c, 1)
            self.tcn = nn.Sequential(*[Bloque(c, d, p) for d in CFG["dilataciones"]])
            self.pos = nn.Parameter(torch.zeros(1, L, c))
            capa = nn.TransformerEncoderLayer(c, CFG["cabezas"], 2 * c, p, batch_first=True, norm_first=True)
            self.trans = nn.TransformerEncoder(capa, CFG["capas_transformer"], enable_nested_tensor=False)
            self.aten = nn.Linear(c, 1)
            self.est = nn.Sequential(nn.Linear(n_est, 64), nn.ReLU(), nn.Dropout(p))
            self.salida = nn.Sequential(nn.Linear(2 * c + 64, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, len(OBJETIVOS)))

        def forward(self, s, e):
            h = self.tcn(self.entrada(s.transpose(1, 2))).transpose(1, 2)
            h = self.trans(h + self.pos)
            w = torch.softmax(self.aten(h), dim=1)
            return self.salida(torch.cat([(w * h).sum(1), h[:, -1], self.est(e)], dim=1))

    return Red()


def entrenar_y_predecir(Str, Etr, Ytr, fechas_tr, Ste, Ete, semilla, dispositivo, epocas, verbose=""):
    import torch
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    corte = np.sort(np.unique(fechas_tr))[int((1 - CFG["frac_validacion"]) * len(np.unique(fechas_tr)))]
    a, v = fechas_tr < corte, fechas_tr >= corte
    T = lambda z: torch.from_numpy(np.ascontiguousarray(z))
    red = crear_modelo(Etr.shape[1]).to(dispositivo)
    opt = torch.optim.AdamW(red.parameters(), lr=CFG["lr"], weight_decay=CFG["decaimiento"])
    perdida = torch.nn.HuberLoss(reduction="none")

    def lotes(S, Es, Y=None, barajar=False):
        orden = np.random.permutation(len(S)) if barajar else np.arange(len(S))
        for i in range(0, len(S), CFG["lote"]):
            j = orden[i:i + CFG["lote"]]
            yield (T(S[j]).to(dispositivo), T(Es[j]).to(dispositivo), None if Y is None else T(Y[j]).to(dispositivo))

    def error(S, Es, Y):
        red.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for s, e, y in lotes(S, Es, Y):
                m = ~torch.isnan(y)
                tot += (perdida(red(s, e), torch.nan_to_num(y)) * m).sum().item()
                n += m.sum().item()
        return tot / max(n, 1)

    mejor, estado, espera = np.inf, None, 0
    for ep in range(epocas):
        red.train()
        t0 = time.time()
        for s, e, y in lotes(Str[a], Etr[a], Ytr[a], barajar=True):
            m = ~torch.isnan(y)                                    # H8 no tiene objetivo a las 14:45
            loss = (perdida(red(s, e), torch.nan_to_num(y)) * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(red.parameters(), 1.0)
            opt.step()
        ev = error(Str[v], Etr[v], Ytr[v])
        print(f"   {verbose} epoca {ep + 1}: validacion interna {ev:.5f} ({time.time() - t0:.0f} s)", flush=True)
        if ev < mejor - 1e-6:
            mejor, espera = ev, 0
            estado = {k_: v_.detach().cpu().clone() for k_, v_ in red.state_dict().items()}
        else:
            espera += 1
            if espera >= CFG["paciencia"]:
                break
    red.load_state_dict(estado)
    red.eval()
    with torch.no_grad():
        return np.concatenate([red(s, e).cpu().numpy() for s, e, _ in lotes(Ste, Ete)])


def predecir(ds, t, barajar=False, rapido=False):
    import torch
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    print(f"Dispositivo: {dispositivo}" + (f" ({torch.cuda.get_device_name(0)})" if dispositivo == "cuda" else
                                          f" ({torch.get_num_threads()} hilos)"), flush=True)
    ok = (t["ok"] & ~t["es_etf"] & t[E.VARS + ["y_cierre"]].notna().all(axis=1)).values
    x = t[ok].reset_index(drop=True)
    S, idx = construir(ds, x)
    x = x.loc[idx].reset_index(drop=True)
    print(f"Muestras con secuencia valida: {len(x)} de {int(ok.sum())}", flush=True)
    Y = np.clip((x[OBJETIVOS].values / x[["rv20"]].values), -5, 5).astype(np.float32)
    rng = np.random.default_rng(0)
    salida = []
    for anio, tr, te in MOD.pliegues(x):
        if rapido:                                               # solo para la prueba: un anio de datos por pliegue
            tr = tr & (x["date"].str[:4] == str(int(anio) - 1)).values
        mu, sd = x.loc[tr, E.VARS].mean().values, x.loc[tr, E.VARS].std().values
        Est = np.clip((x[E.VARS].values - mu) / np.where(sd > 0, sd, 1.0), -5, 5).astype(np.float32)
        Ytr = Y[tr]
        if barajar:
            Ytr = Ytr[rng.permutation(len(Ytr))]
        pred = np.zeros(int(te.sum()))
        semillas = CFG["semillas"][:1] if rapido else CFG["semillas"]
        for s in semillas:
            p = entrenar_y_predecir(S[tr], Est[tr], Ytr, x.loc[tr, "date"].values, S[te], Est[te], s, dispositivo,
                                    2 if rapido else CFG["epocas"], verbose=f"{anio} semilla {s}")
            pred += p[:, OBJETIVOS.index("y_cierre")] / len(semillas)
        o = x.loc[te, ["symbol", "date", "k", "rv20", "y_cierre"]].copy()
        o["y_n"], o["pred_n"] = o["y_cierre"] / o["rv20"], pred
        salida.append(o)
    return pd.concat(salida, ignore_index=True)


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else ""
    if modo not in ("prueba", "ejecutar"):
        raise SystemExit(__doc__)
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("Falta PyTorch: pip install torch (para GPU, ver https://pytorch.org/get-started/locally/)")
    ds, reg = D.abrir_lab()
    t = E.tabla_horaria(ds)
    ses = ds.sessions("dev")
    if modo == "prueba":
        P = predecir(ds, t, barajar=True, rapido=True)
        T = E.operaciones(P, "y_cierre", None)
        m = M.metricas(M.con_costo(T, 0.0), ses)
        print(f"PRUEBA (etiquetas barajadas, debe dar ~0 bruto): {len(T)} ops, {T['date'].nunique()} sesiones, "
              f"media bruta {m['media_por_op']:.2f} bps, t {m['t_conservador']:.2f}")
        return
    hid = reg.registrar(FAMILIA, NOMBRE, {"protocolo": PROTOCOLO, "horizonte": "cierre", "n_por_decision": E.N_OPS,
                                          "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()}},
                        f"TCN + atencion sobre secuencias {PROTOCOLO}; codigo sha256 {_code_sha()}")
    if reg.estado(hid)["dev_evaluada"]:
        raise SystemExit(f"{NOMBRE} ({hid}) ya se evaluo en DEV. No se repite.")
    P = predecir(ds, t)
    P.to_csv(D.SALIDA / "exp4_predicciones_dev.csv", index=False)     # se guarda antes de evaluar, por si algo falla
    T = E.operaciones(P, "y_cierre", None)
    paso, res = M.evaluar_dev(reg, ds, hid, T)
    ic_m, ic_t = E.ic(P)
    ac = E.acierto_direccion(P, "y_cierre")
    bruto = M.metricas(M.con_costo(T, 0.0), ses)
    print(f"IC medio {ic_m:.4f} (t {ic_t:.2f}) | acierto de direccion en todas las predicciones {ac:.1f}% | "
          f"bruto por op {bruto['media_por_op']:.2f} bps (t {bruto['t_conservador']:.2f})")
    T8 = M.con_costo(T, 8.0)
    for clave in ("hora_decision", "direction"):
        tab = T8.groupby(clave).agg(n=("ret_bps", "size"), bruto=("ret_bps", "mean"), neto_8=("net_bps", "mean")).round(2)
        tab.to_csv(D.SALIDA / f"exp4_desglose_{clave}.csv")
        print(tab.to_string())
    m8 = res[8.0]
    pd.DataFrame([{"hipotesis": NOMBRE, "id": hid, "paso_DEV": paso, "n_ops": m8["n_ops"],
                   "media_op_bruta": bruto["media_por_op"], "t_conserv_bruto": bruto["t_conservador"],
                   "media_op_8": m8["media_por_op"], "t_conserv_8": m8["t_conservador"], "PF_8": m8["PF"],
                   "acierto_direccion_pct": ac, "IC": ic_m, "t_IC": ic_t}]).to_csv(D.SALIDA / "exp4_resumen_dev.csv", index=False)
    print(f"\nHipotesis en el registro: {reg.n_hipotesis()} | umbral t: {reg.umbral_t():.2f}")
    print("Sube estos archivos al repositorio: ia/resultados/exp4_*.csv y lab/registro_lab/registro_hipotesis.jsonl")


if __name__ == "__main__":
    main()
