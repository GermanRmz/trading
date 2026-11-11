"""
REVERSION A VWAP INTRADIA - protocolo rev_vwap_v1.

Uso (desde la carpeta de trabajo, con lab_marco.py y sus dependencias):
    python estrategia_reversion_vwap.py diagnostico   # factibilidad; NO crea registro ni mira retornos
    python estrategia_reversion_vwap.py ejecutar      # diagnostico + (si pasa) registra H1-H4 y evalua en DEV

DEFINICIONES FIJAS (no se cambian tras ver resultados; cualquier cambio = otra hipotesis):
  Senal      d_t = 10000*(Close_t/VWAP_t - 1); VWAP acumulado del dia (campo vwap de la vela x volumen;
             si falta, precio tipico) incluyendo la vela t.  z_t = d_t / sigma, con sigma = desviacion
             tipica de d a la MISMA hora en las 20 sesiones validas previas (sin la actual). Si falta
             alguna observacion en esa ventana, no hay senal.
  Direccion  z <= -k: largo.  z >= +k: corto.  (reversion hacia el VWAP)
  Entrada    apertura de la vela siguiente a la senal; su hora debe estar entre 10:30 y 14:30 ET.
             Solo vale si en esa apertura el precio sigue del mismo lado del VWAP de la senal.
             Una operacion por simbolo y sesion: la primera senal con entrada valida.
  Objetivo   VWAP vigente de la vela anterior ya cerrada (lo que una orden limite conoceria).
             Relleno exactamente al nivel del objetivo (sin mejora de precio).
  Stop       (H1, H3) a una distancia D adicional en contra desde el precio de entrada, con
             D = |entrada - VWAP de la senal|. Fijo. Gap en contra: sale a la apertura.
  Misma vela stop y objetivo: gana el stop.
  Tiempo     60 minutos desde la entrada (velas con inicio < entrada+60 min): salida al CIERRE de la
             ultima vela. Cierre forzoso a las 15:45 (redundante, por seguridad).
  Hipotesis  H1 k=2 con stop | H2 k=2 sin stop | H3 k=3 con stop | H4 k=3 sin stop.
  Costos     4, 8 y 12 bps ida y vuelta (descriptivo; el criterio se evalua a 8).
  Nocional   unitario por operacion. Sin limite de posiciones en esta fase (se reportan simultaneas).

DIAGNOSTICO DE FACTIBILIDAD (solo DEV, sin retornos): distancia |entrada - VWAP| en bps de las
operaciones que se abririan (primera senal valida por simbolo y sesion). Se CONTINUA solo si, para
k=2, al menos el 25% de las senales tiene una distancia >= 12 bps. Si falla, se detiene el
experimento sin registrar ninguna hipotesis.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import lab_marco as M

PROTOCOLO = "rev_vwap_v1"
K_GRID = (2.0, 3.0)
HIPOTESIS = [("H1_k2_stop", 2.0, True), ("H2_k2_sin_stop", 2.0, False),
             ("H3_k3_stop", 3.0, True), ("H4_k3_sin_stop", 3.0, False)]
LOOKBACK = 20
BAR_MIN = 5
ENTRY_FROM, ENTRY_TO = 10 * 60 + 30, 14 * 60 + 30   # minutos del dia (ET) de la apertura de la vela de entrada
HOLD_MIN = 60
HARD_CLOSE = 15 * 60 + 45
GATE_K, GATE_DIST_BPS, GATE_SHARE = 2.0, 12.0, 0.25
DIAG_JSON = Path("diagnostico_reversion_vwap.json")


def _code_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# ============================================================ preparacion
CONTEXTO = {"dev": ["dev"], "val": ["dev", "val"]}   # val usa DEV como historia para las 20 sesiones previas


def preparar(ds, sym, split="dev"):
    """Arrays por simbolo con VWAP acumulado, d y z. Solo se generan operaciones dentro de `split`."""
    if split not in CONTEXTO:
        raise NotImplementedError("El tramo sellado se implementara cuando haya un candidato.")
    frames = [ds.bars(sym, s) for s in CONTEXTO[split]]
    df = pd.concat([f for f in frames if len(f)]).sort_index()
    if df.empty:
        return None
    mins = (df.index.hour * 60 + df.index.minute).to_numpy()
    px = df["vwap"].where(df["vwap"] > 0, (df["high"] + df["low"] + df["close"]) / 3)
    vol = df["volume"].fillna(0)
    cv = vol.groupby(df["date"]).cumsum()
    vw = ((px * vol).groupby(df["date"]).cumsum() / cv.where(cv > 0)).to_numpy()
    d = 1e4 * (df["close"].to_numpy() / vw - 1)
    piv = pd.DataFrame({"date": df["date"].to_numpy(), "tod": mins, "d": d}).pivot(index="date", columns="tod", values="d")
    sig = piv.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).std().to_numpy()
    r = piv.index.get_indexer(df["date"]); c = piv.columns.get_indexer(mins)
    sigma = sig[r, c]
    z = np.where(sigma > 0, d / sigma, np.nan)
    dates = df["date"].to_numpy()
    lo, hi = M.SPLITS[split]
    bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    days = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if lo <= dates[bounds[i]] <= hi]
    return {"sym": sym, "idx": df.index, "o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float),
            "l": df["low"].to_numpy(float), "c": df["close"].to_numpy(float), "vw": vw, "z": z, "m": mins,
            "dates": dates, "days": days}


def get_P(ds, cache, sym, split="dev"):
    if (sym, split) not in cache:
        cache[(sym, split)] = preparar(ds, sym, split)
    return cache[(sym, split)]


# ============================================================== senal / simulacion
def primera_senal(o, vw, z, m, k):
    """Primera senal con entrada valida en el dia. Devuelve (e, direccion, vwap_senal, D) o None."""
    n = len(o)
    for j in range(n - 1):
        e = j + 1
        if m[e] - m[j] != BAR_MIN or not (ENTRY_FROM <= m[e] <= ENTRY_TO):
            continue
        zj = z[j]
        if not np.isfinite(zj) or abs(zj) < k:
            continue
        v, E = vw[j], o[e]
        if zj < 0:
            if E < v:
                return e, 1, v, v - E
        else:
            if E > v:
                return e, -1, v, E - v
    return None


def simular(o, h, l, c, vw, m, e, dirn, D, usar_stop):
    """Devuelve (indice_vela_salida, precio_salida, motivo)."""
    n = len(o)
    E = o[e]
    stop = E - dirn * D
    t_fin = m[e] + HOLD_MIN
    ultima = e
    for i in range(e, n):
        if m[i] >= t_fin or m[i] >= HARD_CLOSE:
            break
        ultima = i
        objetivo = vw[i - 1]                        # VWAP de la vela anterior, ya cerrada
        if dirn == 1:
            s_hit, t_hit = usar_stop and l[i] <= stop, h[i] >= objetivo
        else:
            s_hit, t_hit = usar_stop and h[i] >= stop, l[i] <= objetivo
        if s_hit:                                    # stop primero, tambien si coinciden
            gap = (o[i] < stop) if dirn == 1 else (o[i] > stop)
            return i, (o[i] if gap else stop), "stop"
        if t_hit:
            return i, float(objetivo), "objetivo"
    return ultima, c[ultima], "tiempo"


def generar_senales(P, k):
    """Una fila por (simbolo, sesion) con operacion. Sin retornos: solo entrada y distancia."""
    out = []
    for a, b in P["days"]:
        s = primera_senal(P["o"][a:b], P["vw"][a:b], P["z"][a:b], P["m"][a:b], k)
        if s is None:
            continue
        e, dirn, v, D = s
        E = P["o"][a + e]
        out.append({"symbol": P["sym"], "date": P["dates"][a], "a": a, "b": b, "e": e, "direction": dirn,
                    "dist_bps": D / E * 1e4, "z": P["z"][a + e - 1]})
    return out


def generar_trades(P, senales, usar_stop):
    rows = []
    for s in senales:
        a, b, e, dirn = s["a"], s["b"], s["e"], s["direction"]
        o, h, l, c, vw, m = (P[x][a:b] for x in ("o", "h", "l", "c", "vw", "m"))
        E = o[e]
        D = s["dist_bps"] / 1e4 * E
        i, px, motivo = simular(o, h, l, c, vw, m, e, dirn, D, usar_stop)
        ret = dirn * (px - E) / E * 1e4
        rows.append({"symbol": s["symbol"], "date": s["date"], "direction": "largo" if dirn == 1 else "corto",
                     "entry_time": P["idx"][a + e], "exit_time": P["idx"][a + i] + pd.Timedelta(minutes=BAR_MIN),
                     "ret_bps": ret, "risk_bps": s["dist_bps"], "motivo_salida": motivo, "z": s["z"]})
    return rows


# ================================================================ diagnostico
def diagnostico(ds, cache):
    print("=== DIAGNOSTICO DE FACTIBILIDAD (solo DEV; sin retornos) ===")
    sig = {k: [] for k in K_GRID}
    n_symday = 0
    for sym in ds.symbols:
        P = get_P(ds, cache, sym)
        if P is None:
            continue
        n_symday += len(P["days"])
        for k in K_GRID:
            sig[k] += generar_senales(P, k)
    stats, tablas = {}, {}
    ses_dev = len(ds.sessions("dev"))
    for k in K_GRID:
        df = pd.DataFrame(sig[k])
        df["anio"] = df["date"].str[:4]
        q = df["dist_bps"].quantile([.25, .5, .75, .9])
        share = float((df["dist_bps"] >= GATE_DIST_BPS).mean())
        por_dia = df.groupby("date").size().reindex(ds.sessions("dev"), fill_value=0)
        stats[k] = {"n_senales": len(df), "mediana_bps": q[.5], "p25": q[.25], "p75": q[.75], "p90": q[.9],
                    "pct_ge_12bps": 100 * share, "senales_por_simbolo_dia_pct": 100 * len(df) / n_symday,
                    "senales_por_sesion_media": por_dia.mean(), "senales_por_sesion_p90": por_dia.quantile(.9),
                    "senales_por_sesion_max": int(por_dia.max()), "sesiones_sin_senal_pct": 100 * (por_dia == 0).mean()}
        agg = lambda col: df.groupby(col)["dist_bps"].agg(n="count", mediana="median", p75=lambda x: x.quantile(.75),
                                                         p90=lambda x: x.quantile(.9),
                                                         pct_ge_12=lambda x: 100 * (x >= GATE_DIST_BPS).mean()).round(2)
        tablas[k] = {"por_simbolo": agg("symbol"), "por_anio": agg("anio")}
        tablas[k]["por_simbolo"].to_csv(f"diagnostico_rev_vwap_k{k:g}_por_simbolo.csv")
        tablas[k]["por_anio"].to_csv(f"diagnostico_rev_vwap_k{k:g}_por_anio.csv")
    resumen = pd.DataFrame(stats).T.round(2)
    resumen.index.name = "k"
    resumen.to_csv("diagnostico_rev_vwap_resumen.csv")
    print(resumen.T.to_string())
    for k in K_GRID:
        print(f"\n--- k={k:g}: por anio ---"); print(tablas[k]["por_anio"].to_string())
    print(f"\n--- k={GATE_K:g}: por simbolo (distancia en bps) ---"); print(tablas[GATE_K]["por_simbolo"].to_string())
    share = stats[GATE_K]["pct_ge_12bps"] / 100
    paso = bool(share >= GATE_SHARE)
    print(f"\nREGLA: continuar solo si, con k={GATE_K:g}, al menos {GATE_SHARE:.0%} de las senales tiene distancia >= "
          f"{GATE_DIST_BPS:g} bps. Observado: {share:.1%} -> {'CONTINUAR' if paso else 'DETENER EL EXPERIMENTO'}")
    DIAG_JSON.write_text(json.dumps({
        "protocolo": PROTOCOLO, "dataset_id": ds.dataset_id, "fecha": datetime.now().isoformat(timespec="seconds"),
        "codigo_sha256": _code_sha(), "regla": {"k": GATE_K, "dist_bps": GATE_DIST_BPS, "share_min": GATE_SHARE},
        "share_observado": share, "paso": paso, "estadisticas": {str(k): M._clean(v) for k, v in stats.items()}},
        indent=2), encoding="utf-8")
    return paso


# ================================================================ ejecutar DEV
def ejecutar():
    ds = M.Dataset()
    cache = {}
    diag = json.loads(DIAG_JSON.read_text(encoding="utf-8")) if DIAG_JSON.exists() else None
    if diag is None or diag.get("protocolo") != PROTOCOLO or diag.get("dataset_id") != ds.dataset_id:
        paso = diagnostico(ds, cache)
    else:
        paso = diag["paso"]
        print(f"Diagnostico previo ({diag['fecha']}): {'CONTINUAR' if paso else 'DETENER'} "
              f"(observado {diag['share_observado']:.1%}).")
    if not paso:
        print("\nEXPERIMENTO DETENIDO por la regla de factibilidad. No se registro ninguna hipotesis.")
        return

    reg = M.Registro(ds.dataset_id)
    ds.attach(reg)
    sha = _code_sha()
    hids = {}
    for nombre, k, stop in HIPOTESIS:        # se registran las 4 ANTES de evaluar cualquiera
        hids[nombre] = reg.registrar("reversion_vwap", nombre, {"protocolo": PROTOCOLO, "k": k, "stop": stop},
                                     descripcion=f"Reversion a VWAP {PROTOCOLO}; codigo sha256 {sha}")
    print(f"\nHipotesis registradas: {reg.n_hipotesis()} | umbral |t| (Bonferroni 5%): {reg.umbral_t():.3f}")

    resumen = []
    for nombre, k, stop in HIPOTESIS:
        hid = hids[nombre]
        print(f"\n{'=' * 90}\n{nombre} (id {hid}) | k={k:g} | stop={'si' if stop else 'no'}\n{'=' * 90}")
        trades = []
        for sym in ds.symbols:
            P = get_P(ds, cache, sym)
            if P is not None:
                trades += generar_trades(P, generar_senales(P, k), stop)
        tr = pd.DataFrame(trades)
        tr.to_csv(f"trades_rev_vwap_{nombre}_dev.csv", index=False)
        paso_h, res = M.evaluar_dev(reg, ds, hid, tr)
        t8 = M.con_costo(tr, 8.0)
        ses = ds.sessions("dev")
        dg = M.desgloses(ds, t8, ses)
        print("\nMotivos de salida:", tr["motivo_salida"].value_counts(normalize=True).round(3).to_dict())
        print("\nPor anio (8 bps):\n", dg["por_anio"].to_string())
        print("\nPor direccion (8 bps):\n", dg["por_direccion"].to_string())
        print("\nPor mercado (8 bps):\n", dg["por_mercado"].to_string())
        print("\nCon vs sin dias de evento (8 bps):\n", dg["con_vs_sin_eventos"].to_string())
        for nm in ("por_simbolo", "por_sector", "por_anio", "por_direccion", "por_mercado"):
            dg[nm].to_csv(f"desglose_rev_vwap_{nombre}_{nm}.csv")
        m8, m4, m12 = res[8.0], res[4.0], res[12.0]
        resumen.append({"hipotesis": nombre, "id": hid, "paso_DEV": paso_h, "n_ops": m8["n_ops"],
                        "media_op_4": m4["media_por_op"], "media_op_8": m8["media_por_op"], "media_op_12": m12["media_por_op"],
                        "t_conserv_8": m8["t_conservador"], "media_igual_peso_8": m8["media_diaria_igual_peso"],
                        "max_dd_8": m8["max_drawdown"], "max_simult": m8["max_simultaneas"]})
    out = pd.DataFrame(resumen)
    out.to_csv("resultados_rev_vwap_dev.csv", index=False)
    print(f"\n{'=' * 90}\nRESUMEN DEV | umbral |t| = {reg.umbral_t():.3f}\n{'=' * 90}")
    print(out.round(3).to_string(index=False))
    if out["paso_DEV"].any():
        print("\nHay candidatos que pasaron DEV. Siguiente paso: evaluar_val (una sola mirada por candidato).")
    else:
        print("\nNingun candidato paso DEV. No se avanza a VAL ni se abre el sellado.")
        print("No se anaden variantes para 'rescatar' la estrategia: cada una sube el umbral y es otra hipotesis.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ejecutar"
    if cmd == "diagnostico":
        diagnostico(M.Dataset(), {})
    elif cmd == "ejecutar":
        ejecutar()
    else:
        raise SystemExit("Uso: diagnostico | ejecutar")
