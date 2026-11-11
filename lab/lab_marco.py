"""
LABORATORIO COMUN - Marco de pruebas (nucleo).

Hace CUMPLIR la metodologia en codigo:
  * El dataset debe estar congelado e intacto (hash) para poder usarse.
  * Reparto temporal: DEV 2020-2023 | VAL 2024-2025 | SELLADO 2026.
  * Registro global de hipotesis (append-only): cada hipotesis distinta cuenta para la
    correccion de Bonferroni, venga de la familia de estrategias que venga.
  * VAL solo se evalua para candidatos que pasaron DEV, y UNA sola vez por candidato.
  * El tramo SELLADO solo se abre con un candidato que paso DEV y VAL, y solo una vez en
    total (por defecto). Despues, el holdout real es el paper trading.
  * Los criterios de aceptacion quedan hasheados en el registro: si se cambian despues, el
    marco se niega a continuar.
  * Metricas de cartera sobre la serie diaria COMPLETA (dias sin operaciones = 0), t-stat
    sobre la suma diaria, drawdown, exposicion y operaciones simultaneas.
  * Costos de 4, 8 y 12 bps como sensibilidad de la MISMA hipotesis (no suman pruebas).

AVISO HONESTO: esto es una barrera contra errores propios y autoengano accidental, no contra
alguien que edite los archivos a proposito. El sellado es procedimental.

Contrato de operaciones (DataFrame que produce cada estrategia):
    symbol, date (AAAA-MM-DD, hora de Nueva York), entry_time, exit_time,
    ret_bps  = retorno BRUTO de la operacion en bps del precio de entrada (+ = ganancia),
    risk_bps (opcional) = distancia al stop en bps, para expresar tambien en R.
Costo: net_bps = ret_bps - costo_bps (costo ida y vuelta).

Uso:
    python lab_marco.py splits     # muestra el reparto temporal real de tus datos
    python lab_marco.py estado     # dataset, registro, umbral de Bonferroni, candidatos
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

try:
    from lab_dataset_congelar import verify_dataset
except ImportError:
    raise SystemExit("Coloca lab_dataset_congelar.py y lab_descarga_auditoria_v3.py en esta carpeta.")

NY_TZ = "America/New_York"
DATA_DIR = Path("datos_lab_sip_limpio")
MANUAL_EXCLUSIONS = Path("exclusiones_manuales.csv")
REGISTRY_PATH = Path("registro_lab") / "registro_hipotesis.jsonl"

SPLITS = {
    "dev": ("2020-01-01", "2023-12-31"),
    "val": ("2024-01-01", "2025-12-31"),
    "sealed": ("2026-01-01", "2099-12-31"),
}
COSTS_BPS = (4.0, 8.0, 12.0)

# ---- Criterios de aceptacion (se hashean en el registro; no cambiar despues de empezar) ----
CRITERIOS = {
    "dev_min_operaciones": 500,
    "dev_costo_exigido_bps": 8.0,     # debe ser rentable y significativo con este costo
    "rezagos_newey_west": 6,          # kernel Bartlett
    "dev_exige_media_igual_peso_positiva": True,
    "t_para_decidir": "min(t_convencional, t_newey_west)",
    "val_t_min": 1.65,                # t unilateral 5%: un solo candidato declarado
    "val_costo_exigido_bps": 8.0,
    "val_mismo_signo": True,
    "max_aperturas_sellado": 1,
    "alfa": 0.05,
}

# Sector y mercado de listado de cada simbolo (para desgloses). Verificar WMT: el cambio de
# listado a Nasdaq reportado a fines de 2025 solo afecta a los desgloses por mercado.
META = {
    **{s: ("tecnologia", "NASDAQ") for s in ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "AVGO"]},
    **{s: ("finanzas", "NYSE") for s in ["JPM", "BAC", "GS", "MS", "V", "MA", "BRK.B"]},
    **{s: ("consumo_energia", "NYSE") for s in ["XOM", "CVX", "WMT", "KO", "PG", "HD"]},
    "COST": ("consumo_energia", "NASDAQ"),
    **{s: ("salud", "NYSE") for s in ["UNH", "JNJ", "LLY", "ABBV", "MRK"]},
    **{s: ("etf_mercado", "ARCA") for s in ["SPY", "IWM", "DIA", "XLK", "XLF", "XLE"]},
    "QQQ": ("etf_mercado", "NASDAQ"),
}


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _criterios_hash():
    return hashlib.sha256(json.dumps({"c": CRITERIOS, "s": SPLITS, "k": COSTS_BPS},
                                     sort_keys=True).encode()).hexdigest()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o)) else ("inf" if np.isinf(o) else float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ===================================================================== REGISTRO
class Registro:
    """Registro append-only (JSONL) de hipotesis, evaluaciones y aperturas del tramo sellado."""

    def __init__(self, dataset_id, path=REGISTRY_PATH, manual_path=MANUAL_EXCLUSIONS):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events = self._read()
        if manual_path is not None and not Path(manual_path).exists():
            raise FileNotFoundError(f"Falta {manual_path} (exclusiones manuales) en {Path.cwd()}. Colocalo ahi: sin el, "
                                    "el registro quedaria fijado sin las exclusiones declaradas.")
        man_hash = _sha_file(manual_path) if manual_path is not None else None
        if not self.events:
            self._append({"tipo": "init", "criterios_hash": _criterios_hash(), "dataset_id": dataset_id,
                          "splits": SPLITS, "criterios": CRITERIOS, "costos_bps": list(COSTS_BPS),
                          "exclusiones_manuales_sha256": man_hash})
        else:
            init = self.events[0]
            if init["criterios_hash"] != _criterios_hash():
                raise RuntimeError("Los criterios/reparto/costos cambiaron respecto al registro original. "
                                   "No se mueven los postes a mitad de partido: crea un registro nuevo "
                                   "(y declara el motivo) si de verdad lo necesitas.")
            if init["dataset_id"] != dataset_id:
                raise RuntimeError(f"El dataset_id ({dataset_id[:12]}...) no coincide con el del registro "
                                   f"({init['dataset_id'][:12]}...).")
            last = [e for e in self.events if e["tipo"] in ("init", "cambio_exclusiones")][-1]
            if last.get("exclusiones_manuales_sha256", last.get("sha256")) != man_hash:
                raise RuntimeError("exclusiones_manuales.csv cambio desde el registro. Si es legitimo, llama a "
                                   "registro.aceptar_cambio_exclusiones('motivo') y quedara anotado.")

    # -- E/S
    def _read(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _append(self, ev):
        ev = {"ts": datetime.now().isoformat(timespec="seconds"), **_clean(ev)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self.events.append(ev)

    def aceptar_cambio_exclusiones(self, motivo, manual_path=MANUAL_EXCLUSIONS):
        if not motivo or len(motivo) < 10:
            raise ValueError("Describe el motivo del cambio (>= 10 caracteres).")
        self._append({"tipo": "cambio_exclusiones", "motivo": motivo, "sha256": _sha_file(manual_path)})

    # -- hipotesis
    @staticmethod
    def id_hipotesis(familia, nombre, params):
        raw = json.dumps({"f": familia, "n": nombre, "p": params}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def registrar(self, familia, nombre, params, descripcion=""):
        """Registra una hipotesis. Cualquier cambio de parametros crea OTRA hipotesis (y suma una prueba)."""
        hid = self.id_hipotesis(familia, nombre, params)
        if not any(e["tipo"] == "hipotesis" and e["id"] == hid for e in self.events):
            self._append({"tipo": "hipotesis", "id": hid, "familia": familia, "nombre": nombre,
                          "params": params, "descripcion": descripcion})
        return hid

    def hipotesis(self):
        return {e["id"]: e for e in self.events if e["tipo"] == "hipotesis"}

    def n_hipotesis(self):
        return len(self.hipotesis())

    def umbral_t(self):
        n = max(self.n_hipotesis(), 1)
        return NormalDist().inv_cdf(1 - CRITERIOS["alfa"] / (2 * n))

    def eventos(self, hid, tipo):
        return [e for e in self.events if e.get("id") == hid and e["tipo"] == tipo]

    def estado(self, hid):
        if hid not in self.hipotesis():
            raise KeyError(f"Hipotesis {hid} no registrada.")
        dev, val, sel = self.eventos(hid, "eval_dev"), self.eventos(hid, "eval_val"), self.eventos(hid, "eval_sellado")
        return {"dev_evaluada": bool(dev), "dev_paso": bool(dev) and dev[-1]["paso"],
                "val_evaluada": bool(val), "val_paso": bool(val) and val[-1]["paso"],
                "sellado_evaluado": bool(sel)}

    # -- sellado
    def aperturas_sellado(self):
        return [e for e in self.events if e["tipo"] == "apertura_sellado"]

    def token_valido(self, token):
        return bool(token) and any(e["token"] == token for e in self.aperturas_sellado())


# ===================================================================== DATASET
class Dataset:
    def __init__(self, data_dir=DATA_DIR, manual_path=MANUAL_EXCLUSIONS, verify=True, registro=None):
        self.dir = Path(data_dir)
        if verify:
            ok, problems = verify_dataset(self.dir)
            if not ok:
                raise RuntimeError("Dataset no verificado: " + "; ".join(problems))
        man = json.loads((self.dir / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
        self.dataset_id, self.symbols = man["dataset_id"], man["simbolos"]
        self.registro = registro
        self.excluded, self.event_days = {}, set()
        auto = self.dir / "dias_excluidos.csv"
        if not auto.exists():
            raise FileNotFoundError(f"Falta {auto}: el dataset congelado debe incluir su mascara automatica.")
        for r in pd.read_csv(auto).itertuples():
            self.excluded.setdefault(r.symbol, set()).add(str(r.date))
        self.n_auto = sum(len(v) for v in self.excluded.values())
        if manual_path is not None:
            if not Path(manual_path).exists():
                raise FileNotFoundError(f"Falta {manual_path} (exclusiones manuales) en {Path.cwd()}. Sin el, el "
                                        "2023-01-24 de las acciones de NYSE quedaria dentro de los datos.")
            for r in pd.read_csv(manual_path).itertuples():
                if r.decision == "excluir":
                    self.excluded.setdefault(r.symbol, set()).add(str(r.date))
                elif r.decision == "conservar_y_etiquetar":
                    self.event_days.add((r.symbol, str(r.date)))
        self.n_manual = sum(len(v) for v in self.excluded.values()) - self.n_auto
        self._cache, self._sessions = {}, None

    def attach(self, registro):
        self.registro = registro

    def _load(self, symbol, incl_sealed):
        key = (symbol, incl_sealed)
        if key not in self._cache:
            df = pd.read_csv(self.dir / f"{symbol}_5min.csv")
            idx = pd.DatetimeIndex(pd.to_datetime(df.pop("timestamp"), utc=True)).tz_convert(NY_TZ)
            df.index = idx
            df["date"] = idx.strftime("%Y-%m-%d")
            if not incl_sealed:   # los datos sellados ni siquiera se mantienen en memoria
                df = df[df["date"] <= SPLITS["val"][1]]
            self._cache[key] = df
        return self._cache[key]

    def bars(self, symbol, split, token=None):
        """Velas de un simbolo en un tramo, sin los (simbolo, dia) excluidos. 'sealed' exige token."""
        if split not in SPLITS:
            raise ValueError(f"split debe ser uno de {list(SPLITS)}")
        if split == "sealed" and not (self.registro and self.registro.token_valido(token)):
            raise PermissionError("El tramo SELLADO solo se abre con desellar() tras pasar DEV y VAL.")
        df = self._load(symbol, split == "sealed")
        lo, hi = SPLITS[split]
        out = df[(df["date"] >= lo) & (df["date"] <= hi)]
        ex = self.excluded.get(symbol)
        return out[~out["date"].isin(ex)] if ex else out

    def sessions(self, split):
        """Sesiones (fechas) del tramo: union de las fechas con datos. Solo fechas, no precios."""
        if self._sessions is None:
            dates = set()
            for s in self.symbols:
                ts = pd.read_csv(self.dir / f"{s}_5min.csv", usecols=["timestamp"])["timestamp"]
                dates |= set(pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).tz_convert(NY_TZ).strftime("%Y-%m-%d"))
            self._sessions = sorted(dates)
        lo, hi = SPLITS[split]
        return [d for d in self._sessions if lo <= d <= hi]

    def meta(self, symbol):
        return META.get(symbol, ("otro", "desconocido"))


# ===================================================================== METRICAS
def con_costo(trades, cost_bps):
    t = trades.copy()
    t["net_bps"] = t["ret_bps"] - cost_bps
    if "risk_bps" in t.columns:
        t["net_R"] = t["net_bps"] / t["risk_bps"]
    return t


def _max_concurrent(trades):
    if not len(trades):
        return 0
    s = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"], utc=True)).as_unit("ns").asi8
    e = pd.DatetimeIndex(pd.to_datetime(trades["exit_time"], utc=True)).as_unit("ns").asi8
    ev = pd.DataFrame({"t": np.r_[s, e], "d": np.r_[np.ones(len(s)), -np.ones(len(e))]})
    return int(ev.sort_values(["t", "d"]).d.cumsum().max())   # a igual t, las salidas van primero


def _t_newey_west(x, lags):
    """t de la media con varianza de largo plazo Newey-West (kernel Bartlett)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < lags + 5:
        return np.nan
    xc = x - x.mean()
    lrv = (xc @ xc) / n
    for l in range(1, lags + 1):
        lrv += 2 * (1 - l / (lags + 1)) * (xc[l:] @ xc[:-l]) / n
    return x.mean() / np.sqrt(lrv / n) if lrv > 0 else np.nan


def _t_conv(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return x.mean() / (sd / np.sqrt(len(x))) if len(x) >= 5 and sd > 0 else np.nan


def metricas(trades, sessions, col="net_bps"):
    """Metricas de cartera sobre la serie diaria completa de `sessions` (dias sin operar = 0).

    Primaria: suma diaria (nocional fijo por operacion). Complemento: promedio diario de igual peso.
    Inferencia: t convencional y t Newey-West; se decide con el mas conservador.
    """
    sessions = list(sessions)
    has = len(trades) > 0
    extra = set(trades["date"]) - set(sessions) if has else set()
    if extra:
        raise ValueError(f"{len(extra)} fechas de operaciones fuera de las sesiones del tramo, p. ej. {sorted(extra)[:3]}")
    g = trades.groupby("date")[col] if has else None
    s = (g.sum() if has else pd.Series(dtype=float)).reindex(sessions, fill_value=0.0)
    eqw = (g.mean() if has else pd.Series(dtype=float)).reindex(sessions, fill_value=0.0)
    t_c, t_nw = _t_conv(s), _t_newey_west(s, CRITERIOS["rezagos_newey_west"])
    t_cons = min(t_c, t_nw) if not (np.isnan(t_c) or np.isnan(t_nw)) else np.nan
    eq = s.cumsum()
    peak = np.maximum(0.0, eq.cummax())
    x = trades[col] if has else pd.Series(dtype=float)
    gains, losses = x[x > 0].sum(), -x[x < 0].sum()
    traded = (trades.groupby("date").size().reindex(sessions, fill_value=0) > 0) if has else pd.Series(False, index=sessions)
    if has:
        e0 = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"], utc=True)).as_unit("ns").asi8
        e1 = pd.DatetimeIndex(pd.to_datetime(trades["exit_time"], utc=True)).as_unit("ns").asi8
        expo_media = ((e1 - e0) / 6e10).sum() / (390.0 * max(len(sessions), 1))   # unidades de nocional, sobre minutos de sesion
    else:
        expo_media = 0.0
    maxsim = _max_concurrent(trades)
    return {"n_ops": len(trades), "n_sesiones": len(s), "media_por_op": x.mean() if has else np.nan,
            "media_diaria": s.mean(), "t_diario": t_c, "t_newey_west": t_nw, "t_conservador": t_cons,
            "media_diaria_igual_peso": eqw.mean(), "t_igual_peso": _t_conv(eqw),
            "PF": (gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan)),
            "acierto_pct": 100 * (x > 0).mean() if has else np.nan,
            "acumulado": s.sum(), "max_drawdown": (eq - peak).min(),
            "exposicion_pct": 100 * traded.mean() if len(traded) else np.nan,
            "max_simultaneas": maxsim, "exposicion_bruta_max": maxsim, "exposicion_bruta_media": expo_media,
            "ops_por_dia_activo": (len(trades) / max(int(traded.sum()), 1)) if has else 0.0}


def desgloses(ds, trades, sessions, col="net_bps"):
    """Por anio, simbolo, sector, mercado y con/sin dias de eventos etiquetados."""
    t = trades.copy()
    t["anio"] = t["date"].str[:4]
    t["sector"] = t["symbol"].map(lambda s: ds.meta(s)[0])
    t["mercado"] = t["symbol"].map(lambda s: ds.meta(s)[1])
    t["es_evento"] = [(a, b) in ds.event_days for a, b in zip(t["symbol"], t["date"])]
    agg = lambda k: t.groupby(k)[col].agg(n="count", media="mean", total="sum").round(3)
    return {"por_anio": agg("anio"), "por_simbolo": agg("symbol"), "por_sector": agg("sector"),
            "por_mercado": agg("mercado"), "con_vs_sin_eventos": agg("es_evento"),
            **({"por_direccion": agg("direction")} if "direction" in t.columns else {}),
            "metricas_sin_eventos": metricas(t[~t["es_evento"]], sessions, col)}


# =================================================================== EVALUACION
def _prep(ds, trades, split):
    req = {"symbol", "date", "entry_time", "exit_time", "ret_bps"}
    miss = req - set(trades.columns)
    if miss:
        raise ValueError(f"Faltan columnas en las operaciones: {sorted(miss)}")
    lo, hi = SPLITS[split]
    bad = trades[(trades["date"] < lo) | (trades["date"] > hi)]
    if len(bad):
        raise ValueError(f"{len(bad)} operaciones fuera del tramo '{split}' ({lo} a {hi}); no se mezclan tramos.")
    pares = {(a, b) for a, bs in ds.excluded.items() for b in bs}
    ex = np.array([(s, d) in pares for s, d in zip(trades["symbol"], trades["date"])], dtype=bool)
    return trades[~ex].reset_index(drop=True), int(ex.sum())


def _evaluar(ds, trades, split):
    tr, n_drop = _prep(ds, trades, split)
    ses = ds.sessions(split)
    res = {c: metricas(con_costo(tr, c), ses) for c in COSTS_BPS}
    return res, tr, n_drop, ses


def _hash_trades(tr):
    cols = [c for c in ["symbol", "date", "entry_time", "exit_time", "ret_bps"] if c in tr.columns]
    return hashlib.sha256(pd.util.hash_pandas_object(tr[cols].astype(str), index=False).values.tobytes()).hexdigest()[:16]


def imprimir(res, titulo):
    print(f"\n=== {titulo} ===")
    df = pd.DataFrame({f"{c:g} bps": pd.Series(m) for c, m in res.items()}).T
    cols = ["n_ops", "media_por_op", "media_diaria", "media_diaria_igual_peso", "t_diario", "t_newey_west",
            "t_conservador", "PF", "acumulado", "max_drawdown", "exposicion_pct", "max_simultaneas",
            "exposicion_bruta_media"]
    print(df[cols].round(3).to_string())


def evaluar_dev(reg, ds, hid, trades):
    """Evaluacion en DEV (se puede repetir; cada cambio de parametros es OTRA hipotesis registrada)."""
    reg.estado(hid)
    res, tr, n_drop, ses = _evaluar(ds, trades, "dev")
    m = res[CRITERIOS["dev_costo_exigido_bps"]]
    umbral = reg.umbral_t()
    paso = bool(m["n_ops"] >= CRITERIOS["dev_min_operaciones"] and m["media_diaria"] > 0
                and m["t_conservador"] >= umbral
                and (m["media_diaria_igual_peso"] > 0 or not CRITERIOS["dev_exige_media_igual_peso_positiva"]))
    reg._append({"tipo": "eval_dev", "id": hid, "paso": paso, "umbral_t": umbral, "n_hipotesis": reg.n_hipotesis(),
                 "ops_excluidas_por_mascara": n_drop, "hash_operaciones": _hash_trades(tr), "metricas": res})
    imprimir(res, f"DEV {hid} | umbral t = {umbral:.2f} ({reg.n_hipotesis()} hipotesis) | "
                  f"{'PASA' if paso else 'NO PASA'}")
    if n_drop:
        print(f"({n_drop} operaciones en dias excluidos fueron descartadas)")
    return paso, res


def evaluar_val(reg, ds, hid, trades):
    """VAL: solo para candidatos que pasaron DEV, y UNA sola vez por candidato."""
    st = reg.estado(hid)
    if not st["dev_paso"]:
        raise PermissionError("VAL solo se evalua para candidatos que pasaron DEV.")
    if st["val_evaluada"]:
        raise PermissionError("VAL ya se evaluo para este candidato: una sola mirada.")
    res, tr, n_drop, ses = _evaluar(ds, trades, "val")
    m = res[CRITERIOS["val_costo_exigido_bps"]]
    dev_m = reg.eventos(hid, "eval_dev")[-1]["metricas"][str(CRITERIOS["dev_costo_exigido_bps"])]
    mismo = (m["media_diaria"] > 0) == (dev_m["media_diaria"] > 0) if CRITERIOS["val_mismo_signo"] else True
    paso = bool(mismo and m["t_conservador"] >= CRITERIOS["val_t_min"])
    reg._append({"tipo": "eval_val", "id": hid, "paso": paso, "hash_operaciones": _hash_trades(tr),
                 "ops_excluidas_por_mascara": n_drop, "metricas": res})
    imprimir(res, f"VAL {hid} | {'PASA' if paso else 'NO PASA'}")
    return paso, res


def desellar(reg, hid, motivo):
    """Abre el tramo sellado (2026) para UN candidato que paso DEV y VAL. Maximo de aperturas global."""
    st = reg.estado(hid)
    if not (st["dev_paso"] and st["val_paso"]):
        raise PermissionError("Solo se abre el sellado para candidatos que pasaron DEV y VAL.")
    ev = reg.eventos(hid, "eval_dev")[-1]
    t_dev = ev["metricas"][str(CRITERIOS["dev_costo_exigido_bps"])]["t_conservador"]
    if t_dev is None or t_dev < reg.umbral_t():
        raise PermissionError(f"Con las {reg.n_hipotesis()} hipotesis actuales el umbral t es {reg.umbral_t():.2f} y "
                              "el candidato ya no lo supera. No se abre.")
    if len(reg.aperturas_sellado()) >= CRITERIOS["max_aperturas_sellado"]:
        raise PermissionError("El tramo sellado ya se abrio el maximo de veces permitido. 2026 esta 'quemado': "
                              "la validacion real pasa a ser el paper trading.")
    if not motivo or len(motivo) < 10:
        raise ValueError("Describe el motivo (>= 10 caracteres).")
    token = hashlib.sha256(f"{hid}{datetime.now().isoformat()}".encode()).hexdigest()[:20]
    reg._append({"tipo": "apertura_sellado", "id": hid, "motivo": motivo, "token": token})
    print(f"TRAMO SELLADO ABIERTO para {hid}. Esta es la unica apertura permitida.")
    return token


def evaluar_sellado(reg, ds, hid, trades, token):
    if not reg.token_valido(token) or reg.eventos(hid, "apertura_sellado")[-1]["token"] != token:
        raise PermissionError("Token invalido para este candidato.")
    if reg.estado(hid)["sellado_evaluado"]:
        raise PermissionError("El tramo sellado ya se evaluo para este candidato.")
    res, tr, n_drop, ses = _evaluar(ds, trades, "sealed")
    reg._append({"tipo": "eval_sellado", "id": hid, "hash_operaciones": _hash_trades(tr), "metricas": res})
    imprimir(res, f"SELLADO {hid} (resultado final; no se re-optimiza)")
    return res


# ========================================================================== CLI
def _cmd_splits():
    ds = Dataset()
    print(f"Dataset {ds.dataset_id[:16]}... | {len(ds.symbols)} simbolos | pares excluidos: "
          f"{sum(len(v) for v in ds.excluded.values())} ({ds.n_auto} automaticos + {ds.n_manual} manuales) | "
          f"dias de evento etiquetados: {len(ds.event_days)}")
    for name, (lo, hi) in SPLITS.items():
        ses = ds.sessions(name)
        ex = sum(1 for s, ds_ in ds.excluded.items() for d in ds_ if lo <= d <= hi)
        print(f"{name.upper():7s} {ses[0] if ses else '-'} -> {ses[-1] if ses else '-'} | {len(ses):5d} sesiones | "
              f"{ex:3d} pares (simbolo, dia) excluidos")
    print("Anios en DEV:", sorted({d[:4] for d in ds.sessions('dev')}), "| VAL:", sorted({d[:4] for d in ds.sessions('val')}))


def _cmd_estado():
    ds = Dataset()
    reg = Registro(ds.dataset_id)
    ds.attach(reg)
    print(f"Dataset {ds.dataset_id[:16]}... verificado.")
    print(f"Hipotesis registradas: {reg.n_hipotesis()} | umbral |t| (Bonferroni {CRITERIOS['alfa']:.0%}): {reg.umbral_t():.2f}")
    print(f"Aperturas del tramo sellado: {len(reg.aperturas_sellado())}/{CRITERIOS['max_aperturas_sellado']}")
    for hid, h in reg.hipotesis().items():
        print(f"  {hid} {h['familia']}/{h['nombre']} {h['params']} -> {reg.estado(hid)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "estado"
    {"splits": _cmd_splits, "estado": _cmd_estado}.get(cmd, _cmd_estado)()
