"""
RED DE POLITICA - entrenamiento por "backtest diferenciable" y evaluacion.

La red no predice el precio: decide directamente la posicion de cada tramo de 30 minutos y se entrena para
maximizar el Sharpe NETO de costes de esas decisiones (metodo de las Deep Momentum Networks). Como una
cuenta pequena no mueve el precio, el resultado es una funcion derivable de las posiciones, y eso es
aprendizaje por refuerzo directo (Moody y Saffell, 2001) sin el ruido de PPO o DQN.

  ganancia del dia = suma(posicion x retorno del tramo) - coste x (entradas + cambios + salida)

Todo va en unidades de la volatilidad diaria del simbolo. Las sesiones se procesan en trozos de varios
dias seguidos del mismo simbolo: el primero solo sirve de contexto y los demas puntuan.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

import pol_datos as PD
import pol_modelos as PM

J = PD.J


@dataclass
class Config:
    estructura: str = "vlstm"
    h: int = 32
    dropout: float = 0.3
    dias_trozo: int = 6            # sesiones seguidas por trozo
    dias_contexto: int = 1         # las primeras solo dan contexto (no puntuan)
    lote: int = 128                # trozos por lote
    lr: float = 1e-3
    decaimiento: float = 1e-4
    epocas: int = 30
    paciencia: int = 6
    costo_bps: float = 8.0         # ida y vuelta, igual que el criterio del laboratorio
    peso_vol: float = 0.2          # tarea auxiliar: pronosticar la volatilidad del tramo siguiente
    espejo: bool = True            # aumento por "mundo invertido" (impide aprender un sesgo fijo de compra)
    universo: bool = True          # entrenar con todos los simbolos (False: solo con el objetivo)
    frac_validacion: float = 0.15  # ultimo tramo temporal del entrenamiento

    def clave(self):
        base = Config()
        dif = {k: v for k, v in asdict(self).items() if getattr(base, k) != v}
        return "base" if not dif else "_".join(f"{k}={v}" for k, v in sorted(dif.items()))


class Conjunto:
    """Tabla de pasos en memoria, con normalizacion ajustada solo con el entrenamiento."""

    def __init__(self, tabla, dispositivo="cpu"):
        self.t, self.dev = tabla, dispositivo
        self.n = len(tabla["sym"])
        self.fecha, self.sym = tabla["fecha"], tabla["sym"]
        self.anio = np.array([f[:4] for f in self.fecha])
        self.simbolos = tabla["simbolos"]
        self.Y = torch.from_numpy(tabla["Y"]).to(dispositivo)
        self.VT = torch.from_numpy(tabla["VT"]).to(dispositivo)
        self.sig = torch.from_numpy(tabla["sig"]).to(dispositivo)
        self.sym_t = torch.from_numpy(tabla["sym"]).to(dispositivo)
        self.signo = torch.from_numpy(PD.SIGNO).to(dispositivo)
        self.X = None
        # limites [a, b) de cada simbolo (las muestras estan ordenadas por simbolo y fecha)
        corte = np.flatnonzero(np.r_[True, self.sym[1:] != self.sym[:-1], True])
        self.rangos = list(zip(corte[:-1], corte[1:]))

    def escalar(self, mascara_entren):
        """Media y desviacion de cada variable con las muestras de entrenamiento (las de signo no se centran)."""
        x = self.t["X"][mascara_entren].reshape(-1, PD.F)
        mu, sd = x.mean(0), x.std(0)
        mu[PD.CON_MEDIA_CERO] = 0.0
        sd[PD.CON_MEDIA_CERO] = np.sqrt((x[:, PD.CON_MEDIA_CERO] ** 2).mean(0))
        sd[[PD.I_ALTO, PD.I_BAJO]] = sd[[PD.I_ALTO, PD.I_BAJO]].mean()       # el espejo las intercambia
        sd = np.where(sd > 1e-6, sd, 1.0)
        self.mu, self.sd = mu, sd
        self.X = torch.from_numpy(np.clip((self.t["X"] - mu) / sd, -5, 5).astype(np.float32)).to(self.dev)

    def trozos(self, puntuar, dias, contexto, desplazamiento=0):
        """Inicios de trozo y mascara [n_trozos, dias] de las sesiones que puntuan (cada una, una sola vez)."""
        paso = dias - contexto
        inicios, masc = [], []
        for a, b in self.rangos:
            if b - a < dias:
                continue
            ss = list(range(a + desplazamiento, b - dias + 1, paso))
            if not ss or ss[-1] != b - dias:
                ss.append(b - dias)                                           # cola: se solapa con el anterior
            hecho = a + desplazamiento + contexto if desplazamiento else a + contexto
            for s in ss:
                m = np.zeros(dias, bool)
                desde = max(s + contexto, hecho)
                m[desde - s:] = puntuar[desde:s + dias]
                hecho = max(hecho, s + dias)
                if m.any():
                    inicios.append(s)
                    masc.append(m)
        if not inicios:
            return np.zeros(0, int), np.zeros((0, dias), bool)
        return np.array(inicios), np.array(masc)

    def lote(self, inicios, dias):
        idx = torch.as_tensor(inicios[:, None] + np.arange(dias)[None, :], device=self.dev)       # [B, dias]
        return idx, self.X[idx].reshape(len(inicios), dias * J, PD.F), self.sym_t[idx[:, 0]]


def rotacion(pos):
    """Suma de cambios de posicion dentro del dia: entrada, cambios y salida al cierre."""
    return pos[..., 0].abs() + (pos[..., 1:] - pos[..., :-1]).abs().sum(-1) + pos[..., -1].abs()


def _ganancia(pos, y, sig, costo_bps):
    """Ganancia neta por (trozo, dia) en unidades de volatilidad diaria."""
    lado = (costo_bps / 2) * 1e-4 / sig
    return (pos * y).sum(-1) - lado * rotacion(pos)


def _adelante(red, conj, inicios, dias, espejo=None):
    idx, x, sym = conj.lote(inicios, dias)
    y = conj.Y[idx]
    if espejo is not None:                                    # mundo invertido para parte del lote
        e = espejo.view(-1, 1, 1)
        xe = x * conj.signo
        xe[..., PD.I_ALTO], xe[..., PD.I_BAJO] = -x[..., PD.I_BAJO], -x[..., PD.I_ALTO]
        x = torch.where(e, xe, x)
        y = torch.where(e, -y, y)
    pos, vol = red(x, sym)
    return idx, pos.view(len(inicios), dias, J), vol.view(len(inicios), dias, J), y


def predecir(red, conj, puntuar, cfg):
    """Posiciones suaves [n, 12] y pronostico de volatilidad para las sesiones marcadas (NaN en el resto)."""
    red.eval()
    P = np.full((conj.n, J), np.nan, np.float32)
    V = np.full((conj.n, J), np.nan, np.float32)
    inicios, masc = conj.trozos(puntuar, cfg.dias_trozo, cfg.dias_contexto)
    with torch.no_grad():
        for i in range(0, len(inicios), 512):
            s, m = inicios[i:i + 512], masc[i:i + 512]
            idx, pos, vol, _ = _adelante(red, conj, s, cfg.dias_trozo)
            ii = idx.cpu().numpy()[m]
            P[ii], V[ii] = pos.cpu().numpy()[m], vol.cpu().numpy()[m]
    return P, V


def sharpe_cartera(gan, fechas):
    """Sharpe anualizado de la cartera de igual peso: media por fecha y luego media/desviacion entre fechas."""
    if len(gan) == 0:
        return float("nan")
    orden = np.argsort(fechas, kind="stable")
    f, g = fechas[orden], gan[orden]
    cortes = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
    diario = np.add.reduceat(g, cortes) / np.diff(np.r_[cortes, len(g)])
    sd = diario.std()
    return float(diario.mean() / sd * math.sqrt(252)) if sd > 0 and len(diario) > 5 else float("nan")


def ganancia_suave(conj, P, puntuar, costo_bps):
    """Ganancia neta normalizada por sesion con posiciones suaves (numpy)."""
    i = np.flatnonzero(puntuar & np.isfinite(P[:, 0]))
    p = torch.from_numpy(P[i])
    g = _ganancia(p, conj.Y[i].cpu(), conj.sig[i].cpu(), costo_bps)
    return i, g.numpy()


def entrenar(cfg, conj, entren, valid, semilla, objetivo=None, verbose=False):
    """Entrena una red. `entren` y `valid` son mascaras de sesiones. Devuelve (red, historial)."""
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    if not cfg.universo:
        solo = conj.sym == conj.simbolos.index(objetivo)
        entren = entren & solo
    red = PM.crear(cfg.estructura, PD.F, len(conj.simbolos), cfg.h, cfg.dropout).to(conj.dev)
    opt = torch.optim.AdamW(red.parameters(), lr=cfg.lr, weight_decay=cfg.decaimiento)
    mejor, estado, espera, hist = -np.inf, None, 0, []
    paso_t = cfg.dias_trozo - cfg.dias_contexto
    for ep in range(cfg.epocas):
        red.train()
        inicios, masc = conj.trozos(entren, cfg.dias_trozo, cfg.dias_contexto, int(rng.integers(0, paso_t)))
        orden = rng.permutation(len(inicios))
        perdidas = []
        for i in range(0, len(orden), cfg.lote):
            sel = orden[i:i + cfg.lote]
            if len(sel) < 8:
                continue
            m = torch.as_tensor(masc[sel], device=conj.dev)
            esp = torch.as_tensor(rng.random(len(sel)) < 0.5, device=conj.dev) if cfg.espejo else None
            idx, pos, vol, y = _adelante(red, conj, inicios[sel], cfg.dias_trozo, esp)
            g = _ganancia(pos, y, conj.sig[idx], cfg.costo_bps)[m]
            perdida = -g.mean() / (g.std() + 1e-8) * math.sqrt(252)
            if cfg.peso_vol > 0:
                perdida = perdida + cfg.peso_vol * ((vol - conj.VT[idx]) ** 2)[m].mean()
            opt.zero_grad()
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(red.parameters(), 1.0)
            opt.step()
            perdidas.append(float(perdida.detach()))
        P, _ = predecir(red, conj, valid, cfg)
        i, g = ganancia_suave(conj, P, valid, cfg.costo_bps)
        sv = sharpe_cartera(g, conj.fecha[i])
        sv = -99.0 if not np.isfinite(sv) else sv
        hist.append({"epoca": ep + 1, "perdida": float(np.mean(perdidas)) if perdidas else np.nan, "sharpe_val": sv})
        if verbose:
            print(f"      epoca {ep + 1:2d}: perdida {hist[-1]['perdida']:.3f} | Sharpe validacion {sv:.2f}", flush=True)
        if sv > mejor + 1e-4:
            mejor, espera = sv, 0
            estado = {k: v.detach().clone() for k, v in red.state_dict().items()}
        else:
            espera += 1
            if espera >= cfg.paciencia:
                break
    if estado is not None:
        red.load_state_dict(estado)
    return red, {"mejor_sharpe_val": mejor, "epocas": len(hist), "historial": hist}


# ------------------------------------------------------------------ decisiones discretas
def escala(P):
    """Tamano tipico de la posicion suave (la red es libre de escala: el Sharpe no cambia al multiplicarla)."""
    v = P[np.isfinite(P)]
    return float(np.sqrt((v ** 2).mean())) if len(v) else 1.0


def discretizar(P, umbral, q):
    """Compra (+1) / nada (0) / venta (-1): solo se opera cuando la posicion suave supera umbral x escala."""
    return np.where(np.isfinite(P), np.sign(P) * (np.abs(P) > umbral * q), 0.0).astype(np.float32)


def ganancia_dura(tabla, D, costo_bps):
    """Ganancia neta en bps por sesion con posiciones discretas y nocional fijo, y numero de operaciones."""
    rot = np.abs(D[:, 0]) + np.abs(np.diff(D, axis=1)).sum(1) + np.abs(D[:, -1])
    neto = (D * tabla["RB"]).sum(1) - (costo_bps / 2) * rot
    cambio = np.diff(np.c_[np.zeros(len(D)), D], axis=1) != 0
    n_ops = (cambio & (D != 0)).sum(1)
    return neto, n_ops, rot


UMBRALES = (0.0, 0.25, 0.5, 1.0, 1.5)


def elegir_umbral(tabla, P, filas, costo_bps, fechas, min_ops_dia=0.3):
    """Umbral que maximiza el Sharpe neto de cartera en validacion (unidades de volatilidad)."""
    q = escala(P[filas])
    mejor, mejor_u = -np.inf, UMBRALES[0]
    sub = {k: tabla[k][filas] for k in ("RB", "sig")}
    for u in UMBRALES:
        D = discretizar(P[filas], u, q)
        neto, n_ops, _ = ganancia_dura(sub, D, costo_bps)
        if n_ops.mean() < min_ops_dia:
            continue
        s = sharpe_cartera(neto / (1e4 * sub["sig"]), fechas[filas])
        if np.isfinite(s) and s > mejor:
            mejor, mejor_u = s, u
    return mejor_u, q
