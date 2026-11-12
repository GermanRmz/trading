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
    lote: int = 32                 # trozos por lote
    lr: float = 2e-3
    decaimiento: float = 1e-4
    epocas: int = 60
    paciencia: int = 10
    costo_bps: float = 8.0         # ida y vuelta, igual que el criterio del laboratorio
    costo_comun: bool = True       # todos los simbolos entrenan con la relacion coste/volatilidad del objetivo
    peso_vol: float = 0.2          # tarea auxiliar: pronosticar la volatilidad del tramo siguiente
    espejo: bool = True            # aumento por "mundo invertido" (impide aprender un sesgo fijo de compra)
    antisimetrica: bool = False    # simetria exacta: posicion(mundo invertido) = -posicion(mundo real)
    dropout_var: float = 0.0       # apaga variables enteras al azar (obliga a no depender de una sola)
    recorte_x: float = 5.0         # recorte de las variables normalizadas
    recorte_y: float = 0.0         # recorte de los retornos al entrenar, en desviaciones tipicas del tramo (0 = no)
    universo: bool = True          # entrenar con todos los simbolos (False: solo con el objetivo)
    frac_validacion: float = 0.2   # ultimo tramo temporal del entrenamiento

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
        self.sig_costo = self.sig                  # volatilidad con la que se expresa el coste (ver fijar_costo)
        self.X = None
        # limites [a, b) de cada simbolo (las muestras estan ordenadas por simbolo y fecha)
        corte = np.flatnonzero(np.r_[True, self.sym[1:] != self.sym[:-1], True])
        self.rangos = list(zip(corte[:-1], corte[1:]))

    def escalar(self, mascara_entren, recorte=5.0):
        """Media y desviacion de cada variable con las muestras de entrenamiento (las de signo no se centran)."""
        x = self.t["X"][mascara_entren].reshape(-1, PD.F)
        mu, sd = x.mean(0), x.std(0)
        mu[PD.CON_MEDIA_CERO] = 0.0
        sd[PD.CON_MEDIA_CERO] = np.sqrt((x[:, PD.CON_MEDIA_CERO] ** 2).mean(0))
        sd[[PD.I_ALTO, PD.I_BAJO]] = sd[[PD.I_ALTO, PD.I_BAJO]].mean()       # el espejo las intercambia
        sd = np.where(sd > 1e-6, sd, 1.0)
        self.mu, self.sd = mu, sd
        self.X = torch.from_numpy(np.clip((self.t["X"] - mu) / sd, -recorte, recorte).astype(np.float32)).to(self.dev)
        self.sd_y = torch.from_numpy(self.t["Y"][mascara_entren].std(0)).to(self.dev)      # por tramo

    def fijar_costo(self, objetivo, mascara, comun=True):
        """Con `comun`, el coste de todos los simbolos se expresa con la volatilidad tipica del objetivo.

        Un valor de bajo riesgo (8 bps de coste frente a movimientos pequenos) no se puede operar en
        intradia; si se entrenara con su coste real, la red solo aprenderia a no operarlo. Como los
        retornos ya van en unidades de volatilidad, darle a todos la relacion coste/volatilidad del
        objetivo los convierte en "SOXL de practicas" y la red aprende patrones, no a esquivar costes.
        """
        if comun:
            m = mascara & (self.sym == self.simbolos.index(objetivo))
            tipica = float(np.median(self.t["sig"][m])) if m.any() else float(np.median(self.t["sig"][mascara]))
            self.sig_costo = torch.full_like(self.sig, tipica)
        else:
            self.sig_costo = self.sig

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


def _invertir(conj, x):
    xe = x * conj.signo
    xe[..., PD.I_ALTO], xe[..., PD.I_BAJO] = -x[..., PD.I_BAJO], -x[..., PD.I_ALTO]
    return xe


def _adelante(red, conj, inicios, dias, espejo=None, dropout_var=0.0, antisimetrica=False):
    idx, x, sym = conj.lote(inicios, dias)
    y = conj.Y[idx]
    if dropout_var > 0:                                       # la misma variable se apaga en todo el trozo
        vivo = (torch.rand(x.shape[0], 1, x.shape[2], device=x.device) >= dropout_var).float()
        x = x * vivo / (1 - dropout_var)
    if espejo is not None:                                    # mundo invertido para parte del lote
        e = espejo.view(-1, 1, 1)
        x = torch.where(e, _invertir(conj, x), x)
        y = torch.where(e, -y, y)
    pos, vol = red(x, sym)
    if antisimetrica:                                         # la red se evalua tambien en el mundo invertido
        pos_e, vol_e = red(_invertir(conj, x), sym)
        pos, vol = 0.5 * (pos - pos_e), 0.5 * (vol + vol_e)
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
            idx, pos, vol, _ = _adelante(red, conj, s, cfg.dias_trozo, antisimetrica=cfg.antisimetrica)
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
    g = _ganancia(p, conj.Y[i].cpu(), conj.sig_costo[i].cpu(), costo_bps)
    return i, g.numpy()


def entrenar(cfg, conj, entren, valid, semilla, objetivo=None, verbose=False, al_terminar_epoca=None):
    """Entrena una red. `entren` y `valid` son mascaras de sesiones. Devuelve (red, historial).

    Con paciencia = 0 no hay parada temprana: se entrenan `epocas` fijas y se usan los pesos finales
    (la validacion de pocas semanas es demasiado ruidosa para decidir cuando parar).
    """
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
            idx, pos, vol, y = _adelante(red, conj, inicios[sel], cfg.dias_trozo, esp, cfg.dropout_var, cfg.antisimetrica)
            if cfg.recorte_y > 0:
                lim = cfg.recorte_y * conj.sd_y
                y = torch.maximum(torch.minimum(y, lim), -lim)
            g = _ganancia(pos, y, conj.sig_costo[idx], cfg.costo_bps)[m]
            perdida = -g.mean() / (g.std() + 1e-8) * math.sqrt(252)
            if cfg.peso_vol > 0:
                perdida = perdida + cfg.peso_vol * ((vol - conj.VT[idx]) ** 2)[m].mean()
            opt.zero_grad()
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(red.parameters(), 1.0)
            opt.step()
            perdidas.append(float(perdida.detach()))
        if cfg.paciencia > 0 or verbose:
            P, _ = predecir(red, conj, valid, cfg)
            i, g = ganancia_suave(conj, P, valid, cfg.costo_bps)
            sv = sharpe_cartera(g, conj.fecha[i])
            sv = -99.0 if not np.isfinite(sv) else sv
        else:
            sv = float("nan")                              # epocas fijas: no hace falta mirar la validacion
        hist.append({"epoca": ep + 1, "perdida": float(np.mean(perdidas)) if perdidas else np.nan, "sharpe_val": sv})
        if verbose:
            print(f"      epoca {ep + 1:2d}: perdida {hist[-1]['perdida']:.3f} | Sharpe validacion {sv:.2f}", flush=True)
        if al_terminar_epoca is not None:
            al_terminar_epoca(ep + 1, red)
        if cfg.paciencia <= 0:
            mejor = sv
            continue
        if sv > mejor + 1e-4:
            mejor, espera = sv, 0
            estado = {k: v.detach().clone() for k, v in red.state_dict().items()}
        else:
            espera += 1
            if espera >= cfg.paciencia:
                break
    if estado is not None and cfg.paciencia > 0:
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


def ganancia_dura(RB, D, costo_bps):
    """Con posiciones discretas y nocional fijo: ganancia neta en bps por sesion, numero de operaciones
    (rachas del mismo signo) y rotacion. `RB`: retorno de cada tramo en bps."""
    rot = np.abs(D[:, 0]) + np.abs(np.diff(D, axis=1)).sum(1) + np.abs(D[:, -1])
    neto = (D * RB).sum(1) - (costo_bps / 2) * rot
    cambio = np.diff(np.c_[np.zeros(len(D)), D], axis=1) != 0
    return neto, (cambio & (D != 0)).sum(1), rot


def ganancia_dura_norm(conj, D, filas, costo_bps):
    """Lo mismo en unidades de volatilidad y con el coste comun (para juntar simbolos en la validacion)."""
    y = conj.Y[filas].cpu().numpy()
    lado = (costo_bps / 2) * 1e-4 / conj.sig_costo[filas].cpu().numpy()
    rot = np.abs(D[:, 0]) + np.abs(np.diff(D, axis=1)).sum(1) + np.abs(D[:, -1])
    return (D * y).sum(1) - lado * rot


UMBRALES = (1.0, 0.5, 0.0)


def elegir_umbral(conj, P, valid, obj, min_ops_dia=0.55, umbrales=UMBRALES):
    """Umbral de operacion, como multiplo del tamano tipico de la senal.

    Regla fija, sin optimizar nada: el mas exigente de la lista (operar solo cuando la senal supera su
    tamano tipico) que deje en la validacion al menos `min_ops_dia` operaciones por sesion en el
    objetivo. El laboratorio pide un minimo de operaciones para poder juzgar una estrategia; elegir el
    umbral por su resultado en unas pocas semanas de validacion seria ajustar al ruido.
    """
    filas = np.flatnonzero(valid & np.isfinite(P[:, 0]))
    q = escala(P[filas])
    m = conj.sym[filas] == obj
    for u in umbrales:
        D = discretizar(P[filas][m], u, q)
        if ganancia_dura(conj.t["RB"][filas][m], D, 0.0)[1].mean() >= min_ops_dia:
            return u, q
    return umbrales[-1], q
