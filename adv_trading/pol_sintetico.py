"""
RED DE POLITICA - mundos sinteticos para comprobar el metodo sin mirar resultados reales.

  mundo nulo      el retorno real con el signo al azar por (fecha, tramo), comun a todos los simbolos.
                  Conserva volatilidad, horario y correlacion entre simbolos, pero rompe cualquier relacion
                  entre pasado y futuro: ahi ninguna estrategia puede ganar, salvo por azar.
  senal plantada  al mundo nulo se le suma una regla conocida, de tamano `beta`. Sirve para saber si la
                  red encuentra una senal cuando de verdad existe, y cuanta de ella recupera.
"""
import numpy as np

import pol_datos as PD
import pol_entrenar as PE

SENALES = {"momento": 0.06, "regimen": 0.12, "memoria": 0.12}


def senal(T, tipo):
    """Regla conocida g (por muestra y tramo) que se planta en el retorno."""
    v = T["variables"].index
    X = T["X"]
    if tipo == "momento":            # el movimiento del dia continua (relacion suave, visible en una variable)
        return np.tanh(X[:, :, v("r_dia")])
    if tipo == "regimen":            # continua solo por la tarde y en dias de volatilidad alta (interaccion)
        return (np.sign(X[:, :, v("r_cc")]) * (X[:, :, v("rv_rel")] > 0) * (X[:, :, v("paso")] >= 0.6)).astype(np.float32)
    if tipo == "memoria":            # lo ocurrido hasta las 10:00 marca la ultima hora (hay que recordarlo)
        g = np.zeros(X.shape[:2], np.float32)
        g[:, -2:] = np.tanh(X[:, :1, v("r_cc")])
        return g
    raise ValueError(tipo)


def plantar(T, tipo=None, beta=0.0, semilla=123):
    """Copia de la tabla en el mundo nulo; si se da `tipo`, con la senal plantada."""
    rng = np.random.default_rng(semilla)
    _, inv = np.unique(T["fecha"], return_inverse=True)
    signo = rng.choice([-1.0, 1.0], size=(inv.max() + 1, PD.J)).astype(np.float32)[inv]
    y = T["Y"] * signo
    if tipo is not None:
        y = y + beta * T["Y"].std(axis=0, keepdims=True) * senal(T, tipo)
    S = dict(T)
    S["Y"] = np.clip(y, -5, 5).astype(np.float32)
    S["RB"] = (1e4 * S["Y"] * T["sig"][:, None]).astype(np.float32)
    return S


def sharpe_esperado(T, tipo, beta, P, filas, sorteos=30, semilla=7):
    """Sharpe bruto medio de las posiciones P frente a muchos sorteos del mundo nulo (mas la senal).
    Promediar sorteos quita el azar de un unico periodo de prueba."""
    rng = np.random.default_rng(semilla)
    f = T["fecha"][filas]
    _, inv = np.unique(f, return_inverse=True)
    yr, plant = T["Y"][filas], beta * T["Y"].std(axis=0, keepdims=True) * senal(T, tipo)[filas]
    out = []
    for _ in range(sorteos):
        sg = rng.choice([-1.0, 1.0], size=(inv.max() + 1, PD.J))[inv]
        out.append(PE.sharpe_cartera((P * (yr * sg + plant)).sum(1), f))
    return float(np.mean(out))
