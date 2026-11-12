"""
RED DE POLITICA - arquitecturas.

Todas reciben la secuencia de pasos de decision [lote, T, variables] y el indice del simbolo, y devuelven
para CADA paso una posicion en [-1, 1] (tanh) y un pronostico de volatilidad del tramo siguiente.
Todas son CAUSALES: la salida del paso t solo depende de los pasos <= t (hay una prueba que lo verifica).

Base elegida: VSN + LSTM ("vlstm"), la arquitectura con mayor Sharpe en el benchmark de Oxford de 2026
(Saly-Kaufmann, Wood, Calliess y Zohren, arXiv 2603.01820), entrenada como alli: optimizando directamente
el Sharpe. El resto de estructuras son las familias que ese estudio compara, para poder repetir la
comparacion con datos intradia:

  lineal       una capa lineal por paso (sin memoria): referencia minima
  mlp          perceptron por paso (sin memoria)
  lstm         LSTM clasica ("Deep Momentum Network")
  vlstm        seleccion de variables (VSN) + LSTM + enriquecimiento estatico   <- base
  tft          vlstm + auto-atencion causal ("Momentum Transformer")
  transformer  codificador Transformer causal sobre los pasos
  xlstm        sLSTM (compuertas exponenciales) en bloque residual
  vxlstm       VSN + sLSTM
"""
import math

import torch
import torch.nn.functional as Fn
from torch import nn

ESTRUCTURAS = ("lineal", "mlp", "lstm", "vlstm", "tft", "transformer", "xlstm", "vxlstm")


# ------------------------------------------------------------------ bloques
class GateAddNorm(nn.Module):
    """Compuerta GLU + conexion residual + normalizacion."""

    def __init__(self, h, dropout=0.0):
        super().__init__()
        self.dp, self.fc, self.norm = nn.Dropout(dropout), nn.Linear(h, 2 * h), nn.LayerNorm(h)

    def forward(self, x, skip):
        return self.norm(Fn.glu(self.fc(self.dp(x)), dim=-1) + skip)


class GRN(nn.Module):
    """Gated Residual Network (como en TFT), con contexto estatico opcional."""

    def __init__(self, n_in, h, n_out, dropout=0.0, contexto=None):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(n_in, h), nn.Linear(h, h)
        self.ctx = nn.Linear(contexto, h, bias=False) if contexto else None
        self.salto = nn.Linear(n_in, n_out) if n_in != n_out else None
        self.dp, self.puerta, self.norm = nn.Dropout(dropout), nn.Linear(h, 2 * n_out), nn.LayerNorm(n_out)

    def forward(self, x, c=None):
        r = x if self.salto is None else self.salto(x)
        a = self.fc1(x)
        if self.ctx is not None and c is not None:
            a = a + self.ctx(c)
        a = self.fc2(Fn.elu(a))
        return self.norm(Fn.glu(self.puerta(self.dp(a)), dim=-1) + r)


class VSN(nn.Module):
    """Variable Selection Network vectorizada: cada variable tiene su propia mini-red (de dimension `d`)
    y la red decide, en cada paso, cuanto peso dar a cada una (pesos softmax). Misma idea que en TFT,
    sin bucles de Python y con mini-redes pequenas, porque aqui hay muchos menos datos que en futuros."""

    def __init__(self, n_var, h, dropout=0.0, d=8):
        super().__init__()
        self.n_var, self.d = n_var, d
        esc = 1.0 / math.sqrt(d)
        self.w_pre, self.b_pre = nn.Parameter(torch.randn(n_var, d) * 0.5), nn.Parameter(torch.zeros(n_var, d))
        self.w1, self.b1 = nn.Parameter(torch.randn(n_var, d, d) * esc), nn.Parameter(torch.zeros(n_var, d))
        self.w2, self.b2 = nn.Parameter(torch.randn(n_var, d, d) * esc), nn.Parameter(torch.zeros(n_var, d))
        self.wg, self.bg = nn.Parameter(torch.randn(n_var, d, 2 * d) * esc), nn.Parameter(torch.zeros(n_var, 2 * d))
        self.gamma, self.beta = nn.Parameter(torch.ones(n_var, d)), nn.Parameter(torch.zeros(n_var, d))
        self.dp = nn.Dropout(dropout)
        self.seleccion = GRN(n_var * d, min(h, n_var), n_var, dropout, contexto=h)
        self.salida = nn.Linear(d, h)

    def forward(self, x, c=None):
        nb, nt, nv = x.shape
        e = x.unsqueeze(-1) * self.w_pre + self.b_pre                          # [B, T, V, d]
        ev = e.permute(2, 0, 1, 3).reshape(nv, nb * nt, self.d)                # [V, B*T, d]: un producto por variable
        a = Fn.elu(torch.bmm(ev, self.w1) + self.b1.unsqueeze(1))
        a = self.dp(torch.bmm(a, self.w2) + self.b2.unsqueeze(1))
        a = Fn.glu(torch.bmm(a, self.wg) + self.bg.unsqueeze(1), dim=-1)
        a = Fn.layer_norm(a + ev, (self.d,)) * self.gamma.unsqueeze(1) + self.beta.unsqueeze(1)
        pesos = torch.softmax(self.seleccion(e.flatten(-2), None if c is None else c.unsqueeze(1)), dim=-1)
        sal = torch.einsum("vnd,nv->nd", a, pesos.reshape(nb * nt, nv))
        return self.salida(sal).reshape(nb, nt, -1), pesos


class SLSTM(nn.Module):
    """sLSTM de xLSTM (Beck et al., 2024): compuerta de entrada exponencial con estabilizador y estado
    normalizador. Bucle explicito sobre los pasos (las secuencias son cortas)."""

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.wx, self.rh = nn.Linear(h, 4 * h), nn.Linear(h, 4 * h, bias=False)
        with torch.no_grad():
            self.wx.bias[2 * h:3 * h].fill_(2.0)                               # olvido abierto al empezar

    def forward(self, x, estado=None):
        b, t, h = x.shape
        z0 = x.new_zeros(b, h)
        hh, c, n, m = (z0, z0, z0, z0) if estado is None else estado
        gx, out = self.wx(x), []
        for i in range(t):
            z_, i_, f_, o_ = (gx[:, i] + self.rh(hh)).chunk(4, dim=-1)
            lf = Fn.logsigmoid(f_)
            m_n = torch.maximum(lf + m, i_)
            ig, fg = torch.exp(i_ - m_n), torch.exp(lf + m - m_n)
            c = fg * c + ig * torch.tanh(z_)
            n = fg * n + ig
            hh = torch.sigmoid(o_) * (c / (n + 1e-6))
            m = m_n
            out.append(hh)
        return torch.stack(out, dim=1)


class BloqueXLSTM(nn.Module):
    def __init__(self, h, dropout=0.0):
        super().__init__()
        self.n1, self.rec, self.n2 = nn.LayerNorm(h), SLSTM(h), nn.LayerNorm(h)
        hp = int(round(4 * h / 3))
        self.ffn = nn.Sequential(nn.Linear(h, hp), nn.GELU(), nn.Dropout(dropout), nn.Linear(hp, h))
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dp(self.rec(self.n1(x)))
        return x + self.dp(self.ffn(self.n2(x)))


# ------------------------------------------------------------------ redes
class Red(nn.Module):
    """Cabezas comunes: posicion (tanh, arranca cerca de cero) y volatilidad del tramo siguiente."""

    def __init__(self, n_rep):
        super().__init__()
        self.cab_pos, self.cab_vol = nn.Linear(n_rep, 1), nn.Linear(n_rep, 1)
        nn.init.normal_(self.cab_pos.weight, std=1e-3)
        nn.init.zeros_(self.cab_pos.bias)

    def representacion(self, x, sym):
        raise NotImplementedError

    def forward(self, x, sym):
        r = self.representacion(x, sym)
        return torch.tanh(self.cab_pos(r)).squeeze(-1), self.cab_vol(r).squeeze(-1)


class Lineal(Red):
    def __init__(self, n_var, n_sym, h, dropout, **kw):
        super().__init__(n_var)

    def representacion(self, x, sym):
        return x


class _ConEntrada(Red):
    """Proyeccion de entrada + embedding del simbolo (dinamica propia de cada activo)."""

    def __init__(self, n_var, n_sym, h, dropout):
        super().__init__(h)
        self.entrada, self.emb, self.dp = nn.Linear(n_var, h), nn.Embedding(n_sym, h), nn.Dropout(dropout)
        nn.init.normal_(self.emb.weight, std=0.1)

    def _x(self, x, sym):
        return self.entrada(x) + self.emb(sym).unsqueeze(1)


class MLP(_ConEntrada):
    def __init__(self, n_var, n_sym, h, dropout, **kw):
        super().__init__(n_var, n_sym, h, dropout)
        self.fc = nn.Linear(h, h)

    def representacion(self, x, sym):
        return self.dp(Fn.elu(self.fc(self.dp(Fn.elu(self._x(x, sym))))))


class LSTM(_ConEntrada):
    def __init__(self, n_var, n_sym, h, dropout, **kw):
        super().__init__(n_var, n_sym, h, dropout)
        self.rec = nn.LSTM(h, h, batch_first=True)

    def representacion(self, x, sym):
        return self.dp(self.rec(self._x(x, sym))[0])


class XLSTM(_ConEntrada):
    def __init__(self, n_var, n_sym, h, dropout, **kw):
        super().__init__(n_var, n_sym, h, dropout)
        self.bloque, self.norm = BloqueXLSTM(h, dropout), nn.LayerNorm(h)

    def representacion(self, x, sym):
        return self.norm(self.bloque(self._x(x, sym)))


class Transformer(_ConEntrada):
    def __init__(self, n_var, n_sym, h, dropout, t_max=128, capas=2, cabezas=4, **kw):
        super().__init__(n_var, n_sym, h, dropout)
        self.pos = nn.Parameter(torch.zeros(1, t_max, h))
        capa = nn.TransformerEncoderLayer(h, cabezas, 2 * h, dropout, batch_first=True, norm_first=True)
        self.enc, self.norm = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False), nn.LayerNorm(h)

    def representacion(self, x, sym):
        t = x.shape[1]
        mascara = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        return self.norm(self.enc(self._x(x, sym) + self.pos[:, -t:], mask=mascara))


class VLSTM(Red):
    """BASE. VSN -> LSTM (estado inicial dado por el simbolo) -> compuerta residual -> GRN con contexto."""

    recurrente = "lstm"

    def __init__(self, n_var, n_sym, h, dropout, **kw):
        super().__init__(h)
        self.h = h
        self.emb = nn.Embedding(n_sym, h)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.c_sel, self.c_enr = GRN(h, h, h, dropout), GRN(h, h, h, dropout)
        self.vsn = VSN(n_var, h, dropout)
        if self.recurrente == "lstm":
            self.c_h, self.c_c = GRN(h, h, h, dropout), GRN(h, h, h, dropout)
            self.rec = nn.LSTM(h, h, batch_first=True)
        else:
            self.rec = BloqueXLSTM(h, dropout)
        self.puerta, self.grn = GateAddNorm(h, dropout), GRN(h, h, h, dropout, contexto=h)

    def _secuencia(self, x, sym):
        e = self.emb(sym)
        v, self.ultimos_pesos = self.vsn(x, self.c_sel(e))
        if self.recurrente == "lstm":
            o, _ = self.rec(v, (self.c_h(e).unsqueeze(0).contiguous(), self.c_c(e).unsqueeze(0).contiguous()))
        else:
            o = self.rec(v)
        oculto = self.puerta(o, v)
        return self.grn(oculto, self.c_enr(e).unsqueeze(1)), oculto

    def representacion(self, x, sym):
        return self._secuencia(x, sym)[0]


class VXLSTM(VLSTM):
    recurrente = "slstm"


class TFT(VLSTM):
    """Momentum Transformer: la base + auto-atencion causal sobre los pasos."""

    def __init__(self, n_var, n_sym, h, dropout, cabezas=4, **kw):
        super().__init__(n_var, n_sym, h, dropout)
        self.aten = nn.MultiheadAttention(h, cabezas, dropout=dropout, batch_first=True)
        self.p_aten, self.grn2, self.p_sal = GateAddNorm(h, dropout), GRN(h, h, h, dropout), GateAddNorm(h, dropout)

    def representacion(self, x, sym):
        rep, oculto = self._secuencia(x, sym)
        t = x.shape[1]
        mascara = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        a, _ = self.aten(rep, rep, rep, attn_mask=mascara, need_weights=False)
        return self.p_sal(self.grn2(self.p_aten(a, rep)), oculto)


_CLASES = {"lineal": Lineal, "mlp": MLP, "lstm": LSTM, "vlstm": VLSTM, "tft": TFT, "transformer": Transformer,
           "xlstm": XLSTM, "vxlstm": VXLSTM}


def crear(estructura, n_var, n_sym, h=32, dropout=0.3, **kw):
    if estructura not in _CLASES:
        raise ValueError(f"Estructura desconocida: {estructura}. Opciones: {ESTRUCTURAS}")
    return _CLASES[estructura](n_var, n_sym, h, dropout, **kw)


def n_parametros(red):
    return sum(p.numel() for p in red.parameters())


def comprobar_causalidad(estructura, n_var=40, t=24, h=16):
    """Cambia la entrada a partir de un paso y comprueba que las salidas anteriores no se mueven."""
    torch.manual_seed(0)
    red = crear(estructura, n_var, 3, h, 0.0).eval()
    x, sym = torch.randn(4, t, n_var), torch.tensor([0, 1, 2, 0])
    with torch.no_grad():
        p0, v0 = red(x, sym)
        x2 = x.clone()
        x2[:, t // 2:] += 1.0
        p1, v1 = red(x2, sym)
    antes = max((p0[:, :t // 2] - p1[:, :t // 2]).abs().max().item(), (v0[:, :t // 2] - v1[:, :t // 2]).abs().max().item())
    despues = (v0[:, t // 2:] - v1[:, t // 2:]).abs().max().item()
    return antes < 1e-6 and despues > 1e-6
