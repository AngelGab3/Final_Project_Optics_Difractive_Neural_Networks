"""Modelo D2NN de solo fase (Ec. 1-5 del paper)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .optics import AngularSpectrum


# ---------------------------------------------------------------------------
# Detectores en el plano de salida
# ---------------------------------------------------------------------------
def detector_boxes(n: int, layout: str, size_frac: float) -> list[tuple[int, int, int]]:
    """Devuelve cajas (y0, x0, s) en píxeles, una por clase, en orden de clase.

    'paper11': 3 columnas x 5 filas como en la Fig. 3(b) del paper:
        0 1 2 / 3 · 4 / 5 · 6 / 7 · 8 / 9 · X
    '343': 10 detectores en filas de 3-4-3 (diseño típico de Lin et al., 2018).
    """
    s = max(1, int(round(size_frac * n)))
    centers: list[tuple[float, float]] = []
    if layout == "paper11":
        rows = [0.14, 0.32, 0.50, 0.68, 0.86]
        cols = [0.22, 0.50, 0.78]
        centers = [(rows[0], cols[0]), (rows[0], cols[1]), (rows[0], cols[2])]
        for r in rows[1:]:
            centers += [(r, cols[0]), (r, cols[2])]
    elif layout == "343":
        for r, xs in [(0.22, [0.25, 0.50, 0.75]),
                      (0.50, [0.16, 0.39, 0.61, 0.84]),
                      (0.78, [0.25, 0.50, 0.75])]:
            centers += [(r, x) for x in xs]
    else:
        raise ValueError(layout)
    boxes = []
    for cy, cx in centers:
        y0 = int(round(cy * n - s / 2))
        x0 = int(round(cx * n - s / 2))
        boxes.append((min(max(y0, 0), n - s), min(max(x0, 0), n - s), s))
    return boxes


def detector_masks(n: int, boxes) -> torch.Tensor:
    m = torch.zeros(len(boxes), n, n)
    for k, (y0, x0, s) in enumerate(boxes):
        m[k, y0:y0 + s, x0:x0 + s] = 1.0
    return m


# ---------------------------------------------------------------------------
# Red
# ---------------------------------------------------------------------------
class D2NN(nn.Module):
    """Red difractiva de solo fase.

    entrada -> [z0] -> L1 -> [z1] -> L2 -> ... -> L_L -> [z_L] -> plano de salida.
    Cada neurona i de la capa l tiene t = exp(j Phi_i), Phi = phase_max * sigmoid(theta)
    (el paper limita la fase a [0, pi] con una sigmoide para facilitar la fabricación).
    """

    def __init__(self, n: int, dx: float, wl: float, distances: list[float], num_layers: int,
                 boxes, phase_max: float = math.pi, pad_factor: int = 2,
                 bandlimit: bool = True, temperature: float = 10.0,
                 use_checkpoint: bool = False, init_std: float = 0.1):
        super().__init__()
        assert len(distances) == num_layers + 1
        self.n, self.dx, self.wl = n, dx, wl
        self.num_layers = num_layers
        self.phase_max = phase_max
        self.temperature = temperature
        self.use_checkpoint = use_checkpoint
        self.boxes = boxes
        self.distances = list(distances)
        # Propagadores compartidos cuando la distancia se repite (ahorra memoria).
        uniq: dict[float, int] = {}
        props = []
        self.prop_idx = []
        for z in distances:
            if z not in uniq:
                uniq[z] = len(props)
                props.append(AngularSpectrum(n, dx, z, wl, pad_factor, bandlimit))
            self.prop_idx.append(uniq[z])
        self.props = nn.ModuleList(props)
        self.theta = nn.Parameter(init_std * torch.randn(num_layers, n, n))
        self.register_buffer("det", detector_masks(n, boxes), persistent=False)

    # -- utilidades ---------------------------------------------------------
    def phases(self) -> torch.Tensor:
        return self.phase_max * torch.sigmoid(self.theta)

    def prop(self, i: int, u: torch.Tensor) -> torch.Tensor:
        return self.props[self.prop_idx[i]](u)

    def _layer(self, u, phase, i):
        return self.prop(i + 1, u * torch.polar(torch.ones_like(phase), phase))

    # -- paso hacia adelante ------------------------------------------------
    def field_out(self, u: torch.Tensor, phases: torch.Tensor | None = None) -> torch.Tensor:
        """u: (B, n, n) complejo en el plano de entrada -> campo en el plano de salida."""
        ph = self.phases() if phases is None else phases
        u = self.prop(0, u)
        for l in range(self.num_layers):
            if self.use_checkpoint and self.training:
                u = checkpoint(self._layer, u, ph[l], l, use_reentrant=False)
            else:
                u = self._layer(u, ph[l], l)
        return u

    def forward(self, u: torch.Tensor, phases: torch.Tensor | None = None):
        """Devuelve (energía en cada detector (B,K), intensidad de salida (B,n,n))."""
        I = self.field_out(u, phases).abs() ** 2
        E = torch.einsum("bij,kij->bk", I, self.det)
        return E, I

    def logits(self, E: torch.Tensor) -> torch.Tensor:
        """Energía relativa de cada detector, escalada por una temperatura, para la
        softmax-cross-entropy (pérdida usada en el paper)."""
        return self.temperature * E / (E.sum(1, keepdim=True) + 1e-12)
