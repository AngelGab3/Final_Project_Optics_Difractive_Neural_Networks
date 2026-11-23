"""Óptica escalar: propagación por espectro angular (con y sin limitación de banda),
Rayleigh-Sommerfeld directo, soluciones analíticas y la fórmula de conectividad (Ec. 6-9).

Convención: dependencia temporal exp(-j w t); una onda plana avanza como exp(+j k z).
Todas las longitudes en metros.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Fórmulas del paper (Ec. 6-9)
# ---------------------------------------------------------------------------
def phi_max(d_f: float, wl: float) -> float:
    """Semiángulo máximo de difracción de una neurona de tamaño d_f (Ec. 6)."""
    return math.asin(min(1.0, wl / (2.0 * d_f)))


def d_min(n_side: int, d_f: float, wl: float) -> float:
    """Distancia mínima entre capas para conectividad total (Ec. 9).

    D >= sqrt(N) * d_f * sqrt(4 d_f^2 / wl^2 - 1),  con sqrt(N) = neuronas por lado.
    Si d_f <= wl/2 el semiángulo es 90° y cualquier D > 0 cumple.
    """
    r = 4.0 * d_f**2 / wl**2 - 1.0
    return 0.0 if r <= 0 else n_side * d_f * math.sqrt(r)


def critical_distance(n_pad: int, dx: float, wl: float) -> float:
    """Distancia a partir de la cual la función de transferencia del espectro angular
    ingenuo (sin limitar banda) queda submuestreada: z_c = N_pad dx^2 / wl."""
    return n_pad * dx**2 / wl


def band_limit(n_pad: int, dx: float, z: float, wl: float) -> float:
    """Frecuencia límite del espectro angular de banda limitada
    (Matsushima & Shimobaba, Opt. Express 17, 19662 (2009))."""
    df = 1.0 / (n_pad * dx)
    return 1.0 / (wl * math.sqrt((2.0 * df * z) ** 2 + 1.0))


# ---------------------------------------------------------------------------
# Espectro angular
# ---------------------------------------------------------------------------
def asm_transfer(n_pad: int, dx: float, z: float, wl: float, bandlimit: bool = True,
                 dtype=torch.complex64, device=None) -> torch.Tensor:
    """Función de transferencia H(u,v) = exp(j 2 pi z sqrt(1/wl^2 - u^2 - v^2)) (Ec. 3-5).

    Se calcula en float64 porque k*z puede ser ~1e5-1e6 rad y float32 perdería la fase.
    Las ondas evanescentes se incluyen (decaen exponencialmente).
    """
    f = torch.fft.fftfreq(n_pad, d=dx, dtype=torch.float64)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    arg = 1.0 / wl**2 - fx**2 - fy**2
    kz = 2.0 * math.pi * torch.sqrt(arg.abs())
    phase = torch.where(arg >= 0, kz * z, torch.zeros_like(kz))
    amp = torch.where(arg >= 0, torch.ones_like(kz), torch.exp(-kz * abs(z)))
    H = torch.polar(amp, phase)
    if bandlimit:
        fl = band_limit(n_pad, dx, abs(z), wl)
        H = H * ((fx.abs() <= fl) & (fy.abs() <= fl))
    return H.to(dtype=dtype, device=device)


def _pad_center(u: torch.Tensor, n_pad: int) -> torch.Tensor:
    n = u.shape[-1]
    if n_pad == n:
        return u
    out = u.new_zeros(*u.shape[:-2], n_pad, n_pad)
    s = (n_pad - n) // 2
    out[..., s:s + n, s:s + n] = u
    return out


def _crop_center(u: torch.Tensor, n: int) -> torch.Tensor:
    n_pad = u.shape[-1]
    if n_pad == n:
        return u
    s = (n_pad - n) // 2
    return u[..., s:s + n, s:s + n]


class AngularSpectrum(nn.Module):
    """Propaga un campo complejo (..., n, n) una distancia z.

    El campo se rellena con ceros hasta pad_factor*n (evita la convolución circular),
    se multiplica por H en Fourier y se recorta de nuevo a n×n. El recorte equivale a que
    cada capa difractiva tenga una apertura finita del tamaño de la capa (w = n dx).
    """

    def __init__(self, n: int, dx: float, z: float, wl: float, pad_factor: int = 2,
                 bandlimit: bool = True):
        super().__init__()
        self.n, self.dx, self.z, self.wl = n, dx, z, wl
        self.n_pad = int(round(pad_factor * n))
        self.register_buffer("H", asm_transfer(self.n_pad, dx, z, wl, bandlimit), persistent=False)

    def forward(self, u: torch.Tensor, crop: bool = True) -> torch.Tensor:
        U = torch.fft.fft2(_pad_center(u, self.n_pad))
        out = torch.fft.ifft2(U * self.H)
        return _crop_center(out, self.n) if crop else out


def propagate(u: torch.Tensor, dx: float, z: float, wl: float, pad_factor: int = 2,
              bandlimit: bool = True, crop: bool = True) -> torch.Tensor:
    """Versión funcional de AngularSpectrum (para validación y gráficas)."""
    n = u.shape[-1]
    n_pad = int(round(pad_factor * n))
    H = asm_transfer(n_pad, dx, z, wl, bandlimit, dtype=u.dtype, device=u.device)
    out = torch.fft.ifft2(torch.fft.fft2(_pad_center(u, n_pad)) * H)
    return _crop_center(out, n) if crop else out


# ---------------------------------------------------------------------------
# Referencias para validar
# ---------------------------------------------------------------------------
def rayleigh_sommerfeld_direct(u0: np.ndarray, dx: float, z: float, wl: float,
                               out_coords: np.ndarray | None = None) -> np.ndarray:
    """Integral de Rayleigh-Sommerfeld (1ª especie) por suma directa, O(N^2 M).

    h(r) = (z / 2 pi) (1/r - j k) exp(j k r) / r^2.   Solo para mallas pequeñas.
    u0: (n, n) complejo en una malla centrada de paso dx.
    out_coords: (M, 2) puntos (x, y) de observación; por defecto la misma malla.
    """
    n = u0.shape[0]
    c = (np.arange(n) - (n - 1) / 2) * dx
    Y, X = np.meshgrid(c, c, indexing="ij")
    src = np.stack([X.ravel(), Y.ravel()], 1)
    a = u0.ravel()
    mask = a != 0
    src, a = src[mask], a[mask]
    if out_coords is None:
        out_coords = np.stack([X.ravel(), Y.ravel()], 1)
    k = 2 * np.pi / wl
    out = np.zeros(len(out_coords), dtype=np.complex128)
    for i0 in range(0, len(out_coords), 256):
        p = out_coords[i0:i0 + 256]
        dxs = p[:, None, 0] - src[None, :, 0]
        dys = p[:, None, 1] - src[None, :, 1]
        r = np.sqrt(dxs**2 + dys**2 + z**2)
        h = z / (2 * np.pi) * (1 / r - 1j * k) * np.exp(1j * k * r) / r**2
        out[i0:i0 + 256] = (h * a[None, :]).sum(1) * dx * dx
    return out


def circular_aperture_on_axis(z: np.ndarray, a: float, wl: float) -> np.ndarray:
    """Campo exacto (Rayleigh-Sommerfeld 1ª especie) en el eje de una apertura circular
    de radio a iluminada por onda plana unitaria:
        U(z) = exp(jkz) - z/sqrt(z^2+a^2) exp(jk sqrt(z^2+a^2))."""
    k = 2 * np.pi / wl
    R = np.sqrt(z**2 + a**2)
    return np.exp(1j * k * z) - z / R * np.exp(1j * k * R)


def fresnel_square_aperture(x: np.ndarray, y: np.ndarray, half: float, z: float,
                            wl: float) -> np.ndarray:
    """Difracción de Fresnel (analítica, integrales de Fresnel) de una apertura cuadrada
    de lado 2*half, sin el factor de fase global exp(jkz)."""
    from scipy.special import fresnel

    s = np.sqrt(2.0 / (wl * z))

    def one_d(t):
        S1, C1 = fresnel(s * (half - t))
        S2, C2 = fresnel(s * (-half - t))
        return ((C1 - C2) + 1j * (S1 - S2)) / np.sqrt(2)

    # factor (1/j) y producto de las dos integrales unidimensionales
    return (one_d(x) * one_d(y)) / 1j
