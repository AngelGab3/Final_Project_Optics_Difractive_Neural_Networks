"""Datos: MNIST, dígitos 'cambiados' (tapados o alterados) y codificación en amplitud."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def load_mnist(root: str, train: bool):
    from torchvision.datasets import MNIST

    ds = MNIST(root, train=train, download=True)
    return ds.data.clone(), ds.targets.clone()


# ---------------------------------------------------------------------------
# Dígitos cambiados (clase X). El paper no publica este conjunto; lo generamos con
# dos tipos de deformación inspirados en la Fig. 1(a) y 3(a):
#   - 'cover' : se tapa (pone a cero) un rectángulo grande del dígito (p.ej. un 7 tapado)
#   - 'alter' : se le añaden trazos que no pertenecen al dígito (p.ej. un 5 alterado)
# ---------------------------------------------------------------------------
def _bbox(img: np.ndarray):
    ys, xs = np.nonzero(img > 64)
    if len(ys) == 0:
        return 0, 27, 0, 27
    return ys.min(), ys.max(), xs.min(), xs.max()


def _draw_line(img, p0, p1, width, rng):
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) * 2) + 2
    for t in np.linspace(0, 1, n):
        y = p0[0] + t * (p1[0] - p0[0])
        x = p0[1] + t * (p1[1] - p0[1])
        y0, x0 = int(round(y - width / 2)), int(round(x - width / 2))
        img[max(y0, 0):max(y0 + width, 0), max(x0, 0):max(x0 + width, 0)] = 255


def cover_digit(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    y0, y1, x0, x1 = _bbox(img)
    h, w = y1 - y0 + 1, x1 - x0 + 1
    for _ in range(50):
        ch = int(rng.uniform(0.35, 0.6) * h) + 1
        cw = int(rng.uniform(0.5, 1.0) * w) + 1
        cy = rng.integers(y0, max(y0 + 1, y1 - ch + 2))
        cx = rng.integers(x0, max(x0 + 1, x1 - cw + 2))
        trial = out.copy()
        trial[cy:cy + ch, cx:cx + cw] = 0
        removed = 1 - (trial > 64).sum() / max((img > 64).sum(), 1)
        if 0.25 <= removed <= 0.65:
            return trial
    return trial


def alter_digit(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    y0, y1, x0, x1 = _bbox(img)
    for _ in range(rng.integers(2, 4)):
        p0 = (rng.uniform(y0, y1), rng.uniform(x0, x1))
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(8, 16)
        p1 = (np.clip(p0[0] + L * np.sin(ang), 2, 25), np.clip(p0[1] + L * np.cos(ang), 2, 25))
        _draw_line(out, p0, p1, int(rng.integers(2, 4)), rng)
    return out


def make_changed(images: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), count, replace=count > len(images))
    imgs = images[idx].numpy()
    out = np.empty_like(imgs)
    for i, im in enumerate(imgs):
        out[i] = cover_digit(im, rng) if rng.random() < 0.5 else alter_digit(im, rng)
    return torch.from_numpy(out)


def build_split(root: str, train: bool, n_digits: int, n_changed: int, seed: int,
                num_classes: int):
    """Devuelve (imágenes uint8 (M,28,28), etiquetas). La clase X tiene índice 10."""
    X, y = load_mnist(root, train)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    Xd, yd = X[perm[:n_digits]], y[perm[:n_digits]]
    if n_changed > 0:
        assert num_classes == 11
        src = X[perm[n_digits:]] if len(X) - n_digits >= n_changed else X
        Xc = make_changed(src, n_changed, seed + 1)
        Xd = torch.cat([Xd, Xc])
        yd = torch.cat([yd, torch.full((n_changed,), 10, dtype=yd.dtype)])
    return Xd, yd


# ---------------------------------------------------------------------------
# Codificación en el plano de entrada
# ---------------------------------------------------------------------------
class Encoder:
    """Convierte imágenes 28x28 en el campo complejo de entrada (B, n, n).

    El dígito se escala (vecino más cercano, como los objetos pixelados de la Fig. 3(a))
    a digit_frac*n píxeles y se centra. Con binarize=True el objeto es una máscara
    binaria transmisiva (cromo sobre vidrio), que es lo que se fabricó en el paper.
    """

    def __init__(self, n: int, digit_frac: float = 0.7, binarize: bool = True):
        self.n = n
        self.m = max(1, int(round(digit_frac * n)))
        self.binarize = binarize

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        x = imgs.float().unsqueeze(1) / 255.0
        x = F.interpolate(x, size=(self.m, self.m), mode="nearest").squeeze(1)
        if self.binarize:
            x = (x > 0.5).float()
        u = x.new_zeros(x.shape[0], self.n, self.n)
        s = (self.n - self.m) // 2
        u[:, s:s + self.m, s:s + self.m] = x
        return u.to(torch.complex64)

    def plane_wave(self, batch: int = 1, device=None) -> torch.Tensor:
        """Onda plana sin dígito (Fig. 5): apertura cuadrada abierta del tamaño del objeto."""
        u = torch.zeros(batch, self.n, self.n, device=device)
        s = (self.n - self.m) // 2
        u[:, s:s + self.m, s:s + self.m] = 1.0
        return u.to(torch.complex64)
