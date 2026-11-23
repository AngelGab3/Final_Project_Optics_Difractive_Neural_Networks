import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from d2nn import optics
from d2nn.config import WL, Config, fig6_case, scaled_full
from d2nn.data import Encoder, alter_digit, cover_digit
from d2nn.model import D2NN, detector_boxes


def test_eq9_paper_numbers():
    # 1000x1000 neuronas de 4 µm a 632.8 nm -> D_min ~ 5 cm (el paper usa ~5 cm)
    assert abs(optics.d_min(1000, 4e-6, WL) - 0.0504) < 5e-4
    # caso 1 y 2 de la Fig. 6
    assert abs(optics.d_min(80, 0.53 * WL, WL) / WL - 14.9) < 0.1
    assert abs(optics.d_min(80, 6 * WL, WL) / WL - 5740) < 5


def test_eq9_equals_bandlimit_at_nyquist():
    for n in [50, 80, 1000]:
        for nf in [0.7, 1.0, 3.0, 12.0]:
            d = nf * WL
            D = optics.d_min(n, d, WL)
            assert math.isclose(optics.band_limit(2 * n, d, D, WL), 1 / (2 * d), rel_tol=1e-9)


def test_energy_conservation_short_distance():
    torch.manual_seed(0)
    n, dx = 64, 2e-6
    u = torch.zeros(1, n, n, dtype=torch.complex128)
    u[0, 24:40, 24:40] = torch.exp(1j * torch.rand(16, 16, dtype=torch.float64))
    v = optics.propagate(u, dx, 50e-6, WL, crop=False)
    assert abs(v.abs().pow(2).sum() / u.abs().pow(2).sum() - 1) < 1e-3


def test_fresnel_square():
    n, dx, half, z = 512, 1e-6, 50e-6, 2e-3
    c = (np.arange(n) - (n - 1) / 2) * dx
    Y, X = np.meshgrid(c, c, indexing="ij")
    u0 = torch.tensor(((abs(X) <= half) & (abs(Y) <= half)).astype(np.complex128))
    u = optics.propagate(u0[None], dx, z, WL)[0].numpy()
    an = optics.fresnel_square_aperture(X[n // 2], Y[n // 2], half, z, WL)
    I1, I2 = abs(u[n // 2]) ** 2, abs(an) ** 2
    assert abs(I1 - I2).max() / I2.max() < 0.02


def test_model_forward_backward():
    cfg = fig6_case(2, n=32)
    boxes = detector_boxes(cfg.n, cfg.layout, 0.1)
    m = D2NN(cfg.n, cfg.d_f, cfg.wl, cfg.distances, 5, boxes)
    x = torch.rand(3, 32, 32).to(torch.complex64)
    E, I = m(x)
    assert E.shape == (3, 10) and I.shape == (3, 32, 32)
    loss = torch.nn.functional.cross_entropy(m.logits(E), torch.tensor([0, 1, 2]))
    loss.backward()
    assert m.theta.grad is not None and torch.isfinite(m.theta.grad).all()
    ph = m.phases()
    assert ph.min() >= 0 and ph.max() <= math.pi


def test_detectors_disjoint():
    for layout, k in [("paper11", 11), ("343", 10)]:
        b = detector_boxes(1000, layout, 0.05)
        assert len(b) == k
        m = np.zeros((1000, 1000))
        for y0, x0, s in b:
            m[y0:y0 + s, x0:x0 + s] += 1
        assert m.max() == 1


def test_changed_digits_differ():
    rng = np.random.default_rng(0)
    img = np.zeros((28, 28), np.uint8)
    img[5:23, 12:16] = 255
    assert (cover_digit(img, rng) != img).any()
    assert (alter_digit(img, rng) != img).any()


def test_encoder_and_scaled_config():
    enc = Encoder(100, 0.7)
    u = enc(torch.full((2, 28, 28), 255, dtype=torch.uint8))
    assert u.shape == (2, 100, 100) and u.abs().sum() == 2 * 70 * 70
    c = scaled_full(250)
    ref = Config()
    r1 = ref.distance / optics.d_min(ref.n, ref.d_f, ref.wl)
    r2 = c.distance / optics.d_min(c.n, c.d_f, c.wl)
    assert math.isclose(r1, r2)


def test_analytical_wirtinger_gradient():
    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "07_check_gradient.py",
    )
    spec = importlib.util.spec_from_file_location("check_gradient", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.compare_gradients(n=32, batch_size=4, seed=0)
    assert res["total_rel_error"] < 1e-11
    assert all(e < 1e-11 for e in res["layer_rel_errors"])
