"""Verificación numérica del gradiente analítico (cálculo de Wirtinger, Ecs. 3.8-3.13).

Implementa los pasos 1 a 5 de la sección 3.5 a mano en doble precisión sobre una red
de 32x32 neuronas y 5 capas con fases aleatorias, y compara el resultado capa por capa
contra la diferenciación automática de PyTorch.

Uso:  python scripts/07_check_gradient.py [--n 32] [--bs 4] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from d2nn import optics
from d2nn.config import fig6_case
from d2nn.model import D2NN, detector_boxes


def build_double_model(n: int = 32, seed: int = 0) -> D2NN:
    """Construye una D2NN de 5 capas en doble precisión (float64 / complex128)."""
    torch.manual_seed(seed)
    cfg = fig6_case(2, n=n)
    boxes = detector_boxes(cfg.n, cfg.layout, cfg.det_frac)
    model = D2NN(
        cfg.n, cfg.d_f, cfg.wl, cfg.distances, cfg.num_layers, boxes,
        phase_max=cfg.phase_max, pad_factor=cfg.pad_factor,
        bandlimit=cfg.bandlimit, temperature=cfg.temperature,
    ).double()
    for prop_mod in model.props:
        prop_mod.H = optics.asm_transfer(
            prop_mod.n_pad, prop_mod.dx, prop_mod.z, prop_mod.wl,
            cfg.bandlimit, dtype=torch.complex128, device=model.theta.device,
        )
    with torch.no_grad():
        model.theta.uniform_(-2.0, 2.0)
    return model


@torch.no_grad()
def analytical_wirtinger_gradient(model: D2NN, u_in: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calcula dL/dtheta usando exactamente los pasos 1 a 5 del cálculo de Wirtinger."""
    B = u_in.shape[0]
    ph = model.phases()
    sig = torch.sigmoid(model.theta)
    dphi_dtheta = model.phase_max * sig * (1.0 - sig)

    # Propagación hacia adelante guardando el campo W^l justo después de cada capa
    u = model.prop(0, u_in)
    fields_after_layer = []
    for l in range(model.num_layers):
        W = u * torch.polar(torch.ones_like(ph[l]), ph[l])
        fields_after_layer.append(W)
        u = model.prop(l + 1, W)
    y_out = u

    # Paso 1 (Ec. 3.8): sensibilidad respecto a las energías detectoras (incluyendo 1e-12 de logits)
    E = torch.einsum("bij,kij->bk", y_out.abs() ** 2, model.det)
    S = E.sum(1, keepdim=True) + 1e-12
    probs = F.softmax(model.temperature * E / S, dim=1)
    one_hot = F.one_hot(targets, num_classes=model.det.shape[0]).to(E.dtype)
    diff = probs - one_hot
    eps = (model.temperature / (B * S)) * (diff - (diff * (E / S)).sum(1, keepdim=True))

    # Paso 2 (Ec. 3.9) y Paso 3 (Ec. 3.10): campo de error en el plano de salida
    c = torch.einsum("bk,kij->bij", eps, model.det)
    G = c * y_out

    # Pasos 4 y 5 (Ecs. 3.11-3.13): onda de error hacia atrás e interferencia con W^l
    grads_theta = []
    for l in range(model.num_layers - 1, -1, -1):
        prop_mod = model.props[model.prop_idx[l + 1]]
        G_padded = optics._pad_center(G, prop_mod.n_pad)
        G_W = optics._crop_center(torch.fft.ifft2(torch.fft.fft2(G_padded) * prop_mod.H.conj()), model.n)
        dL_dphi = (2.0 * (G_W * fields_after_layer[l].conj()).imag).sum(0)
        grads_theta.append(dL_dphi * dphi_dtheta[l])
        G = G_W * torch.polar(torch.ones_like(ph[l]), -ph[l])

    return torch.stack(list(reversed(grads_theta)))


def compare_gradients(n: int = 32, batch_size: int = 4, seed: int = 0) -> dict:
    """Compara el gradiente analítico contra PyTorch autograd en doble precisión."""
    model = build_double_model(n=n, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    u_in = (
        torch.randn(batch_size, n, n, generator=gen, dtype=torch.float64)
        + 1j * torch.randn(batch_size, n, n, generator=gen, dtype=torch.float64)
    )
    targets = torch.randint(0, model.det.shape[0], (batch_size,), generator=gen)

    model.zero_grad(set_to_none=True)
    E, _ = model(u_in)
    loss = F.cross_entropy(model.logits(E), targets)
    loss.backward()
    g_auto = model.theta.grad.detach().clone()

    g_man = analytical_wirtinger_gradient(model, u_in, targets)

    layer_errors = [
        float(((g_man[l] - g_auto[l]).norm() / g_auto[l].norm()).item())
        for l in range(model.num_layers)
    ]
    total_error = float(((g_man - g_auto).norm() / g_auto.norm()).item())
    return {
        "n": n,
        "batch_size": batch_size,
        "num_layers": model.num_layers,
        "dtype": "float64/complex128",
        "layer_rel_errors": layer_errors,
        "total_rel_error": total_error,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/01_optics/gradient_check.json")
    args = ap.parse_args()

    res = compare_gradients(n=args.n, batch_size=args.bs, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    headers = [f"Capa {i + 1}" for i in range(res["num_layers"])] + ["Total"]
    vals = [f"{e:.2e}" for e in res["layer_rel_errors"]] + [f"{res['total_rel_error']:.2e}"]
    print("Error relativo ||g_analitico - g_PyTorch|| / ||g_PyTorch|| (doble precision):")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    print("| " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
