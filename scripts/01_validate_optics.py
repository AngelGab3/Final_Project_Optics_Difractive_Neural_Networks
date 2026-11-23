"""Fase 1: validación del propagador por espectro angular.

(a) Apertura circular: campo en el eje frente a la solución exacta de Rayleigh-Sommerfeld.
(b) Apertura cuadrada: perfil frente a la solución analítica de Fresnel.
(c) Espectro angular ingenuo vs. de banda limitada vs. Rayleigh-Sommerfeld directo,
    en función de z / z_c (aliasing de la función de transferencia).
(d) Ec. 9 del paper == condición de banda limitada en la frecuencia de Nyquist.
(e) Curva de la Fig. 6(b) con los 4 casos.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from d2nn import optics
from d2nn.config import WL, fig6_case

OUT = "results/01_optics"
os.makedirs(OUT, exist_ok=True)
summary = {}
torch.set_default_dtype(torch.float64)


def grid(n, dx):
    c = (np.arange(n) - (n - 1) / 2) * dx
    return np.meshgrid(c, c, indexing="ij")


# (a) apertura circular ---------------------------------------------------------
n, dx, a = 1024, 0.5e-6, 40e-6
Y, X = grid(n, dx)
u0 = torch.tensor((X**2 + Y**2 <= a**2).astype(np.complex128))
zs = np.linspace(5e-6, 2.5e-3, 120)
num = []
for z in zs:
    u = optics.propagate(u0[None], dx, z, WL, pad_factor=2)[0]
    num.append(u[n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1].mean().item())
num = np.array(num)
ex = optics.circular_aperture_on_axis(zs, a, WL)
# En el campo cercano (z < ~0.3 mm) el borde escalonado del círculo pixelado cambia la fase
# de la onda de borde (delta R ~ a*dx/(2R)); por eso el error se mide para z >= 0.3 mm.
far = zs >= 0.3e-3
err_a = np.abs(np.abs(num[far]) ** 2 - np.abs(ex[far]) ** 2).max() / 4.0
summary["a_circular_max_rel_err_z_ge_0.3mm"] = float(err_a)
zf = np.linspace(zs[0], zs[-1], 4000)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(zf * 1e3, np.abs(optics.circular_aperture_on_axis(zf, a, WL)) ** 2, "k-", lw=1,
        label="Exacta (R-S, en el eje)")
ax.plot(zs * 1e3, np.abs(num) ** 2, "r.", ms=3, label="Espectro angular (banda limitada)")
ax.set_xlabel("z (mm)")
ax.set_ylabel("|U(0,0,z)|²")
ax.set_title(f"Apertura circular a = {a * 1e6:.0f} µm, λ = 632.8 nm  (error máx. {100 * err_a:.1f}% para z ≥ 0.3 mm)", fontsize=9)
ax.legend()
fig.savefig(f"{OUT}/a_apertura_circular.png", bbox_inches="tight")
plt.close(fig)

# (b) apertura cuadrada, Fresnel -----------------------------------------------------
n, dx, half, z = 1024, 1e-6, 100e-6, 8e-3
Y, X = grid(n, dx)
u0 = torch.tensor(((np.abs(X) <= half) & (np.abs(Y) <= half)).astype(np.complex128))
u = optics.propagate(u0[None], dx, z, WL, pad_factor=2)[0].numpy()
row = n // 2
xr = X[row]
an = optics.fresnel_square_aperture(xr, np.zeros_like(xr) + Y[row, 0], half, z, WL)
I_num, I_an = np.abs(u[row]) ** 2, np.abs(an) ** 2
err_b = np.abs(I_num - I_an).max() / I_an.max()
summary["b_square_fresnel_max_rel_err"] = float(err_b)
summary["b_fresnel_number"] = half**2 / (WL * z)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(xr * 1e6, I_an, "k-", lw=1.5, label="Fresnel analítica")
ax.plot(xr * 1e6, I_num, "r--", lw=1, label="Espectro angular")
ax.set_xlim(-250, 250)
ax.set_xlabel("x (µm)")
ax.set_ylabel("Intensidad")
ax.set_title(f"Apertura cuadrada 200 µm, z = 8 mm, N_F = {half**2 / (WL * z):.2f}  "
             f"(error máx. {100 * err_b:.2f}%)")
ax.legend()
fig.savefig(f"{OUT}/b_apertura_cuadrada_fresnel.png", bbox_inches="tight")
plt.close(fig)

# (c) aliasing: ingenuo vs. banda limitada vs. R-S directo -----------------------------
n, dx = 64, 4e-6                                  # neuronas de 4 µm como en el paper
zc = optics.critical_distance(2 * n, dx, WL)
rng = np.random.default_rng(0)
u0 = np.zeros((n, n), complex)
u0[16:48, 16:48] = np.exp(1j * np.pi * rng.random((32, 32)))   # una "capa" con fase aleatoria
# La suma directa de R-S sólo está bien muestreada si dx <= lambda z / (n dx), es decir
# z >= n dx^2 / lambda = z_c / 2 (régimen complementario al del espectro angular ingenuo);
# por eso la comparación empieza en z = z_c.
ratios = np.array([1.0, 1.5, 2.5, 4.0, 7.0, 10.0, 20.0])
err_naive, err_bl = [], []
for r in ratios:
    z = r * zc
    ref = optics.rayleigh_sommerfeld_direct(u0, dx, z, WL).reshape(n, n)
    t = torch.tensor(u0)[None]
    un = optics.propagate(t, dx, z, WL, bandlimit=False)[0].numpy()
    ub = optics.propagate(t, dx, z, WL, bandlimit=True)[0].numpy()
    nr = np.linalg.norm(ref)
    err_naive.append(np.linalg.norm(un - ref) / nr)
    err_bl.append(np.linalg.norm(ub - ref) / nr)
summary["c_zc_m"] = zc
summary["c_ratios"] = ratios.tolist()
summary["c_err_naive"] = [float(e) for e in err_naive]
summary["c_err_bandlimited"] = [float(e) for e in err_bl]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.loglog(ratios, err_naive, "o-", label="Espectro angular ingenuo")
ax.loglog(ratios, err_bl, "s-", label="Banda limitada (Matsushima 2009)")
ax.set_xlabel("z / z_c   (z_c = N_pad dx² / λ)")
ax.set_ylabel("Error relativo vs. R-S directo")
ax.set_title("Neuronas de 4 µm, 64×64, relleno 2×")
ax.legend()
fig.savefig(f"{OUT}/c_aliasing_asm.png", bbox_inches="tight")
plt.close(fig)

# (d) identidad Ec. 9 <-> banda limitada ----------------------------------------------
rows = []
for nside in [80, 200, 1000]:
    for nf in [0.6, 1, 2, 6, 12]:
        d = nf * WL
        D = optics.d_min(nside, d, WL)
        fl = optics.band_limit(2 * nside, d, D, WL) if D > 0 else float("nan")
        rows.append(dict(n=nside, d_f_lambda=nf, D_min_lambda=D / WL,
                         f_limit_times_2d=fl * 2 * d))
summary["d_identity"] = rows
paper = dict(n=1000, d_f=4e-6)
summary["paper_full_D_min_m"] = optics.d_min(1000, 4e-6, WL)
summary["paper_full_zc_naive_pad2_m"] = optics.critical_distance(2000, 4e-6, WL)
summary["paper_full_phi_max_deg"] = math.degrees(optics.phi_max(4e-6, WL))

# (e) Fig. 6(b) -------------------------------------------------------------------------
nf = np.linspace(0.5, 16, 300)
curve = [optics.d_min(80, x * WL, WL) / WL for x in nf]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(nf, curve, "r-", label="Ec. (9): D_min, N = 80×80")
for c in [1, 2, 3, 4]:
    cfg = fig6_case(c)
    x, y = cfg.d_f / WL, cfg.distance / WL
    ok = cfg.distance >= optics.d_min(80, cfg.d_f, WL)
    ax.plot(x, y, "o", color="g" if ok else "r")
    ax.annotate(f"Caso {c}", (x, y), textcoords="offset points", xytext=(5, 5),
                color="g" if ok else "r")
ax.plot(6, 2.4e3, "x", color="gray")
ax.annotate("Caso 3 según la tabla\n(2.4×10³ λ)", (6, 2.4e3), textcoords="offset points",
            xytext=(8, -22), color="gray", fontsize=7)
ax.set_xlabel("Tamaño de neurona d_f (λ)")
ax.set_ylabel("Distancia entre capas D (λ)")
ax.set_title("Fig. 6(b) reproducida")
ax.legend(loc="upper left")
fig.savefig(f"{OUT}/e_fig6b_curva.png", bbox_inches="tight")
plt.close(fig)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
