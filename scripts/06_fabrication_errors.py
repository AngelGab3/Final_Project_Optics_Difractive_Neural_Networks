"""Fase 4: sensibilidad a errores de fabricación y montaje (cf. Fig. S4-S5 del paper),
sin reentrenar:
  (a) cuantización de la fase a 2^b niveles (litografía multi-paso),
  (b) error de profundidad de fase (altura grabada distinta: Phi -> s * Phi),
  (c) desalineación lateral aleatoria de cada capa (en µm),
  (d) error en la distancia entre capas.

python scripts/06_fabrication_errors.py results/04_full/paper_full/model.pt --n-eval 2000
"""
import argparse
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

from d2nn.data import Encoder, build_split
from d2nn.train import build_model, evaluate, load

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--n-eval", type=int, default=2000)
ap.add_argument("--trials", type=int, default=3)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--device", default=None)
ap.add_argument("--data", default="data")
args = ap.parse_args()
if not torch.cuda.is_available():
    torch.set_num_threads(os.cpu_count())

model, cfg = load(args.model, args.device)
dev = model.theta.device
out = os.path.join(os.path.dirname(args.model), "fabricacion")
os.makedirs(out, exist_ok=True)
enc = Encoder(cfg.n, cfg.digit_frac, cfg.binarize)
X, y = build_split(args.data, False, cfg.n_test, cfg.n_test_changed, cfg.seed + 100,
                   cfg.num_classes)
idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:args.n_eval]
X, y = X[idx], y[idx]
ph0 = model.phases().detach()
acc = lambda ph, m=model: evaluate(m, enc, X, y, args.bs, dev, ph)[0]
res = {"ideal": acc(ph0)}
print("ideal", res["ideal"])

# (a) cuantización
bits = [1, 2, 3, 4, 5]
res["quant"] = []
for b in bits:
    L = 2 ** b
    step = cfg.phase_max / (L - 1) if L > 1 else cfg.phase_max
    q = torch.round(ph0 / step) * step
    res["quant"].append(dict(levels=L, acc=acc(q)))
    print("niveles", L, res["quant"][-1]["acc"])

# (b) profundidad
scales = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
res["depth"] = [dict(scale=s, acc=acc(ph0 * s)) for s in scales]
print("profundidad", res["depth"])

# (c) desalineación lateral. La malla tiene una muestra por neurona, así que sólo pueden
# simularse desplazamientos de un número entero de neuronas (uno menor se redondearía a 0).
# Cada capa se desplaza exactamente k neuronas en una dirección aleatoria independiente.
shifts_px = [0, 1, 2, 3, 5]
res["shift"] = []
g = torch.Generator().manual_seed(1)
for k in shifts_px:
    accs = []
    for t in range(args.trials if k else 1):
        ph = ph0.clone()
        for l in range(cfg.num_layers):
            ang = 2 * math.pi * torch.rand(1, generator=g).item()
            d = (int(round(k * math.cos(ang))), int(round(k * math.sin(ang))))
            ph[l] = torch.roll(ph0[l], shifts=d, dims=(0, 1))
        accs.append(acc(ph))
    res["shift"].append(dict(shift_px=k, shift_um=k * cfg.d_f * 1e6,
                             acc=float(np.mean(accs)), std=float(np.std(accs))))
    print("desplazamiento", res["shift"][-1])

# (d) distancia entre capas
rel = [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]
res["distance"] = []
for r in rel:
    cfg2 = type(cfg).from_dict(json.loads(cfg.to_json()))
    cfg2.distance = cfg.distance * (1 + r)
    m2 = build_model(cfg2, dev)
    with torch.no_grad():
        m2.theta.copy_(model.theta)
    m2.eval()
    res["distance"].append(dict(rel=r, acc=acc(ph0, m2)))
    del m2
    print("distancia", res["distance"][-1])

json.dump(res, open(os.path.join(out, "fabricacion.json"), "w"), indent=2)

fig, axs = plt.subplots(1, 4, figsize=(15, 3.3))
axs[0].plot([q["levels"] for q in res["quant"]], [100 * q["acc"] for q in res["quant"]], "o-")
axs[0].set_xscale("log", base=2)
axs[0].set_xlabel("Niveles de fase")
axs[0].set_title("(a) Cuantización")
axs[1].plot([d["scale"] for d in res["depth"]], [100 * d["acc"] for d in res["depth"]], "o-")
axs[1].set_xlabel("Profundidad grabada / nominal")
axs[1].set_title("(b) Error de profundidad")
axs[2].errorbar([d["shift_um"] for d in res["shift"]], [100 * d["acc"] for d in res["shift"]],
                [100 * d["std"] for d in res["shift"]], fmt="o-")
axs[2].set_xlabel("Desalineación por capa (µm)")
axs[2].set_title(f"(c) Desalineación (neurona = {cfg.d_f * 1e6:.1f} µm)")
axs[3].plot([100 * d["rel"] for d in res["distance"]], [100 * d["acc"] for d in res["distance"]], "o-")
axs[3].set_xlabel("Error en D (%)")
axs[3].set_title("(d) Distancia entre capas")
for a in axs:
    a.axhline(100 * res["ideal"], color="gray", ls=":", lw=1)
    a.set_ylabel("Precisión (%)")
fig.savefig(os.path.join(out, "fabricacion.png"), bbox_inches="tight")
print(json.dumps(res, indent=2))
