"""Evalúa un modelo entrenado y genera las figuras equivalentes a las del paper:
  - confusión (Fig. 2a), máscaras de fase (Fig. 1b),
  - salida + energía en detectores para un 3, un 3 alterado, un 4 y un 4 tapado (Fig. 3),
  - cuatro 6 distintos (Fig. 4), onda plana sin dígito (Fig. 5),
  - corte x-z de la intensidad a través de todo el sistema.

python scripts/05_evaluate.py results/04_full/paper_full/model.pt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from d2nn import optics
from d2nn.data import Encoder, alter_digit, build_split, cover_digit, load_mnist
from d2nn.train import evaluate, load
from d2nn.viz import plot_confusion, plot_examples, plot_phases, plot_xz

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--device", default=None)
ap.add_argument("--data", default="data")
ap.add_argument("--xz-steps", type=int, default=40)
ap.add_argument("--bs", type=int, default=16)
args = ap.parse_args()
if not torch.cuda.is_available():
    torch.set_num_threads(os.cpu_count())

model, cfg = load(args.model, args.device)
dev = model.theta.device
out = os.path.dirname(args.model)
enc = Encoder(cfg.n, cfg.digit_frac, cfg.binarize)

# --- prueba completa -------------------------------------------------------------
Xte, yte = build_split(args.data, False, cfg.n_test, cfg.n_test_changed, cfg.seed + 100,
                       cfg.num_classes)
acc, cm, _ = evaluate(model, enc, Xte, yte, args.bs, dev)
plot_confusion(cm, os.path.join(out, "confusion.png"), "Simulación")
metrics = {"test_acc": acc, "n_test": int(len(yte))}
if cfg.num_classes == 11:
    metrics["acc_digitos_sin_cambio"] = float(np.trace(cm[:10, :10]) / cm[:10].sum())
    metrics["acc_clase_X"] = float(cm[10, 10] / cm[10].sum())
print(json.dumps(metrics, indent=2))
json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)

plot_phases(model.phases().detach().cpu().numpy(), cfg.d_f, os.path.join(out, "fases.png"))


# --- ejemplos estilo Fig. 3-5 -----------------------------------------------------
@torch.no_grad()
def run(imgs28):
    u = enc(torch.as_tensor(imgs28).to(dev))
    E, I = model(u)
    return u.abs().cpu().numpy(), I.cpu().numpy(), E.cpu().numpy()


Xm, ym = load_mnist(args.data, False)
rng = np.random.default_rng(1)
pick = lambda d, k=0: Xm[(ym == d).nonzero()[k, 0]].numpy()
ex, titles = [], []
ex.append(pick(3)); titles.append("3")
if cfg.num_classes == 11:
    ex.append(alter_digit(pick(3), rng)); titles.append("3 alterado")
ex.append(pick(4)); titles.append("4")
if cfg.num_classes == 11:
    ex.append(cover_digit(pick(4), rng)); titles.append("4 tapado")
a, I, E = run(np.stack(ex))
plot_examples(a, I, E, model.boxes, titles, os.path.join(out, "fig3_ejemplos.png"))

sixes = np.stack([pick(6, k) for k in range(4)])
a, I, E = run(sixes)
plot_examples(a, I, E, model.boxes, [f"6 (#{k + 1})" for k in range(4)],
              os.path.join(out, "fig4_seises.png"))

with torch.no_grad():
    u = enc.plane_wave(1, dev)
    Ep, Ip = model(u)
plot_examples(u.abs().cpu().numpy(), Ip.cpu().numpy(), Ep.cpu().numpy(), model.boxes,
              ["Onda plana (sin dígito)"], os.path.join(out, "fig5_onda_plana.png"))


# --- corte x-z ------------------------------------------------------------------------
@torch.no_grad()
def xz_slice(u, steps):
    ph = model.phases()
    rows, zs, layer_z = [], [], []
    z0 = 0.0
    for seg, D in enumerate(model.distances):
        for k in range(steps):
            z = D * k / steps
            v = u if k == 0 else optics.propagate(u, cfg.d_f, z, cfg.wl, cfg.pad_factor,
                                                  cfg.bandlimit)
            rows.append((v[0, cfg.n // 2].abs() ** 2).cpu().numpy())
            zs.append(z0 + z)
        u = model.prop(seg, u)
        z0 += D
        if seg < model.num_layers:
            layer_z.append(z0)
            u = u * torch.polar(torch.ones_like(ph[seg]), ph[seg])
    rows.append((u[0, cfg.n // 2].abs() ** 2).cpu().numpy())
    zs.append(z0)
    return np.array(rows), np.array(zs), layer_z


u = enc(torch.as_tensor(pick(3)[None]).to(dev))
xz, zs, lz = xz_slice(u, args.xz_steps)
x = (np.arange(cfg.n) - cfg.n / 2) * cfg.d_f
plot_xz(xz, zs, x, lz, os.path.join(out, "corte_xz.png"),
        "Luz atravesando la red (dígito 3), corte y = 0; líneas: capas L1-L5")
print("Figuras guardadas en", out)
