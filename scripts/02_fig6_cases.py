"""Fase 2: reproduce la Fig. 6 (casos 1-4, N = 80x80, 10 clases) y sus matrices de confusión.

Uso:  python scripts/02_fig6_cases.py [--epochs 20] [--device cuda] [--cases 1 2 3 4 3t]
'3t' es el caso 3 con la distancia literal de la tabla (2.4e3 lambda), debajo de la curva.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from d2nn import optics
from d2nn.config import WL, fig6_case
from d2nn.train import train
from d2nn.viz import plot_confusion

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--device", default=None)
ap.add_argument("--data", default="data")
ap.add_argument("--cases", nargs="+", default=["1", "2", "3", "4", "3t"])
ap.add_argument("--out", default="results/02_fig6")
ap.add_argument("--efficiency-weight", type=float, default=0.0)
args = ap.parse_args()
torch.set_num_threads(os.cpu_count())

paper = {"1": 90.0, "2": 91.0, "3": 91.2, "4": 0.1, "3t": None}
eff_suffix = f"_eff{args.efficiency_weight:g}" if args.efficiency_weight else ""
rows = []
for c in args.cases:
    if c == "3t":
        cfg = fig6_case(3, distance=2.4e3 * WL, name="fig6_case3_tabla", epochs=args.epochs,
                        efficiency_weight=args.efficiency_weight)
    else:
        cfg = fig6_case(int(c), epochs=args.epochs, efficiency_weight=args.efficiency_weight)
    out = os.path.join(args.out, cfg.name + eff_suffix)
    _, hist, cm = train(cfg, out, args.data, args.device, log_every=0)
    plot_confusion(cm, os.path.join(out, "confusion.png"), f"Caso {c}")
    dm = optics.d_min(cfg.n, cfg.d_f, WL)
    rows.append(dict(case=c, d_f_lambda=cfg.d_f / WL, D_lambda=cfg.distance / WL,
                     D_min_lambda=dm / WL, D_over_Dmin=cfg.distance / dm if dm else float("inf"),
                     acc_paper=paper[c], acc_ours=100 * hist[-1]["test_acc"]))

with open(os.path.join(args.out, "tabla_fig6c.json"), "w") as f:
    json.dump(rows, f, indent=2)
lines = ["| Caso | d_f (λ) | D (λ) | D_min Ec.9 (λ) | D/D_min | Paper (%) | Este trabajo (%) |",
         "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['case']} | {r['d_f_lambda']:.2f} | {r['D_lambda']:.3g} | "
                 f"{r['D_min_lambda']:.3g} | {r['D_over_Dmin']:.3g} | "
                 f"{'' if r['acc_paper'] is None else r['acc_paper']} | {r['acc_ours']:.1f} |")
open(os.path.join(args.out, "tabla_fig6c.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
