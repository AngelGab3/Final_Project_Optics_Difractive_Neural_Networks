"""Contribución propia: barrido de D/D_min para comprobar dónde ocurre realmente la
transición que predice la Ec. 9 (el paper sólo la probó con 4 puntos), y prueba
de la hipótesis de conectividad acumulada al variar el número de capas L.

Uso:
  python scripts/03_connectivity_sweep.py [--epochs 20] [--sizes 1 6 12] [--device cuda]
  python scripts/03_connectivity_sweep.py --epochs 20 --layers 1 3 5 8 --device cuda
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from d2nn import optics
from d2nn.config import WL, fig6_case
from d2nn.train import train

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--device", default=None)
ap.add_argument("--data", default="data")
ap.add_argument("--sizes", type=float, nargs="+", default=[1.0, 6.0, 12.0])
ap.add_argument("--layers", type=int, nargs="+", default=None,
                help="si se especifica, barre num_layers para d_f = --layer-df * lambda")
ap.add_argument("--layer-df", type=float, default=6.0)
ap.add_argument("--ratios", type=float, nargs="+",
                default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
ap.add_argument("--out", default="results/03_sweep")
ap.add_argument("--efficiency-weight", type=float, default=0.0)
args = ap.parse_args()
torch.set_num_threads(os.cpu_count())
if args.efficiency_weight:
    args.out = f"{args.out}_eff{args.efficiency_weight:g}"
os.makedirs(args.out, exist_ok=True)


def run_size_sweep():
    res_path = os.path.join(args.out, "sweep.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else []
    done = {(r["d_f_lambda"], r["ratio"]) for r in results}
    for nf in args.sizes:
        dm = optics.d_min(80, nf * WL, WL)
        for r in args.ratios:
            if (nf, r) in done:
                continue
            cfg = fig6_case(2, d_f=nf * WL, distance=r * dm, epochs=args.epochs,
                            efficiency_weight=args.efficiency_weight, name=f"df{nf:g}_r{r:g}")
            _, hist, _ = train(cfg, os.path.join(args.out, "runs", cfg.name), args.data,
                               args.device, log_every=0)
            results.append(dict(d_f_lambda=nf, ratio=r, D_lambda=r * dm / WL,
                                acc=hist[-1]["test_acc"], best=max(h["test_acc"] for h in hist)))
            json.dump(results, open(res_path, "w"), indent=2)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for nf in sorted({r["d_f_lambda"] for r in results}):
        pts = sorted([r for r in results if r["d_f_lambda"] == nf], key=lambda r: r["ratio"])
        ax.semilogx([p["ratio"] for p in pts], [100 * p["acc"] for p in pts], "o-",
                    label=f"d_f = {nf:g} λ")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.axhline(10, color="gray", ls=":", lw=1)
    ax.set_ylim(5, 97)
    ax.text(1.06, 14, "D = D_min (Ec. 9)", fontsize=8)
    ax.text(0.00105, 6.8, "azar (10%)", fontsize=8, color="gray")
    ax.set_xlabel("D / D_min")
    ax.set_ylabel("Precisión de prueba (%)")
    ax.set_title(f"N = 80×80, 5 capas ({args.epochs} épocas): ¿dónde se pierde la conectividad?")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(args.out, "sweep.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(json.dumps(results, indent=2))


def run_layer_sweep():
    nf = args.layer_df
    dm = optics.d_min(80, nf * WL, WL)
    res_path = os.path.join(args.out, "sweep_layers.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else []
    done = {(r["num_layers"], r["ratio"]) for r in results}

    # Reutilizar L=5 de sweep.json si ya fue entrenado con el mismo número de épocas
    base_path = os.path.join(args.out, "sweep.json")
    base_by_ratio = {}
    if os.path.exists(base_path):
        for r in json.load(open(base_path)):
            if abs(r["d_f_lambda"] - nf) < 1e-9:
                hpath = os.path.join(args.out, "runs", f"df{nf:g}_r{r['ratio']:g}", "history.json")
                if os.path.exists(hpath) and len(json.load(open(hpath))) == args.epochs:
                    base_by_ratio[r["ratio"]] = r

    for L in args.layers:
        for r in args.ratios:
            if (L, r) in done:
                continue
            if L == 5 and r in base_by_ratio:
                b = base_by_ratio[r]
                results.append(dict(num_layers=L, d_f_lambda=nf, ratio=r, D_lambda=b["D_lambda"],
                                    acc=b["acc"], best=b["best"]))
                json.dump(results, open(res_path, "w"), indent=2)
                continue
            cfg = fig6_case(2, d_f=nf * WL, distance=r * dm, num_layers=L, epochs=args.epochs,
                            efficiency_weight=args.efficiency_weight, name=f"L{L}_df{nf:g}_r{r:g}")
            _, hist, _ = train(cfg, os.path.join(args.out, "runs", cfg.name), args.data,
                               args.device, log_every=0)
            results.append(dict(num_layers=L, d_f_lambda=nf, ratio=r, D_lambda=r * dm / WL,
                                acc=hist[-1]["test_acc"], best=max(h["test_acc"] for h in hist)))
            json.dump(results, open(res_path, "w"), indent=2)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for L in sorted({r["num_layers"] for r in results}):
        pts = sorted([r for r in results if r["num_layers"] == L], key=lambda r: r["ratio"])
        (line,) = ax.semilogx([p["ratio"] for p in pts], [100 * p["acc"] for p in pts], "o-",
                              label=f"L = {L} ({L + 1} prop.)")
        ax.axvline(1.0 / (L + 1), color=line.get_color(), ls="--", lw=0.9, alpha=0.45)
    ax.axvline(1.0, color="k", ls="--", lw=1.1)
    ax.axhline(10, color="gray", ls=":", lw=1)
    ax.set_ylim(5, 98)
    ax.text(1.06, 14, "D = D_min (Ec. 9)", fontsize=8)
    ax.text(0.00105, 6.8, "azar (10%)", fontsize=8, color="gray")
    ax.set_xlabel("D / D_min")
    ax.set_ylabel("Precisión de prueba (%)")
    ax.set_title(f"N = 80×80, d_f = {nf:g} λ ({args.epochs} épocas): conectividad acumulada")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(os.path.join(args.out, "sweep_layers.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(json.dumps(results, indent=2))


if args.layers is None:
    run_size_sweep()
else:
    run_layer_sweep()
