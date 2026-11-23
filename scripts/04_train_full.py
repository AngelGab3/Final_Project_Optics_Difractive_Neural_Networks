"""Fase 3: entrena el D2NN completo del paper (5 capas de 1000x1000 neuronas de 4 µm,
D = 5 cm, lambda = 632.8 nm, 11 clases, 55k MNIST + 25k dígitos cambiados, 20 épocas).

Ejemplos:
  python scripts/04_train_full.py                       # sistema completo (GPU)
  python scripts/04_train_full.py --bs 16 --checkpoint  # lotes mayores con menos memoria
  python scripts/04_train_full.py --n 250 --epochs 3    # versión reducida para depurar
  python scripts/04_train_full.py --resume              # continuar una corrida cortada
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from d2nn import optics
from d2nn.config import Config, scaled_full
from d2nn.train import train
from d2nn.viz import plot_confusion

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000, help="neuronas por lado (1000 = paper)")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--checkpoint", action="store_true", help="gradient checkpointing por capa")
ap.add_argument("--n-train", type=int, default=55000)
ap.add_argument("--n-train-changed", type=int, default=25000)
ap.add_argument("--n-test", type=int, default=10000)
ap.add_argument("--n-test-changed", type=int, default=1000)
ap.add_argument("--max-minutes", type=float, default=None)
ap.add_argument("--device", default=None)
ap.add_argument("--data", default="data")
ap.add_argument("--out", default=None)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--log-every", type=int, default=200)
ap.add_argument("--efficiency-weight", type=float, default=0.0)
args = ap.parse_args()

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    torch.set_num_threads(os.cpu_count())

cfg = Config() if args.n == 1000 else scaled_full(args.n)
cfg.epochs, cfg.batch_size, cfg.lr = args.epochs, args.bs, args.lr
cfg.use_checkpoint = args.checkpoint
cfg.efficiency_weight = args.efficiency_weight
cfg.n_train, cfg.n_train_changed = args.n_train, args.n_train_changed
cfg.n_test, cfg.n_test_changed = args.n_test, args.n_test_changed
eff_suffix = f"_eff{args.efficiency_weight:g}" if args.efficiency_weight else ""
out = args.out or f"results/04_full/{cfg.name}{eff_suffix}"

dm = optics.d_min(cfg.n, cfg.d_f, cfg.wl)
print(f"n = {cfg.n}, d_f = {cfg.d_f * 1e6:.2f} µm, capa = {cfg.n * cfg.d_f * 1e3:.2f} mm, "
      f"D = {cfg.distance * 1e3:.2f} mm, D_min(Ec. 9) = {dm * 1e3:.2f} mm")
_, hist, cm = train(cfg, out, args.data, args.device, log_every=args.log_every,
                    max_minutes=args.max_minutes, resume=args.resume)
plot_confusion(cm, os.path.join(out, "confusion.png"), "Simulación")
print(f"Precisión final de prueba: {100 * hist[-1]['test_acc']:.2f}%  (paper: 91.57%)")
