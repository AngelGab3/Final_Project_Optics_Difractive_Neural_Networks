"""Mide tiempo y memoria por iteración para estimar cuánto tardará el entrenamiento.

python scripts/benchmark.py --n 1000 --bs 8
python scripts/benchmark.py --n 1000 --bs 16 --checkpoint
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from d2nn.config import Config, scaled_full
from d2nn.train import build_model, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--checkpoint", action="store_true")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--device", default=None)
args = ap.parse_args()

dev = pick_device(args.device)
cfg = Config() if args.n == 1000 else scaled_full(args.n)
cfg.use_checkpoint = args.checkpoint
model = build_model(cfg, dev)
opt = torch.optim.Adam(model.parameters(), 1e-3)
x = torch.rand(args.bs, cfg.n, cfg.n, device=dev).to(torch.complex64)
y = torch.randint(0, cfg.num_classes, (args.bs,), device=dev)


def step():
    E, _ = model(x)
    loss = F.cross_entropy(model.logits(E), y)
    opt.zero_grad()
    loss.backward()
    opt.step()


step()
if dev.type == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
t = time.time()
for _ in range(args.iters):
    step()
if dev.type == "cuda":
    torch.cuda.synchronize()
dt = (time.time() - t) / args.iters
n_img = cfg.n_train + cfg.n_train_changed
ep = dt * n_img / args.bs
print(f"dispositivo {dev}, n = {cfg.n} (malla {cfg.pad_factor * cfg.n}²), lote {args.bs}, "
      f"checkpoint = {args.checkpoint}")
print(f"{dt * 1e3:.1f} ms/iteración  ->  {1e3 * dt / args.bs:.2f} ms/imagen")
print(f"época de {n_img} imágenes: {ep / 60:.1f} min;  20 épocas: {20 * ep / 3600:.2f} h")
if dev.type == "cuda":
    print(f"memoria pico: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
