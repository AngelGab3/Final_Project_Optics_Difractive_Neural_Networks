"""Entrenamiento y evaluación."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import Encoder, build_split
from .model import D2NN, detector_boxes


def pick_device(pref: str | None = None) -> torch.device:
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: Config, device) -> D2NN:
    boxes = detector_boxes(cfg.n, cfg.layout, cfg.det_frac)
    return D2NN(cfg.n, cfg.d_f, cfg.wl, cfg.distances, cfg.num_layers, boxes,
                phase_max=cfg.phase_max, pad_factor=cfg.pad_factor, bandlimit=cfg.bandlimit,
                temperature=cfg.temperature, use_checkpoint=cfg.use_checkpoint).to(device)


def build_data(cfg: Config, root: str):
    tr = build_split(root, True, cfg.n_train, cfg.n_train_changed, cfg.seed, cfg.num_classes)
    te = build_split(root, False, cfg.n_test, cfg.n_test_changed, cfg.seed + 100, cfg.num_classes)
    return tr, te


@torch.no_grad()
def evaluate(model: D2NN, enc: Encoder, X, y, batch_size: int, device, phases=None):
    model.eval()
    K = model.det.shape[0]
    cm = np.zeros((K, K), dtype=np.int64)
    energies = []
    etas = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size]
        E, I = model(enc(xb), phases)
        etas.append((E.sum(1) / (I.sum((1, 2)) + 1e-12)).cpu())
        pred = E.argmax(1).cpu()
        np.add.at(cm, (yb.numpy(), pred.numpy()), 1)
        energies.append(E.cpu())
    acc = np.trace(cm) / cm.sum()
    evaluate.last_efficiency = float(torch.cat(etas).mean())
    return float(acc), cm, torch.cat(energies)


def train(cfg: Config, out_dir: str, data_root: str = "data", device=None,
          log_every: int = 50, max_minutes: float | None = None, resume: bool = False):
    """Entrena según cfg y guarda en out_dir (model.pt, history.json, confusion.npy).
    Con resume=True continúa desde out_dir/model.pt (fase y estado de Adam)."""
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(device)
    torch.manual_seed(cfg.seed)
    (Xtr, ytr), (Xte, yte) = build_data(cfg, data_root)
    enc = Encoder(cfg.n, cfg.digit_frac, cfg.binarize)
    model = build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(cfg.to_json())

    history = []
    start = 0
    ck_path = os.path.join(out_dir, "model.pt")
    if resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu")
        with torch.no_grad():
            model.theta.copy_(ck["theta"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        history = json.load(open(os.path.join(out_dir, "history.json")))
        start = len(history)
        print(f"Reanudando desde la época {start}")
    t0 = time.time()
    g = torch.Generator().manual_seed(cfg.seed)
    for _ in range(start):
        torch.randperm(len(Xtr), generator=g)
    stop = False
    cm = None
    for ep in range(start, cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot, correct, seen = 0.0, 0, 0
        for it, i in enumerate(range(0, len(Xtr), cfg.batch_size)):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            E, I = model(enc(xb))
            loss = F.cross_entropy(model.logits(E), yb)
            if cfg.efficiency_weight:
                eta = E.sum(1) / (I.sum((1, 2)) + 1e-12)
                loss = loss + cfg.efficiency_weight * (1 - eta).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            correct += (E.argmax(1) == yb).sum().item()
            seen += len(idx)
            if log_every and it % log_every == 0:
                print(f"  ep {ep + 1} it {it} loss {loss.item():.4f} "
                      f"acc {correct / seen:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if max_minutes and (time.time() - t0) > 60 * max_minutes:
                stop = True
                break
        test_acc, cm, _ = evaluate(model, enc, Xte, yte, max(cfg.batch_size, 8), device)
        rec = dict(epoch=ep + 1, train_loss=tot / seen, train_acc=correct / seen,
                   test_acc=test_acc, test_efficiency=evaluate.last_efficiency,
                   time_s=time.time() - t0)
        history.append(rec)
        print(f"[{cfg.name}] época {ep + 1}/{cfg.epochs}: pérdida {rec['train_loss']:.4f} "
              f"train {rec['train_acc']:.4f} test {test_acc:.4f} "
              f"eficiencia {rec['test_efficiency']:.3f} ({rec['time_s']:.0f}s)", flush=True)
        save(model, cfg, out_dir, history, cm, opt)
        if stop:
            print("Límite de tiempo alcanzado; se detiene el entrenamiento.")
            break
    return model, history, cm


def save(model: D2NN, cfg: Config, out_dir: str, history, cm, opt=None):
    ck = {"config": json.loads(cfg.to_json()), "theta": model.theta.detach().cpu()}
    if opt is not None:
        ck["optimizer"] = opt.state_dict()
    tmp_pt = os.path.join(out_dir, "model.pt.tmp")
    torch.save(ck, tmp_pt)
    os.replace(tmp_pt, os.path.join(out_dir, "model.pt"))
    tmp_hist = os.path.join(out_dir, "history.json.tmp")
    with open(tmp_hist, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp_hist, os.path.join(out_dir, "history.json"))
    tmp_cm = os.path.join(out_dir, "confusion.tmp.npy")
    np.save(tmp_cm, cm)
    os.replace(tmp_cm, os.path.join(out_dir, "confusion.npy"))


def load(path: str, device=None) -> tuple[D2NN, Config]:
    device = pick_device(device)
    ck = torch.load(path, map_location="cpu")
    cfg = Config.from_dict(ck["config"])
    model = build_model(cfg, device)
    with torch.no_grad():
        model.theta.copy_(ck["theta"])
    model.eval()
    return model, cfg
