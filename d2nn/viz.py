"""Gráficas al estilo de las figuras del paper."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})


def class_names(k: int):
    return [str(i) for i in range(10)] + (["X"] if k == 11 else [])


def plot_confusion(cm: np.ndarray, path: str, title: str = ""):
    k = cm.shape[0]
    names = class_names(k)
    fig, ax = plt.subplots(figsize=(4.8 + 0.1 * k, 4.4 + 0.1 * k))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "#c0392b")
    ax.set_xticks(range(k), names)
    ax.set_yticks(range(k), names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    acc = np.trace(cm) / max(cm.sum(), 1)
    ax.set_title(f"{title}  (acc = {100 * acc:.2f}%)")
    fig.savefig(path)
    plt.close(fig)


def plot_phases(phases: np.ndarray, dx: float, path: str):
    L, n, _ = phases.shape
    ext = n * dx * 1e3
    fig, axs = plt.subplots(1, L, figsize=(2.6 * L, 2.8))
    for l, ax in enumerate(np.atleast_1d(axs)):
        im = ax.imshow(phases[l], cmap="hot", vmin=0, vmax=np.pi, extent=[0, ext, 0, ext])
        ax.set_title(f"L{l + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axs, fraction=0.015, ticks=[0, np.pi])
    cb.ax.set_yticklabels(["0", "π"])
    fig.suptitle(f"Fase de las neuronas (capa de {ext:.2f} mm)")
    fig.savefig(path)
    plt.close(fig)


def _draw_boxes(ax, boxes, names, color="r"):
    for k, (y0, x0, s) in enumerate(boxes):
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), s, s, fill=False, ec=color, lw=0.8))
        ax.text(x0 + s / 2, y0 - 1.5, names[k], color="w", ha="center", va="bottom", fontsize=7)


def plot_examples(inputs, intensities, energies, boxes, titles, path):
    """Fila por ejemplo: objeto de entrada | intensidad de salida con detectores |
    distribución de energía (%) en los detectores. Equivalente a las Fig. 3-4."""
    m = len(inputs)
    k = energies.shape[1]
    names = class_names(k)
    fig, axs = plt.subplots(m, 3, figsize=(10, 3.0 * m), gridspec_kw=dict(width_ratios=[1, 1, 1.3], hspace=0.35))
    axs = np.atleast_2d(axs)
    for i in range(m):
        axs[i, 0].imshow(inputs[i], cmap="gray", vmin=0, vmax=1)
        axs[i, 0].set_title(titles[i])
        I = intensities[i]
        axs[i, 1].imshow(I / I.max(), cmap="gray", vmin=0, vmax=1)
        _draw_boxes(axs[i, 1], boxes, names)
        pred = int(np.argmax(energies[i]))
        axs[i, 1].set_title(f"Salida: detector {names[pred]}")
        for a in axs[i, :2]:
            a.set_xticks([])
            a.set_yticks([])
        pct = 100 * energies[i] / energies[i].sum()
        cols = ["#c0392b" if j == pred else "#7f8c8d" for j in range(k)]
        axs[i, 2].bar(range(k), pct, color=cols)
        axs[i, 2].set_xticks(range(k), names)
        axs[i, 2].set_ylabel("Energía (%)")
        axs[i, 2].set_xlabel("Clase")
    fig.savefig(path)
    plt.close(fig)


def plot_xz(xz: np.ndarray, z_axis: np.ndarray, x_axis: np.ndarray, layer_z, path: str, title=""):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    v = xz / xz.max()
    ax.imshow(v.T ** 0.5, aspect="auto", cmap="inferno", origin="lower",
              extent=[z_axis[0] * 1e3, z_axis[-1] * 1e3, x_axis[0] * 1e3, x_axis[-1] * 1e3])
    for i, zl in enumerate(layer_z):
        ax.axvline(zl * 1e3, color="c", lw=0.8, ls="--")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_title(title or "Intensidad en el corte y = 0 (escala sqrt)")
    fig.savefig(path)
    plt.close(fig)
