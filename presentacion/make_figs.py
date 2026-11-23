"""Genera las figuras SVG/PNG unificadas para presentacion.html a partir de results/."""
from __future__ import annotations

import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

# Paleta alineada con presentacion.html
BLUE = "#2997ff"
TEAL = "#2bb3d9"
GREEN = "#30d158"
GREEN_DARK = "#1a7f37"
AMBER = "#ff9f0a"
RED = "#ff453a"
GRAY = "#86868b"
GRAY_DARK = "#515154"
INK = "#1d1d1f"
GRID = "#e5e5ea"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 13,
    "axes.titlesize": 14.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 13.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#c7c7cc",
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": GRAY_DARK,
    "ytick.color": GRAY_DARK,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.frameon": False,
    "legend.fontsize": 11.5,
    "svg.fonttype": "path",
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(os.path.join(OUT, f"{name}.svg"), bbox_inches="tight", transparent=True)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# 1. Aliasing del propagador ---------------------------------------------------
def make_aliasing() -> None:
    d = json.load(open(os.path.join(RES, "01_optics/summary.json")))
    r = np.array(d["c_ratios"], dtype=float)
    err_naive = np.array(d["c_err_naive"], dtype=float) * 100.0
    err_bl = np.array(d["c_err_bandlimited"], dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhspan(0, 10, color=BLUE, alpha=0.08, zorder=0)
    ax.axhline(10, color=BLUE, lw=1.1, ls=":", alpha=0.65)
    ax.text(2.4, 12.8, "Zona segura (< 10 %)", color=BLUE, fontsize=11.5, fontweight="bold")

    ax.plot(r, err_naive, "o-", color=RED, lw=2.8, ms=7.5,
            label="Espectro angular ingenuo", zorder=3)
    ax.plot(r, err_bl, "s-", color=BLUE, lw=2.8, ms=7.0,
            label="Banda limitada (Matsushima–Shimobaba)", zorder=4)

    # Etiquetas directas sobre las curvas
    ax.annotate("Espectro angular ingenuo\n(aliasing hasta 99.5 %)",
                xy=(7.0, err_naive[4]), xytext=(2.0, 82),
                color=RED, fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5, shrinkA=2, shrinkB=5))
    ax.annotate("Banda limitada\n(3.5 – 9.1 %)",
                xy=(10.0, err_bl[5]), xytext=(4.2, 26),
                color=BLUE, fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5, shrinkA=2, shrinkB=5))

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 10, 20])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.92, 23)
    ax.set_ylim(-2, 108)
    ax.set_xlabel(r"Distancia normalizada $z / z_c$  ($z_c \equiv D_{\min}$)")
    ax.set_ylabel("Error relativo vs. R–S directo (%)")
    save(fig, "aliasing")


# 2. Fig. 6(b) reproducida -----------------------------------------------------
def make_fig6b() -> None:
    tabla = json.load(open(os.path.join(RES, "02_fig6/tabla_fig6c.json")))
    nf = np.linspace(0.505, 16.0, 400)
    dmin_curve = 80.0 * nf * np.sqrt(np.maximum(4.0 * nf ** 2 - 1.0, 0.0))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(nf, dmin_curve, color=INK, lw=2.8, zorder=2)
    ax.fill_between(nf, 5.0, dmin_curve, color=RED, alpha=0.06, zorder=1)
    ax.text(10.5, 95, "régimen sub-conectado\n($D < D_{\\min}$)", color=RED,
            fontsize=11.5, fontweight="bold", ha="center", alpha=0.9)
    ax.text(1.5, 4.5e4, "conectado ($D \\geq D_{\\min}$)", color=GREEN_DARK,
            fontsize=11.5, fontweight="bold", ha="left")
    ax.text(11.3, 3.2e4, r"Ec. (9): $D_{\min}(d_f),\ \sqrt{N}=80$",
            color=INK, fontsize=11, fontweight="bold", rotation=11, ha="center")

    offsets = {
        "1": (10, 6),
        "2": (-58, 5),
        "3": (11, -4),
        "4": (-18, 14),
        "3t": (11, -11),
    }
    labels = {
        "1": "Caso 1",
        "2": "Caso 2",
        "3": "Caso 3 (gráfica)",
        "4": "Caso 4",
        "3t": "Caso 3 (tabla)",
    }

    for row in tabla:
        cid = row["case"]
        x, y = row["d_f_lambda"], row["D_lambda"]
        if cid == "3t":
            ax.plot(x, y, "X", color=GRAY_DARK, ms=10, zorder=5)
            ax.annotate(labels[cid], (x, y), xytext=offsets[cid],
                        textcoords="offset points", color=GRAY_DARK,
                        fontsize=11.5, fontweight="bold")
        else:
            ok = row["D_over_Dmin"] >= 0.99
            col = GREEN_DARK if ok else RED
            ax.plot(x, y, "o", color=col, ms=9.5, mec="white", mew=1.4, zorder=5)
            ax.annotate(labels[cid], (x, y), xytext=offsets[cid],
                        textcoords="offset points", color=col,
                        fontsize=11.5, fontweight="bold")

    # Flecha mostrando discrepancia 10x entre gráfica y tabla
    ax.annotate("", xy=(6.0, 2600.0), xytext=(6.0, 22000.0),
                arrowprops=dict(arrowstyle="<->", color=GRAY_DARK, lw=1.5, ls="--"))
    ax.text(6.45, 13000, r"$10\times$", color=GRAY_DARK, fontsize=11.5, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlim(0, 16.5)
    ax.set_ylim(5.0, 1.3e5)
    ax.set_xlabel(r"Tamaño de neurona $d_f\ (\lambda)$")
    ax.set_ylabel(r"Distancia entre placas $D\ (\lambda)$")
    save(fig, "fig6b")


# 3. Fig. 6 barras comparativas ------------------------------------------------
def make_fig6_barras() -> None:
    tabla = json.load(open(os.path.join(RES, "02_fig6/tabla_fig6c.json")))
    order = ["1", "2", "3", "3t", "4"]
    by_id = {r["case"]: r for r in tabla}
    rows = [by_id[k] for k in order]

    names = ["Caso 1", "Caso 2", "Caso 3\n(gráfica)", "Caso 3\n(tabla)", "Caso 4"]
    sub = [f"$D/D_{{\\min}}={r['D_over_Dmin']:.2f}$" if r["D_over_Dmin"] >= 0.01
           else r"$D/D_{\min}=6.5\times 10^{-4}$"
           for r in rows]

    x = np.arange(len(rows))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(10.0, color=RED, lw=1.2, ls="--", alpha=0.85, zorder=2)
    ax.text(3.38, 12.2, "azar\n(10 %)", color=RED, fontsize=8.6, fontweight="bold",
            ha="center", va="bottom")

    for i, r in enumerate(rows):
        pap = r["acc_paper"]
        ours = r["acc_ours"]
        if pap is not None:
            bx = x[i] - w / 2
            h_draw = max(pap, 0.8) if pap < 1 else pap
            ax.bar(bx, h_draw, w, color="#c7c7cc", edgecolor="none", zorder=3,
                   label="Paper (Chen et al.)" if i == 0 else None)
            if r["case"] == "4":
                ax.annotate("paper:\n«~0.1 %»", xy=(bx, h_draw + 1.5), xytext=(bx - 0.05, 34),
                            color=GRAY_DARK, fontsize=8.5, fontweight="bold", ha="center",
                            arrowprops=dict(arrowstyle="->", color=GRAY_DARK, lw=1.0))
            else:
                ax.text(bx, pap + 1.6, f"{pap:.1f}", ha="center", va="bottom",
                        fontsize=8.6, color=GRAY_DARK, fontweight="bold")
            ox = x[i] + w / 2
        else:
            ox = x[i]

        col = BLUE if r["case"] != "4" else AMBER
        if r["case"] == "3t":
            col = GREEN_DARK
        ax.bar(ox, ours, w, color=col, edgecolor="none", zorder=3,
               label="Este trabajo" if i == 0 else None)
        ax.text(ox, ours + 1.6, f"{ours:.1f}%", ha="center", va="bottom",
                fontsize=8.8, color=col, fontweight="bold")

    ax.set_xticks(x, [f"{n}\n{s}" for n, s in zip(names, sub)])
    ax.tick_params(axis="x", labelsize=8.8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Precisión en prueba (%)")
    ax.legend(loc="upper right", ncol=2)
    save(fig, "fig6_barras")


# 4. Curvas de entrenamiento ---------------------------------------------------
def make_curvas() -> None:
    cases = [
        ("fig6_case1", "Caso 1 (89.8 %)", BLUE, "-"),
        ("fig6_case2", "Caso 2 (89.0 %)", TEAL, "-"),
        ("fig6_case3_tabla", "Caso 3 tabla (91.2 %)", GREEN_DARK, "-"),
        ("fig6_case3", "Caso 3 gráfica (84.8 %)", AMBER, "--"),
        ("fig6_case4", "Caso 4 (14.4 %)", RED, ":"),
    ]
    full_hist = json.load(open(os.path.join(RES, "04_full/paper_full/history.json")))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.3),
                                   gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28})

    for folder, lbl, col, ls in cases:
        h = json.load(open(os.path.join(RES, "02_fig6", folder, "history.json")))
        ep = [r["epoch"] for r in h]
        acc = [r["test_acc"] * 100.0 for r in h]
        ax1.plot(ep, acc, ls=ls, color=col, lw=2.0, label=lbl)

    ax1.set_title("Casos 80×80 (20 épocas)", fontsize=11, color=INK, pad=8)
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Precisión en prueba (%)")
    ax1.set_xlim(1, 20)
    ax1.set_ylim(0, 100)
    ax1.set_xticks([1, 5, 10, 15, 20])
    ax1.legend(loc="center right", fontsize=8.3)

    ep_f = [r["epoch"] for r in full_hist]
    acc_f = [r["test_acc"] * 100.0 for r in full_hist]
    tr_f = [r["train_acc"] * 100.0 for r in full_hist]
    ax2.plot(ep_f, tr_f, "o--", color=GRAY, lw=1.6, ms=4.5, label="Entrenamiento (80k)")
    ax2.plot(ep_f, acc_f, "s-", color=GREEN_DARK, lw=2.4, ms=5.8, label="Prueba (11k)")
    ax2.annotate("90.63 %", xy=(6, acc_f[-1]), xytext=(4.3, 81),
                 color=GREEN_DARK, fontsize=10.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN_DARK, lw=1.2))
    ax2.set_title("Sistema completo 1000×1000", fontsize=11, color=INK, pad=8)
    ax2.set_xlabel("Época (2.93 h total en RTX 5060 Ti)")
    ax2.set_xlim(0.7, 6.4)
    ax2.set_ylim(60, 96)
    ax2.set_xticks(ep_f)
    ax2.legend(loc="lower right", fontsize=8.5)
    save(fig, "curvas_entrenamiento")


# 5. Barrido en d_f y D/D_min --------------------------------------------------
def make_barrido_df() -> None:
    rows = json.load(open(os.path.join(RES, "03_sweep/sweep.json")))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    ax.axvline(1.0, color=GRAY_DARK, lw=1.4, ls="--", zorder=1)
    ax.text(1.09, 44, r"$D = D_{\min}$", color=GRAY_DARK, fontsize=11.5, fontweight="bold")
    ax.axvline(0.1, color=BLUE, lw=1.1, ls=":", alpha=0.75, zorder=1)
    ax.text(0.088, 22, r"$0.1\,D_{\min}$ (88.4 %)", color=BLUE, fontsize=10.5,
            fontweight="bold", ha="right")
    ax.axhline(10.0, color=RED, lw=1.2, ls=":", alpha=0.85)
    ax.text(0.025, 12.2, "azar (10 %)", color=RED, fontsize=10.5, fontweight="bold")

    styles = [
        (1.0, dict(color=BLUE, ls="-", marker="o", ms=6.8, mfc=BLUE, lw=2.5,
                   label=r"$d_f = 1\,\lambda$")),
        (6.0, dict(color=GREEN_DARK, ls="-", marker="s", ms=7.2, mfc=GREEN_DARK, lw=2.8,
                   label=r"$d_f = 6\,\lambda$")),
        (12.0, dict(color=AMBER, ls="--", marker="o", ms=8.5, mfc="white", mew=2.3, lw=2.5,
                    label=r"$d_f = 12\,\lambda$")),
    ]
    for df_val, kw in styles:
        sub = sorted([r for r in rows if r["d_f_lambda"] == df_val], key=lambda r: r["ratio"])
        ax.plot([r["ratio"] for r in sub], [r["acc"] * 100.0 for r in sub], zorder=4, **kw)

    ax.annotate(r"$6\lambda \equiv 12\lambda \;\rightarrow\;$ gobierna $N_{F,n}$",
                xy=(0.088, 86.5), xytext=(0.0011, 74),
                color=INK, fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))

    ax.set_xscale("log")
    ax.set_xlim(0.0008, 3.8)
    ax.set_ylim(5, 100)
    ax.set_xlabel(r"Conectividad geométrica $D / D_{\min}$")
    ax.set_ylabel("Precisión en prueba (%)")
    ax.legend(loc="lower right")
    save(fig, "barrido_df")


# 6. Barrido por número de capas L ---------------------------------------------
def make_barrido_capas() -> None:
    rows = json.load(open(os.path.join(RES, "03_sweep/sweep_layers.json")))
    fig, ax = plt.subplots(figsize=(8.0, 4.4))

    ax.axvline(1.0, color=GRAY_DARK, lw=1.3, ls="--", alpha=0.65, zorder=1)
    ax.text(1.06, 95.0, r"$D_{\min}$", color=GRAY_DARK, fontsize=11, fontweight="bold")
    ax.axvline(0.03, color=INK, lw=1.4, ls="-.", alpha=0.55, zorder=1)
    ax.text(0.032, 94.5, r"$D = 0.03\,D_{\min}$", color=INK, fontsize=11, fontweight="bold")

    # Rampa secuencial azul -> verde
    palette = {1: BLUE, 3: TEAL, 5: "#24a168", 8: GREEN_DARK}
    markers = {1: "^", 3: "D", 5: "s", 8: "o"}
    label_offsets = {
        1: (10, -14),
        3: (10, -10),
        5: (10, -10),
        8: (-52, 8),
    }

    for L in [1, 3, 5, 8]:
        ax.axvline(1.0 / (L + 1), color=palette[L], lw=0.9, ls=":", alpha=0.45, zorder=1)
        sub = sorted([r for r in rows if r["num_layers"] == L], key=lambda r: r["ratio"])
        xs = [r["ratio"] for r in sub]
        ys = [r["acc"] * 100.0 for r in sub]
        ax.plot(xs, ys, marker=markers[L], color=palette[L], lw=2.6, ms=6.8,
                label=f"L = {L} {'placa' if L == 1 else 'placas'}", zorder=4)
        for r in sub:
            if abs(r["ratio"] - 0.03) < 1e-6:
                y03 = r["acc"] * 100.0
                ax.annotate(f"{y03:.1f}%", (0.03, y03), xytext=label_offsets[L],
                            textcoords="offset points", color=palette[L],
                            fontsize=11, fontweight="bold")
            if L == 8 and abs(r["ratio"] - 0.3) < 1e-6:
                ax.annotate(f"{r['acc']*100:.1f}%", (0.3, r["acc"] * 100.0),
                            xytext=(0, 8), textcoords="offset points", ha="center",
                            color=GREEN_DARK, fontsize=11, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(0.0008, 3.8)
    ax.set_ylim(5, 104)
    ax.set_xlabel(r"Conectividad geométrica $D / D_{\min}$  ($d_f = 6\lambda$)")
    ax.set_ylabel("Precisión en prueba (%)")
    ax.legend(loc="lower right")
    save(fig, "barrido_capas")


# 7. Matriz de confusión 1000x1000 ---------------------------------------------
def make_confusion_full() -> None:
    cm = np.load(os.path.join(RES, "04_full/paper_full/confusion.npy"))
    k = cm.shape[0]
    names = [str(i) for i in range(10)] + ["X"]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100.0

    cmap = LinearSegmentedColormap.from_list("apple_green", ["#ffffff", "#d8f5df", GREEN_DARK])

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.grid(False)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color("#d2d2d7")

    ax.imshow(norm, cmap=cmap, vmin=0, vmax=100, aspect="equal")
    for i in range(k):
        for j in range(k):
            val = norm[i, j]
            if i == j or val >= 3.0:
                txt_col = "white" if val > 52.0 else INK
                wt = "bold" if i == j else "normal"
                ax.text(j, i, f"{val:.1f}",
                        ha="center", va="center", fontsize=10.0, fontweight=wt, color=txt_col)

    ax.set_xticks(range(k), names)
    ax.set_yticks(range(k), names)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    save(fig, "confusion_full")


# 8. Tolerancias de fabricación (1x4 original por compatibilidad) --------------
def make_fabricacion() -> None:
    f80 = json.load(open(os.path.join(RES, "02_fig6/fig6_case2/fabricacion/fabricacion.json")))
    f1k = json.load(open(os.path.join(RES, "04_full/paper_full/fabricacion/fabricacion.json")))

    fig, axs = plt.subplots(1, 4, figsize=(13.2, 3.2), sharey=True,
                            gridspec_kw={"wspace": 0.16})

    c80, c1k = AMBER, GREEN_DARK

    for i, ax in enumerate(axs):
        ax.axhline(84.0, color=RED, lw=1.2, ls="--", zorder=1)
        if i == 0:
            ax.text(32.5, 72.0, "experimento Chen (84 %)", color=RED,
                    fontsize=8.2, fontweight="bold", ha="right")
        ax.set_ylim(0, 102)

    # (a) Cuantización
    ax = axs[0]
    lv80 = [r["levels"] for r in f80["quant"]]
    ac80 = [r["acc"] * 100 for r in f80["quant"]]
    lv1k = [r["levels"] for r in f1k["quant"]]
    ac1k = [r["acc"] * 100 for r in f1k["quant"]]
    ax.plot(lv80, ac80, "o--", color=c80, lw=2.0, ms=5.5, label="80×80 (Caso 2)")
    ax.plot(lv1k, ac1k, "s-", color=c1k, lw=2.2, ms=5.5, label="1000×1000 (completa)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lv80, [str(v) for v in lv80])
    ax.set_xlabel("Niveles de fase (litografía)")
    ax.set_ylabel("Precisión (%)")
    ax.set_title("(a) Cuantización", fontsize=10.8, color=INK)
    ax.legend(loc="lower right", fontsize=8.2)

    # (b) Profundidad de grabado
    ax = axs[1]
    sc80 = [r["scale"] for r in f80["depth"]]
    ac80 = [r["acc"] * 100 for r in f80["depth"]]
    sc1k = [r["scale"] for r in f1k["depth"]]
    ac1k = [r["acc"] * 100 for r in f1k["depth"]]
    ax.plot(sc80, ac80, "o--", color=c80, lw=2.0, ms=5.5)
    ax.plot(sc1k, ac1k, "s-", color=c1k, lw=2.2, ms=5.5)
    ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2], ["−20%", "−10%", "0", "+10%", "+20%"])
    ax.set_xlabel("Error de profundidad de grabado")
    ax.set_title("(b) Profundidad Δh", fontsize=10.8, color=INK)

    # (c) Desalineación lateral
    ax = axs[2]
    sh80 = [r["shift_um"] for r in f80["shift"]]
    ac80 = [r["acc"] * 100 for r in f80["shift"]]
    sd80 = [r["std"] * 100 for r in f80["shift"]]
    sh1k = [r["shift_um"] for r in f1k["shift"]]
    ac1k = [r["acc"] * 100 for r in f1k["shift"]]
    sd1k = [r["std"] * 100 for r in f1k["shift"]]
    ax.errorbar(sh80, ac80, yerr=sd80, fmt="o--", color=c80, lw=2.0, ms=5.2, capsize=3)
    ax.errorbar(sh1k, ac1k, yerr=sd1k, fmt="s-", color=c1k, lw=2.2, ms=5.2, capsize=3)
    ax.axvline(4.0, color=GRAY, lw=0.9, ls=":")
    ax.text(4.7, 94.5, "1 neurona\n(~4 µm)", color=GRAY_DARK, fontsize=8.3,
            fontweight="bold", va="top")
    ax.set_xlabel("Desalineación lateral (µm)")
    ax.set_title("(c) Alineación lateral", fontsize=10.8, color=INK)

    # (d) Distancia axial
    ax = axs[3]
    d80 = [r["rel"] * 100 for r in f80["distance"]]
    a80 = [r["acc"] * 100 for r in f80["distance"]]
    d1k = [r["rel"] * 100 for r in f1k["distance"]]
    a1k = [r["acc"] * 100 for r in f1k["distance"]]
    ax.plot(d80, a80, "o--", color=c80, lw=2.0, ms=5.5)
    ax.plot(d1k, a1k, "s-", color=c1k, lw=2.2, ms=5.5)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xlabel("Error de separación ΔD / D (%)")
    ax.set_title("(d) Separación axial", fontsize=10.8, color=INK)

    save(fig, "fabricacion")


# 9. Panel 2x2 de matrices de confusión Fig. 6 ---------------------------------
def make_fig6_confusions() -> None:
    specs = [
        ("fig6_case1", "Caso 1 · 89.8 %"),
        ("fig6_case2", "Caso 2 · 89.0 %"),
        ("fig6_case3_tabla", "Caso 3 (tabla) · 91.2 %"),
        ("fig6_case4", "Caso 4 · 14.4 % (franjas)"),
    ]
    cmap_ok = LinearSegmentedColormap.from_list("blue_clean", ["#ffffff", "#d8edff", BLUE])
    cmap_bad = LinearSegmentedColormap.from_list("amber_clean", ["#ffffff", "#ffebc7", RED])

    fig, axs = plt.subplots(2, 2, figsize=(7.6, 6.2),
                            gridspec_kw={"wspace": 0.24, "hspace": 0.34})
    for idx, (ax, (folder, title)) in enumerate(zip(axs.ravel(), specs)):
        cm = np.load(os.path.join(RES, "02_fig6", folder, "confusion.npy"))
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100.0
        ax.grid(False)
        for sp in ("top", "right", "left", "bottom"):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color("#d2d2d7")
        cmap = cmap_bad if folder == "fig6_case4" else cmap_ok
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=100, aspect="equal")
        ax.set_title(title, fontsize=12.5,
                     color=RED if folder == "fig6_case4" else INK, pad=6)
        ax.set_xticks(range(0, 10, 3))
        ax.set_yticks(range(0, 10, 3))
        ax.tick_params(labelsize=10.5)
        if idx >= 2:
            ax.set_xlabel("Clase predicha", fontsize=11.5)
        if idx % 2 == 0:
            ax.set_ylabel("Clase real", fontsize=11.5)
    save(fig, "fig6_confusions")


# 10. Tolerancias robustas: (a) Cuantización y (b) Profundidad -----------------
def make_fabricacion_robustas() -> None:
    f80 = json.load(open(os.path.join(RES, "02_fig6/fig6_case2/fabricacion/fabricacion.json")))
    f1k = json.load(open(os.path.join(RES, "04_full/paper_full/fabricacion/fabricacion.json")))

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True,
                            gridspec_kw={"wspace": 0.14})
    c80, c1k = AMBER, GREEN_DARK

    for i, ax in enumerate(axs):
        ax.axhline(84.0, color=RED, lw=1.4, ls="--", zorder=1)
        ax.set_ylim(0, 106)
        if i == 0:
            ax.text(31.0, 77.5, "Experimento Chen (84 %)", color=RED,
                    fontsize=10.5, fontweight="bold", ha="right")

    # (a) Cuantización
    ax = axs[0]
    lv80 = [r["levels"] for r in f80["quant"]]
    ac80 = [r["acc"] * 100 for r in f80["quant"]]
    lv1k = [r["levels"] for r in f1k["quant"]]
    ac1k = [r["acc"] * 100 for r in f1k["quant"]]
    ax.plot(lv80, ac80, "o--", color=c80, lw=2.5, ms=7.2, label="80×80 (Caso 2)")
    ax.plot(lv1k, ac1k, "s-", color=c1k, lw=2.8, ms=7.2, label="1000×1000 (completa)")
    ax.axvline(8, color=BLUE, lw=1.2, ls=":", alpha=0.8)
    ax.annotate("8 niveles (3 pasos):\n91.05 % / 88.2 %",
                xy=(8, 91.05), xytext=(9.5, 46),
                color=INK, fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.set_xscale("log", base=2)
    ax.set_xticks(lv80, [str(v) for v in lv80])
    ax.set_xlabel("Niveles de fase (pasos litográficos)")
    ax.set_ylabel("Precisión en prueba (%)")
    ax.set_title("(a) Cuantización de fase", fontsize=13.5, color=INK, pad=8)
    ax.legend(loc="lower right", fontsize=11)

    # (b) Profundidad
    ax = axs[1]
    sc80 = [r["scale"] for r in f80["depth"]]
    ac80 = [r["acc"] * 100 for r in f80["depth"]]
    sc1k = [r["scale"] for r in f1k["depth"]]
    ac1k = [r["acc"] * 100 for r in f1k["depth"]]
    ax.axvspan(0.9, 1.1, color=GREEN, alpha=0.08, zorder=0)
    ax.plot(sc80, ac80, "o--", color=c80, lw=2.5, ms=7.2, label="80×80 (Caso 2)")
    ax.plot(sc1k, ac1k, "s-", color=c1k, lw=2.8, ms=7.2, label="1000×1000 (completa)")
    ax.text(1.0, 58, "±10 % mantiene\n88.8 % – 91.7 %", ha="center",
            color=GREEN_DARK, fontsize=11.5, fontweight="bold")
    ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2], ["−20%", "−10%", "0", "+10%", "+20%"])
    ax.set_xlabel("Error de profundidad de grabado Δh")
    ax.set_title("(b) Error en profundidad Δh", fontsize=13.5, color=INK, pad=8)

    save(fig, "fabricacion_robustas")


# 11. Tolerancias críticas: (c) Alineación lateral y (d) Separación axial ----
def make_fabricacion_criticas() -> None:
    f80 = json.load(open(os.path.join(RES, "02_fig6/fig6_case2/fabricacion/fabricacion.json")))
    f1k = json.load(open(os.path.join(RES, "04_full/paper_full/fabricacion/fabricacion.json")))

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True,
                            gridspec_kw={"wspace": 0.14})
    c80, c1k = AMBER, GREEN_DARK

    for ax in axs:
        ax.axhline(84.0, color=RED, lw=1.4, ls="--", zorder=1)
        ax.axhline(10.0, color=GRAY, lw=1.0, ls=":", zorder=1)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

    # (c) Desalineación lateral
    ax = axs[0]
    sh80 = [r["shift_um"] for r in f80["shift"]]
    ac80 = [r["acc"] * 100 for r in f80["shift"]]
    sd80 = [r["std"] * 100 for r in f80["shift"]]
    sh1k = [r["shift_um"] for r in f1k["shift"]]
    ac1k = [r["acc"] * 100 for r in f1k["shift"]]
    sd1k = [r["std"] * 100 for r in f1k["shift"]]
    ax.errorbar(sh80, ac80, yerr=sd80, fmt="o--", color=c80, lw=2.5, ms=7.0,
                capsize=4, label="80×80 (3.8 µm/píxel)")
    ax.errorbar(sh1k, ac1k, yerr=sd1k, fmt="s-", color=c1k, lw=2.8, ms=7.0,
                capsize=4, label="1000×1000 (4 µm/píxel)")
    ax.axvline(4.0, color=RED, lw=1.3, ls=":")
    ax.annotate("1 neurona (~4 µm):\n83.9 % / 81.4 % ≈ 84 % real",
                xy=(4.0, 82.0), xytext=(6.2, 65),
                color=RED, fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("Desalineación lateral aleatoria (µm)")
    ax.set_ylabel("Precisión en prueba (%)")
    ax.set_title("(c) Desalineación lateral", fontsize=13.5, color=INK, pad=8)
    ax.legend(loc="upper right", fontsize=10.5)

    # (d) Distancia axial
    ax = axs[1]
    d80 = [r["rel"] * 100 for r in f80["distance"]]
    a80 = [r["acc"] * 100 for r in f80["distance"]]
    d1k = [r["rel"] * 100 for r in f1k["distance"]]
    a1k = [r["acc"] * 100 for r in f1k["distance"]]
    ax.plot(d80, a80, "o--", color=c80, lw=2.5, ms=7.0, label="80×80 (ΔD/D ≤ 1.3 %)")
    ax.plot(d1k, a1k, "s-", color=c1k, lw=2.8, ms=7.0, label="1000×1000 (ΔD/D ≤ 0.10 %)")
    ax.annotate("±2 % en 1000² = ±1 mm\n(20× el límite) " + r"$\rightarrow$" + " 12–15 %",
                xy=(2.0, 15.05), xytext=(2.8, 42),
                color=GREEN_DARK, fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN_DARK, lw=1.5))
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xlabel("Error de separación axial ΔD / D (%)")
    ax.set_title("(d) Separación axial ΔD", fontsize=13.5, color=INK, pad=8)
    ax.legend(loc="upper right", fontsize=10.2)

    save(fig, "fabricacion_criticas")


# 12. Compromiso precisión vs. eficiencia energética η -------------------------
def make_eficiencia() -> None:
    # Datos exactos de tesina/capitulos/06_resultados.tex sec. 6.6.1 (Caso 2, 20 épocas)
    weights = [0, 1, 3, 10]
    acc = [89.0, 88.6, 85.8, 64.6]
    eta = [1.6, 2.2, 18.6, 89.4]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(eta, acc, "-", color=BLUE, lw=2.6, zorder=2)

    specs = {
        0:  (RED,        (4.0, 77.5),   "left",  "w = 0 (sin penalización)\nη = 1.6 %, acc = 89.0 %"),
        1:  (BLUE,       (12.0, 92.8),  "left",  "w = 1 (η = 2.2 %, acc = 88.6 %)"),
        3:  (GREEN_DARK, (33.0, 87.2),  "left",  "w = 3: 11× más luz en detectores\n(η = 18.6 %, acc = 85.8 %, −3.2 pp)"),
        10: (BLUE,       (62.0, 60.5),  "center", "w = 10\n(89.4 %, 64.6 %)"),
    }
    for w, x, y in zip(weights, eta, acc):
        col, xy_txt, ha, lbl = specs[w]
        ms = 12.0 if w == 3 else 8.8
        ax.plot(x, y, "o", color=col, ms=ms, mec="white", mew=1.6, zorder=4)
        ax.annotate(lbl, (x, y), xytext=xy_txt, textcoords="data",
                    color=col, fontsize=11, fontweight="bold", ha=ha,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4, shrinkA=3, shrinkB=6))

    ax.set_xlim(-5, 98)
    ax.set_ylim(57, 97)
    ax.set_xlabel("Eficiencia energética en detectores η (%)")
    ax.set_ylabel("Precisión en prueba (%)")
    ax.set_title("Compromiso precisión vs. energía capturada (penalización w(1 − η))", pad=10)
    save(fig, "eficiencia_eta")


def main() -> None:
    make_aliasing()
    make_fig6b()
    make_fig6_barras()
    make_curvas()
    make_barrido_df()
    make_barrido_capas()
    make_confusion_full()
    make_fabricacion()
    make_fig6_confusions()
    make_fabricacion_robustas()
    make_fabricacion_criticas()
    make_eficiencia()
    svgs = sorted(f for f in os.listdir(OUT) if f.endswith(".svg"))
    print(f"Generados {len(svgs)} SVGs en {OUT}: {', '.join(svgs)}")


if __name__ == "__main__":
    main()
