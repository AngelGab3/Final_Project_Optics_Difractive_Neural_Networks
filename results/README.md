# Resultados experimentales (`results/`)

Resumen cuantitativo de las simulaciones numéricas y experimentos en GPU. Todos los resultados cuentan con respaldo en JSON, matrices NumPy y gráficas dentro de sus respectivas subcarpetas.

---

## Organización

| Carpeta | Contenido |
| :--- | :--- |
| `01_optics/` | Validación óptica (`summary.json`), comparación Fresnel/Rayleigh--Sommerfeld, aliasing ASM y verificación del gradiente (`gradient_check.json`) |
| `02_fig6/` | Modelos entrenados, métricas y gráficas de los casos 1–4 y `3t` de la Fig. 6 (`tabla_fig6c.json`) + tolerancias del caso 2 |
| `03_sweep/` | Barrido de conectividad por tamaño de neurona (`sweep.json`) y por número de capas `L` (`sweep_layers.json`) |
| `04_full/` | Modelo completo `1000 × 1000` (`paper_full/model.pt`, `history.json`, `metrics.json`) y tolerancias de fabricación (`fabricacion/fabricacion.json`) |

---

## 1. Validación del propagador e identidad $D_{\min} \equiv z_c$

| Prueba | Referencia | Error relativo máx. | Figura |
| :--- | :--- | :---: | :--- |
| Apertura cuadrada $200\,\mu\text{m}$, $z = 8\text{ mm}$ ($N_F \approx 1.98$) | Solución analítica de Fresnel | **0.04 %** | [`b_apertura_cuadrada_fresnel.png`](01_optics/b_apertura_cuadrada_fresnel.png) |
| Apertura circular $a = 40\,\mu\text{m}$, campo en el eje | R–S exacta: $U = e^{jkz} - \frac{z}{R}e^{jkR}$ | **3.6 %** ($z \ge 0.3\text{ mm}$) | [`a_apertura_circular.png`](01_optics/a_apertura_circular.png) |
| Fase aleatoria ($4\,\mu\text{m}$), $z \in [1, 20]\,z_c$ | R–S por suma directa | **3.5–9.1 %** (banda limitada) vs. 22–99 % (ingenuo) | [`c_aliasing_asm.png`](01_optics/c_aliasing_asm.png) |

* Al imponer $f_{\lim} = 1/(2d_f)$ con relleno $2\times$ se obtiene exactamente **$D_{\min} \equiv z_c$**. En 15 combinaciones de $(\sqrt{N}, d_f)$, el producto $f_{\lim}(D_{\min})\cdot 2d_f$ vale **1.000000** ([`01_optics/summary.json`](01_optics/summary.json)).
* Para el sistema completo ($1000\times 1000$, $d_f = 4\,\mu\text{m}$): $\varphi_{\max} = 4.54^\circ$ y $D_{\min} = 50.41\text{ mm}$.

---

## 2. Gradiente por cálculo de Wirtinger

La implementación manual en doble precisión (`float64`/`complex128`, red $32\times 32$ de 5 capas) reproduce el gradiente de diferenciación automática de PyTorch al nivel de error de máquina ([`01_optics/gradient_check.json`](01_optics/gradient_check.json)):

| Capa 1 | Capa 2 | Capa 3 | Capa 4 | Capa 5 | Error relativo total |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `4.90e-16` | `5.31e-16` | `5.10e-16` | `4.79e-16` | `3.43e-16` | **`4.49e-16`** |

---

## 3. Reproducción de la Fig. 6 ($N = 80\times 80$, 5 capas, 10 clases)

Entrenamiento con 10 000 imágenes de entrenamiento, 500 de prueba, Adam ($\text{lr}=10^{-3}$), 20 épocas ([`02_fig6/tabla_fig6c.json`](02_fig6/tabla_fig6c.json)):

| Caso | $d_f$ ($\lambda$) | $D$ ($\lambda$) | $D_{\min}$ ($\lambda$) | $D/D_{\min}$ | Artículo (%) | Este trabajo (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.53 | 15 | 14.9 | 1.01 | 90.0 | **89.8** |
| 2 | 6.00 | $5.7\times 10^3$ | $5.74\times 10^3$ | 0.99 | 91.0 | **89.0** |
| 3 (gráfica 6b) | 6.00 | $2.4\times 10^4$ | $5.74\times 10^3$ | 4.18 | 91.2 | 84.8 |
| 3 (tabla 6c) | 6.00 | $2.4\times 10^3$ | $5.74\times 10^3$ | 0.42 | 91.2 | **91.2** |
| 4 | 12.00 | 15 | $2.30\times 10^4$ | `6.5e-4` | «~0.1» | **14.4** |

**Inconsistencias detectadas en el artículo:**
1. En el **caso 3**, la tabla 6(c) reporta $D \approx 2.4\times 10^3\lambda$ (donde se obtiene exactamente **91.2 %**), mientras que la gráfica 6(b) ubica el punto en $2.4\times 10^4\lambda$ (donde cae a **84.8 %** por fuga de luz fuera de la abertura finita).
2. En el **caso 4**, «~0.1» corresponde a la fracción $0.1$ ($\approx 10\,\%$ de azar en 10 clases).

---

## 4. Barrido de conectividad y capas ($L \in \{1, 3, 5, 8\}$)

### Por tamaño de neurona ($L = 5$, 20 épocas, [`sweep.json`](03_sweep/sweep.json))

| $D/D_{\min}$ | 0.001 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $d_f = 1\lambda$ (%) | 14.6 | 16.6 | 26.0 | 53.0 | 87.0 | **90.4** | 90.2 | 88.6 |
| $d_f = 6\lambda$ (%) | 15.0 | 16.6 | 26.2 | 52.6 | 88.4 | **89.8** | 89.2 | 86.4 |
| $d_f = 12\lambda$ (%) | 15.0 | 16.4 | 26.2 | 52.6 | 88.4 | **89.8** | 89.2 | 86.6 |

### Por número de capas ($d_f = 6\lambda$, 20 épocas, [`sweep_layers.json`](03_sweep/sweep_layers.json))

| Capas $L$ ($L+1$ prop.) | $1/(L+1)$ | 0.001 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $L = 1$ (2 prop.) | 0.500 | 12.4 | 13.6 | 14.8 | 18.2 | 29.0 | 56.0 | 65.8 | 73.0 |
| $L = 3$ (4 prop.) | 0.250 | 13.6 | 15.4 | 17.8 | 36.4 | 73.2 | 86.0 | **89.2** | 86.0 |
| $L = 5$ (6 prop.) | 0.167 | 15.0 | 16.6 | 26.2 | 52.6 | 88.4 | **89.8** | 89.2 | 86.4 |
| $L = 8$ (9 prop.) | 0.111 | 15.6 | 16.6 | 40.4 | 75.2 | 89.6 | **92.8** | 91.6 | 87.4 |

Las curvas de $6\lambda$ y $12\lambda$ coinciden punto a punto, confirmando el escalamiento por $N_{F,n}$; al crecer $L$, el umbral se recorre hacia menores $D/D_{\min}$ por conectividad acumulada en $L+1$ propagaciones.

---

## 5. Sistema completo ($1000\times 1000$ neuronas, 11 clases) y tolerancias

Entrenado en **NVIDIA GeForce RTX 5060 Ti (16 GiB)** con `--epochs 6 --bs 24` ($20\,004$ iteraciones sobre $80\,000$ imágenes, `7.03 GiB` VRAM pico, `20.32 ms/imagen`, **2.93 h** totales, [`04_full/paper_full/metrics.json`](04_full/paper_full/metrics.json)):

| Métrica (11 000 imágenes de prueba) | Artículo (20 épocas) | Este trabajo (6 épocas) |
| :--- | :---: | :---: |
| Precisión total (11 clases) | 91.57 % | **90.63 %** |
| Dígitos sin cambio (0–9, 10 000 imgs.) | --- | **90.24 %** |
| Clase X (tapados o alterados, 1 000 imgs.) | --- | **94.50 %** |
| Tiempo de entrenamiento | $\approx 20\text{ h}$ (GTX 1080) | **2.93 h** (RTX 5060 Ti) |

* **Tolerancias en caso 2 ($80\times 80$, base 89.0 %, [`02_fig6/fig6_case2/fabricacion/fabricacion.json`](02_fig6/fig6_case2/fabricacion/fabricacion.json)):** 8 niveles conservan **88.2 %**; $\pm 10\%$ de profundidad mantiene $\ge 88.8\,\%$; **1 neurona** ($3.8\,\mu\text{m}$) de desalineación cae a **83.9 %**; $\pm 5\%$ en $D$ conserva $\ge 82.4\,\%$.
* **Tolerancias en $1000\times 1000$ (2 000 imágenes, base ideal 91.35 %, [`04_full/paper_full/fabricacion/fabricacion.json`](04_full/paper_full/fabricacion/fabricacion.json)):** 8 niveles conservan **91.05 %**; $\pm 10\%$ de profundidad mantiene **89.65–91.70 %**; **1 neurona** ($4\,\mu\text{m}$) de desalineación cae a **81.4 ± 3.5 %**; y $\pm 2\%$ en $D = 50\text{ mm}$ ($\pm 1\text{ mm}$) desploma a **12.05–15.05 %** porque la tolerancia axial relativa escala como $1/\sqrt{N}$ ($\Delta D \le 50.6\,\mu\text{m}$, es decir **0.10 %** de $D$).
* **Eficiencia de detección ($\eta$):** con la pérdida relativa original solo **1.1 %** de la energía de salida cae en detectores en $1000^2$ (**1.6 %** en caso 2). Con `--efficiency-weight 3` en caso 2 sube **11×** (**1.6 % $\to$ 18.6 %**) a cambio de ~3 puntos de precisión (89.0 % $\to$ 85.8 %).
