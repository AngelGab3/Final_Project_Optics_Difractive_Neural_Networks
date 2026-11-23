# Scripts de simulación y evaluación (`scripts/`)

Todos los scripts se ejecutan desde la **raíz del repositorio**. Los artefactos generados se guardan en [`../results/`](../results/README.md).

---

## Catálogo de scripts

| Script | Propósito | Salida principal |
| :--- | :--- | :--- |
| [`01_validate_optics.py`](01_validate_optics.py) | Valida `AngularSpectrum` frente a Fresnel analítica, Rayleigh--Sommerfeld en el eje, suma directa, $D_{\min}\equiv z_c$ y curva de la Fig. 6(b) | `results/01_optics/` |
| [`02_fig6_cases.py`](02_fig6_cases.py) | Reproduce la Fig. 6(c) en redes $80\times 80$ (casos `1`, `2`, `3`, `4` y `3t` con el valor de tabla) | `results/02_fig6/` |
| [`03_connectivity_sweep.py`](03_connectivity_sweep.py) | Barrido de $D/D_{\min} \in [10^{-3}, 3]$ variando $d_f \in \{1\lambda, 6\lambda, 12\lambda\}$ o el número de capas $L$ | `results/03_sweep/` |
| [`04_train_full.py`](04_train_full.py) | Entrena el sistema completo de $1000\times 1000$ neuronas y 11 clases en GPU | `results/04_full/paper_full/` |
| [`05_evaluate.py`](05_evaluate.py) | Genera matriz de confusión, máscaras de fase, ejemplos con detectores, dígitos 6, onda plana y corte $x\text{--}z$ | Carpeta del modelo |
| [`06_fabrication_errors.py`](06_fabrication_errors.py) | Evalúa sin reentrenar cuantización, profundidad de grabado, desalineación lateral y error axial | `<modelo>/fabricacion/` |
| [`07_check_gradient.py`](07_check_gradient.py) | Compara el gradiente analítico por cálculo de Wirtinger frente a `autograd` en `float64`/`complex128` | `results/01_optics/gradient_check.json` |
| [`benchmark.py`](benchmark.py) | Mide tiempo por iteración y consumo pico de VRAM para elegir tamaño de lote (`--bs`) | Salida estándar |

---

## Comandos de ejecución

### 1. Pruebas unitarias y verificación matemática

```bash
python -m pytest -q tests
python scripts/07_check_gradient.py
python scripts/01_validate_optics.py
```

Opciones útiles en `07_check_gradient.py`: `--n 32 --bs 4 --seed 0 --out results/01_optics/gradient_check.json`.

### 2. Reproducción de Fig. 6 y barridos de conectividad

```bash
python scripts/02_fig6_cases.py --device cuda --epochs 20
python scripts/03_connectivity_sweep.py --device cuda --epochs 20
python scripts/03_connectivity_sweep.py --device cuda --epochs 20 --layers 1 3 5 8 --layer-df 6.0
```

Flags principales:
* `02_fig6_cases.py`: `--cases 1 2 3 4 3t`, `--epochs 20`, `--efficiency-weight 0.0`, `--out results/02_fig6`
* `03_connectivity_sweep.py`: `--sizes 1.0 6.0 12.0`, `--layers ...`, `--layer-df 6.0`, `--ratios ...`, `--efficiency-weight 0.0`

### 3. Sistema completo (1000 × 1000 neuronas, 11 clases, GPU)

```bash
python scripts/benchmark.py --n 1000 --bs 24 --device cuda
python scripts/04_train_full.py --epochs 6 --bs 24 --max-minutes 185 --device cuda
```

Opciones de entrenamiento en `04_train_full.py`: `--checkpoint` (activa *gradient checkpointing* para reducir VRAM), `--resume`, `--lr 1e-3`, `--efficiency-weight 0.0`, `--n-train 55000`, `--n-train-changed 25000`, `--n-test 10000`, `--n-test-changed 1000`.

### 4. Evaluación visual y tolerancias de fabricación

```bash
# Sobre el modelo reducido del caso 2:
python scripts/05_evaluate.py results/02_fig6/fig6_case2/model.pt
python scripts/06_fabrication_errors.py results/02_fig6/fig6_case2/model.pt

# Sobre el sistema completo 1000x1000:
python scripts/05_evaluate.py results/04_full/paper_full/model.pt --bs 24
python scripts/06_fabrication_errors.py results/04_full/paper_full/model.pt --bs 24 --n-eval 2000 --trials 3
```
