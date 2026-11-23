# Paquete `d2nn/`

Módulos centrales de simulación óptica, entrenamiento en PyTorch y visualización para redes neuronales difractivas en el visible ($\lambda = 632.8\text{ nm}$).

---

## Módulos

| Archivo | Responsabilidad | Símbolos principales |
| :--- | :--- | :--- |
| [`__init__.py`](__init__.py) | Metadatos del paquete y referencia al artículo base | --- |
| [`optics.py`](optics.py) | Difracción escalar, espectro angular (ingenuo y banda limitada), Rayleigh--Sommerfeld directo y soluciones analíticas | `phi_max`, `d_min`, `critical_distance`, `band_limit`, `asm_transfer`, `AngularSpectrum`, `propagate`, `rayleigh_sommerfeld_direct`, `circular_aperture_on_axis`, `fresnel_square_aperture` |
| [`model.py`](model.py) | Modelo físico de la $\text{D}^2\text{NN}$ de solo fase y geometría de detectores | `detector_boxes`, `detector_masks`, `D2NN` |
| [`data.py`](data.py) | Descarga de MNIST, generación de dígitos tapados/alterados (clase X) y codificación en amplitud | `load_mnist`, `cover_digit`, `alter_digit`, `make_changed`, `build_split`, `Encoder` |
| [`config.py`](config.py) | Configuraciones reproducibles para $1000\times 1000$, casos de la Fig. 6 y versiones escaladas | `WL`, `Config`, `fig6_case`, `scaled_full` |
| [`train.py`](train.py) | Bucle de entrenamiento con Adam, evaluación, pérdida de eficiencia y serialización de checkpoints | `pick_device`, `build_model`, `build_data`, `evaluate`, `train`, `save`, `load` |
| [`viz.py`](viz.py) | Generación de figuras en estilo de publicación (confusión, fases, ejemplos, corte $x\text{--}z$) | `class_names`, `plot_confusion`, `plot_phases`, `plot_examples`, `plot_xz` |

---

## Decisiones de implementación no especificadas en el artículo

| Aspecto | Elección en este proyecto | Ubicación |
| :--- | :--- | :--- |
| Propagador | Espectro angular de banda limitada (Matsushima & Shimobaba, 2009), relleno $2\times$, cálculo de $H$ en `float64` | [`optics.py`](optics.py) (`AngularSpectrum`) |
| Tamaño finito de las capas | El campo se recorta a $w\times w$ después de cada propagación | [`optics.py`](optics.py) (`_crop_center`) |
| Distancias entrada--L1 y L5--salida | Iguales a la separación entre capas $D$ | [`config.py`](config.py) (`Config.distances`) |
| Codificación de la entrada | MNIST escalado con vecino más cercano al 70 % del lado y binarizado | [`data.py`](data.py) (`Encoder`) |
| Detectores | 11 cuadrados del 5 % del lado (`paper11` para Fig. 3b); 10 en arreglo 3--4--3 del 8 % para Fig. 6 | [`model.py`](model.py) (`detector_boxes`) |
| Función de pérdida | Entropía cruzada con softmax sobre energía relativa $T\cdot E_k/\sum E_j$ ($T=10$) y término opcional $w(1-\eta)$ | [`model.py`](model.py) (`D2NN.logits`) |
| Parametrización de fase | $\Phi = \pi\,\sigma(\theta)\in [0,\pi]$, con $\theta$ inicial $\sim \mathcal{N}(0, 0.1^2)$ | [`model.py`](model.py) (`D2NN.phases`) |
| Dígitos cambiados (clase X) | 50 % tapados (borra 25–65 % del trazo) y 50 % alterados (2–3 trazos extra) | [`data.py`](data.py) (`make_changed`) |
