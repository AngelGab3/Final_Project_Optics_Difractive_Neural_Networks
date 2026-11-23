"""Configuraciones: sistema completo del paper y los 4 casos de la Fig. 6."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

WL = 632.8e-9  # He-Ne


@dataclass
class Config:
    name: str = "paper_full"
    # óptica
    wl: float = WL
    n: int = 1000                 # neuronas por lado (1000x1000 = 1e6 por capa)
    d_f: float = 4e-6             # tamaño de neurona
    distance: float = 0.05        # distancia entre capas (5 cm)
    num_layers: int = 5
    pad_factor: int = 2
    bandlimit: bool = True
    phase_max: float = 3.141592653589793
    # entrada / salida
    digit_frac: float = 0.7
    binarize: bool = True
    num_classes: int = 11
    layout: str = "paper11"
    det_frac: float = 0.05
    temperature: float = 10.0
    # peso del término de eficiencia: pérdida += w * (1 - energía en detectores / energía de salida).
    # 0 reproduce la versión original (sólo energía relativa entre detectores).
    efficiency_weight: float = 0.0
    # datos (paper: 55k MNIST + 25k cambiados para entrenar; 10k + 1k para probar)
    n_train: int = 55000
    n_train_changed: int = 25000
    n_test: int = 10000
    n_test_changed: int = 1000
    # entrenamiento (paper: Adam lr=1e-3, 20 épocas)
    epochs: int = 20
    batch_size: int = 8
    lr: float = 1e-3
    seed: int = 0
    use_checkpoint: bool = False

    @property
    def distances(self):
        return [self.distance] * (self.num_layers + 1)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def fig6_case(case: int, **kw) -> Config:
    """Casos de la Fig. 6(c): N = 80x80, 10 clases, 10k imágenes de entrenamiento, 500 de prueba.

    Nota: la tabla del paper dice D ~ 2.4e3 lambda para el caso 3, pero en la gráfica 6(b) el
    punto está en ~2.4e4 lambda (encima de la curva). Usamos 2.4e4 (ver README).
    """
    table = {1: (0.53, 15.0), 2: (6.0, 5.7e3), 3: (6.0, 2.4e4), 4: (12.0, 15.0)}
    nf, nd = table[case]
    base = dict(name=f"fig6_case{case}", n=80, d_f=nf * WL, distance=nd * WL,
                num_classes=10, layout="343", det_frac=0.08,
                n_train=10000, n_train_changed=0, n_test=500, n_test_changed=0,
                epochs=20, batch_size=64)
    base.update(kw)
    return Config(**base)


def scaled_full(n: int, **kw) -> Config:
    """Versión reducida del sistema completo conservando la física: mismo d_f y lambda,
    menos neuronas, y D escalado para seguir en el mismo punto relativo de la Ec. 9
    (D/D_min igual que en el paper)."""
    from .optics import d_min

    ref = Config()
    ratio = ref.distance / d_min(ref.n, ref.d_f, ref.wl)
    c = Config(name=f"full_n{n}", n=n, distance=ratio * d_min(n, ref.d_f, ref.wl))
    for k, v in kw.items():
        setattr(c, k, v)
    return c
