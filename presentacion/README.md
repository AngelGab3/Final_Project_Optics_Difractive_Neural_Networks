# Presentación del proyecto (`presentacion/`)

Material de exposición en dos formatos sincronizados: una presentación interactiva en HTML/KaTeX (desplegada en la rama `gh-pages`) y la versión Beamer en PDF.

---

## Entregables y enlaces

| Formato | Archivo | Descripción |
| :--- | :--- | :--- |
| HTML interactiva | [`presentacion.html`](presentacion.html) | 34 diapositivas en 3 actos con diagramas SVG (exportados desde TikZ), figuras de resultados, ecuaciones con KaTeX local y notas de expositor |
| Sitio GitHub Pages | [Presentación Web](https://angelgab3.github.io/Final_Project_Optics_Difractive_Neural_Networks/) | Copia publicada en la rama `gh-pages` (`index.html`, `figs/`, `vendor/`) |
| Beamer PDF | [`presentacion.pdf`](presentacion.pdf) | 16 diapositivas compiladas desde [`presentacion.tex`](presentacion.tex) |

---

## Atajos de teclado en `presentacion.html`

| Tecla | Acción |
| :--- | :--- |
| `ArrowRight` / `PageDown` / `Space` | Siguiente diapositiva |
| `ArrowLeft` / `PageUp` | Diapositiva anterior |
| `Home` / `End` | Ir al inicio o al final |
| `N` | Mostrar u ocultar notas del presentador |
| `F` | Activar o salir de pantalla completa |

---

## Contenido de la carpeta

* `tikz/`: fuentes TikZ standalone para los diagramas físicos y matemáticos.
* `figs/`: diagramas SVG compilados desde TikZ y gráficas PNG/SVG generadas por [`make_figs.py`](make_figs.py).
* `vendor/katex/`: distribución local de KaTeX (permite renderizar todas las ecuaciones sin conexión a internet).
* [`presentacion.tex`](presentacion.tex): fuente LaTeX Beamer.

### Compilación local de Beamer

```bash
cd presentacion
latexmk -pdf presentacion.tex
```
