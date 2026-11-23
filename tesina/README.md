# Tesina (`tesina/`)

Documento técnico en LaTeX sobre redes neuronales difractivas en el visible, redactado por **Alonso Hernández Hernández** y **Ángel Gabriel Munguía Velázquez** (Universidad de las Américas Puebla).

* **Documento compilado:** [`main.pdf`](main.pdf) (55 páginas)
* **Archivo raíz:** [`main.tex`](main.tex)
* **Bibliografía:** [`referencias.bib`](referencias.bib) (16 fuentes en estilo APA 7 mediante `biblatex-apa` + `biber`)

---

## Estructura por capítulos

| Archivo / Carpeta | Contenido |
| :--- | :--- |
| `meta/` | Preámbulo (`preambulo.tex`), portada institucional (`portada.tex`), metadatos (`datos.tex`), resumen/abstract y lista de símbolos (`preliminares.tex`) |
| [`capitulos/01_introduccion.tex`](capitulos/01_introduccion.tex) | Motivación, planteamiento del problema, objetivos y contribuciones |
| [`capitulos/02_difraccion.tex`](capitulos/02_difraccion.tex) | De Maxwell a Helmholtz, Rayleigh--Sommerfeld, espectro angular, Fresnel/Fraunhofer y límite de Nyquist de una neurona |
| [`capitulos/03_modelo.tex`](capitulos/03_modelo.tex) | Arquitectura $\text{D}^2\text{NN}$, relieve en sílice fundida, cálculo de Wirtinger y verificación numérica a $10^{-16}$ |
| [`capitulos/04_conectividad.tex`](capitulos/04_conectividad.tex) | Deducción de $D_{\min}$, casos de la Fig. 6 y escalamiento por número de Fresnel de una neurona ($N_{F,n}$) |
| [`capitulos/05_metodo_numerico.tex`](capitulos/05_metodo_numerico.tex) | Discretización, relleno $2\times$, límite de banda, identidad $D_{\min} \equiv z_c$ y validación del propagador |
| [`capitulos/06_resultados.tex`](capitulos/06_resultados.tex) | Reproducción Fig. 6, inconsistencias del artículo, barridos $D/D_{\min}$ y $L$, sistema $1000^2$, tolerancias y eficiencia |
| [`capitulos/07_conclusiones.tex`](capitulos/07_conclusiones.tex) | Conclusiones, limitaciones y trabajo futuro |
| [`capitulos/A_rayleigh_sommerfeld.tex`](capitulos/A_rayleigh_sommerfeld.tex) | Apéndice A: deducción integral de Rayleigh--Sommerfeld y apertura circular en el eje |
| [`capitulos/B_implementacion.tex`](capitulos/B_implementacion.tex) | Apéndice B: algoritmo de entrenamiento, decisiones de reproducción, comandos y benchmark RTX 5060 Ti |
| `figuras/` | Figuras en PNG utilizadas en el documento |

---

## Compilación

### En local (`pdflatex` + `biber`)

```bash
cd tesina
latexmk -pdf main.tex
```

Para limpiar archivos auxiliares tras compilar:

```bash
latexmk -c && rm -f main.bbl main.run.xml
```

### En Overleaf

Sube el contenido de `tesina/` seleccionando compilador **pdfLaTeX**; Overleaf ejecuta `biber` automáticamente según [`latexmkrc`](latexmkrc).
