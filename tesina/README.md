# Tesina (LaTeX)

Documento de la tesina sobre redes neuronales difractivas en el visible.

- `main.pdf`: versión compilada.
- `meta/datos.tex`: metadatos de la portada (autores, curso, profesor y fecha).
- `capitulos/`: un archivo por capítulo y apéndice.
- `referencias.bib`: bibliografía en APA 7 (`biblatex-apa` + `biber`).

Compilar localmente:

```bash
latexmk -pdf main.tex
```

En Overleaf: sube la carpeta, compilador pdfLaTeX; la bibliografía usa `biber` automáticamente.
