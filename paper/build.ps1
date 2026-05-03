# build.ps1 — compile main.tex to PDF
# Requires a LaTeX distribution (MiKTeX or TeX Live)
# Run from the paper/ directory: .\build.ps1

Set-Location $PSScriptRoot

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

Write-Host ""
Write-Host "Done. Output: paper/main.pdf"
