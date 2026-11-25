#!/usr/bin/env bash
# Build the book from scratch: cover artwork, example figures, generated
# appendices and the PDF (book/_build/main.pdf).
# Requires a TeX Live with the raleway, plex, libertinus and biblatex packages.
set -euo pipefail
cd "$(dirname "$0")"
python cover/make_cover_art.py
python cover/make_cover_art.py 13.4 12.8 cover/banner_art.tex
(cd cover && latexmk -pdf -interaction=nonstopmode -silent cover.tex logo.tex banner.tex >/dev/null)
MPLBACKEND=Agg python build_figures.py
python build_examples_appendix.py
python build_api_appendix.py
latexmk -pdf -outdir=_build -interaction=nonstopmode -silent main.tex
echo "book/_build/main.pdf: $(pdfinfo _build/main.pdf | awk '/Pages/{print $2}') pages"
