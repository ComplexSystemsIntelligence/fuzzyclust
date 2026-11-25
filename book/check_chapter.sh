#!/usr/bin/env bash
# Compile a single chapter in isolation to catch LaTeX errors early.
# Usage: book/check_chapter.sh chapters/05_visualization.tex [more chapters...]
set -euo pipefail
cd "$(dirname "$0")"
tag=$(basename "$1" .tex)_$$
out=_check_tmp/$tag
mkdir -p "$out"
{
  echo '\documentclass[11pt,a4paper]{book}'
  echo '\input{preamble}'
  echo '\addbibresource{references.bib}'
  for f in bib/*.bib; do [ -e "$f" ] && echo "\\addbibresource{$f}"; done
  echo '\begin{document}'
  for ch in "$@"; do echo "\\input{$ch}"; done
  echo '\printbibliography'
  echo '\end{document}'
} > "_check_$tag.tex"
latexmk -pdf -outdir="$out" -interaction=nonstopmode -halt-on-error -silent "_check_$tag.tex" > /dev/null 2>&1 || {
  echo "LaTeX errors:"; grep -A3 -E "^!|Error|Undefined control" "$out/_check_$tag.log" | head -60; exit 1; }
grep -E "Warning: (Citation|Reference)" "$out/_check_$tag.log" | sort -u | head -20 || true
echo "OK: book/$out/_check_$tag.pdf ($(pdfinfo "$out/_check_$tag.pdf" 2>/dev/null | grep Pages | awk '{print $2}') pages)"
rm -f "_check_$tag.tex"
