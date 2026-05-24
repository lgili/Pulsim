#!/usr/bin/env bash
# Build the SPEC 2026 paper PDF.
# Runs the standard 4-pass LaTeX dance:
#   pdflatex → bibtex → pdflatex × 2
# to resolve forward references + bibliography.
#
# Usage:
#   ./build.sh           # build main.pdf
#   ./build.sh clean     # also clean aux files
#
# Requires: pdflatex + bibtex (MacTeX BasicTeX is enough).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

JOB=main

clean() {
  rm -f "${JOB}.aux" "${JOB}.bbl" "${JOB}.blg" "${JOB}.log" \
        "${JOB}.out" "${JOB}.toc" "${JOB}.fls" "${JOB}.fdb_latexmk" \
        "${JOB}.synctex.gz"
}

if [ "${1:-}" = "clean" ]; then
  echo "→ cleaning aux files"
  clean
  rm -f "${JOB}.pdf"
  exit 0
fi

echo "→ pass 1 — pdflatex"
pdflatex -interaction=nonstopmode -halt-on-error "${JOB}.tex" > /dev/null

echo "→ bibtex"
bibtex "${JOB}" || {
  echo "  (bibtex warnings — usually fine; check ${JOB}.blg)"
}

echo "→ pass 2 — pdflatex"
pdflatex -interaction=nonstopmode -halt-on-error "${JOB}.tex" > /dev/null

echo "→ pass 3 — pdflatex (final, resolves all refs)"
pdflatex -interaction=nonstopmode -halt-on-error "${JOB}.tex" > /dev/null

echo
echo "✓ build complete → ${JOB}.pdf ($(du -h ${JOB}.pdf | cut -f1))"
echo
echo "  Quick stats:"
PAGES=$(pdfinfo "${JOB}.pdf" 2>/dev/null | grep '^Pages:' | awk '{print $2}')
echo "    pages: ${PAGES:-unknown} (target ≤ 6 for SPEC 2026)"
if [ -n "${PAGES:-}" ] && [ "${PAGES}" -gt 6 ]; then
  echo "    ⚠️  OVER LIMIT — trim before submitting"
fi
