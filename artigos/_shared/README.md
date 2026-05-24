# Shared assets across papers

This folder holds anything reused by **two or more papers** in the
campaign so we don't fork-and-drift the same content across four
subfolders.

## Planned contents

| File | Used by | Purpose |
|---|---|---|
| `refs.bib` | all papers | canonical BibTeX entries (Nabae NPC 1981, Marquardt MMC 2003, Holmes & Lipo PWM, Erickson & Maksimović PE textbook, etc.) — every paper's own `paper.bib` only adds the paper-specific extras |
| `figures/pulsim_pipeline.pdf` | JOSS + TPEL | architecture diagram of Pulsim's PWL pipeline |
| `figures/mmc_topology.pdf` | JESTPE + (potentially) TPEL §VI | clean SVG/PDF of single-phase MMC topology |
| `scripts/extract_figures.py` | JESTPE + TPEL | pulls figures from executed notebooks at 300dpi PNG + vector PDF |
| `scripts/run_pulsim_bench.py` | TPEL benchmark + JESTPE benchmark | thin wrapper around `pulsim.simulate` that records wall-time + peak memory |

## Why this folder exists

Without it, the canonical `Marquardt MMC 2003` BibTeX entry would
be hand-copied into three `paper.bib` files, with three different
typos and three different DOI fields. With it, one source of truth.

## Conventions

- `refs.bib` keys follow `firstauthor_year_shortvenue` style, e.g.
  `marquardt2003_mmc_pesc`, `nabae1981_npc_iastrans`.
- Figures are committed as **both** PDF (for LaTeX) and PNG (for
  README previews + slides).
- Scripts are dependency-light — they import only `numpy`,
  `matplotlib`, `pulsim`, and stdlib.
