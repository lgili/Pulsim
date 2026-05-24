# SPEC 2026 Conference Paper — MMC Sort-and-Select

**Venue:** [IEEE Southern Power Electronics Conference (SPEC) 2026](https://attend.ieee.org/spec-2026/)
**Location:** Cartagena, Colombia
**Conference dates:** 8–11 December 2026
**Submission deadline:** 31 July 2026
**Working title:** *Sort-and-Select Capacitor Balancing in a
Single-Phase Modular Multilevel Converter: An Open-Source Reference
Implementation*
**Status:** 📝 Drafting LaTeX skeleton

## Why this paper

The MMC project in [`projects/inverters/mmc/`](../../projects/inverters/mmc/)
already produced two highly visual results:

1. **Open-loop drift** — without sort-and-select, the 6 sub-module
   capacitor voltages diverge from the nominal $V_C = V_{dc}/N =
   133.3$ V within 2 line cycles (some caps reach 200+ V, others
   collapse below 50 V), destroying the 4-level staircase output.
2. **Sort-and-select balancing** — with the canonical sort-and-select
   algorithm wired to the simulator via a `step_observer` closure,
   all 6 caps lock within 0.03 V of $V_C$ and the 4-level staircase
   is perfectly preserved.

This is the textbook MMC control challenge in two figures. The paper
turns this into a 6-page IEEE conference submission with:

- A self-contained derivation of the half-bridge SM dynamics + PSC-PWM.
- A formal statement of the sort-and-select algorithm with pseudocode.
- The Pulsim implementation showing how the `step_observer`
  architecture lets the same C++ kernel drive both open-loop and
  closed-loop variants without recompilation.
- Quantitative results: cap voltage spread, fundamental amplitude
  error vs analytical, output THD comparison.

The paper cites the JOSS Pulsim DOI as the canonical software
reference — so it must be submitted **after** the JOSS paper is
accepted (~Sep 2026, well within the 31 Jul submission window).

## Section outline (6 pages, IEEEtran conference, ~3500 words)

| # | Section | Words | Pages | Status |
|:-:|---|---:|:-:|:-:|
|   | Abstract | 200 | 0.25 | ⬜ |
| I | Introduction | 500 | 0.75 | ⬜ |
| II | MMC topology + PSC-PWM | 700 | 1.0 | ⬜ |
| III | Sort-and-select algorithm | 600 | 1.0 | ⬜ |
| IV | Pulsim implementation | 500 | 0.75 | ⬜ |
| V | Simulation results | 800 | 1.75 | ⬜ |
| VI | Conclusion + future work | 200 | 0.25 | ⬜ |
|   | References (10–15 entries) | — | 0.25 | ⬜ |
|   | **Total** | **3500** | **6.0** | |

### Figure plan (extracted from `00_mmc_pulsim_validation.ipynb`)

| Fig | Where | What | Source |
|:-:|---|---|---|
| 1 | §II | MMC topology block diagram | hand-drawn (TikZ) |
| 2 | §II | PSC-PWM carriers + reference signal | from notebook |
| 3 | §III | Sort-and-select decision tree (or pseudocode block) | mmc_model.py |
| 4 | §V.A | Open-loop cap voltages — divergence | from notebook |
| 5 | §V.B | Sort-and-select cap voltages — locked | from notebook |
| 6 | §V.C | 4-level arm voltage + 4-level AC output | from notebook |

`extract_figures.py` regenerates Figs 2, 4, 5, 6 from the MMC
notebook + saves as 300dpi PNG (acceptable by IEEE) or PDF (vector,
preferred). Figs 1 and 3 are hand-drawn in TikZ inside `main.tex`.

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | This tracker |
| `main.tex` | IEEEtran conference paper source |
| `refs.bib` | Bibliography (starter from JOSS, will grow) |
| `figures/` | Extracted + hand-drawn figures |
| `extract_figures.py` | Script to regenerate figures from MMC project |
| `build.sh` | Convenience: `pdflatex → bibtex → pdflatex × 2` |

## Build the paper PDF locally

```bash
cd artigos/02_spec_2026
./build.sh                     # ≈ 3 seconds on macOS
# Open main.pdf
```

The script runs the standard 4-pass LaTeX dance:
`pdflatex → bibtex → pdflatex → pdflatex` to resolve forward
references + bibliography correctly.

## Submission checklist

After draft is polished:

- [ ] All 6 sections written + reviewed
- [ ] Word count within target (~3500 ± 200)
- [ ] All references cited in text (no orphan bib entries)
- [ ] All figures included, captioned, referenced (`\autoref{fig:...}`)
- [ ] PDF compiles with no LaTeX warnings beyond cosmetic
- [ ] PDF passes IEEE PDF eXpress validation (after SPEC sends the
      link, usually with author kit)
- [ ] Author contact info + bio (1-2 lines)
- [ ] Choose up to 3 topic codes when submitting via ConfTool
- [ ] Lecture/dialogue preference declared

## Effort estimate

| Phase | Time |
|---|---|
| Skeleton + outlines (✅ done) | 1 evening |
| Sections I–II first draft | 1 weekend |
| Sections III–IV first draft | 1 weekend |
| Section V (results + figures) first draft | 1 weekend |
| Polish + word-count trim + reference completion | 1 weekend |
| Internal review + PDF eXpress validation | 1 evening |

**Realistic finish: end of June 2026.** Leaves a 1-month buffer
before the 31 Jul deadline.
