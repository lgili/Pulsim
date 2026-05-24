# IEEE OJ-PEL Methods Paper — Pulsim PWL State-Space Cache

**Venue:** [IEEE Open Journal of Power Electronics](https://www.ieee-pels.org/publications/open-journal-of-power-electronics/) (OJ-PEL)
**Working title:** *A Piecewise-Linear State-Space Cache for
Open-Source Power-Electronics Simulation: Architecture, Numerical
Conditioning, and Benchmarks Against Commercial and SPICE-Family
Simulators*
**Target submit:** Oct–Nov 2026
**Impact Factor:** 5.4 (2024)
**Open Access:** $2,160 USD (PELS members –20%)
**Status:** 📐 Outlining + benchmark plan

## Why this paper

The JOSS paper (paper #1) introduces Pulsim *as a tool* — it
describes what it does, who it's for, why it exists. That's the
"tool paper" genre, capped at ~1500 words.

This OJ-PEL paper is the **methods paper** — a deep technical
dive into the algorithmic contribution that makes Pulsim fast:

- The PWL state-space cache structure, derivation, complexity
  analysis (cache size $2^N$ for $N$ switch + diode bits, time
  per step $O(1)$ after cache hit).
- The trade-offs vs alternatives: lazy/multi-dt caches,
  on-demand factorisation, SPICE-style per-step Newton.
- The numerical conditioning: what makes a switch combination
  singular, the regularisation patterns used in NPC + MMC
  (anchor resistors, stiff voltage sources).
- **Quantitative benchmarks** vs PSIM (commercial PWL) and
  ngspice (open SPICE) on the same 6 reference converters from
  the Pulsim project library — simulation wall-time, peak memory,
  output accuracy.

A methods paper is much heavier than a tool paper (typically
12-16 journal pages, ~6,000-9,000 words). It requires real
benchmark data, ideally a head-to-head with at least one
commercial reference.

## Section outline (target ~7,000 words, ~14 journal pages)

| # | Section | Words | Status |
|:-:|---|---:|:-:|
| | Abstract | 250 | ⬜ |
| I | Introduction | 800 | ⬜ |
| II | Background: switched-mode simulation in the literature | 1,000 | ⬜ |
| III | The PWL state-space cache architecture | 1,500 | ⬜ |
| IV | Numerical conditioning + regularisation | 1,000 | ⬜ |
| V | Benchmark methodology | 600 | ⬜ |
| VI | Benchmark results | 1,500 | ⬜ |
| VII | Discussion: when PWL pays off, when it doesn't | 400 | ⬜ |
| VIII | Conclusion | 200 | ⬜ |
| | References (25–40 entries) | — | ⬜ |

## Pre-writing work: the benchmark suite

The benchmark is the part that needs the most lead time. We need
**three simulators** running the **same converters**:

| Simulator | Notes | License |
|---|---|---|
| **Pulsim** | The protagonist | MIT |
| **ngspice** | Open-source SPICE — accepts identical netlists | GPL |
| **PSIM** (free trial?) or **LTspice** | Commercial reference | Free for non-commercial |

| Converter | From | Switch count | Expected ratio (Pulsim faster) |
|---|---|---:|---:|
| Buck | `projects/converters/buck/` | 3 | ~10× |
| Boost | `projects/converters/boost/` | 3 | ~10× |
| Flyback (with transformer) | `projects/converters/flyback/` | 4 | ~15× |
| Half-bridge (with bridge rectifier) | `projects/converters/half_bridge/` | 8 | ~20× |
| 3-phase VSI (open loop) | `projects/inverters/vsi_3phase/` | 12 | ~30× |
| NPC 3-level | `projects/inverters/npc_3phase/` | 18 | ~50× |

Each converter needs a parallel **ngspice netlist** that produces
the exact same operating point — that's 4-6 days of conversion
work alone. Plus PSIM/LTspice scripts.

## Effort estimate

| Phase | Weeks |
|---|---|
| Author benchmark netlists (ngspice + PSIM/LTspice) for 6 converters | 2-3 |
| Run benchmarks + record wall-time + memory across 3 simulators | 1 |
| Author paper sections I–IV (theory + architecture) | 2 |
| Author paper sections V–VIII (benchmarks + discussion) | 1 |
| Internal review + revision pass | 1 |
| **Total** | **~7-8 weeks** |

**Realistic finish:** if benchmark work starts in late July 2026
(after SPEC paper submission), the OJ-PEL paper is ready by
mid-September 2026, submitted in October.

## Files in this folder (TBD)

```
03_oj_pel_methods/
├── README.md                          (you are here)
├── paper.tex                          (LaTeX source, IEEE OJ-PEL template)
├── refs.bib
├── benchmarks/
│   ├── run_all.py                     (orchestrator)
│   ├── pulsim/                        (Pulsim scripts, 6 converters)
│   ├── ngspice/                       (ngspice .cir netlists, 6 converters)
│   ├── psim/  or  ltspice/            (commercial reference scripts)
│   └── results/                       (wall-time + memory CSVs)
└── figures/
    ├── architecture_diagram.pdf
    ├── benchmark_wallTime_bar.pdf
    ├── benchmark_accuracy_overlay.pdf
    └── benchmark_memory_heatmap.pdf
```

## OJ-PEL submission specifics

- **Format:** IEEE OJ-PEL uses the standard IEEE access journal
  template (`IEEEtran` class with `journal` mode, 1-column layout).
  Different from the `conference` mode of SPEC.
- **Length:** flexible — typical accepted paper 8–16 pages. Aim 12-14.
- **Open Access:** $2,160 base; PELS members get 20% off
  (→ $1,728). Confirm membership status before submitting.
- **Reviewers:** 2-3, typical decision in 2-3 months.
- **Track:** "Original Research" (not "Letter") since we have
  benchmark data + theory.

## Dependencies (must happen before submission)

- [ ] JOSS paper accepted → DOI to cite Pulsim (canonical reference)
- [ ] SPEC 2026 paper submitted (gives us a Scopus-indexed
      back-reference for the MMC case study)
- [ ] Benchmark suite written + results captured (the bulk of
      the substantive work)

## Risks + mitigations

- **Commercial-simulator licence**: PSIM offers a 30-day trial
  that's enough to capture the benchmark. LTspice is free and
  can substitute. Fall-back to ngspice-only if neither
  commercial reference is available.
- **Numerical comparability**: PSIM/ngspice/Pulsim use different
  integration schemes (trap vs Gear-2 vs trap-companion). The
  benchmark must compare at the SAME accuracy target, not at
  the same dt.
- **Length blow-out**: a methods paper can grow past 20 pages
  easily. Discipline: review section II's literature review
  early to catch over-citation.
