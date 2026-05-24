# IEEE TPEL Methods Paper — Pulsim PWL State-Space Cache

**Venue:** [IEEE Transactions on Power Electronics](https://www.ieee-pels.org/publications/transactions-on-power-electronics/) (TPEL)
**Working title:** *A Piecewise-Linear State-Space Cache for Fast
Simulation of Switched-Mode Power-Electronic Circuits: Architecture,
Numerical Conditioning, and Benchmarks Against SPICE*
**Target submit:** Oct 2026
**Impact Factor:** 8.2 (2024)
**Cost:** **$0** — subscription route (no APC, no page fees, no
colour fees). Accepted manuscript is self-archived on arXiv per
IEEE's Author Posting Policy.
**Status:** 📐 Outlining + benchmark plan

## Why this paper

The JOSS paper (paper #1) introduces Pulsim *as a tool* — it
describes what it does, who it's for, why it exists. That's the
"tool paper" genre, capped at ~1000 words.

This TPEL paper is the **methods paper** — a deep technical dive
into the algorithmic contribution that makes Pulsim fast:

- The PWL state-space cache structure, derivation, complexity
  analysis (cache size $2^N$ for $N$ switch + diode bits, time
  per step $O(1)$ after cache hit).
- The trade-offs vs alternatives: lazy/multi-dt caches, on-demand
  factorisation, SPICE-style per-step Newton.
- The numerical conditioning: what makes a switch combination
  singular, the regularisation patterns used in NPC + MMC (anchor
  resistors, stiff voltage sources).
- **Quantitative benchmarks** vs ngspice (open-source SPICE) on the
  same 10 reference converters from the Pulsim project library —
  simulation wall-time, peak memory, output accuracy. ngspice is
  used as the open-source reference so the entire benchmark is
  reproducible by any reader without a commercial licence.

A methods paper is much heavier than a tool paper (typically 12–16
journal pages, ~6,000–9,000 words). It requires real benchmark data
that runs on **only-free-software** stacks so reviewers can
re-execute end-to-end.

## Why TPEL (not OJ-PEL)

OJ-PEL was the original target — newer journal, slightly faster
review. **Dropped** because OJ-PEL is mandatory Gold OA (~$2,160
USD APC), which violates our zero-cost constraint.

TPEL's **subscription route** is the right alternative:
- Higher IF (8.2 vs 5.4) — bigger credibility win
- Zero charges to the author at any point in the process
- Wider citation reach (every PELS-affiliated researcher reads TPEL)
- Author retains the right to post the *accepted manuscript* on
  arXiv (not the final IEEE-typeset PDF, but the LaTeX-compiled
  accepted version). See [`../04_arxiv_strategy/`](../04_arxiv_strategy/).

The only "cost" is the IEEE Xplore paywall on the typeset version.
That is fully mitigated by the arXiv companion preprint, which
Google Scholar indexes prominently.

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

The benchmark is the part that needs the most lead time. The
zero-cost constraint also constrains the reference simulators —
we benchmark against **open-source ngspice only**, so reviewers
can reproduce every result on a vanilla Linux box.

| Simulator | Role | License |
|---|---|---|
| **Pulsim** | The protagonist | MIT |
| **ngspice** | Open-source SPICE reference | GPL |

| # | Converter | From project | Switch count | Expected wall-time ratio (Pulsim faster) |
|:-:|---|---|---:|---:|
| 1 | Buck | `projects/converters/buck/` | 1 IGBT + 1 diode = 2 | ~10× |
| 2 | Boost | `projects/converters/boost/` | 1 + 1 = 2 | ~10× |
| 3 | Buck-boost | `projects/converters/buck_boost/` | 1 + 1 = 2 | ~10× |
| 4 | Forward | `projects/converters/forward/` | 1 + 2 = 3 | ~12× |
| 5 | Flyback (with transformer) | `projects/converters/flyback/` | 1 + 1 = 2 | ~12× |
| 6 | Half-bridge LLC | `projects/converters/half_bridge/` | 2 + 4 = 6 | ~20× |
| 7 | Boost PFC | `projects/converters/boost_pfc/` | 1 + 5 = 6 | ~20× |
| 8 | 3-phase VSI | `projects/inverters/vsi_3phase/` | 6 + 6 = 12 | ~30× |
| 9 | NPC 3-level | `projects/inverters/npc_3phase/` | 12 + 18 = 30 | ~50× |
| 10 | MMC (N = 3) | `projects/inverters/mmc/` | 12 + 12 = 24 | ~40× |

Each converter needs a parallel **ngspice netlist** that produces
the exact same operating point — that's 4–6 days of conversion
work alone. The buck case (simplest) is the first deliverable;
see [`benchmarks/buck/`](benchmarks/buck/) (TBD).

## Effort estimate

| Phase | Weeks |
|---|---|
| Author benchmark netlists (ngspice) for 10 converters | 2–3 |
| Run benchmarks + record wall-time + memory | 1 |
| Author paper sections I–IV (theory + architecture) | 2 |
| Author paper sections V–VIII (benchmarks + discussion) | 1 |
| Internal review + revision pass | 1 |
| **Total** | **~7–8 weeks** |

**Realistic finish:** if benchmark work starts in June 2026 (in
parallel with the JOSS review), the TPEL paper is ready by
mid-September 2026, submitted in October.

## Files in this folder (TBD)

```
02_tpel_methods/
├── README.md                          (you are here)
├── paper.tex                          (LaTeX source, IEEEtran journal mode)
├── paper.bib
├── benchmarks/
│   ├── run_all.py                     (orchestrator)
│   ├── pulsim/                        (Pulsim scripts, 10 converters)
│   ├── ngspice/                       (ngspice .cir netlists, 10 converters)
│   └── results/                       (wall-time + memory CSVs)
└── figures/
    ├── architecture_diagram.pdf
    ├── benchmark_wallTime_bar.pdf
    ├── benchmark_accuracy_overlay.pdf
    └── benchmark_memory_heatmap.pdf
```

## TPEL submission specifics

- **Format:** IEEEtran class, `journal` mode, two-column.
- **Length:** typical accepted paper 10–14 pages.
- **Open Access:** *not required.* Hybrid OA available at ~$2,195
  if elected — **we decline**, keeping the paper in the
  subscription tier so the author pays nothing.
- **Reviewers:** typically 3, decision in 3–4 months for first round.
- **Track:** "Regular Paper" (not "Letter") since we have benchmark
  data + theory.
- **Submission portal:** [Manuscript Central — TPEL](https://mc.manuscriptcentral.com/tpel-ieee)

## Dependencies (must happen before submission)

- [ ] JOSS paper accepted → DOI to cite Pulsim (canonical reference)
- [ ] Benchmark suite written + results captured (the bulk of the
      substantive work — see Task #69)
- [ ] Figures extracted from MMC notebook (Task #68) for the
      "complex topology" case study in §VI

## Post-acceptance workflow

Once TPEL accepts the paper:
1. Receive "Decision Letter — Accept" from Manuscript Central
2. Prepare *accepted manuscript* version (LaTeX source as
   submitted, with reviewer revisions, **without** IEEE typesetting)
3. Upload to arXiv (eess.SY category) — see
   [`../04_arxiv_strategy/WORKFLOW.md`](../04_arxiv_strategy/WORKFLOW.md)
4. Cross-link: arXiv preprint header → IEEE Xplore DOI; IEEE
   Author Center profile → arXiv URL.

## Risks + mitigations

- **Numerical comparability**: Pulsim and ngspice use different
  integration schemes (PWL closed-form per cache cell vs SPICE
  trap-companion). The benchmark compares at the **same accuracy
  target**, not the same dt — i.e. tune each simulator's tolerance
  until the output matches to within ε, then measure wall-time.
- **Length blow-out**: a methods paper can grow past 20 pages
  easily. Discipline: review section II's literature review early
  to catch over-citation.
- **Reviewer demands Simulink/PSIM comparison**: pre-emptively
  address in the limitations subsection — explain that the paper
  targets a fully-reproducible benchmark and link to user-reported
  PSIM/Simulink comparisons in the Pulsim repo issues.
