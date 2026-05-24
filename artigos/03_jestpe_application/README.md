# IEEE JESTPE Application Paper — MMC Capacitor-Balancing Benchmark

**Venue:** [IEEE Journal of Emerging and Selected Topics in Power Electronics](https://www.ieee-pels.org/publications/journal-of-emerging-and-selected-topics-in-power-electronics/) (JESTPE)
**Working title:** *An Open-Source Benchmark of Capacitor-Balancing
Strategies for Modular Multilevel Converters: From N = 3 to N = 20*
**Target submit:** Dec 2026 / Jan 2027
**Impact Factor:** 6.3 (2024)
**Cost:** **$0** — subscription route (no APC). Accepted manuscript
on arXiv per IEEE Author Posting Policy.
**Status:** 🧭 Concept stage

## Why this paper

The TPEL paper (paper #2) sells Pulsim as a *simulator* — it's the
methods paper that establishes the algorithmic contribution.

This JESTPE paper is the **application paper** — Pulsim is the
*instrument*, the substantive contribution is in power-electronics
research itself. The story: capacitor balancing in modular
multilevel converters (MMC) is one of the most-studied control
problems in PE, but the published landscape is fragmented:

- Different authors use different N (sub-module count per arm)
- Different testbenches (PSIM, Simulink, custom C, hardware)
- Rarely-reproducible benchmark setups
- Different metrics (max-min cap deviation, circulating-current
  THD, switching loss)

We use Pulsim to run **a single, reproducible benchmark suite**
that:
1. Implements four canonical balancing strategies — naive
   sort-and-select, hysteresis-bounded sort-and-select, rotating
   sort-and-select, second-order circulating-current suppression
2. Sweeps **N = 3, 5, 10, 15, 20** sub-modules per arm
3. Reports cap voltage deviation, circulating-current THD,
   switching frequency per sub-module, and CPU wall-time
4. Cross-checks against published HVDC field/lab data where
   available (e.g. Trans Bay Cable HVDC station, the
   widely-cited Marquardt/Lesnicar reference)

The dramatic open-loop drift vs sort-and-select comparison already
captured in `projects/inverters/mmc/00_mmc_pulsim_validation.ipynb`
is the seed figure.

## Why JESTPE (not a second TPEL submission)

- **Different reviewer pool** → independent verdicts. Reviewers who
  reviewed our TPEL methods paper are unlikely to re-review for
  JESTPE.
- **JESTPE explicitly welcomes "emerging topic" papers** and
  tool-driven application studies.
- **Avoids the appearance** of self-citing TPEL twice in the same
  calendar year.
- **Still PELS, still IF > 6** — high credibility, indexed in
  Scopus + IEEE Xplore + Web of Science.

## Section outline (target ~7,000 words, ~12 journal pages)

| # | Section | Words | Status |
|:-:|---|---:|:-:|
| | Abstract | 250 | ⬜ |
| I | Introduction + literature review | 1,000 | ⬜ |
| II | MMC model + the four balancing strategies | 1,200 | ⬜ |
| III | Benchmark protocol (N sweep, metrics, simulator) | 800 | ⬜ |
| IV | Results: cap deviation + THD + switching freq | 1,800 | ⬜ |
| V | Comparison with published HVDC data | 800 | ⬜ |
| VI | Discussion: which strategy for which application | 700 | ⬜ |
| VII | Conclusion + reproducibility statement | 250 | ⬜ |
| | References (35–50 entries) | — | ⬜ |

## Pre-writing work

| Task | Weeks |
|---|---|
| Extend `mmc_model.py` to parametric N (currently hard-coded N = 3) | 1 |
| Implement 4 balancing controllers (3 are sort-and-select variants) | 1.5 |
| Define metric extraction: cap deviation, THD, fsw per SM, CPU time | 0.5 |
| Run full N sweep × 4 controllers (20 runs total) | 1 |
| Identify published HVDC data point (Trans Bay Cable or BorWin1) | 0.5 |
| Generate figures + tables | 1 |
| Write paper (sections I–VII) | 3 |
| Internal review + revision pass | 1 |
| **Total** | **~9–10 weeks** |

Realistic schedule: starts after TPEL submission (Oct 2026),
finishes Dec 2026 / Jan 2027. Aligns with the goal of having two
IEEE Transactions submissions in review by Q1 2027.

## Files in this folder (TBD)

```
03_jestpe_application/
├── README.md                          (you are here)
├── paper.tex                          (LaTeX source, IEEEtran journal mode)
├── paper.bib
├── controllers/
│   ├── naive_sort_select.py
│   ├── hysteresis_sort_select.py
│   ├── rotating_sort_select.py
│   └── second_order_suppression.py
├── runs/
│   ├── sweep_orchestrator.py          (run all 20 configurations)
│   └── results_N{3,5,10,15,20}.csv
└── figures/
    ├── topology_diagram.pdf
    ├── cap_deviation_vs_N.pdf
    ├── thd_vs_N.pdf
    ├── fsw_per_sm_vs_N.pdf
    ├── waveform_overlay_N5.pdf
    └── hvdc_data_comparison.pdf
```

## JESTPE submission specifics

- **Format:** IEEEtran class, `journal` mode, two-column.
- **Length:** typical accepted paper 10–14 pages.
- **Open Access:** *not required.* Hybrid OA available at ~$2,195
  if elected — **we decline**, subscription tier.
- **Reviewers:** typically 3, decision in 3–4 months for first round.
- **Submission portal:** [Manuscript Central — JESTPE](https://mc.manuscriptcentral.com/jestpe-ieee)

## Dependencies (must happen before submission)

- [ ] JOSS paper accepted → Pulsim has a DOI to cite
- [ ] TPEL methods paper *submitted* (doesn't need to be accepted
      — JESTPE can cite "submitted to IEEE TPEL" if needed)
- [ ] Controllers implemented + benchmark suite reproducible

## Post-acceptance workflow

Same as TPEL — see [`../04_arxiv_strategy/WORKFLOW.md`](../04_arxiv_strategy/WORKFLOW.md).

## Risks + mitigations

- **HVDC field-data access**: HVDC operator data is rarely public.
  Mitigation: lean on published papers from CIGRÉ working groups
  + the Lesnicar/Marquardt 2003 reference for synthetic field
  conditions; document the discrepancy frankly.
- **Scope creep on N**: tempting to push N up to 50 (Trans Bay
  Cable). Cap at N = 20 for paper #1 — anything larger is future
  work and can become a follow-up letter.
- **Controller IP**: all four strategies are published in the open
  literature, so no novelty concerns on the *controllers*. The
  novelty is in the *reproducible benchmark*.
