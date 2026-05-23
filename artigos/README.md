# Pulsim — Publication Plan

Strategic roadmap for the first wave of peer-reviewed publications about
Pulsim. Goal: build **credibility + visibility** as a research-grade
open-source power-electronics simulator.

Each subfolder holds one paper in active preparation, with its own
`paper.md` / LaTeX source, `paper.bib`, figures, supplementary code,
and a paper-specific README tracking submission status.

## Folder layout

```
artigos/
  README.md                       ← this file (the strategic plan)
  shared/                         ← BibTeX + figures + assets shared across papers
  01_joss_tool_paper/             ← Journal of Open Source Software (FIRST)
  02_epe_ecce_europe_2026/        ← EPE-ECCE Europe 2026 conference
  03_oj_pel_methods/              ← IEEE Open Journal of Power Electronics
  04_tpel_application/            ← IEEE Transactions on Power Electronics
```

## Strategic priorities

1. **Establish a canonical citation FAST.** The longer Pulsim circulates
   without a citable paper, the more downstream papers will hand-wave
   the reference. JOSS gets us a DOI in ~12 weeks at $0 cost.
2. **Pair high-throughput venues with high-impact ones.** Submit the
   tool paper (JOSS) and a conference paper (EPE-ECCE Europe 2026) in
   parallel — different audiences, different timelines. Then channel
   the conference feedback into a stronger journal submission.
3. **Don't over-invest in any single venue.** Spread bets across 4
   paper types (tool / conference / methods / application) and venue
   tiers ($0 OA → top-tier IF 8) so a single rejection doesn't stall
   the credibility-building campaign.
4. **Make the simulator itself the supplementary material.** Every
   paper should link to a tagged Pulsim release + reproducibility
   notebooks. JOSS forces this discipline; the others reward it.

## Pipeline summary

| # | Venue | Paper type | IF | Target submit | Cost |
|:-:|---|---|---:|---|---|
| 1 | **JOSS** | Tool paper | n/a (CiteScore ~3.2) | Jun 2026 | **$0** |
| 2 | **EPE-ECCE Europe 2026** | Conference (case study) | — | 8 Jun 2026 deadline | conf fee |
| 3 | **IEEE OJ-PEL** | Methods (PWL cache) | 5.4 | Sep–Oct 2026 | $2.2k OA |
| 4 | **IEEE TPEL** or **JESTPE** | Application (MMC benchmark) | 8.2 / 6.3 | Dec 2026 | OA optional |

## Per-paper plans

### 1. JOSS — tool paper (in active prep)

**Folder:** [`01_joss_tool_paper/`](01_joss_tool_paper/)

The first and most important paper. JOSS specialises in publishing
research software (open peer review on GitHub, ~12-week timeline,
Diamond OA = $0 to authors and readers). The paper itself is short
(250–1000 words) — the *software* is the substantive artefact, and the
review process audits the repo (CI, tests, docs, license, statement of
need) more than the paper text.

**Why JOSS first:**
- Zero cost → no risk
- Open peer review hardens the repo (reviewers file GitHub issues)
- Fast (12 weeks median) → DOI in hand before any other paper publishes
- Citable by everyone (other papers cite Pulsim → JOSS DOI)

**What we need to do before submitting:**
- [ ] CI green on Linux + macOS (already mostly there)
- [ ] `CITATION.cff` at repo root
- [ ] `README.md` install section tested in a clean container
- [ ] `paper.md` finalised (in `01_joss_tool_paper/`)
- [ ] `paper.bib` with all referenced prior work
- [ ] Tag a Pulsim release (e.g. `v1.1.0`) that matches the paper

**Target submit:** Jun 2026 — leaves headroom before EPE-ECCE Europe
full-paper deadline (also 8 Jun).

---

### 2. EPE-ECCE Europe 2026 — conference paper (case study)

**Folder:** [`02_epe_ecce_europe_2026/`](02_epe_ecce_europe_2026/)

EPE-ECCE Europe is the major European power-electronics conference;
the 2026 edition is in Valencia (14–18 Sep). **Full-paper deadline:
8 June 2026** — currently open.

**Strategy:** don't make this a tool paper (conferences reject
"here's our tool" papers). Make it a **case study** using Pulsim to
analyse one of the multilevel topologies in the project library — best
candidate is the **NPC 3-level vs 2-level VSI** comparison or the
**MMC capacitor balancing** study. The paper showcases the *use* of
Pulsim while Pulsim itself is cited (JOSS DOI) as the tool.

**Format:** 6 pages, double column.

**Why it matters:** conference presentation = face time with PE
researchers + an indexed Scopus citation + the deadline forces us to
ship the JOSS paper first (synergy, not competition).

---

### 3. IEEE OJ-PEL — methods paper (PWL state-space cache)

**Folder:** [`03_oj_pel_methods/`](03_oj_pel_methods/)

The IEEE Open Journal of Power Electronics is **gold OA**, newer than
TPEL (~5.4 IF in 2024), and explicitly welcomes novel simulation
methodologies validated against real hardware or other simulators.
Target: a deep paper on Pulsim's PWL (piecewise-linear) state-space
cache — the algorithmic contribution that gives Pulsim its
performance edge vs general-purpose SPICE-likes.

**Structure (draft):**
1. State-of-the-art in switched-mode power-electronics simulation
   (SPICE family, PSIM, PLECS, SimPowerSystems, ngspice, gEDA)
2. PWL state-space cache: derivation, enumeration strategy,
   numerical conditioning
3. Benchmark suite: simulation wall-time vs PSIM and ngspice on the
   10 reference converters (buck → MMC)
4. Discussion of cache-build cost vs run-time savings
5. Reproducibility: link to tagged Pulsim release + benchmark notebooks

**APC:** $2,160 (PELS members get 20% off → ~$1,728).

**Target submit:** Sep–Oct 2026.

---

### 4. IEEE TPEL / JESTPE — application paper (MMC benchmark)

**Folder:** [`04_tpel_application/`](04_tpel_application/)

A "full" application paper using Pulsim as the research instrument.
Two candidate angles:

* **TPEL (IF 8.16):** systematic study of MMC capacitor-balancing
  algorithms (sort-and-select variants, second-order suppression, etc.)
  across N = 3 to N = 20, comparing against published HVDC field data.
  Pulsim is the enabling tool; the paper is "real" PE research.
* **JESTPE (IF 6.26):** a multi-topology benchmark (NPC vs MMC vs T-type
  at the same power level), with THD/efficiency/cost comparisons. More
  "emerging topics" flavour.

**Target submit:** Dec 2026.

---

## Timeline (calendar)

```
   2026                        2027
   May  Jun  Jul  Aug  Sep  Oct  Nov  Dec  Jan  Feb  Mar
   │                                                  │
   ├─ CI hardening + CITATION.cff
   │
   ├─ JOSS draft ───► submit (early Jun)
   │
   ├──── EPE-ECCE Europe full paper (deadline 8 Jun)
   │
   │    ├─── PSIM/SPICE benchmark runs (Jul–Aug)
   │    │
   │    ├─── JOSS reviews + iteration (Jul–Aug)
   │    │
   │    │    ├─── JOSS accept → DOI minted (Sep)
   │    │    │
   │    │    ├─── EPE-ECCE Europe in Valencia (14–18 Sep)
   │    │    │
   │    │    │    ├─── OJ-PEL methods paper drafted (Sep–Oct)
   │    │    │    │
   │    │    │    │    ├── OJ-PEL submit ───► review (Nov 2026 – Apr 2027)
   │    │    │    │    │
   │    │    │    │    │    ├── TPEL application paper drafted (Oct–Dec)
   │    │    │    │    │    │
   │    │    │    │    │    │    ├── TPEL submit (Dec) ───► review (Jan–Jun 2027)
   │    │    │    │    │    │    │
   │    │    │    │    │    │    │    ├── APEC 2027 digest (~Aug 2026)
   │    │    │    │    │    │    │    │
   │    │    │    │    │    │    │    └─ ECCE 2027 digest (~Jan 2027)
```

**Realistic outcome by Dec 2026:** 1 JOSS DOI + 1 conference paper
published + 1 OJ-PEL in review + 1 TPEL in writing. Four citation
sources alive on Google Scholar.

## Tracking

Each subfolder has a `README.md` tracking that paper's:
- Current status (drafting / pre-submission / in review / accepted)
- Action items
- Submission date + venue link
- Reviewer comments (when applicable)

## Bibliography conventions

Shared BibTeX entries live in [`shared/refs.bib`](shared/refs.bib) so
all four papers cite the same canonical sources (Marquardt MMC 2003,
Nabae NPC 1981, Holmes & Lipo PWM textbook, Erickson & Maksimović,
etc.). Per-paper-only references go in each subfolder's `paper.bib`.

## Useful links

- [JOSS submission page](https://joss.theoj.org/papers/new)
- [JOSS author guidelines](https://joss.readthedocs.io/en/latest/submitting.html)
- [JOSS example papers](https://joss.theoj.org/papers/published) — read
  10 of these before writing ours.
- [EPE-ECCE Europe 2026 CFP](https://www.ecce-europe.org/2026/authorsreviewers/call-for-papers/)
- [IEEE OJ-PEL](https://www.ieee-pels.org/publications/open-journal-of-power-electronics/)
- [IEEE TPEL submission portal](https://mc.manuscriptcentral.com/tpel-ieee)
