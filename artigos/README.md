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
  02_spec_2026/                   ← IEEE SPEC 2026 conference (Cartagena, Dec)
  03_oj_pel_methods/              ← IEEE Open Journal of Power Electronics
  04_tpel_application/            ← IEEE Transactions on Power Electronics
```

> **Pivot history.** The original plan targeted **EPE-ECCE Europe
> 2026** (Valencia, Sep) for paper #2, but the extended-digest
> deadline (9 Mar 2026) had already passed when work began. EPE-ECCE
> requires the 2-stage digest-then-full-paper process; without an
> accepted digest, the 8 Jun full-paper deadline is unavailable.
> Re-targeted to **IEEE SPEC 2026** (Southern Power Electronics
> Conference, Cartagena, Colombia, 8-11 Dec; submission deadline
> 31 Jul 2026 = 69 days at the time of pivot) — same IEEEtran
> conference template, same MMC topic, comfortable timeline.

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
| 1 | **JOSS** | Tool paper | n/a (CiteScore ~3.2) | 7 Jun 2026 | **$0** |
| 2 | **IEEE SPEC 2026** | Conference (case study) | — | **31 Jul 2026 deadline** | conf fee |
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

### 2. IEEE SPEC 2026 — conference paper (case study)

**Folder:** [`02_spec_2026/`](02_spec_2026/)

The IEEE Southern Power Electronics Conference (SPEC) is the
PELS-flagship conference for Latin America and the southern
hemisphere. **2026 edition: Cartagena, Colombia, 8–11 December.
Submission deadline: 31 July 2026** — comfortably open. Same
IEEEtran conference template as EPE-ECCE / APEC / ECCE, so the
LaTeX work transfers across any future PELS conference target.

**Topic chosen:** *Sort-and-Select Capacitor Balancing in a
Single-Phase Modular Multilevel Converter: An Open-Source Reference
Implementation.* This builds directly on the MMC project in
`projects/inverters/mmc/` and exploits the dramatic side-by-side
visualisation of open-loop drift (caps diverging from ±100 V) vs
sort-and-select control (all 6 caps locked within 0.03 V) already
captured in `00_mmc_pulsim_validation.ipynb`.

**Format:** 6 pages, double column, A4, IEEEtran `conference` class.

**Why it matters:** conference presentation = face time with PE
researchers + indexed Scopus citation; SPEC is also the most
realistic in-person venue for a Brazil-based author.

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

## Timeline (calendar, post-EPE-ECCE pivot)

```
   2026                            2027
   May Jun Jul Aug Sep Oct Nov Dec Jan Feb Mar
   │                                       │
   ├ JOSS submission package locked ✅
   │
   │  [7 Jun] JOSS public-history gate opens
   │  ├──► submit JOSS
   │  │
   │  ├ SPEC 2026 paper drafted in parallel
   │  │
   │  │   [31 Jul] SPEC 2026 deadline
   │  │   ├──► submit SPEC
   │  │   │
   │  │   ├ OJ-PEL methods paper benchmark work
   │  │   │ (PSIM/ngspice/Pulsim head-to-head runs)
   │  │   │
   │  │   │  [~Sep] JOSS accept → DOI minted
   │  │   │
   │  │   │  ├──► OJ-PEL draft + submit (~Oct)
   │  │   │  │
   │  │   │  │  [Aug 2026] APEC 2027 digest deadline
   │  │   │  │  ├──► submit APEC digest
   │  │   │  │  │
   │  │   │  │  │  [Dec] SPEC 2026 in Cartagena
   │  │   │  │  │
   │  │   │  │  │  ├──► TPEL/JESTPE app paper drafted (Dec-Feb)
   │  │   │  │  │  │
   │  │   │  │  │  │  [Mar 2027] APEC 2027 conference
   │  │   │  │  │  │
   │  │   │  │  │  └──► submit TPEL (Q1 2027)
```

**Realistic outcome by Mar 2027:** 1 JOSS DOI + 1 SPEC paper
presented + 1 OJ-PEL in review + 1 APEC accepted + 1 TPEL in
writing. Five independent citation sources alive on Google Scholar
within ~10 months of starting.

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
