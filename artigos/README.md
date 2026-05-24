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
  _shared/                        ← scripts + figures + bib reused across papers
  01_joss_tool_paper/             ← Journal of Open Source Software (FIRST)
  02_tpel_methods/                ← IEEE TPEL methods paper (subscription route, free)
  03_jestpe_application/          ← IEEE JESTPE application paper (subscription route, free)
  04_arxiv_strategy/              ← arXiv companion-preprint policy + workflow
```

> **Constraint check.** The author has explicit constraints:
> (1) **$0 publishing cost** — no APCs, no page fees;
> (2) **no travel** — no resources for conference attendance / virtual
> registration.
>
> **Pivot history:**
> - Paper #2 was originally **EPE-ECCE Europe 2026** (Valencia, Sep).
>   Missed the 9-Mar-2026 digest deadline → dropped.
> - Paper #2 was then re-targeted to **IEEE SPEC 2026** (Cartagena,
>   Dec). Dropped because conference attendance / virtual
>   registration fees violate constraint (2).
> - Paper #3 was originally **IEEE Open Journal of Power Electronics**
>   (OJ-PEL). Dropped because OJ-PEL is **mandatory Gold OA**
>   (~$2,160 USD) → violates constraint (1).
>
> **Settled pipeline.** All four papers below are **free to publish**
> and **require zero travel**: JOSS is Diamond OA; the two IEEE
> Transactions go through the **subscription route** (no APC; paper
> sits behind IEEE Xplore paywall, but author self-archives the
> accepted manuscript on arXiv per IEEE Author Posting Policy);
> the arXiv companion strategy itself is documented for use across
> the campaign.

## Strategic priorities

1. **Establish a canonical citation FAST.** The longer Pulsim circulates
   without a citable paper, the more downstream papers will hand-wave
   the reference. JOSS gets us a DOI in ~12 weeks at $0 cost.
2. **Stack independent journal submissions.** With conferences off
   the table (travel/registration cost), our high-impact slots are
   two IEEE Transactions in parallel: one *methods* paper (TPEL —
   the PWL cache algorithm) and one *application* paper (JESTPE —
   MMC capacitor-balancing benchmark). Different reviewer pools,
   different angles, both indexed in IEEE Xplore + Scopus.
3. **Always pair a paywalled submission with an arXiv preprint.**
   IEEE's Author Posting Policy permits depositing the accepted
   manuscript on arXiv. This neutralises the only real downside of
   the subscription route (paywall) without paying an APC.
4. **Make the simulator itself the supplementary material.** Every
   paper should link to a tagged Pulsim release + reproducibility
   notebooks. JOSS forces this discipline; the IEEE Transactions
   reward it through "Reproducibility-Enabled Research" badges.

## Pipeline summary

| # | Venue | Paper type | IF | Target submit | Cost | Travel |
|:-:|---|---|---:|---|:-:|:-:|
| 1 | **JOSS** | Tool paper | CiteScore 3.2 | 7 Jun 2026 | **$0** | none |
| 2 | **IEEE TPEL** (subscription) | Methods (PWL cache + benchmark) | 8.2 | Oct 2026 | **$0** | none |
| 3 | **IEEE JESTPE** (subscription) | Application (MMC + balancing) | 6.3 | Dec 2026 | **$0** | none |
| 4 | **arXiv** (companion preprints) | Open preprint workflow | — | continuous | **$0** | none |

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

**Target submit:** 7 Jun 2026 — first day the Pulsim repo clears
JOSS's "≥ 6 months of public history" gate (repo went public on
6 Dec 2025).

---

### 2. IEEE TPEL — methods paper (PWL state-space cache)

**Folder:** [`02_tpel_methods/`](02_tpel_methods/)

The IEEE *Transactions on Power Electronics* (TPEL) is the flagship
PELS journal (IF 8.2 in 2024) and the canonical home for a deep
methods paper on switched-mode simulation. We target the **subscription
route** (no APC, no page fees, no colour fees) and pair it with an
arXiv preprint of the accepted manuscript per IEEE's Author Posting
Policy — readers without IEEE Xplore access still find the paper on
Google Scholar via the arXiv mirror.

**Topic:** *A Piecewise-Linear State-Space Cache for Fast Simulation
of Switched-Mode Power-Electronic Circuits.* The algorithmic
contribution behind Pulsim's performance edge over general-purpose
SPICE-likes.

**Structure (draft):**
1. State-of-the-art in switched-mode PE simulation (SPICE family,
   PSIM, PLECS, SimPowerSystems, ngspice, gEDA)
2. PWL state-space cache: derivation, enumeration strategy, numerical
   conditioning, event-detection coupling
3. Benchmark suite: wall-time + accuracy vs ngspice on the 10
   reference converters in `projects/` (buck → MMC)
4. Discussion of cache-build cost vs run-time savings; sweet spot
   in terms of switch count
5. Reproducibility: link to tagged Pulsim release + benchmark
   notebooks + raw ngspice netlists

**Cost:** $0 (subscription route — IEEE collects no APC; the paper
sits behind the IEEE Xplore paywall, but the accepted manuscript is
self-archived on arXiv).

**Target submit:** Oct 2026 (after JOSS DOI is minted, so the
benchmark notebooks can cite Pulsim's DOI directly).

---

### 3. IEEE JESTPE — application paper (MMC benchmark)

**Folder:** [`03_jestpe_application/`](03_jestpe_application/)

The IEEE *Journal of Emerging and Selected Topics in Power Electronics*
(JESTPE) is the second-tier PELS journal (IF 6.3 in 2024) and welcomes
application-driven studies that use a novel tool. Same subscription
route as TPEL — no APC. Different reviewer pool from TPEL, so the two
submissions don't compete for reviewer attention.

**Topic:** *An Open-Source Benchmark of Capacitor-Balancing Strategies
for Modular Multilevel Converters: From N = 3 to N = 20.* Builds on
the MMC project in `projects/inverters/mmc/` and the side-by-side
open-loop drift vs sort-and-select comparison already captured in
`00_mmc_pulsim_validation.ipynb`. Extends to (a) larger N, (b)
second-order circulating-current suppression, (c) sort-and-select
variants (rotating, hysteresis-bounded), (d) comparison against
published HVDC field data where available.

**Why JESTPE rather than a second TPEL submission:**
- Different reviewer pool → independent verdicts
- JESTPE explicitly welcomes "emerging topic" and tool-driven studies
- Avoids the appearance of self-citing TPEL twice in the same year

**Cost:** $0 (subscription route).

**Target submit:** Dec 2026 / Jan 2027.

---

### 4. arXiv companion-preprint workflow

**Folder:** [`04_arxiv_strategy/`](04_arxiv_strategy/)

Not a paper — this folder documents the **policy and workflow** for
pairing every IEEE Transactions submission with an arXiv preprint.

**Why it's a first-class artefact:**
- IEEE's Author Posting Policy (2024 revision) explicitly permits
  posting the *accepted* manuscript (not the final IEEE-typeset
  version) on arXiv and personal/institutional repositories.
- arXiv preprints accrue Google Scholar citations long before the
  IEEE Xplore version is indexed.
- A preprint with a clear DOI cross-reference back to the journal
  version ensures every citation lands on Pulsim's canonical DOI.

**Contents (planned):**
- `WORKFLOW.md` — step-by-step: submit to TPEL → revise → accept →
  prepare accepted manuscript → submit to arXiv → cross-link
- `IEEE_POSTING_POLICY.md` — verbatim quotes + URL of the relevant
  IEEE policy clauses, dated, so we don't lose them to a website
  refresh
- `arxiv_metadata_template.txt` — pre-filled arXiv metadata (authors,
  affiliation, MSC/PACS codes, eess.SY category) reused across
  submissions

---

## Timeline (calendar, free + no-travel pipeline)

```
   2026                            2027
   May Jun Jul Aug Sep Oct Nov Dec Jan Feb Mar
   │                                       │
   ├ JOSS submission package locked ✅
   │
   │  [7 Jun] JOSS public-history gate opens
   │  ├──► submit JOSS
   │  │
   │  ├ Buck Pulsim-vs-ngspice benchmark (foundation for TPEL)
   │  │
   │  │   [Jul–Sep] Extend benchmark to 10 reference converters
   │  │   │         (boost, buck-boost, flyback, … MMC)
   │  │   │
   │  │   │  [~Sep] JOSS accept → DOI minted
   │  │   │
   │  │   │  ├──► TPEL methods paper drafted (Aug–Sep)
   │  │   │  ├──► submit TPEL (Oct)
   │  │   │  │
   │  │   │  │   [Oct–Nov] JESTPE application paper drafted
   │  │   │  │   ├──► submit JESTPE (Dec/Jan)
   │  │   │  │   │
   │  │   │  │   │   [continuous] arXiv companion preprints
   │  │   │  │   │   posted on each "accept" (per IEEE policy)
   │  │   │  │   │
   │  │   │  │   │   [Q1 2027] TPEL first-round reviews back
   │  │   │  │   │   └──► revise + resubmit
```

**Realistic outcome by Mar 2027:** 1 JOSS DOI minted + 1 TPEL
in first-round revision + 1 JESTPE in initial review + 2 arXiv
preprints accruing Google Scholar citations. Four independent
citation sources alive on Google Scholar within ~10 months of
starting — zero dollars spent, zero trips taken.

## Tracking

Each subfolder has a `README.md` tracking that paper's:
- Current status (drafting / pre-submission / in review / accepted)
- Action items
- Submission date + venue link
- Reviewer comments (when applicable)

## Bibliography conventions

Shared BibTeX entries live in [`_shared/refs.bib`](_shared/refs.bib)
so all papers cite the same canonical sources (Marquardt MMC 2003,
Nabae NPC 1981, Holmes & Lipo PWM textbook, Erickson & Maksimović,
etc.). Per-paper-only references go in each subfolder's `paper.bib`.

## Useful links

**JOSS**
- [JOSS submission page](https://joss.theoj.org/papers/new)
- [JOSS author guidelines](https://joss.readthedocs.io/en/latest/submitting.html)
- [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html#submission-requirements)
- [JOSS example paper](https://joss.readthedocs.io/en/latest/example_paper.html)
- [JOSS published papers](https://joss.theoj.org/papers/published) — read
  10 of these before writing ours.

**IEEE Transactions (subscription route — $0)**
- [IEEE TPEL homepage](https://www.ieee-pels.org/publications/transactions-on-power-electronics/)
- [IEEE TPEL submission portal (Manuscript Central)](https://mc.manuscriptcentral.com/tpel-ieee)
- [IEEE JESTPE homepage](https://www.ieee-pels.org/publications/journal-of-emerging-and-selected-topics-in-power-electronics/)
- [IEEE JESTPE submission portal](https://mc.manuscriptcentral.com/jestpe-ieee)
- [IEEE Author Posting Policy](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/post-publication-policies/)
  — confirms self-archiving accepted manuscript on arXiv is permitted.

**arXiv**
- [arXiv submission portal](https://arxiv.org/submit)
- [arXiv eess.SY category](https://arxiv.org/list/eess.SY/recent) —
  Systems and Control, the right primary category for power-electronics
  simulation work.
