# TPEL Methods Paper — Section I (Introduction) Outline

**Working title:** *A Piecewise-Linear State-Space Cache for Fast
Simulation of Switched-Mode Power-Electronic Circuits*
**Target length:** ~800 words (8 paragraphs at ~100 words each)
**Goal of §I:** Hook → motivate the problem → state the contribution
→ summarise the paper structure. Reviewers stop reading at the end
of §I if it doesn't land; treat it as the single most important
section.

## Paragraph-by-paragraph plan

### ¶1 — The hook (~90 words)

**Move:** Stat-driven opening. The state of switched-mode PE
simulation in 2026: every PhD student, EV powertrain team, and grid
researcher reaches for the same closed-source tools (PLECS, PSIM,
Simscape, Saber) or for general-purpose SPICE that wasn't designed
for switching. Closed-source means: no audit, no modification, no
distribution of reproducibility artefacts in papers.

**Citations:**
- @plecs_2024 (PLECS, Plexim)
- @psim_2024 (PSIM, Powersim — now Altair)
- @mathworks_simscape (Simscape Electrical, MathWorks)
- @saber_2024 (Synopsys Saber)

**Transition:** "This forces a choice: speed *or* reproducibility."

### ¶2 — Why SPICE isn't the answer (~100 words)

**Move:** Set up the foil. SPICE-family solvers (ngspice, Xyce,
LTspice) are open-source and freely redistributable, but were
architected in 1973 for op-amp + transistor amplifier design. Their
per-step Newton iteration on a single large MNA system pays a
~5–50× wall-time penalty on switching converters: every switching
edge triggers a full Jacobian refactorisation, and the trapezoidal
companion model exhibits well-documented numerical ringing on hard
switching transitions.

**Citations:**
- @nagel1973_spice (Nagel SPICE thesis)
- @ngspice_manual (ngspice user manual)
- @keiter2022_xyce (Xyce)
- @demarco1997_trapezoidal_ringing (trapezoidal companion ringing
  in power-electronic simulation)

### ¶3 — The PWL alternative (~100 words)

**Move:** Introduce the contrast. The closed-source tools that
*do* work for PE — PLECS in particular — exploit a structural
observation: between switching events, a power-electronic circuit
is linear and time-invariant. The system matrix changes only at
switching events, so the per-step solve reduces to a matrix-vector
product against a *pre-computed* exponential. This is the
"piecewise-linear state-space" approach. It has been part of the
PE simulation literature since the late 1970s (Wong & Owen) but
never given an open, citable, peer-reviewed open-source reference
implementation.

**Citations:**
- @wong1979_pwl_simulation (Wong & Owen, PWL state-space)
- @maksimovic_2001_state_space_avg (state-space averaging vs PWL)
- @plecs_simulation_methods_2018 (Plexim white paper)
- @schweizer2013_simulation_review (review of PE sim methods)

### ¶4 — The gap this paper fills (~100 words)

**Move:** Sharpen the problem statement. The literature describes
the *idea* of PWL state-space simulation but does not provide a
reproducible artefact:
1. The cache enumeration strategy is rarely documented in detail
   (how do you handle $2^N$ switch combinations when $N > 10$?)
2. The numerical conditioning of degenerate switch combinations
   (all-off floating nodes, switch-short collapsing rank) is
   typically treated as proprietary IP.
3. There is no published wall-time benchmark against open-source
   SPICE on a shared, reproducible reference suite of converters.

**Citations:**
- @bartoszewicz_2019_sim_review (recent review confirms gap)
- @plecs_simulation_methods_2018 (closed-source — gap as data)

### ¶5 — Contributions (~110 words, bulleted)

**Move:** The standard TPEL Introduction format — explicit
enumerated contributions. Five items:

1. **Algorithmic.** A complete description of Pulsim's PWL
   state-space cache: lazy enumeration on first encounter,
   numerical conditioning patterns (anchor resistors,
   stiff-source substitution), and a fast event-detection coupling
   that avoids the cache thrashing common to naive PWL caches.
2. **Open-source artefact.** A peer-reviewed reference
   implementation (Pulsim, MIT-licensed, cited via JOSS DOI) that
   readers can install with `pip install pulsim` and re-run every
   benchmark in §VI in under 5 minutes.
3. **Benchmark suite.** Ten reference converters
   (buck, boost, buck-boost, forward, flyback, half-bridge LLC,
   boost PFC, 3-φ VSI, NPC 3-level, MMC N = 3) reproducibly
   simulated in both Pulsim and ngspice, with wall-time + memory
   + accuracy data published as machine-readable CSV.
4. **Limits analysis.** Honest documentation of the regime where
   PWL caching does *not* pay off (low switch count, see §VII).
5. **Future-work substrate.** A foundation other PE simulation
   researchers can extend — adding their own integrators, switch
   models, or hardware-in-the-loop bindings.

### ¶6 — Related work (~90 words)

**Move:** Survey by paragraph rather than table (the full table
goes in §II). Three buckets:
- **Commercial PWL solvers** (PLECS, PSIM, Simscape Electrical) —
  closed, but established the approach.
- **Open-source SPICE family** (ngspice, Xyce, LTspice) —
  general-purpose; not specialised for switching.
- **Academic codebases** (PSEUDO, SimSCAPE-academic forks,
  one-off MATLAB scripts) — usually unmaintained, single-paper
  artefacts.

Pulsim is the first MIT-licensed, peer-reviewed, actively-maintained
PWL solver targeting the PE community.

**Citations:**
- @plecs_2024, @psim_2024, @ngspice_manual, @keiter2022_xyce,
  @pseudo_2015 (representative academic codebase)

### ¶7 — The independent-researcher angle (~80 words)

**Move:** A short, honest paragraph on context. This work is
funded out-of-pocket by an independent researcher. That constraint
shaped two design choices: (a) every dependency must be free for
both authors and readers; (b) every result must be reproducible
without specialised hardware. Both constraints align Pulsim with
the JOSS / openness movement in PE simulation — see also
@katz_2020_open_software_research_software.

### ¶8 — Roadmap (~80 words)

**Move:** Section guide. "The remainder of this paper is
organised as follows. Section II surveys the existing landscape
of switched-mode PE simulators. Section III derives the PWL
state-space cache architecture. Section IV addresses the
numerical conditioning of degenerate switch combinations. Section
V describes the benchmark protocol; Section VI presents the
results. Section VII discusses when PWL caching pays off and
when it does not — informing future-work directions outlined in
Section VIII."

## Citation budget for §I alone

~12 unique bib entries — they go in `paper.bib`:

| Key | Source | Used in ¶ |
|---|---|:-:|
| `plecs_2024` | Plexim PLECS user manual / website, 2024 | 1, 6 |
| `psim_2024` | Altair PSIM (formerly Powersim) reference, 2024 | 1, 6 |
| `mathworks_simscape` | Simscape Electrical reference | 1 |
| `saber_2024` | Synopsys Saber reference | 1 |
| `nagel1973_spice` | L. W. Nagel, "SPICE2: A computer program …," 1975 ERL Memo | 2 |
| `ngspice_manual` | ngspice user manual v45 | 2, 6 |
| `keiter2022_xyce` | Keiter et al., Xyce overview | 2, 6 |
| `demarco1997_trapezoidal_ringing` | classic ngspice trap-rule ringing analysis | 2 |
| `wong1979_pwl_simulation` | Wong & Owen, PWL state-space for PE | 3 |
| `maksimovic_2001_state_space_avg` | state-space averaging textbook chapter | 3 |
| `plecs_simulation_methods_2018` | Plexim PWL white paper | 3, 4 |
| `schweizer2013_simulation_review` | review of PE simulation methods | 3 |
| `bartoszewicz_2019_sim_review` | newer simulation-methods review | 4 |
| `pseudo_2015` | representative academic open-source codebase | 6 |
| `katz_2020_open_software_research_software` | Katz et al. on RSE/JOSS movement | 7 |

(These overlap heavily with the JOSS paper's `paper.bib` — most
can be copied over.)

## Writing tactics for §I

- **One graph allowed.** §I can support at most one figure;
  candidate is a single bar chart of wall-time × switch-count
  showing both simulators across the 10 reference converters
  (also goes in §VI as the headline result, repeated in §I as a
  teaser). Do NOT duplicate the figure caption — reference it as
  "see Fig. 1 in §VI."
- **Past tense for prior work, present tense for the paper's
  own contributions.** TPEL house style.
- **No "we" until ¶5 (Contributions).** Active voice elsewhere,
  but keep the subject technical ("Pulsim caches …") not
  authorial.
- **Bound the scope.** Explicitly say "this paper considers
  ideal-switch + linear-passive models; nonlinear magnetics and
  detailed loss models are deferred to follow-up work." This
  pre-empts reviewer 2's "but what about saturation?" objection.

## When to write §I

Per the timeline in `../README.md`:

1. **First:** finish the benchmark suite for at least 4 of the 10
   converters (buck ✅, plus boost, half-bridge, NPC) so §I can
   cite real numbers in ¶5 rather than "expected ratios."
2. **Then:** draft §III (cache architecture) — this is the
   technical core that §I summarises.
3. **Finally:** write §I as the *last* prose draft — that's the
   only way to make ¶5 honest about what the paper actually
   delivers.

## Open questions for the user

- Should we include the "independent researcher" paragraph (¶7)?
  Pro: differentiates the paper, signals openness.
  Con: TPEL house style is impersonal — could be seen as
  out-of-place. Default: include it; cut if the first internal
  review flags it as out-of-tone.
- Single-author or invite a co-author for credibility?
  TPEL accepts single-author papers; impact-factor not affected.
