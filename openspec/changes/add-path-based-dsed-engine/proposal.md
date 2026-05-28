# Add Path-Based DSED Engine

**Change ID:** `add-path-based-dsed-engine`
**Status:** 🟡 Proposal (awaiting approval)
**Author:** Luiz Carlos Gili
**Target release:** v2.0.0 (breaking — `engine` selector added; default stays `pwl`)
**Estimated effort:** 5–6 months of focused work, 7 gates with validation criteria each.

---

## Why

The Pulsim PWL state-space cache + path-based partial-refactor framework
(shipped in v1.3.0–v1.4.0) eliminated per-step Jacobian re-factorisation
between mask transitions. The remaining wall-time bottleneck for the
converters Pulsim targets is **fixed-step integration**: the user must
pick a single `dt` that resolves the fastest dynamic in the circuit
(switching edge, parasitic ringing), and every slow phase of the
simulation pays that cost.

For canonical converters this is catastrophically wasteful:

- **LLC resonant** needs $\Delta t \le 10$~ns to capture soft-switching
  transitions, but spends 99% of cycle time in regimes where $\Delta t
  = 1~\mu$s would be plenty.
- **Buck DCM** needs fine $\Delta t$ to catch zero-current-detection
  events for body-diode commutation; current Pulsim users either
  use a very small $\Delta t$ everywhere or risk missing events
  entirely.
- **MMC arm-scale** ($\nstate \ge 100$) integration step cost scales
  with $\nstate$ regardless of dynamics — trapezoidal pays the same
  for fast and slow regions.
- **Inrush / start-up transients** (100~ms+) typically dwarf
  steady-state simulation time even though the dynamics are
  predominantly linear and slow.

The published literature has converged on **discrete-state
event-driven (DSED) simulation** as the right answer to all four
problems. The Tsinghua line (Yan 2019; Wang 2021 backward DSED;
Zhao 2023 hybrid-time; Liu 2025 PI-controlled step) reports
$10\times\text{--}1000\times$ wall-clock speedups over PLECS on
real converters. The commercial spin-off DSIM (Powersys) is built
on this paradigm. **No open-source DSED implementation exists.**

Simultaneously, our v1.3.0–v1.4.0 work on path-based partial
refactorisation (Chan-Brandwajn-Tinney heritage) gives us
$2.7\text{--}2.9\times$ speedup specifically on the
single/multi-bit mask transitions that DSED's event handler
triggers. DSED still needs to solve linear systems between events,
and its event handler still triggers mask transitions —
**both are exactly the workloads our path-based update is optimised
for.**

This change proposes the **first open-source DSED simulator with a
path-based partial-refactor linear-solve kernel**. The combination
is genuinely novel: no published paper combines the two threads.
Captured benchmarks on the ten reference converters bundled with
Pulsim are projected at $5\text{--}15\times$ geometric-mean
speedup over the v1.4.0 fixed-step PWL cache, with no accuracy
regression.

The work is the foundation for a follow-up IEEE TPEL methods paper
(working title *"A Path-Based Discrete-State Event-Driven Simulator
for Switched-Mode Power Electronics"*), targeting Q3 2027
submission after the current v1.4.0 TPEL paper lands Q1 2027.

---

## What Changes

### New capability: `dsed-engine`

A new `pulsim::dsed` namespace + corresponding Python facade,
implementing variable-step, event-driven simulation that consumes
the existing `PwlStateSpaceCache` + `PulsimSparseLuSolverT` as its
linear-solve kernel.

### User-facing API additions

- New `simulate(builder, t_end, *, engine='dsed'|'pwl', ...)`
  selector. Default remains `engine='pwl'` (backward-compatible).
  `engine='dsed'` opts in to the new simulator.
- New parameters for `engine='dsed'`:
  - `dt_max` (default $\Delta t / 10$ of the smallest natural
    time constant) — upper bound for the step
  - `rtol`, `atol` (defaults $10^{-6}$, $10^{-9}$) — error tolerance
  - `event_triggers` — list of user-defined zero-crossing predicates
    (in addition to auto-detected gate edges + diode commutations)
- New `result.event_log` — chronological list of events fired with
  $(t, \text{event\_type}, \text{old\_mask}, \text{new\_mask})$.

### New C++23 modules (header-only, in `core/include/pulsim/dsed/`)

- `event_predictor.hpp` (~300 lines) — Newton-based event
  prediction over a set of armed predicates.
- `adaptive_integrator.hpp` (~400 lines) — RK45 (Dormand-Prince)
  + BDF2 + Krylov-Φ matrix-exponential schemes, behind a unified
  `Step(...)` interface.
- `step_controller.hpp` (~80 lines) — PI-controlled step-size
  adaptation per arXiv 2503.09898.
- `scheduler.hpp` (~500 lines) — outer loop wiring everything to
  the existing `PwlStateSpaceCache` and `PulsimSparseLuSolverT`.

### No changes to existing modules

The existing `PulsimSparseLuSolverT`, `PwlStateSpaceCache`,
trapezoidal-companion assembler, MNA stamper, and Python facade
are pure consumers from the DSED engine's perspective. **Zero
modifications to existing behaviour** on `engine='pwl'`.

### Tests + benchmarks

- New `core/tests/dsed/` test suite (15+ tests on event
  prediction, integrator accuracy, step controller, event
  handling).
- New benchmark category `[dsed][microbench]` in
  `pulsim_benchmarks` that runs the same N-switch chain as
  `[rank1][microbench]` but in event-driven mode.
- Per-converter benchmark suite on the 10 reference projects in
  `projects/`, comparing PWL vs DSED engines on:
  - Wall-clock per simulated second
  - RMSE on output voltage waveform (after a 5-cycle settling
    transient)
  - Pulsim path-based hit rate at events
  - Event count + average step size

### Documentation

- New `docs/how-pulsim-works/11-discrete-state-event-driven.md`
  chapter documenting the DSED engine for the methods-paper
  reader.
- Update `docs/performance-tuning.md` with `engine='dsed'`
  guidance on when to use it.
- Update `docs/quickstart.md` Python example to show DSED opt-in.
- Update `artigos/02_tpel_methods/paper/sections/09_conclusion.tex`
  "Future work" subsection to reference this OpenSpec.

---

## Implementation gates

Implementation proceeds through **7 sequential gates**, each with
explicit validation criteria. Subsequent gates do not start until
the prior gate's criteria pass. See `tasks.md` for the full
checklist per gate.

| Gate | Topic | Duration | Validation Criterion |
|-----:|-------|---------:|----------------------|
|  0   | Literature deep dive + design memo | 1 week | `notes/DSED_FOUNDATIONS.md` reviewed by user; design.md in this OpenSpec extended with concrete formulas |
|  1   | Python prototype (buck CCM) | 3 weeks | RMSE ≤ 0.1 % vs v1.4.0 baseline on buck steady-state; wall-clock ≤ 2× v1.4.0 (no regression target on first attempt) |
|  2   | C++23 port + path-based LU integration | 4 weeks | All v1.4.0 unit tests still pass; new `[dsed][unit]` tests pass; `[dsed][microbench]` ≥ 1× v1.4.0 on Gray-code chain |
|  3   | DCM + body-diode commutation handling | 3 weeks | Buck DCM simulation matches PSIM reference within 1 % output ripple; ≥ 5× v1.4.0 wall-clock on the same simulation |
|  4   | Stiffness detection + BDF backward DSED | 3 weeks | LLC resonant simulation completes in ≤ 50 % v1.4.0 wall-clock; geometric-mean speedup ≥ 3× across 5 converters (buck, boost, half-bridge, NPC, MMC N=3) |
|  5   | Per-converter benchmark suite | 2 weeks | All 10 reference converters validated; CSV captured; geometric-mean speedup ≥ 5× across the 10 |
|  6   | Paper draft + v2.0.0 release | 4 weeks | Draft TPEL paper #2 in `artigos/04_dsed_methods/paper/`; v2.0.0 tagged; migration guide written |

**Total: ~5–6 months.** Each gate's validation criterion must pass
before the next gate starts. The user is the approver at each gate.

---

## Why this is genuinely novel

| Existing work | What it has | What it lacks |
|---|---|---|
| Tsinghua DSED line (2019–2025) | Algorithm published; ~7 papers | Closed implementation; no path-based LU integration |
| DSIM (Powersys, commercial) | DSED + own solver | Closed; expensive; no open-source baseline to cite |
| PLECS / PSIM / Simscape | Variable-step solvers (closed) | No DSED specifically; no documented partial refactor |
| Pulsim v1.4.0 (today) | Path-based partial refactor (open) | Fixed-dt; no event-driven mode |

Pulsim + DSED + path-based partial refactor is the **first
MIT-licensed open-source implementation of the DSED paradigm**
combined with a **first-of-its-kind integration of path-based
partial refactor as the event-handler's linear-solve kernel**.

---

## Out of scope

- Multi-rate co-simulation (FMI / FMU) — separate OpenSpec.
- GPU/FPGA back-end — explicitly not pursued (see survey
  conclusion in the v1.4.0 TPEL paper's §VIII.D placeholder).
- Auto-regularisation of degenerate switch combinations
  (`add-auto-regularization`, Phase 2) — tracked separately.
- ML-based event predictor — out of scope for v2.0.0; could be a
  v3.0.0 follow-up.

---

## Risks + mitigations

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| Event-prediction Newton root-finding expensive per step | Medium | Medium | Cache "armed" predicates; only check predicates where $\dot g_i$ has the right sign; benchmark in Gate 1 |
| Stiffness-detection picks wrong integrator | High | High | Conservative thresholds; auto-fall-back to BDF on convergence failure; expose `force_integrator=` override |
| Variable-step output makes downstream waveform analysis hard | Low | Medium | Optional `output_interp_dt` parameter that interpolates to fixed-dt for user convenience |
| Tsinghua / DSIM patent concerns | Low | High | Algorithm published in academic papers (no patent encumbrance); legal review at Gate 5 |
| Path-based partial refactor doesn't help DSED much (because DSED has fewer events than steps in fixed-dt) | Medium | High | Validated early in Gate 2; if hit-rate < 30 %, paper claim shifts from "path-based DSED" to "DSED-with-fast-fallback" |
| Implementation complexity blows up | Medium | Medium | 7 gates with hard validation criteria; user is approver; abort + de-scope if Gate 3 doesn't show ≥ 5× speedup |

---

## Approval criteria

The user signs off on:
1. The "Why" framing — that DSED is the right next algorithmic
   investment for Pulsim (vs ML, GPU, or smaller scope work).
2. The 7-gate structure with explicit validation criteria.
3. The estimated 5–6 month timeline.
4. The v2.0.0 breaking-change positioning (default stays `pwl`,
   `dsed` is opt-in).

Once approved, implementation begins at Gate 0.
