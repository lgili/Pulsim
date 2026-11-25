## Context

TR-BDF2 with γ = 2 − √2 solves both stages with the trapezoidal matrix
at `dt = γh`, so one factor per `(mask, h)` serves a whole step
(`solver/trbdf2_transient.hpp:25-33`). The engine reaches that factor
through `PwlStateSpaceCache::solve_at(mask, γh, …)`. Because the cache
is built with a placeholder `dt`, every TR-BDF2 solve goes through the
event-entry table, never through the primary segments (the probe shows
0 primary segments built).

An event entry holds the `(G, C, b)` split, one solver analysed once and
one factor at `current_dt`. A new `dt` on a resident mask is a numeric
refactor; a new mask is a full build; the table is capped at 8 entries.

Measured today: 1.23 refactors per accepted step on the buck, 1.00 on
the VSI plus 3,198 full builds, 0.67 on a ladder with no switch at all.

Constraints: the LTE test and its tolerances are not touched; the
thread-local solve workspace stays; `-ffp-contract=off` stays; step
bounds and estimator belong to `update-trbdf2-error-control`.

## Goals / Non-Goals

- Goals: a mask is built once; a `(mask, h)` pair that recurs is a hit;
  a refactor allocates nothing; parts 1 and 2 bit-identical; part 3
  with an explicit accuracy gate and an off switch.
- Non-Goals: changing the controller's law, exponent or safety factor;
  fewer events; a shared symbolic analysis across masks; reuse of an
  old factor at a different `h` (modified Newton).

## Decisions

- **Byte budget, one knob.** `set_segment_budget_bytes` already bounds
  the lazy segment table; the same value now bounds the event table,
  each table accounted separately. `EventEntry::approx_bytes()` counts
  `G`, `C`, `b` and every resident factor. Eviction is LRU and happens
  after everything that can throw, as today.
  - Alternative: raise the constant to 128. Rejected: an MMC visits
    thousands of masks; a count says nothing about memory.
  - Alternative: a second setter. Rejected: no user asked for one.
- **A factor is a pure function of `(mask, dt)`.** The matrix is
  `G + (1/dt)·C` on both the build and the refactor branch, the column
  ordering depends only on the pattern, and `factorize` has no memory
  of earlier values. So which entries are resident cannot change a
  result. This is why part 1 is bit-identical; the goldens verify it.
- **In-place combine keeps Eigen's arithmetic.** `G` and `C` are
  scattered once onto the union pattern, absent entries stored as
  `+0.0`; `J[k] = G[k] + inv_dt · C[k]` is then the per-entry operation
  of `e.G + (Real{1} / dt) * e.C`.
- **Factorisation work vectors become solver members.** `pinv`, `x`,
  `xi`, `rstack`, `pstack`, `visit` are kept between calls; the L/U
  arrays are cleared, not released. `factorize` is non-const and
  already mutates the factor, so it was never safe to call concurrently
  on one solver; `solve` keeps its thread-local buffer.
- **Grid: `m` levels per octave, default 4, rounding down.**
  - Rounding down keeps `h_used ≤ h_proposed`, so the controller's
    safety margin only grows and the LTE test is unchanged.
  - `m = 4`: a step loses at most 16 % (about 8 % on average), and the
    step grows only when the controller asks for ×1.19 or more. That
    deadband is the mechanism: BDF codes hold `h` until the proposed
    ratio is large enough to pay for a new factorisation.
  - Alternatives: `m = 2` (loses up to 29 %, fewer levels), `m = 8`
    (8 %, twice the distinct factors). Task 5.1 measures 2, 4 and 8
    before the default is fixed.
  - Alternative: round to nearest. Rejected: it can exceed the proposal
    by up to 9 % and turn an accepted step into a rejected one.
- **The grid is absolute and computed without libm.** `h = c_j · 2^e`
  seconds with `c_j` four literal constants. Quantisation is
  `frexp` + table compare + `ldexp`, all exact, so the step sequence is
  the same on every platform and does not depend on `t_end` or
  `h_init`. Alternative: `exp2(floor(m·log2 h)/m)`. Rejected: libm
  results differ in the last bit between platforms.
- **One quantisation point.** The step is quantised once, at the top of
  the step loop after the clamp to `[h_lo, h_max]`
  (`trbdf2_transient.hpp:1086`) and before the landing truncations.
  Accept, reject and Newton-failure paths all pass through it.
  Exemptions: `h == h_max` (a constant, itself a perfect key); a result
  below `h_lo` (use `h_lo`); landing steps.
- **Four grid slots and one scratch slot per mask.** Each slot is a
  solver with its own factor, keyed by the exact `dt`. The stepper
  tells `solve_at` whether the solve is on the grid. Grid solves use
  the grid slots (LRU among them); off-grid solves use the scratch slot
  only, so a landing step never evicts a reusable factor.
  - Alternative: one slot per mask (today). Rejected: around an edge a
    mask alternates between the post-edge step and the steady step.
  - Alternative: no hint, promote on second use. Rejected: both stages
    of a step use the same `dt`, so every landing step would be
    promoted.
  - Without `update-linear-time-assembly` each slot runs its own
    symbolic analysis on first use (`event_analyses` counts them); with
    it the slots share one.
- **Off switch.** `step_grid_per_octave = 0` disables quantisation and
  the hint (every solve uses one slot), which reproduces parts 1 + 2
  exactly. It is the oracle for bit-identity and the migration path.

## Risks / Trade-offs

- Rounding down costs steps: up to 19 % more accepted steps in the
  worst case → gate `n_accept ≤ 1.20 × baseline` per reference circuit;
  the saving must outweigh it in wall time.
- A coarser `h` sequence can interact with the edge-parity step memory
  and with the chatter guard → the chatter, forced-accept and
  held-diode counters must not rise on the reference set.
- Memory: up to five factors per mask → inside the byte budget (a
  fixed-engine VSI segment measured 1.5 KB; the default budget is
  1 GiB).
- Two **BREAKING** TR-BDF2 changes (`update-trbdf2-error-control` is
  the other) would regenerate references twice → land both in one
  release.
- The targets in tasks.md, and the 0.30 ratio in the spec delta, are
  estimates → baseline first; a missed target is reported, not waived,
  and the spec delta is revised before the default flips.

## Migration Plan

1. Counters and baseline.
2. Part 1, then part 2, each against the `trbdf2` goldens.
3. Part 3 behind `step_grid_per_octave`, default 0 until the accuracy
   gate passes and the owner signs off; then default 4, references
   regenerated, CHANGELOG entry under numerical changes.
4. Rollback: set the option to 0.

## Open Questions

- Grid density `m` (2, 4 or 8) and slot count: accept "decide by the
  measurement in task 5.1", or fix `m = 4` now?
- Should part 3 ship in the same release as
  `update-trbdf2-error-control`, so `trbdf2` references change once?
- Keep `step_grid_per_octave` as a permanent option, or remove it one
  release after the default flips?
- Is one byte budget applied to each table acceptable, or should the
  two tables share a single total?
