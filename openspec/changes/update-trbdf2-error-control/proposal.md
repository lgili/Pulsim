## Why

Four properties of the variable-step engine's error control
(`solver/trbdf2_transient.hpp`) were measured in the core review of
2026-11 (`raw/numerics.md` items 7 and 11; probes in
`/Users/lgili/pulsim_wt/core_review_2026-11/numerics/`):

- **The estimate is half the local error.** The estimate at `:1327`
  uses the constant of `HistoryState::trbdf2_lte_const()`
  (`pwl/history_state.hpp:251`) without the factor 2 of Bank et al.
  (1985). `p10_lte.py`: true local error / estimate = 1.94, 1.97, 1.98
  at hλ = 0.1, 0.05, 0.025. The tolerance the user gets is twice the
  one asked for.
- **A stiff mode reads as error at any step.** The raw estimate grows
  with `|hλ|`: prototype `p25_trbdf2_errctl_proto.py`, one step of
  `y' = λy`, estimate / amplitude = 1.2e3 at hλ = −5e3 and 1.2e6 at
  hλ = −5e6, where the true error is 9.6e-4 and 9.7e-7. The code works
  around it with a step floor (`h_chatter`, `:932`). Hosea & Shampine
  (1996) give the remedy: one back-solve with the matrix already
  factored.
- **The answer depends on `t_end`.** `h_max = span/1000` (`:360`),
  `h_min = span·1e-11` (`:400`) and four more floors and caps (`:409`,
  `:796`, `:932`, `:985`) are fractions of the run length. `p17_span.py`: the output error at 40 T
  is 8.2e-4 V in a 40 T run and 1.04e-3 V in a 400 T run, with 1541
  and 761 steps.
- **A source jump is found by rejection.** `p13b.py`, pulse source
  into an RC, default tolerances: the steps before the 3.3 µs jump
  shrink to 6.3e-11 s, the first steps after it are 1.4e-14, 1.3e-14,
  6.4e-14 s, 32 steps are rejected and 8 accepted steps are below
  1e-10 s. A step of 1e-14 s is harmless on an RC and is below the
  Newton roundoff floor of an ordinary inductor.

One claim of the review is not confirmed and is not used here:
`raw/numerics.md` item 11 says the pulse-driven RC "floors at 1.8e-4 V
whatever the rtol". `p13b.py` sweeps the tolerance: 2.69e-3, 7.36e-4,
1.82e-4, 4.0e-5, 8.7e-6 V at rtol 1e-4 … 1e-8, the `rtol^(2/3)` law of
a second-order method. Landing the jump does not change the error
(prototype: 1.824e-4 V at rtol 1e-6 either way); it removes the
rejections and the femtosecond steps.

## What Changes

- **Estimator scale (BREAKING).** The local error estimate is
  multiplied by 2 (Bank et al.), so `rtol` and `atol` mean what they
  say. At a given tolerance a run takes about 1.26× the steps where
  the estimate binds and its error falls by about 1.6× (estimates,
  `2^(1/3)` and `2^(2/3)`).
- **Stiff filter.** The estimate is filtered once through
  `(I − (γ/2)·h·J)⁻¹` using the stage factor (Hosea & Shampine): one
  extra back-solve per attempted step on the linear path. `h_chatter`
  is kept and is removed only if its own regression circuits pass
  without it.
- **Bounds that do not depend on `t_end` (BREAKING, step sequence).**
  The step floor, the probe floor and its retry cap, the chatter floor
  (and the window derived from it) and the gate-pulse floor are
  expressed against the resolved step ceiling
  and the time resolution `16·ε·|t|`, not against the span. The
  default ceiling is the circuit's own cap (periodic sources, a
  schedule's edges, a learned gate pulse) when one exists and
  `span/1000` only when none does. `TrBdf2Stats` reports which.
- **Source jumps and corners are time events.** A table of source
  breakpoints (pulse edges and ramp corners, PWM voltage source edges)
  is landed exactly like a gate edge; the landed step reads the
  jumping source's left limit.

## Impact

- Affected specs: `transient-timestep`.
- Affected code: `solver/trbdf2_transient.hpp` (estimate, filter,
  bounds, landing), `pwl/history_state.hpp` (constant, error
  right-hand side), `pwl/cache.hpp` (a solve with a caller-supplied
  right-hand side on an existing factor), `sources/` (breakpoint
  table; shared with `add-discontinuity-restart-rule`, whichever lands
  first introduces it), `python/bindings.cpp`,
  `python/pulsim/__init__.py` (statistics, docstrings), docs,
  `CHANGELOG.md`.
- Depends on: nothing. The filter on Newton circuits needs the factor
  that `update-newton-persistent-solver` keeps alive; until then it is
  applied on the linear path only. The post-jump step is sized
  properly only with `update-trbdf2-post-event-restart`.
- Golden baselines: fixed-step and `dsed` baselines are untouched.
  Variable-step results move within tolerance (they get more
  accurate), step counts rise by up to about 26 % at a given
  tolerance, and tests that pin step counts or more digits than the
  tolerance are re-pinned. Users are told through a CHANGELOG
  **BREAKING** entry: to reproduce the previous step density, double
  `rtol` and `atol`.
