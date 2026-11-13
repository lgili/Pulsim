## Why

`simulate(engine='dsed')` with the default `integrator='auto'` and no
PWL diode picks the least accurate integrator the engine has for the
circuits where it matters most.

- The auto scheduler (`core/include/pulsim/dsed/scheduler_auto.hpp:227`)
  integrates every mode its stiffness detector flags with fixed-step
  BDF2 at `h_bdf2 = 1 µs` (`python/pulsim/_dsed_dispatch.py:77`). It
  never asks `system_.has_exact_step()`, although the RK45 scheduler in
  the same folder does (`scheduler.hpp:338-342`) and then advances with
  the closed-form propagator, which is exact for any step.
- The first BDF2 step is a Crank–Nicolson bootstrap
  (`core/include/pulsim/dsed/bdf2_integrator.hpp:102-116`), which is not
  L-stable: on a mode with time constant τ its amplification is
  `(1 − h/2τ)/(1 + h/2τ)`, about −0.96 at `h/τ = 100`.

Measured, probe
`/Users/lgili/pulsim_wt/core_review_2026-11/numerics/p8c.py` (100 V
source, 10 Ω, a 1 mΩ switch to ground with 1 nF across it, 100 kHz,
50 % duty; τ = 10 ns while the switch is open):

| `integrator` | samples | `v(sw)` range | max error in the open interval |
|---|---|---|---|
| `'auto'` (default) | 42 | **−99.98 V … 196.08 V** | 96.07 V |
| `'rk45'` (takes the exact stepper) | 42 | 0.010 V … 100.000 V | 1.0e-6 V |
| `'bdf2'` | 402 | −99.98 V … 166.67 V | 66.66 V |

A capacitor charged from 100 V reads 196 V, with no warning. The value
is the Crank–Nicolson bootstrap exactly: `100·2z/(1 + z)` with
`z = h/2τ = 50` is 196.08 V; the explicit `'bdf2'` run takes 402
samples over 40 µs, a 0.1 µs step, for which `z = 5` gives 166.67 V.

## What Changes

- **`integrator='auto'` uses the exact stepper whenever the current mode
  has one** (`has_exact_step()`: LTI mode, sources the stepper supports,
  non-defective eigenbasis), whatever the stiffness detector says. Steps
  are capped at `dt_max` as on the RK45 scheduler. Only modes without an
  exact stepper go to the detector's choice between DOPRI5 and BDF2.
- **The BDF2 bootstrap becomes backward Euler**,
  `(I − h·A)·y₁ = y₀ + h·b(t + h)`, which is L-stable. It applies to the
  auto scheduler's BDF2 segments and to `integrator='bdf2'`.
- The result reports `n_exact_steps` beside `n_rk45_steps` and
  `n_bdf2_steps`.
- **BREAKING (numbers):** `engine='dsed'` runs that went through BDF2
  segments change. On the circuit above `'auto'` goes from a 96 V error
  to the exact-stepper error (1e-6 V measured on that path); `'bdf2'`
  starts each segment at `100·(h/τ)/(1 + h/τ)` — 90.91 V at the 0.1 µs
  step that run takes — instead of 166.67 V, and peaks at 102.77 V (the
  BDF2 recurrence after a first-order start, computed, not measured). Runs with PWL diodes already route to the RK45
  scheduler and are unchanged.

## Impact

- Affected specs: `transient-timestep`.
- Affected code: `core/include/pulsim/dsed/scheduler_auto.hpp`,
  `core/include/pulsim/dsed/bdf2_integrator.hpp`,
  `python/dsed_bindings.cpp` (`n_exact_steps` in the stats dict),
  `python/pulsim/_dsed_dispatch.py` (result attribute, docstring of
  `integrator`), `core/tests/dsed/`, `python/tests/test_dsed*.py`,
  docs, CHANGELOG, stubs if the result attribute is part of them.
- Dependencies: none. Making the exact propagator a first-class
  strategy with its robustness gaps closed is
  `add-exact-propagator-strategy`; this change only stops the default
  from avoiding it.
- Users affected: anyone on `engine='dsed'` without diodes whose
  circuit has a mode faster than about `h_bdf2/10`. Their results move
  toward the exact solution; no keyword changes. `integrator='bdf2'`
  remains available and remains second order after the first step.
