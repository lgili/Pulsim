## Why

Both implicit integrators of the kernel are second order, and second
order is expensive where the answer is an oscillation. Measured (audit
`raw/numerics.md` item 10, probe `numerics/p4_lc.py`): an ideal LC tank
(100 µH, 100 nF, 50.3 kHz, 10 V) over 1000 periods on `trbdf2` at the
default tolerances takes **121 steps per period**, loses **0.99 % of
its energy** and ends 8.4 V out of phase on a 10 V amplitude; at
`rtol = 1e-8` it takes 1250 steps per period. The fixed trapezoidal
engine conserves the energy exactly but carries a phase error of
`(ωh)²/12` per radian (audit §3).

Per step, on `x' = iωx` (`probes/esdirk43_dispersion.py`):

| Steps per period | TR-BDF2: amplitude, phase (rad) | Fourth-order ESDIRK below: amplitude, phase (rad) |
|---|---|---|
| 50 | −9.2e-7, −8.0e-5 | −1.1e-10, −2.7e-8 |
| 20 | −3.5e-5, −1.2e-3 | −2.5e-8, −2.6e-6 |
| 10 | −5.4e-4, −9.7e-3 | −1.4e-6, −8.2e-5 |

The exact propagator removes the problem for circuits that are linear
per mask (`add-exact-propagator-strategy`), but not where a Newton
device sits in the resonant path (a nonlinear `Coss` in an LLC or a
phase-shifted bridge, a saturable inductor, a Level-1 MOSFET) or where
the circuit is too large for a dense propagator. The alternatives in
the tree do not fit:

- `integrators/radau_iia3.hpp` (bound at `python/bindings.cpp:43`,
  used by no module of `python/pulsim`; `add-adaptive-runge-kutta-
  solvers`) is fully implicit: its two stages couple into a system of
  twice the size, so it cannot use the per-(mask, h) factor of the
  companion matrix.
- `integrators/dormand_prince5.hpp` is explicit, hence limited by
  stability on the stiff modes every switched circuit has (owner note:
  RK45 ground at h ≈ 2e-10 on a DCM idle mode).

TR-BDF2 was chosen because both of its stages solve with the
trapezoidal matrix at `γh`
(`core/include/pulsim/solver/trbdf2_transient.hpp:26-37`). A
fourth-order method with the same property exists.

Prototype (`probes/esdirk43_lc.py`, `probes/esdirk43_stability.py`;
numpy and sympy, the same norm and the same I-controller for both
methods; steps, linear solves and energy on the LC tank above):

| rtol | TR-BDF2: steps, solves per period; energy; max error | ESDIRK4(3): steps, solves per period; energy; max error |
|---|---|---|
| 1e-5 | 153, 336; −0.49 %; 5.4 V | 14.4, 72; −0.50 %; 1.9 V |
| 1e-6 | 336, 701; −0.048 %; 1.2 V | 25.9, 130; −0.028 %; 0.18 V |
| 1e-8 | 1576, 3164; −4.7e-4 %; 0.053 V | 83.7, 418; −8.2e-5 %; 0.0017 V |

At equal tolerance the fourth-order method takes about 10–19 times
fewer steps and 4.7–7.6 times fewer linear solves, with a smaller
error. The prototype's TR-BDF2 column is close to the kernel's own
figures (121 and 1250 steps per period), not identical: the kernel's
estimator is a factor of two optimistic (audit item 7).

## What Changes

- **`Esdirk43` integrator strategy**: the six-stage, fourth-order,
  stiffly accurate, L-stable ESDIRK4(3)6L[2]SA of Kennedy & Carpenter
  (the implicit part of ARK4(3)6L[2]SA, Appl. Numer. Math. 44, 2003),
  diagonal `γ = 1/4`, embedded third-order solution, stage order 2. The
  coefficients are in design.md and were verified in exact rational
  arithmetic.
- **One factor per (mask, h)**: every implicit stage solves with
  `G + C/(γh)`, which is the trapezoidal companion matrix at
  `dt = h/2`; the five stages reuse `solve_at(mask, h/2)`.
- **Error control**: the embedded pair, an I-controller with exponent
  −1/4, and one extra back-solve as a stiff filter (the embedded
  solution is not L-stable: `R̂(−∞) = −3/20`).
- **Events**: the explicit first stage needs the derivative after the
  event, taken from the consistent post-event solve; state events are
  bracketed by a Hermite interpolant of the step and landed by a
  re-step.
- **Devices**: stages are expressed through `f` and `q` of the unified
  device interface (`(q(X_i) − ψ_i)/(γh) + f(X_i) = 0`), so no device
  carries a formula for this method.
- **Selection**: `integrator='esdirk43'` on request. `engine='auto'`
  keeps TR-BDF2 until the rule of design.md ("accuracy-limited steps")
  has been measured on the reference set.
- No breaking change: opt-in; existing engines and results unchanged.

## Impact

- Affected specs: `transient-timestep`.
- Affected code: a new integrator in the unified stepper
  (`core/include/pulsim/solver/…`), the stage-context type of the
  device interface (a general `(β·h, ψ)` stage beside trapezoidal and
  BDF2), `pwl/cache.hpp` only through `solve_at`, `python/bindings.cpp`,
  `python/pulsim/__init__.py` (the `integrator=` option), docs
  (integrator guide), stubs.
- Depends on:
  - `refactor-unified-transient-stepper` (Phase 3): the integrator
    interface. Adding a third hand-written driver beside the 1588-line
    TR-BDF2 function is what that change exists to stop.
  - `add-unified-device-interface` (Phase 3): generic stages for
    stateful devices.
  - `update-trbdf2-post-event-restart` (Phase 2): the post-event
    derivative.
  - `update-trbdf2-error-control` (Phase 2): the stiff filter and the
    estimator scale it shares.
  - `update-trbdf2-factor-reuse` (Phase 1): the step-size grid and the
    dt-keyed factor cache, which this method uses unchanged.
  - `restore-reference-benchmark-harness` (Phase 0): the counter gate.
- Ships on today's structure (tasks 1–2): the tableau with its
  verification tests, and a kernel integrator for linear `(G, C, b)`
  systems without events, tested against closed forms. It is not
  reachable from `simulate()` until the stepper exists.
- Relation to `add-adaptive-runge-kutta-solvers` (18 of 33 tasks done):
  this change provides the stiff high-order method that change wanted
  from Radau IIA(3); whether to close its remaining tasks is an owner
  decision (design.md, Open Questions). The roadmap already retires
  `integrators/` in the stepper refactor.
- Honest limit: on circuits whose step is set by events and not by
  accuracy (a hard-switched buck takes 19 samples per period on
  `trbdf2`), five solves per step against two make this method about
  2.5 times slower per step with nothing to gain.
