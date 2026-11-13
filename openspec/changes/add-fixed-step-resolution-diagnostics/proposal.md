## Why

The fixed engine gives no sign when `dt` does not resolve the circuit.
Measured (core review 2026-11, `raw/numerics.md` item 6; `p13.py`,
re-run on this branch): an RC stepped at `dt = 10·τ` is 1.87 V off on a
10 V step and overshoots to 11.11 V, with zero warnings. The guide's
only advice is "halve `dt` and compare" (§6.3), which costs a second
run and, on today's engine, is misleading when the error is erratic in
`dt` (`raw/numerics.md` item 3).

The two symptoms of an under-resolved trapezoidal run are cheap to
observe inside the step loop:

- **truncation error**: the local error `(h³/12)·|y‴|` of each
  capacitor voltage and inductor current, available from three
  consecutive derivative slots the companion already holds;
- **ringing at the Nyquist frequency**: a mode with `τ < h/2` has a
  negative trapezoidal amplification factor, so its contribution
  changes sign every step.

A numpy prototype (`p24_resolution_proto.py`) on today's step gives,
for the RC: worst single-step truncation 0.35 of the variable's
magnitude and a run of 6 sign-alternating steps at `dt = 10τ`; 2.7e-2
at `dt = τ`; 3.3e-4 at `τ/10`; 3.3e-6 at `τ/100`. A buck in continuous
conduction stays below 2.7e-4 from T/20 to T/400 with no ringing; the
DCM buck's idle interval (the measured 0.50 V / 26.15 V alternation)
shows runs of 49 and 195 alternating steps at T/100 and T/400; the
hard-switched capacitor loop shows 0.67 and runs of hundreds of steps.

## What Changes

- **Per-step truncation estimate.** For every linear capacitor and
  inductor the fixed engine computes
  `e = (h/12)·|f_{k+1} − 2·f_k + f_{k−1}|` (`f = i_C/C` or `v_L/L`)
  and records, per stored sample, the largest `e / peak|y|` since the
  previous sample. The stencil is restarted at the start of the run
  and at every step whose stamped mask changed or that restarted.
- **Nyquist-ringing detector.** A state entry (node voltage or branch
  current) whose step-to-step difference changes sign on four or more
  consecutive steps, each difference above 1 % of the entry's own
  magnitude, is recorded with its name, first time, run length and
  amplitude.
- **Result and warning.** `SimulationResult::resolution` carries the
  per-sample trace, the worst value with its branch and time, and the
  ringing records. Python exposes it as `res.resolution` and emits one
  `pulsim.ResolutionWarning` per run when the worst truncation exceeds
  1e-2 or any ringing was recorded, naming the variable and suggesting
  a smaller `dt` or the variable-step engine.
- `simulate(resolution_diagnostics=True)`; `False` skips the
  bookkeeping.
- No numerical result changes: the diagnostics read the state and
  write only their own records.

## Impact

- Affected specs: `transient-timestep`, `python-bindings`.
- Affected code: `solver/run_transient.hpp` (two small per-step
  passes), `solver/options.hpp`, `solver/result.hpp`,
  `python/bindings.cpp`, `python/pulsim/__init__.py` (warning class,
  result view, `simulate` kwarg), `python/pulsim/_result_views.py`,
  stubs, `AI_GUIDE.md` §6.3, docs, `CHANGELOG.md`.
- Depends on: nothing. With `add-discontinuity-restart-rule` the
  ringing that follows events disappears and the detector is left with
  the modes `dt` really does not resolve.
- Golden baselines: none moves; states are bit-identical. New
  `ResolutionWarning`s appear on runs that are under-resolved, which
  is the purpose; test suites that run with warnings as errors need
  the listed circuits reviewed (tasks).
- `engine='trbdf2'` and `engine='dsed'` are not concerned; their
  `res.resolution` is `None`.
