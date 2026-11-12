## Why

`add_nonlinear_diode` (`models::IdealDiode`) sources power. Its forward
law, `i_on = α·(v − V_F0)/R_d` with `α = logistic(κ·(v − V_F0))`
(`core/include/pulsim/models/ideal_diode.hpp:69-81`), is negative for
every `v < V_F0`, so below the knee the diode pushes current out of its
anode while the anode is positive.

Measured with the default parameters (`V_F0 = 0.7 V`, `R_d = 10 mΩ`,
`κ = 20 /V`), probe
`/Users/lgili/pulsim_wt/core_review_2026-11/models/p2_ideal_diode_iv.py`:

| v | law | kernel (forced voltage) |
|---|---|---|
| 0 V | −5.8e-5 A | −1e-4 A |
| +0.5 V | −0.360 A | −0.3597 A |
| +0.636 V | **−1.392 A** (minimum) | −1.3923 A |
| +0.7 V | 0 A | 0 A |

The law is not monotone, so Newton has a region of negative
conductance. Consequence on a circuit (`p1_engine_matrix.py`, 20 V /
1 kHz sine, 10 Ω, diode to ground, 2 ms): `engine='pwl'` at
`dt = 1 µs` aborts with `SimulationAborted`; `engine='trbdf2'` finishes
with RMS 9.634 V where the Shockley and PWL diodes give 10.01 V.

The SPICE importer sends every LTspice `Ron/Roff/Vfwd` diode to this
device (`python/pulsim/spice_import.py:645`), so an imported netlist with
a piecewise-linear diode inherits the defect. LTspice's model is an
ideal switch with a series drop, which is exactly the PWL
`SwitchedDiode` with `V_f` added by `add-switched-diode-forward-drop`.
The defect is not listed in `docs/known-limitations.md`.

## What Changes

- **`IdealDiode::current` becomes passive and monotone**, in softplus
  form:
  `i(v) = [sp(κ·(v − V_F0)) − sp(−κ·V_F0)]/(κ·R_d) + G_off·v`,
  `sp(x) = ln(1 + eˣ)` evaluated without overflow. Then `i(0) = 0`,
  `di/dv = logistic(κ·(v − V_F0))/R_d + G_off > 0`, `i·v ≥ 0`, and the
  forward asymptote is still `(v − V_F0)/R_d`. Parameters and their names
  are unchanged.
- `numeric::softplus` (scalar and AD) beside `numeric::logistic`.
- **The SPICE importer maps `Ron/Roff/Vfwd` diodes to
  `add_diode(g_on = area/Ron, g_off = area/Roff, V_th = 0, V_f = Vfwd)`**
  instead of `add_nonlinear_diode`. The device becomes a PWL switch
  branch: no Newton iteration, accepted by `engine='dsed'`, visible to
  `res.i` and `device_loss_summary`.
- `docs/known-limitations.md`, `KNOWN_LIMITATIONS.md` and
  `AI_GUIDE.md` state what the smooth diode is (a soft-knee diode with a
  reverse floor of `sp(−κ·V_F0)/(κ·R_d)`, 4.2 µA at the defaults) and
  when to prefer `add_diode(V_f=)` or `add_shockley_diode`.
- **BREAKING (numbers):** every circuit with `add_nonlinear_diode`
  changes near the knee. At the defaults the current at 0.636 V goes
  from −1.392 A to +1.227 A, at `V_F0` from 0 A to 3.466 A
  (`ln 2/(κ·R_d)`), at 1.0 V from 29.93 A to 30.01 A. The clamp above is
  expected at RMS 10.010 V (static-law value 10.0096 V).
- **BREAKING (import):** imported PWL diodes change device kind
  (`StoredKind.NonlinearDiode` → `StoredKind.Diode`) and conduct with a
  sharp knee at `Vfwd`.

## Impact

- Affected specs: `device-models`, `python-bindings`.
- Affected code: `core/include/pulsim/models/ideal_diode.hpp`,
  `core/include/pulsim/numeric/` (new `softplus.hpp`),
  `python/pulsim/spice_import.py`,
  `python/pulsim/__init__.py` (the smooth-diode branch of the `res.i`
  reconstruction, if it re-implements the law), C++ tests under
  `core/tests/layer4_v3`, `layer5_v4`, `layer2`, `layer5` that pin the
  old law, `python/tests`, docs, CHANGELOG. No binding signature
  changes, so stubs are unchanged.
- Dependencies: `add-switched-diode-forward-drop` (the `V_f` parameter
  the importer now uses) is already implemented on this branch. No
  dependency on other roadmap changes. `add-engine-conformance-matrix`
  lists `NonlinearDiode` on `pwl` as supported and relies on this fix
  for the clamp cell.
- Users affected: direct callers of `add_nonlinear_diode` / YAML
  `nonlinear_diode` (39 files under `python/`, `core/tests`, `examples` and `docs`
  mention it), and anyone
  importing LTspice netlists with `Ron/Roff/Vfwd` diode models.
  Migration: none needed for the importer; for the smooth diode, results
  move toward the physical ones and no parameter needs to change.
