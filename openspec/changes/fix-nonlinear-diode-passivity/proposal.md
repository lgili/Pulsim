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

A passive softplus law was implemented first and kept in
`implementation-attempt.patch`; it moved every mA-level operating point
by 0.35–0.4 V and removed the premise of 27 hostile-circuit fixtures
(`DECISIONS.md`). The owner decided (2026-11-12) to **retire** the
device instead of changing its law.

## What Changes

- **The law is not changed.** `IdealDiode` stays in the kernel, used by
  the DC continuation strategies, the MOSFET Level-1 body diode and the
  C++ tests.
- **BREAKING:** `CircuitBuilder.add_nonlinear_diode` (Python) raises
  `pulsim.RetiredDeviceError` (a `ValueError`). The message names the
  device, quotes −1.39 A at +0.636 V and names the replacements:
  `add_diode(..., V_f=)` (PWL diode with a forward drop) and
  `add_shockley_diode` (smooth junction). The keyword
  `allow_non_passive=True` (in the style of `allow_synthetic=True` for
  parts) still builds the legacy device; the hostile-circuit test
  fixtures pass it.
- **BREAKING:** YAML `type: nonlinear_diode` is refused unless the
  device sets `allow_non_passive: true`.
- **BREAKING (import):** the SPICE importer maps `Ron/Roff/Vfwd` diodes
  to `add_diode(g_on = area/Ron, g_off = area/Roff, V_th = 0,
  V_f = Vfwd)`. Imported PWL diodes change kind
  (`StoredKind.NonlinearDiode` → `StoredKind.Diode`).
- `pulsim.catalog()`, the API reference, the migration guide, the
  notebooks and `AI_GUIDE.md` stop offering the device and name it
  retired. `CHANGELOG.md` and `docs/known-limitations.md` record it.

## Impact

- Affected specs: `device-models`, `python-bindings`.
- Affected code: `python/pulsim/_builder_ergonomics.py`,
  `python/pulsim/__init__.py`, `python/pulsim/spice_import.py`,
  `python/pulsim/discovery.py`, `core/include/pulsim/yaml/loader.hpp`,
  the Python tests that build the device on purpose (opt-in added),
  `core/tests/yaml/test_yaml_loader.cpp`, docs, notebooks, CHANGELOG.
  The C++ `CircuitBuilder::add_nonlinear_diode` and the binding
  signature are unchanged, so stubs are unchanged.
- Dependencies: `add-switched-diode-forward-drop` (the `V_f` parameter
  the importer now uses) is already implemented on this branch.
- Users affected: Python and YAML callers of `add_nonlinear_diode`;
  LTspice netlists with `Ron/Roff/Vfwd` diode models.
