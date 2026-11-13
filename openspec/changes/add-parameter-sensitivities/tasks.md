Reference values.

- RC step (10 V, 1 kΩ, 1 µF) at `t = RC = 1 ms`:
  `∂v/∂R = −V/(R·e) = −3.678794e-3 V/Ω`,
  `∂v/∂C = −V/(C·e) = −3.678794e6 V/F`.
- DCM buck of `numerics/p2_buck.py` (R = 50 Ω, D = 0.3), from zero,
  state at 40 T (`../add-periodic-steady-state-shooting/probes/shooting_dcm.py`):
  `∂v_C/∂L = 1758.1254 V/H`, `∂v_C/∂R = 2.5551883e-2 V/Ω`,
  `∂v_C/∂D = 43.093591 V`.
- Same circuit, periodic orbit: `∂v_C*/∂L = −3.4575154e4 V/H`,
  `∂v_C*/∂R = 6.92255e-2 V/Ω`, `∂v_C*/∂D = 23.0895697 V`.
- Shockley rectifier of `probes/forward_sensitivity_mna.py`
  (trapezoidal, 4000 steps over 40 ms): forward sensitivities of the
  capacitor voltage equal central differences within 3.4e-7 (R), 1e-8
  (C, R_load), 1.1e-8 (`I_s`), 8e-9 (`n·V_t`).

## 1. Characterisation
- [x] 1.1 Port `probes/fd_gradient_today.py` as a documented test of today's finite-difference gradients (fixed engine at T/100: relative error between −13 and +2; `trbdf2`: about 1e-4), kept as the "why" record
- [x] 1.2 Finite-difference harness for tests: central difference with step selection and a Richardson check, usable on any engine
- [x] 1.3 Port the two prototypes as reference implementations with the values above as constants

## 2. Exact path, linear-element parameters and duty (today's `dsed` adapter)
- [x] 2.1 `∂A/∂p`, `∂b/∂p` per mask for affine stamps (R, L, C, source value) from the exact difference of two assemblies
- [x] 2.2 `∂Φ/∂p`, `∂(Γ₀b)/∂p` by the Daleckii–Krein formula with φ₁ divided differences
- [x] 2.3 Segment recurrence and jump correction at diode events and at parameter-dependent gate edges
- [x] 2.4 Tests: RC values above within 1e-9 relative; DCM buck transient values within 1e-7 relative; every one equal to a central difference of the same exact map within 1e-7; `∂v_C/∂D` computed without the gate-edge jump term is 0 (regression guard), with it 43.093591 V

## 3. Steady-state gradients (with slice 1 of `add-periodic-steady-state-shooting`)
- [x] 3.1 `∂P/∂p` by one sensitivity period; `dx*/dp = (I − M)⁻¹·∂P/∂p`
- [x] 3.2 Tests: the three orbit gradients above within 1e-7 relative, and equal to central differences of the steady-state solve within 1e-7 (prototype: 3.4e-10, 5.5e-9, 3.5e-10)

## 4. Parameters on the AD scalar (needs `add-unified-device-interface`)
- [ ] 4.1 `Params<S>` for every device; parameter seeding helpers in `ad/ad_scalar.hpp`
- [ ] 4.2 Per-device unit tests: `∂f/∂p`, `∂q/∂p` by AD against a central difference of the law within 1e-8 relative, for every parameter of every device kind
- [ ] 4.3 Test: `Params<Real>` build is bit-identical on the golden hashes; `static_assert` on layout

## 5. Recurrence in the stepper (needs `refactor-unified-transient-stepper`)
- [ ] 5.1 Stage recurrence of D1 for backward Euler, trapezoidal, TR-BDF2 (and ESDIRK when present); history of `dq/dp`
- [ ] 5.2 Sensitivity solve iterated to the Newton tolerance when the factor is stale
- [ ] 5.3 Counters: `n_sens_solves`, `n_sens_iterations`
- [ ] 5.4 Tests: RC values within 1e-5 relative on TR-BDF2 at rtol 1e-8; Shockley rectifier within 1e-6 of central differences of the same fixed-step scheme (prototype 3.4e-7 worst); states bit-identical with and without `sensitivities`
- [ ] 5.5 Measurement gate (counters): the number of factorizations is identical with and without sensitivities; `n_sens_solves = n_p × stages × accepted steps` on a linear circuit; on the Shockley rectifier `n_sens_iterations ≤ 2` per stage

## 6. Events (needs `add-unified-event-queue`, `add-fixed-step-event-splitting`)
- [ ] 6.1 Jump correction for state events and for parameter-dependent time events; `dτ/dp` from modulators
- [ ] 6.2 Refusals of the applicability rules, each with a test
- [ ] 6.3 Tests: DCM buck transient values within 1e-5 relative on TR-BDF2 (rtol 1e-9) and on the fixed engine with event splitting; a rectifier with `V_th = V_f + 1e-3` (field jump at turn-on) within 1e-5 of a central difference on the exact path; the fixed engine without event splitting raises naming the diode

## 7. KPI gradients and optimisation interface
- [ ] 7.1 `res.sens`, `res.sensitivity.states`, `res.kpi(…, with_grad=True)` for final value, average, RMS, energy, peak, ripple; `non_unique` flag
- [ ] 7.2 Orbit KPIs with the second sensitivity period (D5)
- [ ] 7.3 `pulsim.value_and_grad`
- [ ] 7.4 Tests: average and RMS gradients of `v_out` on the DCM buck against central differences of the exact path within 1e-6; `scipy.optimize.minimize(jac=True)` on an LC output filter (two parameters, ripple target) converges and its reported gradient at the start equals a central difference within 1e-6
- [ ] 7.5 Measurement gate: a gradient with respect to 8 parameters costs one run with 8 back-substitutions per stage, against 16 runs by central differences (counted as period runs or steps, not wall time)

## 8. Controller gains (needs `add-compiled-control-graph`)
- [ ] 8.1 Block parameters as parameters; discrete-state sensitivities updated at ticks
- [ ] 8.2 Test: discrete P loop on an RC (`K_p = 2`, 100 µs): `∂v[10]/∂K_p` equals the derivative of the closed-form recurrence within 1e-9

## 9. Docs and verification
- [ ] 9.1 Sensitivity guide (what is differentiated, the event rule, when it is refused, discrete against continuous gradient), sweep guide, AI_GUIDE, CHANGELOG
- [ ] 9.2 ruff, pytest, ctest (`-j2`), `openspec validate --all --strict`, stub check
