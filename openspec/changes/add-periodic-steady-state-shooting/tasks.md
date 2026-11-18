Reference values. Buck of `numerics/p2_buck.py` (24 V, 100 µH, 47 µF,
`g_on = 1e3`, `g_off = 1e-9`, 100 kHz), state `(i_L, v_C)` at the rising
gate edge:

| Case | Orbit | Multipliers | Source |
|---|---|---|---|
| DCM, R = 50 Ω, D = 0.3 | `(6.00750582388986e-9 A, 8.99624708728943 V)`, diode off at `0.799830698444468·T` | 0 and 0.988977 | 50 digits; `probes/shooting_dcm.py` |
| CCM, R = 1 Ω, D = 0.5 | `(11.6878790893 A, 11.9877289409 V)` | `0.894557 ± 0.089619j` (modulus 0.899035) | `probes/shooting_dcm.py` |
| Peak-current-mode, L = 100 µH, C = 4700 µF, R = 2 Ω, `I_ref = 7 A` | `(6.4083405593 A, 13.4083540775 V)`, D = 0.5590 | −1.267293 and 0.998949 (analytic `−D/(1−D)` = −1.2674) | `probes/peak_current_mode.py` |

Steady-state gradient used by the small-signal test:
`∂v_C*/∂D = 23.0895697 V` on the DCM buck (prototype, equal to a
central finite difference to 3.5e-10).

Slice 1 (tasks 1–4, 8) is implemented; decisions and deviations are in
`DECISIONS.md`. The spec's source refusals ("a source that is not
`T`-periodic") were true only for sine voltage sources until the
integration fix of DECISIONS 12 extended them to the current sources. Notes on ticked items: 3.3's CCM Bode is checked on the
Q = 8 buck of `test_sampled_ac.py`; 6.3's saltation guard is covered in
slice 1 only by a hysteresis-band rectifier (M against a finite
difference; without saltation it is off by 3.1e-5); 8.1's guides are a
section of `docs/api-reference.md` (there is no separate steady-state or
Bode guide).

## 1. Characterisation (no behaviour change)
- [x] 1.1 Tests that pin today's behaviour on the two bucks: `steady_state` CCM passes, DCM raises "no periodic orbit found"; `run_periodic_shooting` DCM returns 8.8998 V (−1.07 %); mark the DCM cases as the ones this change flips
- [x] 1.2 Port `probes/shooting_dcm.py` and `probes/peak_current_mode.py` into the test tree as reference implementations (numpy only), with the table above as constants

## 2. Solver on the exact path (today's `dsed` adapter)
- [x] 2.1 Binding: per-mask `(A, b)`, the recovery row of each predicate, and an event log entry `(t, predicate, mask before, mask after)` for one period started from a given reduced state
- [x] 2.2 `periodic_steady_state`: map evaluation, `M = Π S_k·Φ_k` with `Φ_k` from the modal form, saltation from `f⁻`, `f⁺`, `∇g`
- [x] 2.3 Newton with the scaled norm, backtracking, transient fallback, budget, verification run (D4)
- [x] 2.4 Singular `I − M` and grazing-event diagnostics (D6, D2)
- [x] 2.5 Tests: DCM orbit within 1e-8 of the reference from the zero state in at most 8 period runs (prototype: 5, plus the verification run); the iteration passes through a CCM pattern; CCM orbit within 1e-9 in at most 4 period runs; agreement with a 4000-period transient within 1e-9; residual sequence shows quadratic convergence (each residual at most the square of the previous one times 1e3; prototype 2.2e-4, 3.0e-7, 5.7e-13)

## 3. Outputs
- [x] 3.1 Result type of D5; multipliers with their type; `stable`
- [x] 3.2 `sampled_model(inputs, outputs)` with `B_d = ∂P/∂u` from the same propagators (duty and source amplitude as inputs)
- [x] 3.3 Tests: DCM multipliers 0 (within 1e-9) and 0.988977 (within 1e-6); `M` equals a central finite difference of the map within 1e-7; `H(z = 1)` of duty-to-`v_C` equals 23.0895697 V within 1e-6 relative; duty-to-output of a 100 kHz CCM buck below 5 kHz within 1 % of `V_in/(1 + sL/R + s²LC)` (0.77 % today, `sampled_ac.py` docstring)

## 4. Delegation on the exact path
- [x] 4.1 `steady_state` and `frequency_response` call the solver when the circuit is within the exact stepper's coverage; otherwise the existing code runs unchanged
- [x] 4.2 Tests: `steady_state` on the DCM buck (PWL diode, C++ schedule) returns the reference orbit; `frequency_response` on the DCM buck returns a finite response whose DC gain matches task 3.3

## 5. Fixed and TR-BDF2 engines (needs `add-discontinuity-restart-rule`, `add-fixed-step-event-splitting`)
- [x] 5.1 Map on the true state with a restart at the section `t₀`
- [x] 5.2 `M` by finite differences with Broyden updates; refresh rule on the convergence rate
- [x] 5.3 `run_periodic_shooting`, `run_harmonic_balance` delegate; old implementations removed; deprecation note on `run_harmonic_balance`
- [x] 5.4 Tests: on each engine the orbit equals that engine's own 4000-period transient within 1e-8 of the state scale; fixed engine at T/100 and T/101 agree within 1e-4 on the DCM buck (today −0.023 % against −0.50 %); `run_periodic_shooting` on the DCM buck within 1e-4 of 8.996247 V (today 8.8998 V)
- [x] 5.5 Measurement gate (period runs, counted): DCM buck from zero ≤ 12 runs on the fixed engine, against `(N + 1)` per iteration today

## 6. State events with a field jump (needs `add-unified-event-queue`, `add-unified-device-interface`)
- [ ] 6.1 Saltation for comparator-driven gate edges, hysteresis-band diodes and reset maps; flat device-state vector in `x₀`
- [ ] 6.2 Variational recurrence in the kernel for the implicit path (shared with `add-parameter-sensitivities`), replacing finite differences
- [ ] 6.3 Tests: peak-current-mode buck of the table — multipliers −1.267293 and 0.998949 within 1e-5, `stable = False`, and with `I_ref` lowered to give D < 0.5 `stable = True`; with saltation disabled in the test harness the multipliers are 0.9993 ± 0.0146j (the regression this guards); a rectifier with `V_th = V_f + 1e-3` has `M` equal to a finite difference of the map within 1e-6

## 7. Controller states (needs `add-compiled-control-graph`)
- [ ] 7.1 Controller states of the control graph in `x₀`; refusal for Python closures with hidden state
- [ ] 7.2 Tests: voltage-mode buck with a PI at the switching rate — the orbit has zero mean error within 1e-9 and equals a long transient within 1e-8

## 8. Docs and verification
- [x] 8.1 Steady-state guide (true state, section, what a multiplier outside the unit circle means), Bode guide, AI_GUIDE, CHANGELOG with the before/after numbers
- [x] 8.2 ruff, pytest, ctest (`-j2`), `openspec validate --all --strict`, stub check (slice 1: ctest not run — no core header changed and the ctest build does not compile the bindings)
