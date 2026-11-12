Reference values.

- RL (10 V, 1 Ω, 1 mH), ideal switch closed at t = 0:
  `i(10 ms) = 10·(1 − e^{−10}) = 9.99954600 A` (the conductance form
  with `g_on = 1e3` settles at 9.98955 A).
- Same circuit, switch opened at 5 ms with no other path:
  `i_L(5 ms⁻) = 9.93262053 A`, transferred flux
  `L·i = 9.93262053e-3 V·s`, energy `½·L·i² = 49.3285 mJ`.
- 1 µF at 10 V paralleled with 3 µF at 2 V through `R_on = 0`:
  4.000000 V, 6.0 µC transferred, 24.0 µJ lost.
- DCM buck of `numerics/p2_buck.py` (R = 50 Ω, D = 0.3) with an ideal
  switch and an ideal diode (`probes/ideal_dcm_buck.py`, 40 digits):
  orbit `v_C = 8.9963970815477 V` at the rising edge, diode off at
  `0.79983739563545·T`, `i_L = 0` in the idle interval.

## 1. Characterisation (no behaviour change)
- [ ] 1.1 Port `probes/current_kernel_goff0.py` as tests of today's behaviour: `g_on = inf` aborts with "numerically singular"; `g_off = 0` opened on 9.93 A gives ±397 V on `pwl` and above 1e10 V on `trbdf2`, with no warning
- [ ] 1.2 Port `probes/branch_form.py` as the dense reference for the stamp, the cut-set and the capacitor-loop cases

## 2. Validation of the conductance form
- [ ] 2.1 `add_switch`, `add_diode`, YAML: `g_on` finite and > 0, `g_off` finite and ≥ 0; error names the device and the field (**BREAKING** for invalid values)
- [ ] 2.2 Tests: each invalid value is refused at build time; every existing test circuit still builds

## 3. Branch stamp and linear algebra
- [ ] 3.1 `stamp_switch_branch` (D1) shared by the transient and DC assemblers; branch-current unknown allocation in the pool; structural entries kept when zero
- [ ] 3.2 Shared symbolic analysis across masks; partial pivoting per mask; rank-one row update for a single flip; counted fallback to a full refactor
- [ ] 3.3 Tests: on a four-switch bridge the pattern (column pointers and row indices) is identical for all 16 masks; every mask factorizes with residual `‖Jx − b‖/‖b‖ ≤ 1e-12` including all-ON with `R_on = 0`; a single-flip update equals a full refactor within 1e-12; pivot growth recorded
- [ ] 3.4 Test: with `R_on = 1e-3`, `g_off = 1e-9` the branch form reproduces the conductance form within 1e-10 on the buck, the bridge rectifier and the three-phase inverter

## 4. API, floating nodes, first slice
- [ ] 4.1 `add_ideal_switch`, `add_diode(form='branch')`, YAML `form: branch`, `components()` fields, stubs
- [ ] 4.2 Per-mask connectivity pass (D3) with capacitors for the transient matrix and without for DC; loop of closed ideal switches and voltage sources; errors name mask, nodes and switches
- [ ] 4.3 Conformance-table row for the branch form; refusal on engines that do not take it
- [ ] 4.4 Tests (`R_on = 0`, `g_off = 1e-9`): RL current 9.99954600 A at 10 ms within the engine tolerance on `trbdf2` (rtol 1e-9: within 1e-6) and, on the fixed engine at `h = τ/80`, max error at most 2e-3 A over the run (prototype 1.51e-3); two open ideal switches in series with nothing between them raise the floating-node error naming the node; two closed ideal switches in parallel raise the loop error

## 5. Dependent states (needs `add-discontinuity-restart-rule`, `update-trbdf2-post-event-restart`)
- [ ] 5.1 Enable `g_off = 0` and `R_on = 0` across capacitors on engines whose first post-event step is backward Euler
- [ ] 5.2 Inconsistency detection from the state jump across the restart sub-steps; event record `{t, switch, kind, Λ, ΔW}`; `on_impulse` policy; pulse sample marked in the result
- [ ] 5.3 Tests, inductor cut-set: `i_L = 0` (exactly) from the first step after the event; the node-voltage pulse has area `−L·i_L(t⁻)` of the run within 1e-9 relative, `i_L(t⁻)` being 9.93262053 A within the engine tolerance; `|v| < 1e-9 V` from the second step after the restart (no ±397 V ringing); the record reports `½·L·i_L(t⁻)²` within 1e-9 relative (49.3285 mJ for the exact current); `on_impulse='error'` raises naming the switch and the inductor
- [ ] 5.4 Tests, capacitor loop: both capacitors at 4.000000 V within 1e-12 after the event; transferred charge 6.0 µC and loss 24.0 µJ within 1e-9 relative in the record; switch current 0 within 1e-9 A after the restart
- [ ] 5.5 Test: a consistent commutation (diode off at `i = 0` in DCM) produces no impulse record

## 6. Branch-form diode, outputs, losses
- [ ] 6.1 ON signal from `i_s`; `decide` / `crossing_signal` with a form tag at every decision site (event iteration, DC, seeding, sub-step watcher, TR-BDF2 crossing and held checks)
- [ ] 6.2 `res.i`, `device_loss_summary` (`R_on·i² + V_f·i`, impulse energy as an event energy), `_diode_state`
- [ ] 6.3 Tests: ideal DCM buck orbit 8.9963970815 V within 1e-6 on `trbdf2` (rtol 1e-9) with `i_L = 0` exactly in the idle interval (6.0e-9 A with the resistor pair) and `v(sw) = v_out` there; the half-wave scenario of the `transient-timestep` spec with `V_f = 0.7`, `R_on = 0.05 Ω` in branch form matches the conductance form (`g_on = 20`) within 1e-9; forced 10 A through an ideal diode with `V_f = 0.7` gives `v_D = 0.7 V` and 7.0 W

## 7. Measurement gate, docs, verification
- [ ] 7.1 Counters on the three-phase inverter and the buck, branch against conductance form: unknowns (+1 per switch), factorizations per run (equal within the fallback count), partial-refactor fallbacks, solves per step (equal); on TR-BDF2 the smallest committed step in the DCM buck is reported for both forms (8.26e-13 s with the resistor pair in the audit) and the branch form's must not be the smaller; ns/step reported as information
- [ ] 7.2 Docs: device guide ("ideal switch: when, and what the new errors mean"), PSIM / PLECS porting note, AI_GUIDE trap list, CHANGELOG
- [ ] 7.3 ruff, pytest, ctest (`-j2`), `openspec validate --all --strict`, stub check

## 8. Exact strategy with ideal switches (needs `add-exact-propagator-strategy`)
- [ ] 8.1 Extraction with per-mask constraints (the reduced state loses the dependent states); explicit projection `x⁺ = x⁻ − M⁻¹Kᵀ(KM⁻¹Kᵀ)⁻¹(Kx⁻ − d)` at the event; saltation with the projection as reset map
- [ ] 8.2 Tests: ideal DCM buck orbit within 1e-9 of 8.9963970815477 V and diode-off instant within 1e-12·T of `0.79983739563545·T`; Floquet multipliers 0 and 0.9889768767 within 1e-8; capacitor-loop and cut-set records equal to the implicit engines' within 1e-9
