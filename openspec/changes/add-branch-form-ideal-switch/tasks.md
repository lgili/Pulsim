Reference values.

- RL (10 V, 1 Ω, 1 mH), ideal switch closed at t = 0:
  `i(10 ms) = 10·(1 − e^{−10}) = 9.99954600 A` (the conductance form
  with `g_on = 1e3` settles at 9.98955 A).
- Same circuit, switch opened at 5 ms with no other path:
  `i_L(5 ms⁻) = 9.93262053 A`, transferred flux
  `L·i = 9.93262053e-3 V·s`, energy `½·L·i² = 49.3285 mJ`. At the
  fixed step `h = τ/20` the run's own `i_L(t⁻)` is 9.93096483 A and the
  record is 49.3120 mJ (`probes/sdirk_restart_ideal.out` (1)).
- 1 µF at 10 V paralleled with 3 µF at 2 V through `R_on = 0`:
  4.000000 V, 6.0 µC transferred, 24.0 µJ lost.
- DCM buck of `numerics/p2_buck.py` (R = 50 Ω, D = 0.3) with an ideal
  switch and an ideal diode (`probes/ideal_dcm_buck.py`, 40 digits):
  orbit `v_C = 8.9963970815477 V` at the rising edge, diode off at
  `0.79983739563545·T`, `i_L = 0` in the idle interval.
- Same, fixed step `T/N`, SDIRK×3 restart, 300 periods from the orbit:
  `v_C` error −2.442e-6, −6.591e-7, −1.709e-7, −4.345e-8 V at
  N = 50, 100, 200, 400 (`probes/sdirk_restart_ideal.out` (3)).

## 1. Characterisation (no behaviour change)
- [ ] 1.1 Port `probes/current_kernel_goff0.py` as tests of today's behaviour: `g_on = inf` aborts with "numerically singular"; `g_off = 0` opened on 9.93 A gives ±397 V on `pwl` and above 1e10 V on `trbdf2`, with no warning
- [ ] 1.2 Port `probes/branch_form.py` and `probes/sdirk_restart_ideal.py` as dense references for the stamp, the cut-set, the capacitor loop and the ideal DCM buck

## 2. Validation of the conductance form
- [ ] 2.1 `add_switch`, `add_diode`, YAML: `g_on` finite and > 0, `g_off` finite and ≥ 0; error names the device and the field (**BREAKING** for invalid values)
- [ ] 2.2 Tests: each invalid value is refused at build time; every existing test circuit still builds

## 3. Branch stamp and linear algebra
- [ ] 3.1 `stamp_switch_branch` (D1) shared by the transient and DC assemblers; branch-current unknown allocation in the pool; structural entries kept when zero
- [ ] 3.2 Shared symbolic analysis across masks; partial pivoting per mask; rank-one row update for a single flip; counted fallback to a full refactor
- [ ] 3.3 Tests: on a four-switch bridge the pattern (column pointers and row indices) is identical for all 16 masks; every mask factorizes with residual `‖Jx − b‖/‖b‖ ≤ 1e-12` including all-ON with `R_on = 0`; a single-flip update equals a full refactor within 1e-12; pivot growth recorded
- [ ] 3.4 Test: with `R_on = 1e-3`, `g_off = 1e-9` the branch form reproduces the conductance form within 1e-10 on the buck, the bridge rectifier and the three-phase inverter

## 4. API, structural checks, first slice
- [ ] 4.1 `add_ideal_switch`, `add_diode(form='branch')`, YAML `form: branch`, `components()` fields, stubs
- [ ] 4.2 Per-mode structural pass (D3): islands bounded by OFF diodes marked as gauge; islands bounded only by open gated switches → `UnsupportedError(engine, 'floating_node', switch, 'switch')`; loops of closed ideal switches and voltage sources → `UnsupportedError(engine, 'ideal_voltage_loop', switch, 'switch')`; capacitors for the transient matrix, not for DC
- [ ] 4.3 Capability-table row for the branch form; `g_off = 0` and `R_on = 0` across a capacitor refused on the implicit strategies with `feature = 'dependent_states'` until group 5
- [ ] 4.4 Tests (`R_on = 0`, `g_off = 1e-9`): RL current 9.99954600 A at 10 ms within 1e-6 on `trbdf2` (rtol 1e-9) and, on the fixed engine at `h = τ/80`, max error at most 2e-3 A (prototype 1.51e-3); two open ideal switches in series with nothing between them raise `floating_node` naming the node and both switches; two closed ideal switches in parallel across a source raise `ideal_voltage_loop`; each check runs once per mode (5 runs for a converter visiting 5 modes)

## 5. Re-initialisation service on the implicit strategies (needs the SDIRK×3 restart of `add-discontinuity-restart-rule` and `update-trbdf2-post-event-restart`)
- [ ] 5.1 `ReinitService`: dependent-state detection from the D3 pass, projection, record `{t, devices, kind, Λ, ΔW}`, `on_impulse` policy, pulse sample marked; on the implicit strategies the projection is realised by the restart (the explicit projection is computed for the record and as an assertion)
- [ ] 5.2 Enable `g_off = 0` and `R_on = 0` across capacitors on `pwl` and `trbdf2`
- [ ] 5.3 Tests, inductor cut-set at `h = τ/20` on `pwl`: `i_L = 0` exactly from the first step after the event; the node-voltage samples are 0 within 1e-9 V from the second step on (no ±397 V ringing); the SDIRK quadrature of the node voltage over the first step equals `−L·i_L(t⁻)` within 1e-12 relative; the record reports `½·L·i_L(t⁻)²` within 1e-9 relative (49.3120 mJ at this step); `on_impulse='error'` raises naming the switch and the inductor
- [ ] 5.4 Tests, capacitor loop: both capacitors at 4.000000 V within 1e-12; record 6.0 µC and 24.0 µJ within 1e-9 relative; switch current below 1e-9 A from the second step after the event
- [ ] 5.5 Same two tests on `trbdf2`, and the explicit projection equals the restart's end state within the engine tolerance
- [ ] 5.6 Test: a consistent commutation (diode off at `i = 0` in DCM) produces no record

## 6. Branch-form diode with complementarity, outputs, losses (needs `add-complementarity-mode-selection` groups 2–6)
- [ ] 6.1 Crossing signals `i_s` (ON) and `w` (OFF); located crossings handed to the mode selector at every decision site
- [ ] 6.2 Ideal DCM buck on `pwl` with SDIRK×3: `v_C` error at N = 50, 100, 200, 400 within a factor 1.5 of the prototype values (−2.442e-6 … −4.345e-8 V), observed order ≥ 1.85 between N = 200 and 400, `i_L = 0` exactly in the idle interval
- [ ] 6.3 Ideal DCM buck on `trbdf2` (rtol 1e-9): orbit 8.9963970815 V within 1e-6
- [ ] 6.4 Losses (D8): conduction `R_on·i² + V_f·i` time-weighted; forced 10 A through an ideal diode with `V_f = 0.7` gives `v_D = 0.7 V` and 7.0 W; switching energies read from `E_on/E_off(i, v, T_j)` tables at each event — a buck with a tabulated `E_on = 10 µJ·(i/1 A)·(v/24 V)` at 0.5 A, 24 V books 5.0 µJ per turn-on within 1e-12 relative; impulse energy in its own category
- [ ] 6.5 `res.i` is the solved unknown; `device_loss_summary`, `_diode_state` updated

## 7. Measurement gate (before the flip)
- [ ] 7.1 Counters on the three-phase inverter and the buck, branch against conductance form on the implicit route: unknowns (+1 per switch), factorizations per run, partial-refactor fallbacks, solves per step; ns/step reported
- [ ] 7.2 On TR-BDF2 the smallest committed step in the DCM buck for both forms (8.26e-13 s with the resistor pair in the audit); the branch form's must not be the smaller
- [ ] 7.3 Gate for the flip: ns/step of the 12-switch inverter on the implicit route at most 1.5 times the conductance form's; if not met, the flip is held and the measurement recorded here

## 8. Exact route with ideal switches (needs `add-exact-propagator-strategy` groups 1–6)
- [ ] 8.1 Reduced state per mode (dimension `n_x − rank(K)`), explicit projection at mode entry, gauge directions from `add-complementarity-mode-selection` D10; saltation with the projection as reset map
- [ ] 8.2 Tests: ideal DCM buck orbit within 1e-9 of 8.9963970815477 V and diode-off instant within 1e-11·T of `0.79983739563545·T` (prototype: 4.5e-12 V, 2.0e-12·T); Floquet multipliers 0 and 0.9889768767 within 1e-8; capacitor-loop and cut-set records equal to the implicit strategies' within 1e-9

## 9. Default flip (in the single number-changing release)
- [ ] 9.1 Migration audit: every example, tutorial and test circuit built with the ideal default; every refusal listed with its remedy and fixed in the repository
- [ ] 9.2 Default form = branch, ideal (D6); `form='conductance'` warns; topology helpers follow the default
- [ ] 9.3 Re-baseline through the reference harness together with the other number-changing defaults; CHANGELOG lists each changed number and its cause

## 10. Docs and verification
- [ ] 10.1 Device guide ("the ideal switch: what the new refusals and records mean"), PSIM / PLECS porting note, AI_GUIDE trap list, CHANGELOG
- [ ] 10.2 ruff, pytest, ctest (`-j2`), `openspec validate --all --strict`, stub check
