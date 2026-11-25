## 1. Tables in the kernel
- [x] 1.1 `thermal/loss_table.hpp`: `LossTable3D` and `ConductionTable` (ascending axes, single-point axes, multilinear, linear extrapolation, clamp at 0); refusals by name
- [x] 1.2 Bindings and stubs; `LossTable.to_kernel()` in `python/pulsim/loss_tables.py`
- [x] 1.3 Test: 10 000 random `(v, i, T_j)` points, inside and outside the grid, equal `LossTable.__call__` within 1e-12; a table `E = 1 mJ·(1 + 0.01·(T − 25))` on `tj = [25, 125]` returns 1.5 mJ at 75 °C and 2.5 mJ at 175 °C

## 2. Thermal port and event energy on `pwl`
- [x] 2.1 `thermal/device_losses.hpp`: per-device spec (junction node or fixed `Tj`, tables, conduction mode), pending-heat accumulator, event record
- [x] 2.2 `CircuitBuilder::set_device_losses` with validation by name (unknown device, unsupported kind, both or neither of `junction` / `Tj`, `conduction="table"` without `V_on`, junction node with a purely resistive path to a temperature source)
- [x] 2.3 Reading rule at each mask change in `solver/run_transient.hpp` (gate-commanded and kernel-decided events), one helper shared by all callers
- [x] 2.4 Deposition over the following step; compensated energy accumulator; `res.loss_events`, `res.device_energy` in `solver/result.hpp` and the bindings
- [x] 2.5 Test (adiabatic): `C_th = 0.01 J/K`, `R_th = 1e9 K/W`, `E_on = E_off = 1 mJ`, 10 events → `ΔT_j = 1.000 K` within 1e-6; injected energy equals the table sum within 1e-12
- [x] 2.6 Test (open loop): `R_th = 1 K/W`, 10 kHz, duty 0.5, `Tj = 25` → time-weighted period mean of the junction node after 100 ms is 45.0 °C ± 0.05 K
- [x] 2.7 Test (closed loop): `junction=`, `E(T) = 1 mJ·(1 + 0.01·(T − 25))` → 50.0 °C ± 0.1 K
- [x] 2.8 Conduction heat `electrical`: 10 A, `V_f = 0.7 V`, `g_on = 20 S` → 12.0 W and `T_j = 37.0 °C` ± 0.01 K; `table`: `V_on = 1.0 + 0.002·(T − 25)` V at 10 A → 35.20 °C ± 0.01 K, with `E_cond_electrical` and `E_cond_table` both reported
- [x] 2.9 Parity test: a three-phase VSI with fixed `Tj = 125`; kernel `E_on`, `E_off` per device equal `device_loss_summary(switch_specs=...)` within 1e-9
- [x] 2.10 Diode recovery: a PWL diode with an `E_off` (E_rec) table deposits at each natural turn-off; count equals the diode's commutation count
- [x] 2.11 No-losses identity: a circuit without `set_device_losses` is bit-identical to the build before this change

## 3. Other engines, YAML, parts
- [x] 3.1 Hook on the commutation callback of `add-unified-event-queue`; `trbdf2` pulse deposition without refining the thermal kink (the shared LossTracker at the trbdf2 commit / landing / gate-edge sites and its event limits; DECISIONS 13)
- [x] 3.2 `dsed` state-jump deposition and conduction heat between events, or the conformance refusal (refusal by name, DECISIONS 8)
- [x] 3.3 Engine agreement test: closed-loop mean `T_j` within 0.2 K across `pwl`, `trbdf2`, `dsed`; parity at fixed `Tj` within 1e-6 on the variable-step engines (test_exact_junction_loop::test_closed_loop_mean_junction_temperature_agrees_across_the_three_engines; add-exact-propagator-strategy DECISIONS session 25)
- [x] 3.4 Conformance rows (`add-engine-conformance-matrix`): `device_losses[event]`, `device_losses[conduction]` × {pwl, trbdf2, dsed}; no silent cell (pwl honours; trbdf2, dsed, bdf1, auto refuse by name: `test_device_losses_cell`, DECISIONS 8)
- [x] 3.5 YAML `losses:` block with strict keys; loader tests (valid block; unknown key refused naming the key; `junction` and `Tj` together refused)
- [x] 3.6 `Part.attach_losses(builder, device, *, T_amb, heatsink_node=None, allow_synthetic=False)`: builds the part's Cauer chain and sets its tables; synthetic part refused without the flag
- [x] 3.7 Test with a PLECS XML fixture (`import_plecs_xml` → `attach_losses`): the on-state table reaches the kernel uncollapsed (heat at two currents and two temperatures equals the table within 1e-9)

## 4. Electrical feedback (optional)
- [ ] 4.1 Re-fit `V_f`, `1/g_on` from `V_on(i, T_j)` when `|T_j − T_ref| > dT_refresh`; parametric refactor; refresh count in the result
- [ ] 4.2 Test: two paralleled devices with unequal `R_th` share current unequally with feedback on and equally with it off

## 5. Python summaries read kernel records
- [x] 5.1 `device_loss_summary` and `device_thermal_summary` use `res.loss_events` when present; identical output at fixed `Tj` (regression on the existing loss tests) — seen 2026-11-25 on 527e644d + this port (DECISIONS 18): at fixed `Tj` the summary read from the kernel records equals the post-hoc `switch_specs` path on a twin circuit without a port within 1e-9 on pwl, pwl with `discontinuity_restart=True` and dsed, 1e-5 on trbdf2 (whole run and a window; both summaries); the 22 existing loss / thermal test files pass unchanged (292 tests); a 1e-6 error injected into the kernel read fails 15 of the 28 new tests

## 6. Documentation
- [x] 6.1 `docs/losses-and-thermal.md` (online versus post-hoc, the reading rule table, the energy-balance note), `AI_GUIDE.md` (which path to use, `Tj` versus `junction`), `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` (event energy is not drawn from the circuit; Newton devices excluded), `CHANGELOG.md`
- [x] 6.2 Tutorial: PLECS XML → closed-loop T_j on a buck — `docs/tutorials/11-electrothermal-plecs-buck.md` (synthetic XML inline; ported from gemini/wip-2026-11-24 edf6efc9 without its electrical-feedback claims, since 4.1 is open); seen 2026-11-25 on 527e644d: every block runs in `test_doc_snippets` (junction 20 K above ambient, closed-loop energies above the frozen-`Tj` run, heat ledger = recorded energy within 1e-12)

## 7. Verification
- [x] 7.1 `ruff check`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check

## 8. The exact steady-state route (period post-processing, no engine file)
- [x] 8.1 The reading rule on the periodic orbit: every commutation of the steady period read from the period map's left and right limits (turn-on: `v` before, `i` after; turn-off the reverse); a diode's `E_off_table` in `switch_specs` is its recovery, read at each turn-off; the sampled `device_loss_summary` refuses a diode there by name
- [x] 8.2 Parity at fixed `T_j = 125 °C`: a three-phase SPWM inverter with dead time, kernel `E_on` / `E_off` per switch equal `device_loss_summary` on a settled `dsed` period within 1e-9 (measured 1.8e-13); each diode's recovery count equals its turn-off count (`test_design_thermal_kernel.py`)
- [x] 8.3 The junction fixed point `T_j = T_amb + R_th P(T_j)` in the kernel (`dsed/period_thermal.hpp`), runaway detected by `R_th dP/dT_j >= 1`; oracles: `(T_amb + R_th P0 (1 - 25 a)) / (1 - R_th P0 a)` within 0.01 K (measured 1e-12) and a runaway at `R_th P0 a = 1.2` (`test_period_thermal.cpp`, `test_design_thermal_kernel.py`)
- [x] 8.4 Conduction heat from a `ConductionTable` linear in `i` in closed form over the on-state current, or the circuit's own closed-form integrals; tables that are not closed-form refused by name
