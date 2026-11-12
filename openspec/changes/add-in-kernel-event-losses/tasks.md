## 1. Tables in the kernel
- [ ] 1.1 `thermal/loss_table.hpp`: `LossTable3D` and `ConductionTable` (ascending axes, single-point axes, multilinear, linear extrapolation, clamp at 0); refusals by name
- [ ] 1.2 Bindings and stubs; `LossTable.to_kernel()` in `python/pulsim/loss_tables.py`
- [ ] 1.3 Test: 10 000 random `(v, i, T_j)` points, inside and outside the grid, equal `LossTable.__call__` within 1e-12; a table `E = 1 mJ·(1 + 0.01·(T − 25))` on `tj = [25, 125]` returns 1.5 mJ at 75 °C and 2.5 mJ at 175 °C

## 2. Thermal port and event energy on `pwl`
- [ ] 2.1 `thermal/device_losses.hpp`: per-device spec (junction node or fixed `Tj`, tables, conduction mode), pending-heat accumulator, event record
- [ ] 2.2 `CircuitBuilder::set_device_losses` with validation by name (unknown device, unsupported kind, both or neither of `junction` / `Tj`, `conduction="table"` without `V_on`, junction node with a purely resistive path to a temperature source)
- [ ] 2.3 Reading rule at each mask change in `solver/run_transient.hpp` (gate-commanded and kernel-decided events), one helper shared by all callers
- [ ] 2.4 Deposition over the following step; compensated energy accumulator; `res.loss_events`, `res.device_energy` in `solver/result.hpp` and the bindings
- [ ] 2.5 Test (adiabatic): `C_th = 0.01 J/K`, `R_th = 1e9 K/W`, `E_on = E_off = 1 mJ`, 10 events → `ΔT_j = 1.000 K` within 1e-6; injected energy equals the table sum within 1e-12
- [ ] 2.6 Test (open loop): `R_th = 1 K/W`, 10 kHz, duty 0.5, `Tj = 25` → time-weighted period mean of the junction node after 100 ms is 45.0 °C ± 0.05 K
- [ ] 2.7 Test (closed loop): `junction=`, `E(T) = 1 mJ·(1 + 0.01·(T − 25))` → 50.0 °C ± 0.1 K
- [ ] 2.8 Conduction heat `electrical`: 10 A, `V_f = 0.7 V`, `g_on = 20 S` → 12.0 W and `T_j = 37.0 °C` ± 0.01 K; `table`: `V_on = 1.0 + 0.002·(T − 25)` V at 10 A → 35.20 °C ± 0.01 K, with `E_cond_electrical` and `E_cond_table` both reported
- [ ] 2.9 Parity test: a three-phase VSI with fixed `Tj = 125`; kernel `E_on`, `E_off` per device equal `device_loss_summary(switch_specs=...)` within 1e-9
- [ ] 2.10 Diode recovery: a PWL diode with an `E_off` (E_rec) table deposits at each natural turn-off; count equals the diode's commutation count
- [ ] 2.11 No-losses identity: a circuit without `set_device_losses` is bit-identical to the build before this change

## 3. Other engines, YAML, parts
- [ ] 3.1 Hook on the commutation callback of `add-unified-event-queue`; `trbdf2` pulse deposition without refining the thermal kink
- [ ] 3.2 `dsed` state-jump deposition and conduction heat between events, or the conformance refusal
- [ ] 3.3 Engine agreement test: closed-loop mean `T_j` within 0.2 K across `pwl`, `trbdf2`, `dsed`; parity at fixed `Tj` within 1e-6 on the variable-step engines
- [ ] 3.4 Conformance rows (`add-engine-conformance-matrix`): `device_losses[event]`, `device_losses[conduction]` × {pwl, trbdf2, dsed}; no silent cell
- [ ] 3.5 YAML `losses:` block with strict keys; loader tests (valid block; unknown key refused naming the key; `junction` and `Tj` together refused)
- [ ] 3.6 `Part.attach_losses(builder, device, *, T_amb, heatsink_node=None, allow_synthetic=False)`: builds the part's Cauer chain and sets its tables; synthetic part refused without the flag
- [ ] 3.7 Test with a PLECS XML fixture (`import_plecs_xml` → `attach_losses`): the on-state table reaches the kernel uncollapsed (heat at two currents and two temperatures equals the table within 1e-9)

## 4. Electrical feedback (optional)
- [ ] 4.1 Re-fit `V_f`, `1/g_on` from `V_on(i, T_j)` when `|T_j − T_ref| > dT_refresh`; parametric refactor; refresh count in the result
- [ ] 4.2 Test: two paralleled devices with unequal `R_th` share current unequally with feedback on and equally with it off

## 5. Python summaries read kernel records
- [ ] 5.1 `device_loss_summary` and `device_thermal_summary` use `res.loss_events` when present; identical output at fixed `Tj` (regression on the existing loss tests)

## 6. Documentation
- [ ] 6.1 `docs/losses-and-thermal.md` (online versus post-hoc, the reading rule table, the energy-balance note), `AI_GUIDE.md` (which path to use, `Tj` versus `junction`), `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` (event energy is not drawn from the circuit; Newton devices excluded), `CHANGELOG.md`
- [ ] 6.2 Tutorial: PLECS XML → closed-loop T_j on a buck

## 7. Verification
- [ ] 7.1 `ruff check`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check
