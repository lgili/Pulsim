## 1. Law and stamp
- [ ] 1.1 `models/gated_diode.hpp`: `Params {g_on, g_off, V_f, V_th, I_H, mode, reverse_blocking}`; `turn_on_voltage`, `branch_current`, `decide(c, g, v)`, `crossing_signal(c, g, v)`; validation by name
- [ ] 1.2 C++ law tables: every `(mode, c, g)` row of the design table at `V_f ± ε`, `I_H ± ε`; `reverse_blocking=False`, `V_f = 0` gives `c = g`
- [ ] 1.3 Stamp through `stamp_switch_fixed(..., v_drop_on)`; C++ stamp-delta test (`J` unchanged between modes, `b` gains `∓g_on·V_f` only when `c = 1`)

## 2. Pool, mask and record
- [ ] 2.1 `DevicePool::add_gated_diode`; event state carries `(c, g)`; effective-mask construction (commanded bit → `g`, cache key ← `c`)
- [ ] 2.2 Result records the commanded and the effective state per step under distinct names
- [ ] 2.3 Test: `SwitchStateMask(num_switches)` with `.set(i, True)` for the gate; control case proves forward conduction before any blocking assertion

## 3. IGBT
- [ ] 3.1 `CircuitBuilder::add_igbt(..., V_ce0, reverse_blocking, V_th)`; `FutureWarning` when `reverse_blocking` is left at its (stage-1) default
- [ ] 3.2 Test: +10 A, `V_ce0 = 0.9 V`, `R_on = 10 mΩ` → `v_CE = 1.000 V`, 10.0 W (DC op, `pwl`, `trbdf2`), 1e-9
- [ ] 3.3 Test: −100 V applied, gate on → `i = −1e-7 A`; gate off with +100 V → `i = +1e-7 A`
- [ ] 3.4 Test: `reverse_blocking=False`, `V_ce0 = 0` bit-identical to the pre-change `add_igbt` on a three-phase VSI (states, mask record, loss summary)
- [ ] 3.5 Test: hold at `V_ce0 ± ε` with the gate on, `min(i) ≥ −1e-9 A` on every engine
- [ ] 3.6 Preflight / abort message names the IGBT when an inductive current has no path

## 4. Decision sites
- [ ] 4.1 `pwl`: event iteration, sub-step watcher, seeding, DC operating point
- [ ] 4.2 `trbdf2`: crossing signal, held tolerance, forbidden-state check
- [ ] 4.3 `dsed`: predicates and settle
- [ ] 4.4 Per-site test: gate held off under forward bias → `|i| ≤ g_off·|v|` on `pwl`, `trbdf2`, `dsed`
- [ ] 4.5 Conformance rows (`add-engine-conformance-matrix`): `igbt[gated]`, `thyristor`, `gto`, `triac` × {pwl, trbdf2, dsed, DC, AC}

## 5. Thyristor, GTO, TRIAC
- [ ] 5.1 `add_thyristor`, `add_gto`, `add_triac`; `switch_indices_of`
- [ ] 5.2 Half-wave rectifier, 100 V peak, 10 Ω, 60 Hz, dt = 10 µs: mean 29.699 V (α = 30°) and 15.915 V (α = 90°) within 0.5 %; load voltage never below −1e-6 V (the case in which the Python closure gives −2.140 V)
- [ ] 5.3 Latch test: 100 µs gate pulse into an RL load → conduction ends at the current zero
- [ ] 5.4 Holding-current test: `I_H = 0.05 A`, gate removed → OFF at the first accepted state with `i ≤ 0.05 A`; with the gate held on, conduction continues down to `i = 0`
- [ ] 5.5 `sources/phase_control_fn.hpp` and its binding
- [ ] 5.6 Six-pulse bridge, 400 V line-to-line, 60 Hz, 20 A constant-current load, gate edges on the grid: 540.2, 467.8, 270.1 V at α = 0°, 30°, 60° within 0.5 %; with 1 mH per phase at α = 30°: 460.6 V within 0.5 %
- [ ] 5.7 Engine agreement on 5.6 within 0.5 % (`pwl`, `trbdf2`; `dsed` with a resistive-inductive load, since it rejects a current source)
- [ ] 5.8 TRIAC: AC phase control on a resistive load, RMS load voltage `V_rms·sqrt(1 − α/π + sin(2α)/(2π))` = 69.68 V for 100 V peak, α = 30°, within 0.5 %

## 6. Python and YAML
- [ ] 6.1 `components()` exports the new parameters; `_diode_state` law extended to the gated modes; `res.i`, `device_loss_summary` (`V_ce0·I + R_on·I²` = 10.0 W at 10 A), `device_thermal_summary`
- [ ] 6.2 Bindings, stubs
- [ ] 6.3 YAML: `igbt` keys `V_ce0`, `reverse_blocking`; types `thyristor`, `gto`, `triac`; strict validation naming the device
- [ ] 6.4 `make_thyristor_switch_fn` and `add_thyristor_to_chain` emit `DeprecationWarning` naming `add_thyristor`

## 7. Documentation
- [ ] 7.1 `AI_GUIDE.md` (device table, the freewheel-path trap, the stopgap series composite), `docs/losses-and-thermal.md` (IGBT conduction law), `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` (non-goals; the Python closure's `i_holding = 0` failure until removal), `CHANGELOG.md` (staged BREAKING)

## 8. Default flip (next release)
- [ ] 8.1 `reverse_blocking=True` default; update helpers and examples that relied on reverse conduction

## 9. Verification
- [ ] 9.1 `ruff check`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check
