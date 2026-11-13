## 1. Failing tests first
`python/tests/test_composite_diode_forward_drop.py` and C++ twins in `core/tests/builder/`.
- [x] 1.1 `add_mosfet("Q", drain, source, 1e-3, 1e9, True, 0.7)`, 10 A forced source → drain, gate off: `|v_SD| = 0.710 V ± 1e-6` (0.7 + 10/1e3) on `pwl`, `trbdf2` and at the DC operating point; instantaneous diode power 7.10 W. Fails today (0.010 V)
- [x] 1.2 Same, gate on: `|v_SD| = 0.010 V`, `Q_body` current below 1e-6 A, all loss in `Q` (0.0993 W over the 100 µs run). Passes today and must keep passing
- [x] 1.3 `V_F = 0.0`: `|v_SD| = 0.010 V` gate off; `V_F = 1.2`: 1.210 V; `V_F = -0.1` and `nan` raise naming the device and `V_F`
- [x] 1.4 `add_mosfet_with_body_diode(..., V_F=0.7, g_on_diode=20.0)`: 10 A → 1.200 V, `device_loss_summary` 12.0 W for `Q_body`
- [x] 1.5 `add_mosfet_level1(..., V_F=0.7)`, gate at 0 V, 10 A source → drain: 0.710 V (today 0.010 V); `V_F=1.0`: 1.010 V
- [x] 1.6 `add_igbt_level1(..., with_fwd=True, V_F=0.7)`, gate off, 10 A emitter → collector: 0.710 V; default keyword gives the same
- [x] 1.7 Turn-on voltage: ramp `v_SD` on an OFF body diode; it turns on at 0.700 V ± one step, not below (PWL composites: unchanged from today; level-1: was 0.5 V)
- [x] 1.8 Buck (24 V, D = 0.5, 100 kHz, 100 µH, 100 µF, 1.2 Ω, low side `add_mosfet` held off), 400 periods, time-weighted over the last 100: V_out = 11.640 V ± 0.005 V and `P(QL_body)` = 3.442 W ± 0.01 W (3.479 W is the 0–400-period average; see DECISIONS.md) on `pwl` (`dt = T/200`), `trbdf2` and `dsed`. Fails today (11.990 V, 0.052 W)
- [x] 1.9 H-bridge of four `add_mosfet` with dead time on `dsed`: settles (no "did not settle" error) or raises the named band error; record which, and decide Open Question 2 from it — settles (2402 steps / 200 periods); OQ2: no `V_th` keyword (DECISIONS.md)
- [x] 1.10 YAML: `mosfet_with_body_diode` with `V_F: 0.7` and `mosfet_level1` with `V_F: 1.0` load and reproduce 1.1 and 1.5

## 2. Builders
- [x] 2.1 `add_mosfet_with_body_diode`: `pool_.add_diode(b_body, g_on_diode, g_off_diode, /*V_th=*/0, /*V_f=*/V_F)`; validation by name; doc comments of it and of `add_mosfet`
- [x] 2.2 `add_mosfet_level1`, `add_igbt_level1`: `V_F` argument, diode built the same way; remove the 0.5 V literals
- [x] 2.3 `yaml/loader.hpp`: `V_F` for `mosfet`, `mosfet_with_body_diode`, `mosfet_level1`, `igbt_level1`
- [x] 2.4 Bindings: `V_F` keyword on the two level-1 builders; docstrings of all four say "conduction drop"

## 3. Callers
- [x] 3.1 Run the suite; for each failing expectation in the 24 Python and 5 C++ files of design.md, re-derive the value (`V_F·I` term stated in the test) or pin `V_F=0.0` where the test is about something else — one expectation moved (`test_mosfet_third_quadrant`, gate-off body diode: -10 mV → -(0.7 + i/g_on)); the C++ twin `test_circuit_builder.cpp` reads the diode params
- [ ] 3.2 Examples and `projects/`: re-run each script and notebook of design.md; update printed expectations and any asserted number
- [x] 3.3 `python/pulsim/spice_import.py` and `discovery.py`: check what they pass for the body diode and whether an imported `.model` forward voltage should feed `V_F` — no change: a MOS card has no forward-voltage parameter; with IS/JS the importer already builds a Shockley body diode, otherwise the level-1 default (now a 0.7 V drop) applies; `discovery.py` only lists names

## 4. Docs
- [x] 4.1 `CHANGELOG.md` [Unreleased]: BREAKING entry with the forced-current and buck figures and the migration snippet
- [x] 4.2 `python/pulsim/AI_GUIDE.md`: trap 1 (`add_mosfet(V_F=)` is now a drop), the signature lines, the executed snippet if it prints a diode voltage
- [x] 4.3 `docs/gotchas.md`, `docs/api-reference.md`, `docs/internals/layer2-v1-mosfet-igbt.md`, `docs/internals/layer8-yaml-parser.md`, tutorials 02 and 06
- [x] 4.4 `KNOWN_LIMITATIONS.md`: zero-width band of the composite diodes; `add_igbt` still has no knee and no diode
- [x] 4.5 Stubs via `scripts/generate_stubs.py`

## 5. Verification
- [x] 5.1 `ruff check`
- [x] 5.2 `pytest python/tests`
- [ ] 5.3 `ctest -j2`
- [x] 5.4 `openspec validate --all --strict`
- [x] 5.5 Stub check
- [x] 5.6 Re-run `p6b.py`, `probes/d9b_forced.py`, `probes/d9_buck_emulated.py`; the "today" rows must read as the "future" rows; then delete `probes/` (tests 1.1–1.8 replace it) — `d9b_forced.py` and `d9_buck_emulated.py` now print today = future (0.7100 V; 11.6403 V); `p6b.py` is not in this repository; probes deleted
