## 1. Stamps and kinds
- [x] 1.1 `stamping/stamp_controlled.hpp`: `stamp_vccs`, `stamp_cccs`, `stamp_ccvs`, `stamp_gyrator`; C++ stamp tests on a 4-node matrix (entries and signs of the design table)
- [x] 1.2 `models/controlled_sources.hpp` (`VCCS`, `CCCS`, `CCVS`, `Gyrator` params and traits); `DevicePool` kinds; `assemble` and `dc_assemble` call the stamps
- [x] 1.3 `CircuitBuilder::add_vccs / add_ccvs / add_cccs / add_gyrator`; controlling-branch resolution; refusals by name (non-finite parameter; controlling branch without a current unknown, listing the valid kinds)
- [x] 1.4 DC tests: VCCS `g_m = 2 mS`, 5 V, 1 kΩ → +10.000 V; CCVS 10 mA, 100 Ω → 1.000 V; CCCS 1 mA, β = 50, 100 Ω → 5.000 V; gyrator `g = 0.01 S` with 50 Ω → 200 Ω; all 1e-12

## 2. Transient and AC
- [x] 2.1 (pwl equals the inductor to 1e-9; 63.21 mA is met on trbdf2/dsed — pwl reads 63.19 mA, its first-order start, DECISIONS 7) `pwl`: gyrator + 1 µF against a 10 mH inductor, 1 V step through 10 Ω, same grid → 1e-9; `i(1 ms) = 63.21 mA`
- [x] 2.2 Gyrator power `v_1 i_1 + v_2 i_2` ≤ 1e-12·`|v_1 i_1|` at every step
- [x] 2.3 (trbdf2 within 1.5e-8 of the analytic current; against pwl it converges at first order, DECISIONS 7) `trbdf2`: same cases, agreement with `pwl` within 1e-6
- [x] 2.4 AC sweep (`analysis/mna_sweep.hpp`): gyrator + 1 µF → 62.83 Ω, +90° at 1 kHz within 1e-9; VCCS transconductance stage gain `g_m·R` flat to 1e-9
- [x] 2.5 (trbdf2 and dsed; pwl steps exactly 3·dt after its first step, DECISIONS 7) A VCCS integrator: unit-gain VCCS into 1 F from a 3 V source → `v(t) = 3·t` within 1e-9 (the `dθ/dt = ω` building block)

## 3. Waveform current sources
- [x] 3.1 `models/pwl_waveform.hpp` (`PwlWaveform`: value, breakpoints, validation); reuse of the sine and pulse evaluators for node-row injection
- [x] 3.2 `add_sine_current_source`, `add_pulse_current_source`, `add_pwl_current_source`
- [x] 3.3 Tests: sine into 10 Ω equals `10·i(t)` at every grid point (1e-12); pulse equals the pulse voltage source across 1 Ω (1e-12); PWL triangle 0 → 1 A → 0 over 2 ms into 1 mF → 1.000 V (1e-9 on `pwl` with knots on the grid)
- [x] 3.4 Breakpoints on `trbdf2` (local clamp until `update-trbdf2-error-control`): the PWL triangle with knots off any natural step → 1.000 V within 1e-6
- [x] 3.5 Refusal tests: one knot, decreasing `t`, NaN value, `periodic` with unequal end values — each names the source
- [x] 3.6 DC operating point uses the value at `t = 0`

## 4. Surface
- [x] 4.1 Bindings and stubs for the seven methods; `components()` entries
- [x] 4.2 `res.i` for the new kinds (`port=` for the gyrator); test against a 1 mΩ series resistor's current (1e-9)
- [x] 4.3 YAML types `vccs`, `ccvs`, `cccs`, `gyrator`, `sine_current_source`, `pulse_current_source`, `pwl_current_source` (`t`/`i` lists or `file:`), strict keys; loader tests including an unknown key and an unknown `control` name
- [x] 4.4 YAML / Python parity test: each kind built both ways gives identical `components()`

## 5. Conformance
- [ ] 5.1 (the matrix of `add-engine-conformance-matrix` is not in the tree yet; the rows are in CHANGELOG and `docs/known-limitations.md`) Rows in the matrix of `add-engine-conformance-matrix`: seven kinds × {pwl, trbdf2, dsed, DC, AC}
- [x] 5.2 (the 25 v2 example netlists, 43 runs bit-identical, 7 identical refusals; the v1 benchmark set does not load, DECISIONS 9) Existing-circuit identity: the benchmark set's results are bit-identical
- [x] 5.3 (dsed runs all seven, DECISIONS 6) `dsed`: determine whether `dsed/builder_adapter.hpp` can take the constant stamps and the time-varying node sources; implement, or refuse through the conformance error; record the outcome in the matrix

## 6. Documentation
- [x] 6.1 (AI_GUIDE got the signatures and the terminal convention but no per-kind example: it sits at 32744 of 32768 bytes; each kind's law and convention is in `docs/api-reference.md`, and the executed examples are `python/tests/test_controlled_sources.py`) `AI_GUIDE.md` (source table with the terminal convention and one example each), `docs/api-reference.md`, `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` (no behavioural sources; `dsed` status), `CHANGELOG.md`
- [x] 6.2 Reconcile the two comments that describe the current-source direction (`models/current_source.hpp:9-10` and `stamping/stamp_current_source.hpp:50-56`) with the measured convention

## 7. Verification
- [x] 7.1 `ruff check`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check
