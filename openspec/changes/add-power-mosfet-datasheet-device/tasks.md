## 1. Charge laws
- [x] 1.1 `models/charge_table.hpp`: `ChargeTable` from triples `(v, Q, C)` and from pairs `(v, C)`; Hermite `charge(v)`, exact `capacitance(v)`, linear tails, Fritsch–Carlson check and clamp; refusals by name (fewer than 3 knots, no `v = 0` knot, `v` not increasing, `C ≤ 0`)
- [x] 1.2 C++ test: junction law C0 = 2 nF, V0 = 25 V, m = 0.5 at 64 knots with exact slopes → Q(400 V) = 312.31 nC within 1e-4, C(400 V) = 485.07 pF within 1e-3; `C(v)` equals the finite difference of `Q(v)` within 1e-7
- [x] 1.3 C++ test: `Q(v)` strictly increasing at 1 pV spacing around 0 V, 1 V and 400 V
- [x] 1.4 `vdmos_cgd` closed form with the stable `ln cosh`; C++ test against the ten ngspice points of `probes/cgd.cir` (1e-6) and `C = dQ/dv` (1e-9)
- [x] 1.5 `NonlinearCapacitor::Params` law variant (`junction | table | vdmos_cgd`); history and refresh call `charge` / `capacitance` through it; every existing `NonlinearCapacitor` test bit-identical
- [ ] 1.6 Charge-conservation test: 0 → 400 V → 0 ramp through 1 kΩ, ∮ i dt = 0 within 1e-9 relative on `pwl` and `trbdf2`, for each law
  - Not ticked: implemented (`test_charge_is_conserved_over_a_closed_cycle`), but 1e-9 is not reachable: measured pwl 2.2e-8..8.9e-8 for all three laws (the 2.0 junction law included), trbdf2 6.6e-5..2.6e-4 (output-grid trapezoid vs TR-BDF2 quadrature); pinned 1e-6 / 2e-3. Owner to accept the tolerance (DECISIONS.md 17).

## 2. Composite device
- [x] 2.1 `CircuitBuilder::add_power_mosfet` (internal nodes only when the element is non-zero; validation by name: `R_d ≥ 0`, `R_g ≥ 0`, `L_s ≥ 0`, `C_gs > 0`, the temperature factor `r(T) > 0`)
  - `C_gs ≥ 0` instead of `> 0`, with a "no charge" refusal when R_g > 0 and both gate capacitances are 0 (DECISIONS.md 5).
- [x] 2.2 `components()` reports one `power_mosfet` entry with its parameters and sub-element names
- [x] 2.3 Binding, `ChargeTable` binding, stubs; YAML type `power_mosfet` (strict keys)
- [x] 2.4 Python test: the composite with linear `C_gs = 1 nF`, `C_gd = 100 pF`, `R_g = 10 Ω` reproduces `p4_dpt.py` row 3 (E_on 136.14 µJ, E_off 158.96 µJ, plateau 64.8 ns) within 1e-9
- [x] 2.5 `pulsim.double_pulse_test`; integrals with `np.trapezoid` on the result's own grid; test: plateau within 10 % of `Q_gd / I_g` at R_g = 10 Ω and 47 Ω (62.9 ns and the 47 Ω hand value)
  - Measured (trbdf2, seeded off-state start, DECISIONS.md 16): 64.57 ns vs 62.94 ns at 10 Ω, 300.56 ns vs 295.80 ns at 47 Ω.
- [x] 2.6 Engine agreement: `pwl` against `trbdf2` within 0.5 % in E_on and E_off, with and without the Lauritzen body diode
  - Measured 0.058 % / 0.079 % (E_on / E_off), same with the Lauritzen body diode; pinned 0.2 %.
- [x] 2.7 Third-quadrant test: 10 A reverse with the gate on gives `v_DS = −I·R_DS(on)`; with the gate off the body diode carries it
- [x] 2.8 `T_j` test: at 0.1 A and the datasheet V_GS, `v_DS(125 °C) / v_DS(25 °C)` equals `r(125 °C)` = 1.700 (`tc1 = 5e-3`, `tc2 = 2e-5`) within 1e-3
- [ ] 2.9 Conformance rows (`add-engine-conformance-matrix`): `power_mosfet` × {pwl ok, trbdf2 ok, dsed refused, DC ok, AC refused}; `nonlinear_capacitor[table]`, `nonlinear_capacitor[vdmos_cgd]` likewise
  - Not ticked: the conformance matrix belongs to Phase 0 (`add-engine-conformance-matrix`). Rows to add: `power_mosfet`, `nonlinear_capacitor[table]`, `nonlinear_capacitor[vdmos_cgd]` × {pwl ok, trbdf2 ok, dsed refused, DC ok, AC refused}.

## 3. VDMOS import
- [x] 3.1 `_vdmos` builds `add_power_mosfet` when the card carries any of CGS, CGDMAX, CGDMIN, A, CJO, TT, RG; mapping table in the docstring
- [x] 3.2 `charge="drop"` restores the Level-1 device; the summary warning lists what was kept and what is still dropped (subthreshold)
- [x] 3.3 Test: a card with CGS = 1 nF, CGDMAX = 1 nF, CGDMIN = 50 pF, A = 0.5, RG = 3 Ω imports with those values in `components()`; an existing capacitance-free card imports bit-identically
- [x] 3.4 Fix the `T_M / TT` ratio: sweep it against ngspice Q_rr on a diode-recovery deck, record the measured ratio and error in the importer docstring
  - `T_M = 0.01·TT`: against `pm_diode_recovery.cir` I_rrm −0.09 %, Q_rr +0.11 % (DECISIONS.md 15).

## 4. Fitters and parts
- [x] 4.1 `python/pulsim/datasheet.py`: `fit_charge_table`, `fit_mosfet_channel`, `fit_rdson_temperature`, `fit_body_diode_recovery`, `power_mosfet_from_datasheet`; each returns its residual and refuses on a residual above its documented limit
- [x] 4.2 Round-trip tests: curves generated from known parameters are recovered (K, V_T within 1e-6; `tc1`, `tc2` within 1e-9; table within `tol`)
- [x] 4.3 Part schema `power_mosfet` block; loader refuses intersecting `fitted_on` / `validated_on`, a missing datasheet reference, and a `validated` flag without a `double_pulse` block
- [x] 4.4 `pulsim.lib.power_mosfet(number)` and `Part.add_to(builder, name, drain, source, gate, T_j=)`
- [x] 4.5 PLECS "table and formula": import-time evaluation with a whitelisted expression grammar (no `eval`), `variables=` argument, provenance note; formulas using `lookup(...)` stay refused
- [ ] 4.6 First real parts (owner decision on which and where they live); CI acceptance: double pulse at the datasheet condition within the declared tolerance
  - Not ticked: owner decision; no part without datasheet provenance ships (owner note *shipped-parts-are-synthetic*). The CI loop over `validated` parts is in place and runs empty.

## 5. Table channel and single device (after `add-unified-device-interface`)
- [ ] 5.1 `ChannelTable` with monotonicity checks and the terminal-swap third quadrant; C¹ test at `V_DS = 0`
- [ ] 5.2 The power MOSFET as one f/q device; composite retired behind the same API; results within 1e-9 of step 2
- [ ] 5.3 Generic outputs: `res.i`, loss rows; AC from f and q
  - Not ticked (5.1–5.3): need `add-unified-device-interface` (Phase 3). Until then `res.i` / loss rows do not cover the composite; `components()` lists its sub-elements.

## 6. Documentation
- [x] 6.1 `AI_GUIDE.md` (device choice table, the time-step trap), `docs/losses-and-thermal.md`, `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` (non-goals), `CHANGELOG.md` (BREAKING importer default)
- [ ] 6.2 Tutorial: datasheet → fitters → double pulse
  - Not ticked: tutorial not written.

## 7. ngspice cross-validation
- [x] 7.1 Decks in `benchmarks/spice/ngspice`: Level-1 output and transfer curves, Shockley curve; tolerance stated per region (the sigmoid blend differs from SPICE's hard region boundaries within a few `1/kappa`)
- [x] 7.2 Deck: Coss junction law and `vdmos_cgd` capacitance sweeps (1e-6)
- [x] 7.3 Deck: VDMOS double pulse, same card on both sides; record measured differences and set the gate from them
  - Measured (trbdf2): E_on +0.15 %, E_off +0.07 %, plateau +0.35 %, dv/dt_on −0.08 %, I_rr −0.01 %; di/dt −0.6 % / +1.2 %. Gates: 0.5 % energies, 1 % plateau and I_rr, 5 % di/dt. The design's 5 %/5 %/3 % targets were not used for tuning.
- [ ] 7.4 Wire into the nightly job of `restore-reference-benchmark-harness`
  - Not ticked: the nightly job belongs to `restore-reference-benchmark-harness`; the ngspice-side tests skip cleanly without ngspice on PATH.

## 8. Verification
- [ ] 8.1 `ruff check`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check
