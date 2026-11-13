# Decisions taken during implementation (owner away)

Each entry: question → option taken → why → how to reverse.

## D1. Where the composites live

- Question: tasks 1.1, 2.1 and 3.2 name the C++ `CircuitBuilder`; the
  composites need no new kernel code, and the C++ builder, bindings and
  YAML loader were being edited by other changes at the same time.
- Option: `python/pulsim/passives.py` wraps `CircuitBuilder.add_capacitor`
  and `add_inductor` with keyword-only `esr`, `esl`, `c_par`, `rac`, and
  installs `add_transmission_line` (same mechanism as
  `_builder_ergonomics` uses for `add_shockley_diode`). With every new
  parameter zero the wrapper calls the native method with the same
  arguments, so the result is the native one bit for bit.
- Why: one implementation, no rebuild, nothing to merge in shared C++.
- Reverse: port the expansion into `circuit_builder.hpp` with the same
  sub-element names, bind it, delete the wrappers; tests unchanged.

## D2. YAML keys `esr`, `esl`, `c_par`, `rac`; type `transmission_line`

- Option: not done (task 1.8 and 3.2's YAML half left unticked). The
  loader is C++ and needs D1's reversal first.
- Risk recorded: today's loader does not check per-device keys, so a
  YAML `capacitor` with `esr: 0.05` loads and silently ignores it.
  That predates this change; it is listed for the YAML follow-up.

## D3. Where the ESR loss is reported (task 1.7)

- Option: as the resistor row `{name}.esr` of `device_loss_summary`
  (the sub-element is an ordinary resistor). No change to `losses.py`.
- Why: the row is named after the component and its value is exactly
  `i²·esr`; folding it into a `{name}` row would need a composite
  registry.
- Reverse: aggregate `{name}.*` rows in `losses.py`.

## D4. ESR as one number (Open Question 3)

- Option: one number (the design's text).

## D5. `c_par` on the inductor now (Open Question 4)

- Option: yes, as designed.

## D6. Bergeron line (section 5, Open Questions 1 and 2)

- Option: not implemented; `model="bergeron"` is refused by name with a
  pointer to `model="pi"`. `model="pi"` is the default of the Python
  call (the design's default is `bergeron`, which does not exist yet).
- Why: the design makes it a kernel device (`models/transmission_line.hpp`,
  `pwl/transmission_line_history.hpp`, a pool kind, a per-step history
  source in `run_transient`, a step cap in `trbdf2`). Built from
  today's `b_extra_fn` + `step_observer` hooks it would be a Python
  observer on `pwl` only — the pattern `add-mna-native-machines` is
  retiring — and invisible to `trbdf2`.
- Reverse: implement section 5 as designed; flip the default.

## D7. π line return conductor

- Option: shunt capacitances go to `k_neg`; when `m_neg != k_neg` a
  0 V source `{name}.ret` joins them (the return conductor's impedance
  is lumped into the forward conductor; differential mode only).

## D8. `fit_rac_ladder` R_dc

- Option: when `R_dc` is not given it is a non-negative least-squares
  unknown alongside the cells; the fit minimises the relative error.

## D9. DC-bias capacitance table (section 4)

- Option: not started; it depends on the `ChargeTable` /
  charge-law `NonlinearCapacitor` of `add-power-mosfet-datasheet-device`,
  being implemented by another change.

## D10. Default `n_cells` (task 3.4)

- Measured on the round-wire skin-effect curve over two decades around
  the skin corner (R_ac/R_dc from 1.0002 to 1.853): 1 cell 39 %
  (refused), 2 cells 1.75 %, 4 cells 0.60 %, 6 cells 0.18 %. The
  non-negative fit keeps only 2 non-zero cells of the 4.
- Option: keep the design's default `n_cells = 4` (2 cells reach 5 % on
  two decades; 4 leave margin for wider bands at no cost, since unused
  cells come out zero and are dropped).
- Reverse: change the default.

## D11. Tolerances that differ from the validation plan

- Phasor test (task 1.4): 1e-3 as designed; measured 5.5e-5, 3.3e-9 and
  7.1e-5 at 20, 159.15 and 800 kHz (200 steps per period).
- ESR step (task 1.5): measured with a ±1000 V pulse behind 1 kΩ (no
  pulse current source yet), expected `esr·2000/(1000 + esr)`, 1e-6.
- ESR loss (task 1.7): the design's 1e-9 is a transient average; the
  test bound is 1e-6 (measured 50.000 mW to 6.5e-11 relative).
- Ladder resistance (task 3.3): the design's 10.009 / 55.00 / 99.11 mΩ
  are rounded; they are checked to 2e-4, and the AC sweep against the
  ladder's analytic impedance to 1e-9.
- `c_par` resonance (task 1.6): 5.033 MHz is the rounded value (1e-3);
  the peak of the AC sweep equals `1/(2π√(LC))` to 1e-6 (with
  `esr = 0.1 Ω` so the peak is finite).
