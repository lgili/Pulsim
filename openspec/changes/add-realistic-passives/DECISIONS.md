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

- Option: `esr`, `esl` (capacitor) and `esr`, `c_par` (inductor) are
  read by the C++ loader, which expands them into the same elements as
  the Python keywords (pinned equal by
  `python/tests/test_yaml_composites.py`). `rac:` is refused by name
  (a ladder is a fitted object; fitting at load needs Python). No
  `transmission_line` YAML type (section 5.10 belongs to the Bergeron
  line).
- Risk recorded: the loader still does not check other per-device
  keys; an unknown key is ignored without a warning. That predates
  this change.

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
- Update (integration, after both changes met): 4.1 and 4.3 done as
  `passives.odd_charge_table(v, Q, C)` (knots for v >= 0 mirrored into
  a `ChargeTable`; tails are the table's end-knot slopes). Open: 4.2
  `fit_mlcc_dc_bias`, 4.4 (ripple, charge conservation at 1e-9, which
  add-power-mosfet-datasheet-device measured as unreachable on pwl,
  2.2e-8, its DECISIONS 17), 4.5 YAML.

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

## D12. Bergeron line (section 5) stays deferred to the device interface

- Question (2026-11-18 session): implement section 5 now in today's
  drivers, or after `add-unified-device-interface` lands?
- Option: deferred; `model="bergeron"` keeps refusing by name.
- Why: the line is a stateful kind (a ring buffer of both ends, commit
  per accepted step, rollback on a retried step, snapshot block, a
  step cap and stage-time lookup on `trbdf2`). In today's tree a
  stateful kind is wired by hand into `run_transient.hpp` (20 sites for
  the IGBT tail history) and `trbdf2_transient.hpp` (10), plus the
  snapshot struct, the pool, the bindings and the dsed refusals; the
  device-interface branch (`feat/core-p3-device-interface`, slice 2,
  "How to add a device kind") replaces exactly those sites with one
  `StatefulKinds` list, and is being merged in batch 12. Wiring the line
  into the old sites now would conflict with that merge and be ported
  again right after. Unlike the Newton kinds the line also needs a
  linear history source in `b` (constant `1/Z0` in the cached matrix),
  which the interface's kind categories do not describe yet.
- Reverse: implement section 5 as a stateful kind on the interface
  (`pwl/transmission_line_history.hpp`, a `StateBlock`, an entry in
  `StatefulKinds`, the `b`-source hook); tasks 5.1–5.10 unchanged.

## D13. YAML `rac:` reads ladder cells only

- Option: `rac: {R_dc, L_hf, R: [...], L: [...]}` is expanded by the C++
  loader exactly as `add_inductor(rac=)` (same names, `components()`
  equal); `L_hf + ΣL` must equal `L` to 1e-9, cells > 0, `esr` and `rac`
  exclusive (as in Python). An `(f, R_ac)` table is refused by name.
- Why: fitting is a non-negative least squares in Python
  (`fit_rac_ladder`); the loader is C++. Write the fitted cells.
- Reverse: a Python pre-pass over the YAML that fits and rewrites `rac:`.

## D14. Charge conservation measured with a charge meter

- Question: task 4.4 asks for zero net charge over a ±20 V cycle to 1e-9;
  the output-grid trapezoid is not the engine's quadrature (trbdf2) and
  the 1e-12 S placeholder that `pwl/assemble.hpp` stamps on every
  NonlinearCapacitor node leaks (add-power-mosfet-datasheet-device
  DECISIONS 20).
- Option: the law in series with a 10 µF linear capacitor, no bleeder
  (`auto_regularize=False`); both are stamped in charge form by the same
  rule and carry the same current, so the meter's C·Δv is the law's
  charge under the engine's own rule; the placeholder leak at the meter
  node (1e-12 S × ∫v dt) is added to the balance. Measured 1.6e-15 (pwl)
  and 2.5e-15 (trbdf2) of Q(20 V).
- Reverse: remove the placeholder from the assembly (changes every
  result with a nonlinear capacitor) and drop the leak term.

## D15. Session 2026-11-18 scope

- Done: 3.2 (cells), 4.2, 4.4, 4.5, 6.1. Not done: section 5 (D12),
  6.2's AI_GUIDE part (the guide is at its 32 KiB limit: it gained the
  `rac=` keyword and a pointer to `fit_mlcc_dc_bias`; the ESL step and
  π-versus-Bergeron guidance are in `docs/known-limitations.md` only).
- No frequency-dependent ESR, dielectric absorption, core loss or
  temperature dependence: non-goals of this change (design.md), not open
  tasks.

## D16. A STRICT analytic case for the ESR expansion

- Question: the realistic-passive tests compare against closed forms
  inside `test_passives.py`; the reference harness had no passive with
  a parasitic.
- Option: `rc_esr_charge` in `benchmarks/analytic` (10 V through 1 kΩ
  into 1 µF with `esr=250 Ω`, observed at the capacitor terminal:
  `v = 10(1 − 0.8·e^{−t/1.25 ms})`); `k` scales the ESR, so the 1 %
  non-vacuity check bites on the parasitic itself. The terminal voltage
  jumps at t = 0 (an algebraic observable), so `exclude_t0`. Ceilings
  at twice the measured error (pwl at dt = 1 µs: at 10 µs its first-order
  start error, 1.4e-3, hid a 1 % ESR change), and a pwl golden added.
  The manifest block was inserted by hand from `freeze_analytic.py`'s
  printed numbers: its `--write` also rewrites the dsed entries of the
  static and coupled cases as "ran although refused" (the case table and
  the manifest disagree there; not this change's to settle).
- Reverse: delete the case, its manifest entry and golden; count back to 20.

## D17. Bergeron line after the device interface landed (closeout 2026-11-19)

- Question: D12 deferred section 5 to the device interface; it has
  landed (`device::StatefulKind`, `pwl::StatefulKinds`). Is section 5
  doable now without touching a time loop?
- Finding: not on the contract as it stands. A stateful kind declares
  `kFlatPerDevice`, which must equal its block's compile-time
  `reals_per_instance` (`StatefulCheck::reals`); a delay line carries
  `ceil(τ/dt) + 1` samples of `(v, i)` per end, fixed per run but not per
  kind. And every stateful kind must sit on a Newton row of `kKindCaps`
  (`StatefulCheck::newton`); the line is linear (`1/Z0` in the cached
  matrix, history sources in `b`), so as a Newton kind it would push every
  circuit holding a line onto the Newton path.
- What closes it: a variable-length state block (the in-run rollback and
  the snapshot's flat vector are already `std::vector<Real>`; the block
  table and the reader need a per-instance length), a linear "b-source"
  category whose stamp adds to `b` only (the cached matrix keeps `1/Z0`
  from the build), then tasks 5.1–5.10 unchanged. The trbdf2 step cap at
  τ can be set by the run plan as `dt_max` (above the dispatch, no loop
  hunk); stage-time lookup reads `StepCtx::t`.
- Option: left open as NOT DONE (not attempted in the closeout session);
  `model="bergeron"` keeps refusing by name.
