# Decisions — add-controlled-source-set

Each entry: question → option taken → why → how to reverse.

## 1. PWL-table voltage source in this change?

- Option: no; only `add_pwl_current_source`.
- Why: neither option is marked proposed, so the conservative one. The
  proposal and tasks list seven methods. A table voltage source also
  needs the voltage-source overlay in four engine paths
  (`run_transient`, trbdf2 `accumulate_sources`, the dsed adapter and the
  binding's `compute_time_varying_b_extra`), which is a separate
  surface.
- Reverse: add `PwlVoltageSource{PwlWaveform w}` beside
  `PulseVoltageSource`, overlay it where the pulse voltage source is
  overlaid, and add a breakpoint case to the trbdf2 clamp.

## 2. Gyrator parameter `g` or `r`?

- Option: `g` in siemens only; no `r` alias. YAML `r:` is an unknown key
  and is refused.
- Why: `g` is the stamp's natural form, and the magnetic domain will pass
  `g = 1/N`. One spelling avoids two keywords that could disagree.
- Reverse: add an `r` keyword (and YAML key) that sets `g = 1/r`, and
  refuse both being given.

## 3. Python argument names `n_pos` / `n_neg`, YAML `from` / `to`

- Option: the current-output kinds take `n_pos`, `n_neg` in Python
  (`from` is a Python keyword) and `from`, `to` in YAML, as
  `add_current_source` already does.
- Why: sibling pattern.
- Reverse: none needed.

## 4. A control must be added before the controlled source

- Option: `control` is resolved when `add_cccs` / `add_ccvs` runs. An
  unknown name, or a branch without a current unknown, is refused there,
  and the message lists the valid kinds and says "added before".
- Why: the builder resolves names eagerly everywhere else (`node_id_of`,
  `add_inductor_coupling`), and an eager error points at the line that
  is wrong.
- Reverse: store the name and resolve it at `graph()` finalisation.

## 5. YAML table `file:` is relative to the working directory

- Option: the path is taken as given (relative to the working
  directory). `components()` reports the knots, not the path.
- Why: `load_yaml_string` has no file to be relative to, and one rule for
  both loaders is the conservative choice. The knots are what was
  simulated.
- Reverse: thread `LoadedCircuit::source`'s directory into `load_device`
  and resolve against it when loading from a file.

## 6. dsed outcome (task 5.3)

- Option: dsed runs all seven kinds; no refusal.
  - The controlled stamps reach it through the cached matrix that the
    adapter extracts.
  - The waveform current sources reach it through
    `build_time_varying_b_extra_`.
  - `detect_sine_only_` returns false when a current waveform is
    present, so the sine-only exact stepper cannot drop it.
- Why: measured.
  - Gyrator with 1 µF: within 1e-15 of the exponential.
  - Every controlled kind in one circuit: within 1e-7 of trbdf2.
  - Sine, pulse and table current sources into an RC: within 1e-6 of
    trbdf2.
- Reverse: refuse the kinds in the dsed dispatcher's support check.

## 7. Validation tolerances that the design table could not meet as written

- Question: the plan asks pwl within 1e-9 of `i(1 ms) = 63.21 mA`,
  trbdf2 within 1e-6 of pwl, and the VCCS integrator `v = 3·t` within
  1e-9. Should these be met on the fixed engine?
- Option: no. Each engine is held to the analytic waveform at the
  accuracy it actually has, and the fixed engine's error is pinned
  exactly.
  - pwl: within 1e-9 of an identical inductor circuit on the same grid
    (the design's own comparison).
  - pwl against the exponential: the error is pinned at exactly
    `dt/(2τ)`.
  - trbdf2: within 1e-6 of the exponential (measured 1.5e-8).
  - dsed: within 1e-12 (measured 1e-15).
  - Integrator: `v = 3·t` within 1e-9 on trbdf2 and dsed. On pwl, every
    step after the first is exactly `3·dt`, and the first is `1.5·dt`.
  - trbdf2 against pwl: the error shrinks by more than 6.7× from
    `dt = 1e-7` to `1e-8`.
- Why: the fixed engine starts first order from an inconsistent state
  (guide trap 9, `test_first_step_order.py`). A plain 10 mH inductor
  reads 63.19 mA at `dt = 1e-6`, the same as the gyrator. Meeting 1e-6
  against pwl would need `dt = 1e-9`. These are properties of the
  engine, not of the stamps.
- Reverse: when `update-fixed-step-loop-bookkeeping` (or a consistent
  start) lands, tighten the pwl assertions to the design numbers.

## 8. `auto_regularize=False` in the capacitor-only tests

- Option: the integrator and table tests pass `auto_regularize=False`.
- Why: a node whose only DC path is a current source and a capacitor is
  floating at DC, so the regulariser ties it to ground with 1 GΩ. That
  leaks 2e-9 of the 1 V table charge over 3 ms, which is above the 1e-9
  assertion. Without the tie the error is 6e-14.
- Reverse: none; the tie is documented behaviour (guide trap 7).

## 9. Bit identity (task 5.2) on the v2 example netlists

- Option: the 25 v2 netlists in `examples/` (`circuit:` schema) were run
  on pwl and trbdf2, and the times and states hashed before and after
  the change.
  - 43 runs are bit-identical.
  - 7 raise the same refusal (a controlled switch with no `switch_fn` on
    trbdf2) in both builds.
- Why: `benchmarks/circuits/` holds 126 `schema: pulsim-v1` netlists,
  which the 2.0 loader refuses, and the benchmark runner's backend
  module (`pulsim_python_backend`) is not in the tree. The full pytest
  suite is the second identity check.
- Reverse: run the benchmark runner once it loads v2 netlists.

## 10. `PwlWaveform` in its own header

- Option: `models/pwl_waveform.hpp` (task 3.1); `controlled_sources.hpp`
  includes it.
- Why: the task names the file.
- Reverse: none needed.
