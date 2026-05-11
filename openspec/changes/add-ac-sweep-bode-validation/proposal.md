## Why
The kernel already exposes `set_ac_perturbation(source, amplitude, frequency, phase)` (we saw it in `runtime_circuit.hpp` around line 2402), so the lower-level "inject a small-signal probe and recover H(jω) via Goertzel" plumbing exists. What's missing is the user-facing layer that lets a benchmark sweep frequencies, automatically extract the Bode response, and validate it against a known analytical model.

This is **the** feature that distinguishes a design-grade simulator (PLECS, PSIM Smart Control) from a transient-only tool. Compensator design, loop-gain measurement, and stability margins all need it.

## What Changes
- Add a Python helper `benchmarks/frequency/` that wraps the AC perturbation API: given a circuit YAML + source name + frequency range, run a series of short transients with the perturbation enabled and return the recovered `H(jω)` per frequency point.
- Implement `compare_to_analytical(measured_bode, model_fn)` that scores |H_meas| vs |H_model| and ∠H_meas vs ∠H_model.
- New `frequency_response_*` benchmark category (parallel to `closed_loop` and `linear`) whose validation type is `bode` rather than `reference`:
  - `bode_buck_plant` — small-signal V_out/d transfer function of an open-loop buck, validated against Erickson § 7.2 averaged model (analytical).
  - `bode_boost_plant` — same for a boost, including the RHP zero predicted by the averaged model.
  - `bode_buck_pi_loop_gain` — closed-loop gain T(s) = G·H with PI compensator; recover gain/phase margin and compare to hand-computed margin within ±2 dB / ±5°.
  - `bode_rlc_low_pass` — purely passive LC tank as a sanity baseline; second-order rolloff at −40 dB/dec above f₀.
- Extend `scripts/closed_loop_dashboard.py` (or add a parallel `frequency_dashboard.py`) that shows pass / fail of the Bode tests and the measured GM / PM.

## Impact
- Affected specs: `ac-analysis` (extend with Bode validation), `benchmark-suite` (new validation type).
- Affected code: new `benchmarks/frequency/` package, new benchmark YAMLs, runner extensions for the `bode` validation type, and (optionally) a dashboard.
- The C++ kernel needs no changes — the `set_ac_perturbation` hook is already there.
