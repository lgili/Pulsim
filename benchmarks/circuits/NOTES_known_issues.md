# Known Pulsim Solver Issues — Surfaced by the pulsim-bench Suite

This document catalogues circuits that were constructed during the
ngspice-parity validation effort and **exposed real numerical issues**
in Pulsim 0.10.0. The YAML + .cir files are kept on disk as
**regression-test fixtures**: once the underlying solver work
addresses the issue, the YAML can be re-added to
`benchmarks/benchmarks.yaml` and the test should pass.

Each entry below lists:
- **Files**: where the fixture lives.
- **Symptom**: what Pulsim does wrong.
- **ngspice baseline**: what the correct behaviour looks like.
- **Hypothesis**: which layer of Pulsim probably has the bug.
- **Discovered**: which validation batch surfaced it.
- **Related OpenSpec change**: existing planned work that would
  likely fix it (if any).

---

## 1. vcswitch + L + freewheel diode commutation — numerical divergence

- **Canonical fixture (snubber-equipped, stronger evidence)**:
  - `benchmarks/circuits/chopper_with_rc_snubber.yaml`
  - `benchmarks/ngspice/chopper_with_rc_snubber.cir`
- **Topology**: Vcc (20 V DC) → vcswitch S1 → L (1 mH) → R_load (10 Ω) → 0,
  with a freewheel diode from 0 → sw AND an RC snubber (R_sn = 10 Ω,
  C_sn = 100 nF) from sw → vcc. vcswitch driven by PWM at 5 kHz / 50 %.
- **Symptom**: snubber damps the first commutation spike correctly
  (±6.5 V instead of catastrophic), but the residual numerical
  perturbation accumulates **exponentially** over subsequent periods:
  ±1 kV at t = 305 µs → ±24 kV at t = 1.5 ms. The Vcc source-current
  measurement at that point reads ±15 MA — clearly nonphysical. The
  simulation is diverging numerically, not physically.
- **ngspice baseline**: same topology settles into the expected
  steady-state ripple within ~5 periods; V(sw) stays bounded between
  about -V_F and Vcc + 5 V (the snubber's contribution).
- **Hypothesis**: the bug is **solver-side, not topology-side**.
  Even when a real physical damping element (the snubber) is present,
  Pulsim's `gmres_trbdf2` + ideal-vcswitch + ideal-diode combination
  produces an unstable numerical mode that grows over time. The
  earlier `chopper_inductive_freewheel` test (no snubber, dropped in
  batch 3) was symptomatic of the same root cause; the addition of
  the snubber CONFIRMED the bug is not just "missing damping".
- **Discovered**: validation batches 3 and 4 (commits `140c635` and
  the current batch).
- **Related OpenSpec change**: `refactor-pwl-switching-engine`
  (Roadmap Fase 0.1, the PLECS-killer PWL state-space cache) and
  `refactor-unify-robustness-policy` (Fase 4.2). Both changes intend
  to harden the solver against this class of inductive
  ideal-switching numerical issue. The PWL state-space approach
  bypasses Newton on stable topology segments entirely — which is
  the right architectural fix for this symptom.

### Earlier related fixture (no snubber)

A version without the snubber (`chopper_inductive_freewheel`) was
attempted in batch 3 and exhibited even more severe ringing
(±5 kV at dt = 1 µs, worsening to ±50 kV at dt = 100 ns). It was
not committed; the snubber-equipped version supersedes it as the
canonical regression test because (a) snubber damps the most
obvious failure mode yet the divergence still happens, and (b)
the snubber-equipped circuit is closer to real-world practice
(no production design would use ideal switching without snubbing).

---

## How to revive these tests after a fix

1. Restore the entry in `benchmarks/benchmarks.yaml` (search for
   the `NOTE:` comment mentioning the ID and uncomment the matching
   block).
2. Run `pulsim-bench parity --backend ngspice --only <id>` to
   confirm the issue is gone.
3. If the new behaviour matches ngspice within the documented
   thresholds (`max_error: 2.0`), the test goes back into the
   regular suite.
4. Update this NOTES file to record the resolution (commit
   reference, what the fix was, who reviewed).

---

## Tests that PASSED but are still informative

These are NOT bugs — they passed the parity test — but the relatively
loose thresholds document **measurable model differences** between
Pulsim and ngspice that future readers may want to know about:

| Circuit | Threshold | Cause |
|---|---|---|
| `diode_clamp_overvoltage` | 1.0 V | Diode V_F model gap: Pulsim g_on=200 → V_F ≈ 5 mV; ngspice DIODEMOD → V_F ≈ 0.7 V |
| `diode_half_wave_rl_load` | 1.0 V | Same V_F model gap |
| `back_to_back_diodes_sine` | 1.0 V | Same V_F model gap, applied to a clipper |
| `lc_resonant_tank` | 0.5 V | Phase drift between Pulsim trbdf2 + ngspice trap on lossless oscillation (windowed to 3 cycles) |
| `high_q_rlc_ringdown` | 1.0 V | Damping mismatch over 50 cycles at Q = 316 |
| `rlc_overdamped_step` | 0.2 V | Fixed-step vs adaptive timestep timing on the diode-knee transient |
