# FRA — Frequency Response Analysis

> Status: shipped. PWL-admissible circuits supported. Behavioral / PWM-loop
> FRA on Newton-DAE deferred (`add-frequency-domain-analysis` Phase 1.2).

FRA is the *empirical* Bode tool — it injects a small-signal sinusoid on
top of a named DC source, runs a transient simulation for several
periods, and extracts the fundamental at the perturbation frequency via
a single-bin Goertzel DFT. The result is `H(jω)` in the same form as
`run_ac_sweep`, but measured rather than analytically derived.

## When to use FRA vs AC sweep

| Question | Use AC sweep | Use FRA |
|---|---|---|
| Is my linearized model correct? | ✓ | (validation only) |
| Does PWM modulation interact with the loop? | ✗ | ✓ |
| Is saturation kicking in at large signals? | ✗ | ✓ |
| Do switching dead-times shift my phase? | ✗ | ✓ |
| What's the small-signal control-to-output bandwidth? | ✓ | ✓ |
| I want it fast (200 freq points in ms). | ✓ | (slower — runs transient per freq) |

On a strictly linear circuit the two methods must agree within ≤ 1 dB
/ ≤ 5° — that's the gate G.1 contract pinned by
`test_frequency_analysis_phase3.cpp::Phase 3.5`.

## TL;DR

```python
fra = pulsim.FraOptions()
fra.f_start = 100.0
fra.f_stop  = 1e5
fra.points_per_decade  = 5
fra.perturbation_source    = "Vref"
fra.perturbation_amplitude = 0.01     # 1% of the source's nominal
fra.measurement_nodes      = ["vout"]
fra.n_cycles          = 8
fra.discard_cycles    = 3
fra.samples_per_cycle = 64

result = sim.run_fra(fra)
fig, _ = pulsim.bode_plot(result, title="FRA: closed-loop buck")
```

## How it works

For each frequency `f_k`:

1. **Configure perturbation** — call
   `Circuit::set_ac_perturbation(name, ε, f_k, φ)` which overlays
   `ε·cos(2π·f_k·t + φ)` on the named source's RHS row(s) during
   `assemble_state_space`. Cosine is intentional: the input phasor
   becomes `ε·e^{jφ}` (real for `φ=0`), matching the AC sweep
   B-column convention so `H = Y/ε` directly aligns with AC.
2. **Run transient** — fixed-dt = `period / samples_per_cycle`,
   tstop = `n_cycles · samples_per_cycle · dt`, integrator forced to
   trapezoidal (most phase-accurate). The simulator's adaptive
   timestep machinery is overridden for the duration of the FRA
   call.
3. **Discard warmup** — drop the first `discard_cycles` cycles to let
   any step-response transient die out.
4. **Mean-subtract** — remove the DC operating-point offset so the
   Goertzel result reflects only the small-signal response.
5. **Goertzel single-bin DFT** at `f_k` — returns the complex Fourier
   coefficient `Y(f_k)`.
6. **Apply half-step phase correction** — multiply by `e^{-j·ω·dt/2}`.
   The trapezoidal integrator uses the perturbation averaged at the
   step midpoint `t + dt/2`, while the `SimulationCallback` captures
   the output at the step end `t + dt`. Net effect = `+ω·dt/2` virtual
   phase shift; the correction reverses it.
7. **Divide by ε** — gives `H(jω) = Y/ε` in the same convention as
   AC sweep.

## Tuning knobs

### `perturbation_amplitude`

Default: 0.01 (1% of source units). Pick small enough to stay in the
linear region, large enough to dominate numerical noise. For a buck
converter with `Vref = 1 V` reference, `ε = 0.01 V` is typical. For a
high-current `Vbus = 400 V` source, you might use `ε = 4 V` (still
1%).

If the circuit has saturating magnetics or strong nonlinearity, drop
the amplitude until FRA converges to a frequency-independent transfer
function — that's the small-signal regime.

### `n_cycles` / `discard_cycles`

Default: 6 / 2. The first `discard_cycles` periods absorb the
step-response transient kicked off when the cosine perturbation starts
at its peak (t=0 → cos(0) = 1). The remaining `n_cycles -
discard_cycles` periods feed the Goertzel DFT.

For circuits with long settling times (slow loops, inductive rails),
bump both: `n_cycles = 12, discard_cycles = 6`. The Goertzel result
will stabilize as the post-transient window grows.

### `samples_per_cycle`

Default: 32. Sets the per-period sample density for the DFT. Goertzel
quantization scales as `2π/samples_per_cycle ≈ 11° at 32`, narrowing to
`≈ 5° at 64` and `≈ 1.5° at 256`. For tight phase margins
(`≤ 2°`) crank this up to 128–256.

The sampling rate is also tied to the transient `dt` used inside the
FRA call. Higher `samples_per_cycle` = smaller `dt` = more transient
work per frequency point.

## Result shape

`FraResult` mirrors `AcSweepResult`'s schema:

```python
result.frequencies               # list[float], Hz
result.measurements[k].node      # str
result.measurements[k].magnitude_db
result.measurements[k].phase_deg
result.measurements[k].real_part
result.measurements[k].imag_part
result.total_transient_steps     # FRA-specific: total simulator steps
result.wall_seconds
```

That means everything in the [`ac-analysis.md`](ac-analysis.md) export
/ plotting section works on `FraResult` too:

```python
pulsim.bode_plot(fra_result)
pulsim.export_fra_csv(fra_result, "fra.csv")
pulsim.export_fra_json(fra_result, "fra.json")
pulsim.fra_overlay(ac_result, fra_result, title="AC vs FRA")  # validation
```

## Validation contract

On linear circuits FRA agrees with `run_ac_sweep` within ≤ 1 dB / ≤ 5°
at every frequency point — gate G.1 of the change spec. The
`fra_overlay` helper draws both on the same Bode axes so the gap is
visualizable.

When the gap exceeds the gate, the failure mode is usually one of:

1. **Settling not done** — bump `discard_cycles`.
2. **Quantization** — bump `samples_per_cycle`.
3. **Nonlinearity active** — bump `perturbation_amplitude` smaller.
4. **Real bug** — the circuit has a feedback path AC sweep doesn't see
   (e.g., a saturating regulator). FRA is correct in that case; AC
   sweep is the misleading one.

## Performance

FRA is fundamentally slower than AC sweep because each frequency point
runs a full transient simulation. For a buck-shaped circuit at 1 kHz
with `n_cycles = 6, samples_per_cycle = 64`:

```
period   = 1 ms
dt       = 1 ms / 64 ≈ 16 µs
n_steps  = 6 · 64 = 384
wall     ≈ 1-5 ms / freq point
```

100 frequency points = ~ 0.1 to 0.5 s wall-clock. Per-point cost
dominates — there's no analyze-pattern-once trick for FRA because each
frequency runs a different transient.

If you need fast Bode data, prefer AC sweep. Use FRA only for what AC
sweep can't do (PWM loop validation, nonlinear converters, switching
dead-time effects).

## Failure modes

`FraResult.success = false` when:

- DC operating point fails (`failure_reason = "fra_dc_op_failed"`).
- Circuit has Behavioral devices the PWL path doesn't support yet
  (`"fra_non_admissible_behavioral_device"`).
- Perturbation source not found
  (`"ac_sweep_perturbation_source_not_found:<name>"`, shared with AC
  sweep).
- Measurement node not found
  (`"fra_measurement_node_not_found:<name>"`).
- Transient diverges at a specific frequency
  (`"fra_transient_failed_at_f:<value>"`) — usually means the
  perturbation amplitude is too large or the integrator is unstable
  for the chosen `dt`.

## See also

- [`ac-analysis.md`](ac-analysis.md) — the analytical complement.
- [`convergence-tuning-guide.md`](convergence-tuning-guide.md) — for
  tuning the transient integrator that FRA runs internally.
- Runnable example: `examples/python/02_fra_vs_ac_sweep.py` — overlays
  the empirical FRA against the analytical AC sweep on an RC low-pass
  and prints the ΔdB / Δdeg per frequency point.
