# AC Analysis

> Status: shipped. PWL-admissible circuits supported on the segment-engine
> linearization path. Behavioral / Newton-DAE linearization deferred to a
> follow-up (`add-frequency-domain-analysis` Phase 1.2).

Pulsim's AC small-signal analyzer linearizes the circuit around an
operating point and sweeps frequency. For each `f` it solves
`(jω·E − A) X = B` and returns `H(jω) = X / U` per requested
measurement node — magnitude in dB, phase in degrees, plus the raw
real / imaginary parts.

This is the analytical Bode tool (`.AC` in PSIM, `Analysis Tools → AC
Sweep` in PLECS). The empirical equivalent (transient + DFT) is
documented separately in [`fra.md`](fra.md).

## TL;DR

```python
import pulsim
ckt, opts = pulsim.YamlParser().load("netlist.yaml")
sim = pulsim.Simulator(ckt, opts)

ac = pulsim.AcSweepOptions()
ac.f_start = 1.0
ac.f_stop  = 1e6
ac.points_per_decade = 30
ac.perturbation_source = "Vin"
ac.measurement_nodes   = ["vout"]

result = sim.run_ac_sweep(ac)
fig, _ = pulsim.bode_plot(result, title="Buck output transfer")
fig.savefig("bode.png")
```

A 200-point sweep of a 4-state LC filter completes in ≤ 1 ms on
Release+LTO; analyze-pattern is computed once for the whole sweep, so
per-frequency cost is one factorize + one solve.

## Three layers, three call paths

| Layer | API | Returned object | Use when… |
|---|---|---|---|
| **Linearization only** | `Simulator.linearize_around(x_op, t_op)` | `LinearSystem { E, A, B, C, D }` | You want the descriptor matrices for downstream tools (eigenvalue analysis, observer design, custom solvers). |
| **Analytical AC sweep** | `Simulator.run_ac_sweep(opts)` | `AcSweepResult { frequencies, measurements[] }` | The standard Bode plot of a linear (or PWL-linearized) circuit. |
| **Empirical FRA** | `Simulator.run_fra(opts)` | `FraResult { frequencies, measurements[] }` | The same Bode shape, but measured by transient injection — needed when nonlinearity / PWM modulation matters. See [`fra.md`](fra.md). |

All three live on `Simulator` so the same DC operating point is reused
across analyses (the `dc_operating_point()` call inside each method is
idempotent for a fixed circuit).

## State-space form

The linearization is in descriptor (DAE) form:

```
E · dx/dt = A · x + B · u
y         = C · x + D · u
```

Mapping from Pulsim's MNA-trapezoidal state-space `M·dx/dt + N·x = b(t)`:

- **`E = M`** (descriptor / mass matrix)
- **`A = -N`** (dynamics matrix — sign flip because the LHS sees `+N·x`)
- **`B`** carries one column per perturbation source. Single-source
  sweeps (Phase 2/3) collapse to a one-column matrix:
  - **Voltage source** `V_k`: `B[branch_k, k] = +1`
  - **Current source** `I_k` between `(npos, nneg)`: `B[npos, k] = +1`,
    `B[nneg, k] = -1`
- **`C`** = identity by default (output is the full state vector). Per-
  node selection happens at the AC sweep call site via
  `measurement_nodes`.
- **`D`** = 0 in MNA — sources contribute to dynamics, not direct
  feedthrough.

## Frequency grid

`AcSweepOptions::scale` selects the spacing; `AcSweepOptions::f_start`
and `f_stop` set the bounds.

- `Logarithmic` (default): `points_per_decade` log-spaced points; the
  total point count is `ceil(log10(f_stop/f_start) · points_per_decade)
  + 1`.
- `Linear`: `num_points` uniformly-spaced points (defaults to
  `points_per_decade` if not set).

The internal complex pencil `K(ω) = jω·E - A` has the same sparsity
pattern across `ω`, so `Eigen::SparseLU::analyzePattern` runs **once**
at the start of the sweep. Per-`ω` cost is `factorize(K) + solve(B)`.

## MIMO transfer-function matrix

`AcSweepOptions::perturbation_sources` (vector) replaces the single
`perturbation_source` for multi-input sweeps. The result carries one
`AcMeasurement` per `(source, node)` pair, with both labels populated:

```python
ac.perturbation_sources = ["Va", "Vb"]
ac.measurement_nodes    = ["m1", "m2"]
result = sim.run_ac_sweep(ac)
# result.measurements has 2 × 2 = 4 entries:
#   measurements[0]: node='m1', perturbation_source='Va'
#   measurements[1]: node='m1', perturbation_source='Vb'
#   measurements[2]: node='m2', perturbation_source='Va'
#   measurements[3]: node='m2', perturbation_source='Vb'
```

Order is (output, input) — outer loop is the measurement node.

## YAML schema

Top-level `analysis:` array, each entry maps onto an `AcSweepOptions`
or `FraOptions`:

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
analysis:
  - type: ac
    name: open_loop_buck
    f_start: 1.0
    f_stop: 1e6
    points_per_decade: 30
    scale: log
    perturbation_source: Vref
    measurement_nodes: [vout]
  - type: fra
    name: closed_loop_validation
    f_start: 100.0
    f_stop: 1e5
    points_per_decade: 5
    perturbation_source: Vref
    perturbation_amplitude: 0.01
    measurement_nodes: [vout]
    n_cycles: 8
    discard_cycles: 3
    samples_per_cycle: 64
components:
  - { type: voltage_source, name: Vref, nodes: [in, gnd], value: 1.0 }
  ...
```

Strict mode (default) rejects unknown `type:` and unknown per-entry
fields. The parser populates `SimulationOptions::ac_sweeps` and
`SimulationOptions::fra_sweeps`; the user iterates and calls the
existing `run_ac_sweep` / `run_fra` methods. Multiple analyses with the
same `Simulator` instance share the DC operating point.

## Plotting helpers

`pulsim.bode_plot(result)` returns matplotlib `(fig, (ax_mag, ax_phase))`.
matplotlib is imported lazily — installing it is optional unless you
actually plot.

```python
fig, (ax_mag, ax_phase) = pulsim.bode_plot(result)
fig, ax = pulsim.nyquist_plot(result, unit_circle=True)
fig, _  = pulsim.fra_overlay(ac_result, fra_result)
```

`bode_plot` accepts both `AcSweepResult` and `FraResult` because they
expose the same shape (per-measurement `magnitude_db`, `phase_deg`,
`real_part`, `imag_part`).

## Export / load

```python
pulsim.export_ac_csv (result, "result.csv",  format="magphase")  # or "complex"
pulsim.export_ac_json(result, "result.json")
pulsim.export_fra_csv(result, "fra.csv")
pulsim.export_fra_json(result, "fra.json")

# Round-trip: load a CSV back into a result-shaped container that
# bode_plot accepts directly.
loaded = pulsim.load_ac_result_csv("result.csv")
fig, _ = pulsim.bode_plot(loaded)
```

## Performance

Per-frequency cost on AppleClang 17 / Release+LTO / Apple Silicon, for a
4-state LC filter:

| Stage | Wall-clock |
|---|---|
| `analyzePattern(K)` | once at sweep start, ≈ 5 µs |
| `factorize(K(ω_k))` | per frequency, ≈ 2 µs |
| `solve(B)` | per frequency, ≈ 1 µs |
| **200-point sweep** | **≈ 600 µs total** |

The contract is "200-point AC sweep ≤ 50 ms" (regression floor); the
measured value sits two orders of magnitude under it.

## Failure modes

`AcSweepResult.success = false` when:

- DC operating point fails (`failure_reason = "ac_sweep_dc_op_failed"`).
- Circuit has Behavioral-mode devices PWL state-space doesn't support
  yet (`"ac_sweep_non_admissible_behavioral_device"`). AD-driven
  Behavioral linearization is the Phase 1.2 follow-up.
- Named perturbation source not found in the circuit
  (`"ac_sweep_perturbation_source_not_found:<name>"`).
- Named measurement node not found
  (`"ac_sweep_measurement_node_not_found:<name>"`).
- Frequency range invalid
  (`"ac_sweep_invalid_frequency_range"`).
- Numerical breakdown at a specific frequency
  (`"ac_sweep_factorization_failed_at_f:<value>"`).

Always check `result.success` before accessing `result.measurements`.

## See also

- [`fra.md`](fra.md) — empirical Bode via transient injection.
- [`linear-solver-cache.md`](linear-solver-cache.md) — the same
  analyze-pattern-once architecture powering the segment-primary cache.
- Runnable examples: `examples/python/01_ac_sweep_rc.py` (Bode of an RC
  low-pass), `examples/python/11_yaml_ac_analysis.py` (loading the
  circuit from YAML).
