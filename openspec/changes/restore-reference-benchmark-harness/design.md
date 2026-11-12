## Context

Inventory on `docs/core-roadmap-openspec` (counted for this proposal):

| Item | Count |
|---|---|
| `benchmarks/circuits/**/*.yaml`, all `schema: pulsim-v1` | 126 |
| … with only passive elements, sources and diodes | 53 |
| … with gated switches, open loop | 55 |
| … with control blocks (`pi_controller`, `pwm_generator`, `pll`, …) | 18 |
| `benchmark.validation.type`: `reference` / `analytical` / `bode` / `none` or absent | 74 / 8 / 1 / 43 |
| `benchmarks/spice/ngspice/*.cir` (63 share a circuit's file name) | 66 |
| `benchmarks/spice/ltspice/*.cir` | 8 |
| `benchmarks/baselines/*.csv.gz` | 70 |
| `benchmarks/manifests/benchmarks.yaml` entries (58 name an ngspice netlist) | 108 |
| `benchmarks/tools/*.py` | 7 files, 5489 lines |

Component types used by the 126 files (occurrences): `voltage_source`
276 (waveforms: `dc` 116, `pwm` 108, `sine` 33, `pulse` 19), `resistor`
213, `vcswitch` 116, `capacitor` 113, `inductor` 108, `diode` 83,
`pwm_generator` 14, `switch` 11, `pi_controller` 11, `coupled_inductor`
8, `mosfet` 4, `current_source` 3, `pll` 3, and 17 further types used
once or twice.

The 2.0 YAML loader (`yaml/loader.hpp`) reads `circuit: devices:` and
`simulation:` only. It has no way to say how a switch is gated and no
control blocks; `chain:` is handled by a separate Python call. A v1
`vcswitch` is a switch whose state follows a control node, usually
driven by a `pwm` voltage source.

Engine facts the harness must respect (from the 2026-11 review):

- `pwl` is first order at every gate edge and at an inconsistent start;
  `trbdf2` grids cluster at events; `dsed` is exact per mode.
- On `pwl` and `trbdf2` the first stored sample of an algebraic node is
  the initial state, not the solution at `t = 0⁺`: an RL step reads 0 V
  across the inductor at `t = 0` where the answer is 10 V
  (`probes/h1_analytic.py`). `dsed` returns 10 V.

Probe written for this proposal: `probes/h1_analytic.py` in this change
folder (to become the first three cases of `benchmarks/analytic/`).

## Goals / Non-Goals

- Goals: the 126 circuits run or are skipped by name; any two engines
  can be compared on any circuit with one metric; 20 circuits have an
  exact answer; the fixed engine is pinned; performance regressions are
  detected without a clock; ngspice is consulted every night.
- Non-Goals:
  - A v2 YAML schema for gate drive or control
    (`add-immutable-circuit-ir`, `add-compiled-control-graph`).
  - Fixing what the harness finds. Each disagreement becomes an issue or
    a triage entry.
  - PSIM, PLECS or LTspice comparison (no reference data in the
    repository).
  - Porting the KPI tools (`kpi_gate.py`, `freeze_kpi_baseline.py`,
    `stress_suite.py`, `local_limit_suite.py`); see Open Questions.
  - Wall-time benchmarking.

## Decisions

### 1. Read v1 in the harness; do not rewrite the files

`v1_reader.load(path) -> BenchCase{builder, switch_fn, closed_loops,
t_end, dt, observables, validation, notes}`.

| v1 | 2.0 |
|---|---|
| `resistor`, `capacitor`, `inductor`, `current_source`, `diode`, `switch` | the same builder call |
| `voltage_source` + `dc` / `sine` / `pulse` / `pwm` waveform | `add_voltage_source`, `add_sine_voltage_source`, `add_pulse_voltage_source`, `add_pwm_voltage_source` |
| `vcswitch` whose control node is driven only by a `pwm` or `pulse` source | `add_switch` + a native `SwitchSchedule` with the same period, duty, phase and threshold |
| `mosfet`, `pmos`, `igbt` | `add_mosfet` / `add_igbt` + schedule from the gate node as above |
| `coupled_inductor`, `transformer` | `add_transformer` |
| `saturable_inductor`, `snubber_rc` | builder equivalents |
| `pi_controller` → `pwm_generator` → one switch | `bind_pi_to_switch` |
| any other control block, or a control node driven by the circuit | `skipped: <reason naming the block>` |
| `simulation:` `tstop`, `dt`, `uic` | `t_end`, `dt`, initial conditions; solver-selection keys (`integrator`, `newton`, `fallback`, …) are recorded and ignored |

`benchmarks/manifests/v2_triage.yaml` has one entry per circuit:
`status: runs | skipped`, `reason`, `tol` overrides. A test asserts
there are 126 entries, that every `runs` circuit loads and simulates on
`pwl`, and that a `skipped` circuit still fails to load for its stated
reason (so an entry cannot go stale). 108 circuits use only component
types with a direct mapping; how many of them run is the first result
of the implementation, not a number this proposal can state.

Alternatives considered:
- *Convert the files to v2 YAML.* The v2 schema cannot express the gate
  of 116 `vcswitch` instances or any control block, so 73 files would
  need a schema that later phases own. It would also rename 126 files'
  worth of manifest and ngspice pairings in a change whose purpose is a
  safety net.
- *Teach the kernel loader to read v1.* The loader refuses v1 on
  purpose; a legacy reader in the kernel is the opposite of the
  roadmap's direction.
- *Start a new suite.* Discards 126 circuits and 66 paired netlists.

### 2. One metric module, time-weighted

`benchmarks/tools/metrics.py`:

- `err_rms(t, y, ref)` — `sqrt(∫(y − ref)² dt / (t_end − t_0)) / span`,
  trapezoidal on the trace's own time grid when `ref` is a function
  (analytic), and on the union of both grids with linear interpolation
  when `ref` is another trace. `span = max(ref) − min(ref)`, floored at
  an absolute scale given per observable.
- `err_max(...)` at the same points.
- Scalars (mean, RMS over a window) through `pulsim.time_average` and
  `pulsim.time_rms`.
- The window excludes `t = t_start` for observables that are not state
  variables (Context, second bullet).

An unweighted mean over samples is never used: a ramp sampled 1000
times in its first 1 % and 10 times after reads 0.0104 unweighted and
0.5000 time-weighted (`probes/h1_analytic.py`).

### 3. Differential runner

`differential_runner.py --circuits <glob> --engines pwl,trbdf2,dsed`.
For each circuit and engine: run; classify the outcome.

- `refused` — the engine raised its typed refusal. Compared with the
  capability table when `pulsim.engine_capabilities()` exists.
- `agree` / `disagree` — `err_rms` against the circuit's reference is
  within / outside the tolerance.
- `error` — anything else; always a failure.

Reference, first available: (a) the analytic solution; (b) the ngspice
waveform (nightly only); (c) `trbdf2` at `rtol = 1e-8`; (d) `pwl` at
`dt/4`, accepted only if `dt/2` and `dt/4` agree within a quarter of
the tolerance. The choice is written in the report.

Default tolerances on `err_rms`, overridable per circuit in the triage
manifest: circuits with no switching event 1e-3; switched circuits
2e-2 on `pwl` (first order at edges; the review measured 1.18e-2 A to
7.4e-4 A on a CCM buck from T/50 to T/800) and 1e-3 on `trbdf2` and
`dsed`. A pull request may tighten a tolerance; loosening one requires
a reason in the manifest.

Output: `results.json` with one record per cell (outcome, reference,
both metrics, counters, wall time for information).

### 4. Analytic reference set (20 circuits)

`benchmarks/analytic/<id>.py`, each exporting `build()`, `t_end`,
`observable`, `exact(t)` and tolerances per engine.

| # | Circuit | Closed form |
|---|---|---|
| 1 | RC charge, 10 V, 1 kΩ, 1 µF | `v = V(1 − e^{−t/RC})` |
| 2 | RC discharge from `v₀ = 5 V` | `v = v₀·e^{−t/RC}` |
| 3 | RL step, 10 V, 10 Ω, 1 mH | `i = (V/R)(1 − e^{−Rt/L})` |
| 4 | Series RLC underdamped, 1 Ω, 1 mH, 10 µF | `v_C = V[1 − e^{−αt}(cos ω_d t + (α/ω_d) sin ω_d t)]`, `α = R/2L`, `ω_d = √(1/LC − α²)` |
| 5 | Series RLC overdamped, 300 Ω | `v_C = V[1 − (s₂e^{s₁t} − s₁e^{s₂t})/(s₂ − s₁)]`, `s₁,₂ = −α ± √(α² − ω₀²)` |
| 6 | Lossless LC from `v₀ = 10 V` | `v = v₀ cos ω₀t`; `½Cv² + ½Li²` constant |
| 7 | Sine-driven RC from rest | `v = (V/√(1+(ωτ)²))[sin(ωt − φ) + sin φ·e^{−t/τ}]`, `φ = atan ωτ` |
| 8 | Sine-driven series RL from rest | `i = (V/|Z|)[sin(ωt − φ) + sin φ·e^{−Rt/L}]` |
| 9 | Wheatstone bridge (`bridge_resistive_pure`) | `V(a) = 6.6667 V`, `V(b) = 4.0000 V` |
| 10 | Pulse-driven RC | piecewise exponential between source breakpoints |
| 11 | Coupled inductors, k = 0.99, resistive source and load, step | two exponentials from the 2×2 state matrix |
| 12 | Ideal transformer n:1, source resistance, resistive load | `v_p = V·R'/(R_s + R')`, `R' = R_L/n²`, `v_s = n·v_p` |
| 13 | VCVS, gain 2, resistive load, sine input | `v_o = 2·v_i` |
| 14 | Shockley diode + R from a DC source | `i = (nV_T/R)·W((I_S R/nV_T)·e^{(V + I_S R)/nV_T}) − I_S` (Lambert W) |
| 15 | PWL diode, 10 A forced, `g_on = 20 S`, `V_f = 0.7 V` | `v = 1.2 V`, `P = 12 W` |
| 16 | Half-wave rectifier, resistive load, `V_f = 0.7 V` | `v_o = max(0, (v_in − V_f)·R/(R + 1/g_on))` |
| 17 | Half-wave rectifier, RL load | `i = (V_m/|Z|)[sin(θ − φ) + sin φ·e^{−θ/tan φ}]` until the extinction angle β, one scalar root |
| 18 | Hard-switched RC (`p8_caploop.py`) | `v = 100 − 99.99·e^{−(t − t_off)/10 ns}` in each open interval |
| 19 | Buck, CCM, PWL devices, open loop | periodic steady state as the fixed point of the period map built from one matrix exponential per interval |
| 20 | Buck, DCM | as 19 with the diode turn-off instant as a scalar root |

Cases 19 and 20 are exact piecewise-LTI solutions computed with
`scipy.linalg.expm` and `brentq`, the method the numerics audit used
(its DCM buck reads 8.9962 V). Case 2 runs on `pwl` and `trbdf2` only,
because `dsed` refuses initial conditions; cases 14 and the diode laws
follow the refusals of the capability table.

Measured on three of them (`probes/h1_analytic.py`, `err_rms` after
interpolation to 2001 points):

| Case | `pwl` | `trbdf2`, `rtol = 1e-6` | `dsed` |
|---|---|---|---|
| 1, 5 ms, `dt = 10 µs` | 1.59e-3 | 6.0e-7 | 2.8e-8 |
| 4, 10 ms, `dt = 1 µs` | 6.13e-4 | 9.24e-5 | 1.5e-6 |

Initial tolerances are set from the first full run with a factor of 2
of margin and written to `benchmarks/manifests/analytic.yaml`. They are
ceilings: later phases lower them (the restart rule is expected to take
case 1 on `pwl` to second order).

### 5. Golden checkpoints of the fixed engine

`benchmarks/goldens/pwl/<circuit>.json`, for the 20 analytic circuits
and at least 20 circuits of the triaged suite chosen to cover every
topology family and every Newton device (the exact list is a task):

```json
{"dt": 1e-6, "t_end": 2e-3, "n_steps": 2000,
 "switch_digest": "sha256:…", "n_commutations": 40,
 "checkpoints": [{"t": 2.5e-4, "x": {"out": 10.9091, "L1": 1.0909}}, …8],
 "rtol": 1e-9, "atol": 1e-12, "generated_on": "macos-arm64, clang 17"}
```

- Integers and the digest of the stored switch-state record are
  compared exactly: they are discrete and do not depend on the
  platform's floating-point library.
- State values are compared with `|x − g| ≤ atol + rtol·|g|`. `rtol` is
  the smallest decade that passes on every CI platform, at most 1e-9
  for circuits without Newton devices and 1e-6 with them. A circuit
  that cannot meet the cap is left out with a named reason.
- `--update-goldens` rewrites the files; the diff is reviewed. Changes
  marked **B** in the roadmap re-freeze on purpose and say so.

Alternative *bit-exact hash of the state trajectory*: the roadmap row
says "golden hashes". A hash cannot pass on three platforms (the 9e-5
`dsed` difference between Windows and macOS is the evidence), and when
it fails it does not say by how much.

### 6. Counter-based performance gate

`RunCounters` on every result: `n_steps`, `n_rejects`, `n_solves`,
`n_factorizations`, `n_symbolic_analyses`, `n_newton_iterations`,
`n_segment_builds`, `n_event_refactors`. Existing sources are reused
(`CacheMetrics`, `TrBdf2Stats`, the DSED stats dict); the missing ones
are the factorisation and Newton-iteration totals of the fixed and
Newton paths.

`benchmarks/manifests/perf_counters.yaml` freezes them for 12 circuits
× engines (small linear, 500-node ladder, buck, three-phase inverter,
one per Newton device). Gate: on `pwl`, each counter ≤ the frozen
value; on `trbdf2` and `dsed`, ≤ 1.02 × the frozen value (their step
count depends on floating-point comparisons in the error controller).
A counter that improves by more than 5 % prints a reminder to
re-freeze. Wall time is written to the report only.

Existing wall-clock assertions in `python/tests` (8 files use
`perf_counter`; `test_subsystems.py:299` asserts `< 0.05 s`) are
inventoried and each is converted to a count or a ratio, or removed.

### 7. ngspice job

`.github/workflows/nightly-reference.yml`, `schedule` plus
`workflow_dispatch`: install ngspice, run the differential runner on
all `runs` circuits × engines, then `benchmark_ngspice.py` on the
circuits with a netlist, publish `results.json` and a summary table,
and open or update one issue when a cell changes outcome. It does not
gate pull requests. The ngspice version is recorded in the artifact
(45.2 locally).

### 8. Pull-request job

One job in `ci.yml`: analytic set on three engines, golden checkpoints,
counter gate. It must stay short on a shared runner; the budget and the
circuit subset are tuned when it first runs.

## Risks / Trade-offs

- The reader is a second netlist front end. Mitigation: it lives under
  `benchmarks/`, is never imported by `pulsim`, and maps only what the
  126 files use.
- A v1 circuit may map to a slightly different 2.0 circuit (switch
  threshold, default `g_off`). Mitigation: the ngspice comparison is the
  arbiter where a netlist exists; elsewhere the mapping table is the
  specification and every mapped default is printed in the report.
- Goldens freeze today's defects (first-order edges, the 0 V first
  sample). That is their job: they pin behaviour for refactors, they do
  not certify it. The analytic set is what measures correctness.
- Tolerances chosen from one run can be too tight on another platform →
  the first CI run on all platforms sets them, before the job is made
  blocking.
- 126 × 3 engines nightly on a hosted runner: duration unknown until
  measured; the nightly may need sharding.

## Migration Plan

1. `metrics.py`, `v1_reader.py`, triage manifest; `benchmark_runner.py`
   on `pulsim.simulate`.
2. Analytic set and its test.
3. Differential runner over the triaged suite; file the disagreements.
4. `RunCounters`, counter gate.
5. Goldens, after the Phase 0 changes that move `pwl` numbers.
6. Workflows; make the pull-request job blocking after one green week.

Each step is useful alone; rollback is deletion of the step's files.

## Open Questions

1. Reader in the harness (proposed) or a one-time rewrite of the 126
   files to a new benchmark schema? If the reader: keep the header
   `schema: pulsim-v1` or rename it (for example `pulsim-bench-1`) to
   stop it looking like a kernel netlist?
2. The 18 circuits with control blocks: map the `pi_controller` →
   `pwm_generator` pattern through `bind_pi_to_switch` now (proposed),
   or mark all 18 skipped until `add-compiled-control-graph`?
3. The 70 legacy baselines captured from the v1 engine: keep as an
   informational "drift from v1" column (proposed), regenerate from
   2.0, or delete?
4. The v1 scenario names in `benchmarks.yaml` (`direct_trap`,
   `gmres_trbdf2`, `jfnk_trbdf2`, `rosenbrockw`, `direct_bdf1`, …)
   select solvers that no longer exist. Replace them with the engine
   list (proposed) or keep the file as history and add a new manifest?
5. The KPI tools and `benchmarks/kpi_baselines/` (frozen 2026-02):
   retire, or port to the counters of decision 6?
6. Golden generator platform (macOS arm64 proposed) and the tolerance
   caps (1e-9 linear, 1e-6 Newton).
7. Nightly failure policy: an issue (proposed) or a failing badge; and
   whether to pin the ngspice version on the runner.
8. Should the first stored sample of algebraic nodes on `pwl` /
   `trbdf2` (0 V instead of the `t = 0⁺` value) be treated as a defect
   to fix, or documented and excluded from comparisons as proposed
   here?
