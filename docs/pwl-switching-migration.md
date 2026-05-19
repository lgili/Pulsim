# PWL Switching Engine — Migration Guide

> **Status (v0.11):** PWL engine is now the **default**. As of
> `simplify-and-harden-numerical-surface` Phase 11,
> `SwitchingMode::Auto` resolves to **Ideal** automatically — the
> state-space fast path engages out of the box on any switching
> converter where every device declares `supports_pwl`. The previous
> Behavioral semantics remain available by setting
> `opts.switching_mode = SwitchingMode::Behavioral` explicitly.
>
> Known stability gaps on a small set of legacy buck-converter /
> diode-loss circuits (vout overshoots, forward-bias diode spikes)
> are tracked in the follow-up OpenSpec change
> `openspec/changes/harden-pwl-ideal-buck-diode/`. The 16 affected
> tests pin Behavioral explicitly with inline comments referencing
> that follow-up.

Pulsim's switching engine has two execution paths for diodes, MOSFETs,
IGBTs, and voltage-controlled switches:

| Path | Per-step cost | Newton iterations | Accuracy on switching topologies |
|------|---------------|--------------------|----------------------------------|
| **Behavioral** (`SwitchingMode::Behavioral`) | Newton iteration with smooth `tanh` / Shichman-Hodges stamps | Multiple per step | Smooth transitions; Newton sometimes thrashes near commutation |
| **PWL / Ideal** (`SwitchingMode::Ideal`) | One linear solve per step on the cached topology factorization | **Zero** in stable topology windows | Sharp transitions; events detected at end of step (bisection follow-up below) |

`refactor-pwl-switching-engine` lands the infrastructure and end-to-end
contract tests for the PWL path. On the in-tree buck benchmark (5 PWM
cycles @ 100 kHz, `dt = 100 ns`) `Ideal` mode runs roughly **240× faster
wall-clock** than `Behavioral` on AppleClang 17 / Release+LTO. The buck
test [`test_pwl_speedup_benchmark.cpp`](https://github.com/lgili/Pulsim/blob/main/core/tests/test_pwl_speedup_benchmark.cpp)
asserts a loose `≤3×` ratio to absorb CI noise; the full numbers are
logged via Catch2 `INFO`.

## SwitchingMode and how it resolves

Every PWL-eligible device carries a `mode_` field with three values:

- `SwitchingMode::Ideal` — opt-in to the PWL state-space fast path.
- `SwitchingMode::Behavioral` — force the legacy smooth-stamp / Newton path.
- `SwitchingMode::Auto` — defer to the circuit-level default supplied at
  Simulator construction time.

The Simulator pushes `SimulationOptions::switching_mode` into
`Circuit::default_switching_mode_`. At each segment-engine and
event-scan call site, devices in `Auto` resolve via
`resolve_switching_mode(device_mode, circuit_default)`. The current
shipped default for the circuit-level resolution is **Behavioral**.

## Three ways to opt in

### 1. Simulation-level YAML field

Applies to every PWL-eligible device that is still in `Auto`:

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstop: 100e-6
  dt: 100e-9
  switching_mode: ideal   # also accepts: auto, pwl, behavioral, smooth
components:
  - { type: voltage_source, name: Vin, nodes: [vin, 0], value: 12.0 }
  - { type: switch,         name: SW,  nodes: [vin, sw], initial_state: false }
  - { type: diode,          name: D,   nodes: [0, sw] }
  - { type: inductor,       name: L,   nodes: [sw, vout], value: 22e-6 }
  - { type: capacitor,      name: C,   nodes: [vout, 0], value: 100e-6 }
  - { type: resistor,       name: R,   nodes: [vout, 0], value: 5.0 }
```

### 2. Per-device YAML override

Overrides the simulation-level default for one device. Useful when most
of the circuit is in `Ideal` mode but a single device needs the smooth
stamp (control-loop linearization, large-signal stability work):

```yaml
simulation:
  switching_mode: ideal
components:
  - { type: diode, name: D_fast, nodes: [a, b] }                        # follows simulation default → Ideal
  - { type: diode, name: D_soft, nodes: [c, d], switching_mode: behavioral }  # forced Behavioral
```

The per-device field is honored on `diode`, `switch`, `vcswitch`,
`mosfet`, and `igbt`. Aliases mirror the simulation-level parser:
`pwl == ideal` and `smooth == behavioral`. Strict mode rejects unknown
values with diagnostic `Invalid <device>.switching_mode: '<value>'`.

### 3. Python / C++ API

```python
import pulsim as ps

circuit = ps.Circuit()
# ... add devices ...

# Whole-circuit:
circuit.set_switching_mode_for_all(ps.SwitchingMode.Ideal)

# Per-device:
circuit.set_switching_mode("D1", ps.SwitchingMode.Ideal)
circuit.set_switching_mode("D_softstart", ps.SwitchingMode.Behavioral)

# Or through simulation options (Auto-mode devices resolve to this):
options = ps.SimulationOptions()
options.switching_mode = ps.SwitchingMode.Ideal
```

The C++ surface mirrors the Python names: `Circuit::set_switching_mode(name, mode)`,
`Circuit::set_switching_mode_for_all(mode)`, and the
`SimulationOptions::switching_mode` field.

## Caveats and follow-up work

### Choose `BDF1` to actually engage the segment engine today

`apply_auto_transient_profile` rewrites `Integrator::Trapezoidal` →
`Integrator::TRBDF2` for switching circuits, and TRBDF2's multi-stage
Newton path bypasses the segment-primary fast path. To run the PWL
state-space engine on switching circuits today, set the integrator
explicitly:

```python
options.integrator = ps.Integrator.BDF1
```

Adding segment-primary support to TRBDF2 (multistage segment integration)
is logged for a follow-up change.

### Event timing is first-order today

When a PWL device's `should_commute(PwlEventContext)` predicate fires
after a step is accepted, the commutation is committed at the *end* of
that step. The Tustin step itself ran with the old topology spanning the
full `dt`, so the post-event state can drift by up to one `dt` worth of
old-topology integration before the new topology takes over.

The architecture for bisection-to-event time (full second-order accuracy
at the commutation instant) is in place: `Circuit::scan_pwl_commutations(x)`
returns the candidate set, and `Circuit::commit_pwl_commutations(events)`
applies them. The remaining work is to:

1. Bisect linearly in `α ∈ [0, 1]` between `x_prev` and `x_now` to find
   the earliest `α` where any candidate device crosses its threshold
   within `event_tolerance`.
2. Reuse the existing event-aligned step-splitting infrastructure
   (`find_switch_event_time` plus the `event_request.threshold_crossing_time`
   path) so the next solve_step lands exactly at the bisected time with
   the old topology, then commits.

The `pwl_event_bisections` telemetry counter is already wired and reads
zero under the first-order scheduler; it will increment under the
bisecting scheduler.

### Caching of topology factorization is unbounded for now

The segment-stepper caches per-topology factorizations forever. For
circuits with k switching devices, the worst case is `2^k` topologies in
the cache. For k ≤ 12 (typical interleaved 3φ converter with control
sensing), this is comfortable. For larger grids, LRU eviction and a
`PwlTopologyExplosion` diagnostic land in `refactor-linear-solver-cache`
(already archived) — the cache key design is now owned end-to-end by
that change.

## Migration checklist

When converting an existing netlist to PWL mode:

- [ ] Set `simulation.switching_mode: ideal` (or set
      `SimulationOptions.switching_mode = SwitchingMode.Ideal` in code).
- [ ] Set `simulation.integrator: bdf1` to bypass the TRBDF2 auto-profile
      and engage the segment engine end-to-end.
- [ ] If a specific device must remain on the smooth path (e.g., for a
      control-loop linearization), set its
      `components[].switching_mode: behavioral`.
- [ ] Run the simulation and verify `backend_telemetry.state_space_primary_steps`
      is non-zero — every step that lands on the fast path increments it,
      and `dae_fallback_steps` increments when the segment engine bails
      out (mixed Ideal / Behavioral devices, transformer in the netlist,
      etc.).
- [ ] Compare waveforms to the Behavioral baseline; the engine targets
      `≤ 0.5%` parity vs LTspice on PSIM-style switching benchmarks (gate
      G.2 in the change's `tasks.md`).

## Deprecation of the Python retry wrapper

The Python `run_transient` wrapper layer historically applied three
retries with shrinking `dt`, auto-installed high-value bleeder resistors,
and re-tuned Newton/linear solver options to coax difficult
switching circuits through Newton iteration. Once PWL Ideal mode is the
default for switching circuits, the wrapper becomes a no-op cost.

The first step of that deprecation ships behind environment variable
`PULSIM_LEGACY_RETRY_FALLBACK` (default `1` to preserve behavior). When
set to `0`, the wrapper bypasses retries / bleeders entirely and calls
through to a single transient pass:

```bash
PULSIM_LEGACY_RETRY_FALLBACK=0 python my_simulation.py
```

A `DeprecationWarning` fires on the first retry of any run when the
variable is unset, pointing here. The wrapper itself is removed in
`refactor-unify-robustness-policy` once Auto resolves to Ideal by default.

## See also

- [`refactor-pwl-switching-engine`](https://github.com/lgili/Pulsim/blob/main/openspec/changes/refactor-pwl-switching-engine/proposal.md)
  — change proposal, design, and tasks list.
- [Backend Architecture](backend-architecture.md) — where the segment
  engine sits in the solver pipeline.
- [Convergence Tuning Guide](convergence-tuning-guide.md) — what to
  tweak when neither path converges (large signal or pathological
  algebraic loops).
- [Linear-Solver Cache](linear-solver-cache.md) — how the per-topology
  KLU factorization cache amortizes the cost across PWM cycles.
