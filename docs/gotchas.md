# Gotchas

Most v2 simulations Just Work. The handful that don't almost always fail for one of the reasons below. This page is the field-debugging guide.

## Newton convergence

### Symptom: "Newton iteration did not converge"

The transient throws `std::runtime_error` partway through. Common causes:

#### (a) Instantaneous gate edges on an inductive load

A `PulseVoltageSource` with `rise_time = fall_time = 0` driving a MOSFET/IGBT gate, with an inductor in the drain/collector path, will diverge within a few PWM cycles. The trapezoidal companion model for `L` can't follow a discontinuous current command.

**Fix:** add realistic ramps:

```python
b.add_pulse_voltage_source(
    "Vg", "g", "gnd",
    v_pulsed=15.0,
    rise_time=100e-9, fall_time=100e-9,   # ← critical
    pulse_width=..., period=...,
)
```

100 ns is realistic for a real gate driver and gives Newton ~5 iterations to follow each edge smoothly at `dt = 20 ns`.

#### (b) MOSFET without a body diode

When the high-side MOSFET opens, the inductor needs a freewheel path. Without one, the MNA matrix briefly becomes structurally singular.

**Fix:** use `add_mosfet_level1(..., with_body_diode=True)` (proposal #3.1) or add an explicit anti-parallel `add_diode("Body", source, drain, ...)`.

#### (c) Cold start at `x = 0`

Some op-amp / feedback circuits have a non-trivial DC operating point that `x = 0` is far from. Newton can wander before converging.

**Fix:** seed from the DC OP:

```python
res = p.simulate(b, ..., start_from_dc_op=True)
```

This calls `compute_dc_op(graph, pool, mask, opts.t_start)` first and feeds the result into `HistoryState`.

#### (d) Sharp sigmoid (`kappa` too high)

MOSFETs / IGBTs use sigmoid blending between regions; `kappa` controls how sharp. A `kappa = 50` is more accurate at the boundary but makes the Jacobian condition number worse.

**Fix:** dial `kappa` down to 10–15.

### Symptom: residual oscillates but doesn't shrink

You hit a Newton "limit cycle" — the iterate ping-pongs between two faces of a non-smooth feature.

**Fix:** the trust-region Newton (V5) and pseudo-transient continuation (V10) are designed for this. Set `opts.max_newton_iterations = 50` and let v2 escalate strategies internally.

## Cache size + build time

### Symptom: `cache.build(dt)` takes forever

The cache enumerates all `2^N` reachable switch combinations. With `N = 8` switches that's 256 entries; with `N = 12` it's 4096. Each entry stamps the MNA matrix + KLU-factorizes it.

**Fix:** lazy-build (Layer 4 V6) is the default — only states reached by `switch_fn` are factored. If yours isn't lazy, check that you haven't pinned all bits via a custom switch driver. For pathological topologies with very many switches, consider partitioning into sub-circuits.

### Symptom: KLU singular-matrix error

Cache build fails with `compute_dc_op: DC matrix structurally singular for mask ...`.

This usually means:
- A node is dangling (no connection to ground).
- All sources are current sources and the topology has no DC path to ground.
- A nonlinear branch (`BranchKind::Nonlinear`) has no diagonal contribution. v2 already adds a 1e-12 G_min for `SaturableInductor`; for other custom devices you may need to add your own.

**Fix:** check the topology with a small dump (`pool.state_size(graph)` should equal `num_active_nodes + num_sources + num_inductors`). Add a 1 µΩ resistor between any floating node and ground.

## Time-step (`dt`) choices

### Symptom: ringing or instability with reasonable circuits

If `dt` is too large compared to the system's smallest time constant, the trapezoidal companion can ring (overshoot a true exponential by 10-20 %).

**Heuristic:** use `dt ≤ min(τ) / 10` where `τ` are the RC and L/R time constants. For switching circuits, also enforce `dt ≤ T_PWM / 200` (catch the ON/OFF edge within 1 sample even when the PWM frequency drifts).

### Symptom: wall-clock time too long

`run_transient` is linear in `N_steps`. If a 100 µs simulation at `dt = 1 ns` takes too long, you have 100,000 steps — even at 10 µs per step, that's a second.

**Fix:** profile to confirm where time is spent. The cache lookup is O(1); the trap solve is O(state_size) for a triangular back-sub. Newton refresh dominates for nonlinear circuits.

## Builder / API pitfalls

### Symptom: "branch_id N is not a Resistor"

You called the wrong typed accessor on `DevicePool`. The `kind_of(b_id)` casts the variant index directly to `StoredKind`.

**Fix:** always go through the typed accessor matching `pool.kind_of(b_id)`, or use the variant directly.

### Symptom: forgotten ground

Every isolated subnet must have a path to ground. Even galvanically-isolated transformer secondaries need a 1 µΩ resistor to primary ground (or use a `BranchKind::Source` like a 0 V source to tie the secondary somewhere).

**Fix:** add `b.add_resistor("R_iso", "sec_gnd", "gnd", 1e-6)`. The current flowing through `R_iso` will be in the picoamps and doesn't perturb the result.

### Symptom: switch_fn never fires the bit I expect

`SwitchStateMask` indexes by **switch order in the graph**, not by `branch_id`. The mapping is established by `add_switch` / `add_diode` insertion order.

**Fix:** inspect `b.graph.num_switches` and check the insertion sequence. The first switch added gets bit 0, the next gets bit 1, etc.

## Python-specific pitfalls

### Symptom: `AttributeError: module 'pulsim.v2' has no attribute 'simulate'`

The installed `pulsim` wheel is stale relative to the source tree.

**Fix:** if you're developing locally, set `PYTHONPATH=build/python:$PYTHONPATH` or do `pip install -e .`. Confirm via `print(pulsim.__file__)` that you're using the development tree.

### Symptom: NumPy-array vs. list confusion

`SimulationResult.states[k]` is a NumPy 1-D array; `SimulationResult.times` is a Python list of floats.

**Fix:** pass `times` through `np.asarray(...)` if you need to do vectorized arithmetic. For state, just index directly: `res.states[k][node_id]`.

## When to ask for help

If you hit something not listed here, the best signals are:

1. **Minimum repro YAML or 10-line Python snippet.**
2. **`opts.max_newton_iterations = 50; print(res.commutations)`** — the event log shows which switch flipped when, often pointing at the root cause.
3. **`cache.dt()` value** — confirms you didn't accidentally pass `dt = 0` (static-only cache).

Open an issue with those three pieces of evidence and we can usually pinpoint the problem in one round-trip.
