# Gotchas

Most pulsim simulations Just Work. The handful that don't almost always fail for one of the reasons below. This page is the field-debugging guide.

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

Since v2.0 this is not a single solve that either works or aborts the run. It is a cascade, tried in order until one rung answers:

| Rung | What it varies | What it fixes |
|---|---|---|
| naive | nothing | the common case; costs one solve |
| gmin stepping | a conductance from every node to ground, 1e-2 S down to 1e-12 S by decades | a badly-pivoted matrix, and a Newton with no basin to start from |
| source stepping | every independent source amplitude, 0 → nominal | a Newton basin problem |
| pseudo-transient | an artificial `dx/dt = -F` | resistive-nonlinear problems with no constraint rows |

Both homotopies bisect when a step is too wide, and each rung warm-starts from the last. If all four fail you get one message naming every rung and why it stopped — and a pointer to `strategy="settle"`, which runs an actual transient and is the only route to a *switching* steady state.

You can drive the same cascade directly:

```python
report = []
x = p.compute_dc_op(b, report=report)     # strategy="auto" by default
print(report[0].summary())                # which rung answered
```

**`compute_dc_op` stamps nonlinear devices.** Before v2.0 it did not: a diode fed from 5 V through 1 kΩ was reported at 5.000 V instead of 0.700 V, because `dc_assemble` treats `BranchKind::Nonlinear` as an open circuit and nothing put it back. If you want the linear system, ask for it with `enable_nonlinear_refresh=False`.

#### (c2) gmin, and what it will not do for you

Every DC solve stamps a 1e-12 S conductance from each node to ground — SPICE's `GMIN`. On a 12 V node that is 12 pA, three decades below the 1 GΩ reference ties the topology preflight inserts, so the two never compete. Pass `gmin=0` to reproduce the un-augmented system exactly.

What it deliberately will **not** do is give a *floating* node an equation. A conductance on every node would make an unreferenced node solvable and report a confident 0 V for a node whose voltage is undefined — so every DC entry point checks the un-augmented system for structural rank deficiency first, and the named error wins:

```
compute_dc_op: DC matrix structurally singular for mask 0b0 N=0
  — node 's1' is in a 2-node subnet with no connection to ground
  through any device, so its voltage is undefined rather than merely
  hard to compute...
```

gmin conditions matrices; the preflight repairs topology. Neither covers for the other.

#### (d) Sharp sigmoid (`kappa` too high)

MOSFETs / IGBTs use sigmoid blending between regions; `kappa` controls how sharp. A `kappa = 50` is more accurate at the boundary but makes the Jacobian condition number worse.

**Fix:** dial `kappa` down to 10–15. Since v2.0 the DC cascade recovers a 12-diode chain at `kappa = 50` on its own (gmin stepping gets it), but `kappa = 5000` defeats every rung — at that sharpness you are asking a smooth model to be an ideal switch, and `add_diode()` (the PWL switch diode) is the right device instead.

### Symptom: residual oscillates but doesn't shrink

You hit a Newton "limit cycle" — the iterate ping-pongs between two faces of a non-smooth feature.

**Fix:** the trust-region Newton (V5) and pseudo-transient continuation (V10) are designed for this. Set `opts.max_newton_iterations = 50` and let the solver escalate strategies internally.

## Cache size + build time

### Symptom: `cache.build(dt)` takes forever

The cache enumerates all `2^N` reachable switch combinations. With `N = 8` switches that's 256 entries; with `N = 12` it's 4096. Each entry stamps the MNA matrix + KLU-factorizes it.

**Fix:** `simulate()` builds lazily (v1.8+) — only states actually reached by `switch_fn` are factored, so many-switch circuits start immediately. If you drive the cache DIRECTLY, `cache.build(dt)` is the eager 2^N path; call `cache.build_lazy(dt)` instead (`cache.num_built_segments()` shows how many states the run really visited). For pathological topologies, consider partitioning into sub-circuits.

### Symptom: KLU singular-matrix error

Cache build fails with `compute_dc_op: DC matrix structurally singular for mask ...`.

This usually means:
- A node is dangling (no connection to ground).
- All sources are current sources and the topology has no DC path to ground.
- A nonlinear branch (`BranchKind::Nonlinear`) has no diagonal contribution. Pulsim already adds a 1e-12 G_min for `SaturableInductor`; for other custom devices you may need to add your own.

**Fix (v2.0 and later): none needed.** `simulate()` runs a topology
preflight that finds unreferenced subnets and ties each one to ground
through a 1 GΩ resistor, then tells you what it inserted. Read it with
`result._preflight`, or pass `auto_regularize=False` to get the error
back. What follows is what that pass does for you, and what to type if
you build the cache yourself.

**Manual fix:** check the topology with a small dump (`pool.state_size(graph)` should equal `num_active_nodes + num_sources + num_inductors`). For an isolated subnet (e.g. a transformer secondary) add a HIGH-value tie to ground — `b.add_resistor("R_iso", "sec_gnd", "gnd", 1e9)` — which gives the MNA a DC reference without bonding the nets. A 1 µΩ resistor is a deliberate galvanic BOND: applied to a live floating node it silently changes the circuit.

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

Every isolated subnet must have a DC reference. Since v2.0 `simulate()`
supplies one for you — see the preflight note above — so this section is
about the cases where you want to place the tie yourself, or where you
need a BOND rather than a reference.

A galvanically-isolated transformer secondary needs a tie to primary ground: use a HIGH-value resistor (1 MΩ–1 GΩ) when you only need the MNA reference, or a deliberate low-value bond (1 µΩ / 0 V source) when the nets are truly common. The two are NOT interchangeable — a 1 µΩ tie on a node that carries signal silently shorts it, and preflight will never insert one: it only ever adds references.

**Fix:** nothing, if you let `simulate()` do it. Otherwise
`b.add_resistor("R_iso", "sec_gnd", "gnd", 1e9)` for a reference-only tie (leakage ~nA, invisible in the waveforms), or `1e-6` only when you intend an actual bond between the grounds.

### Symptom: switch_fn never fires the bit I expect

`SwitchStateMask` indexes by **switch order in the graph**, not by `branch_id`. The mapping is established by `add_switch` / `add_diode` insertion order.

**Fix:** inspect `b.graph.num_switches` and check the insertion sequence. The first switch added gets bit 0, the next gets bit 1, etc.

## Python-specific pitfalls

### Symptom: `AttributeError: module 'pulsim' has no attribute 'simulate'`

The installed `pulsim` wheel is stale relative to the source tree.

**Fix:** if you're developing locally, set `PYTHONPATH=build/python:$PYTHONPATH` or do `pip install -e .`. Confirm via `print(pulsim.__file__)` that you're using the development tree.

### Symptom: NumPy-array vs. list confusion

`SimulationResult.states[k]` is a NumPy 1-D array; `SimulationResult.times` is a Python list of floats.

**Fix:** pass `times` through `np.asarray(...)` if you need to do vectorized arithmetic. `res.states` is already a 2-D numpy array (v2.0), so `res.states[k][node_id]`, `res.states[:, node_id]` and `res.states[k, node_id]` all work directly — but it is a READ-ONLY view over kernel memory, so `.copy()` first if you need to modify it in place.

## When to ask for help

If you hit something not listed here, the best signals are:

1. **Minimum repro YAML or 10-line Python snippet.**
2. **`opts.max_newton_iterations = 50; print(res.commutations)`** — the event log shows which switch flipped when, often pointing at the root cause.
3. **`cache.dt()` value** — confirms you didn't accidentally pass `dt = 0` (static-only cache).

Open an issue with those three pieces of evidence and we can usually pinpoint the problem in one round-trip.
