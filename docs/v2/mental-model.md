# The mental model

Pulsim v2 has **four** big pieces. Once you understand what they do, the rest of the code reads like a translation of standard MNA + Newton textbook material.

```
   ┌──────────────┐
   │ CircuitBuilder │  ──┐
   └──────────────┘     │   "name → node id" + ergonomic add_* helpers
                        │
                        ▼
   ┌──────────────┐  ┌─────────────┐
   │    Graph     │  │ DevicePool  │   topology (kind of each branch) +
   │ branches:    │  │ params:     │   per-device parameter storage
   │   from, to,  │  │   R, L, C,  │   (Resistor, Capacitor, Inductor,
   │   kind       │  │   diode, …) │   VoltageSource, MOSFET, IGBT, …)
   └──────┬───────┘  └─────┬───────┘
          │                │
          ▼                ▼
       ┌───────────────────────────┐
       │  PwlStateSpaceCache       │   For each REACHABLE switch
       │  (one factorized matrix   │   combination + dt, build the MNA
       │   per switch mask + dt)   │   matrix once, KLU-factorize once.
       └────────────┬──────────────┘
                    │
                    ▼
       ┌───────────────────────────┐
       │       run_transient       │   Fixed-dt loop: lookup the cache
       │  + optional Newton refresh │   entry for the current switch
       │  + optional event detect   │   state, solve, record sample,
       └───────────────────────────┘   call nl_refresh if any nonlinear
                                        device is present.
```

## 1. Graph + DevicePool — the circuit description

The `Graph` stores topology: each branch has a `from` node id, a `to` node id, and a `kind` enum:

```cpp
enum class BranchKind {
    PassiveLinear,   // R, L, C
    Source,          // V-source, I-source, PWM, Sine, Pulse, VCVS
    Switch,          // controlled switch + ideal-blend diode
    Nonlinear,       // smooth IdealDiode, SH1 MOSFET, IGBT, SaturableInductor
};
```

The `DevicePool` stores **parameters** for each branch in a `std::variant`. The `StoredKind` enum mirrors the variant index so that `pool.kind_of(branch_id)` returns the actual device type at O(1) cost.

You almost never touch `Graph` or `DevicePool` directly — you go through `CircuitBuilder` (or its Python alias) and use `b.add_resistor`, `b.add_mosfet_level1`, etc.

## 2. PwlStateSpaceCache — the trick that makes v2 fast

A switching converter spends 99 % of its time in just a few **discrete switch configurations** (e.g. "high-side ON, low-side OFF, freewheel diode reverse-blocking" for a buck during the ON portion). For each combination, the MNA matrix is fixed.

The cache enumerates every reachable combination (`SwitchStateMask`), stamps the MNA matrix once, and KLU-factorizes it. Then `run_transient` just does:

```
for k in 0 .. N:
    t = t_start + k*dt
    mask = switch_fn(t)               # what's the switch state?
    cache_entry = cache.lookup(mask)  # O(1) hash
    cache_entry.solver.solve(b)       # already factored → just back-sub
    record sample
```

Trapezoidal-rule companion models for L and C are pre-baked into each cached entry, so the entire dynamic step is just a single triangular solve.

**Consequence:** the first build can be slow (and grows exponentially with the number of switches), but every subsequent solve is cheap. Circuits with a handful of switches (typical for SMPS) build in milliseconds and run hundreds of thousands of steps per second.

## 3. Newton refresh — for nonlinear devices

Smooth-blend `IdealDiode`, SH1 `MosfetLevel1`, Level-1 `IgbtLevel1`, and `SaturableInductor` are **not** pre-baked — their stamping depends on the operating point, which Newton has to find iteratively.

For these devices, each time step runs a Newton loop:

```
x_k+1 = x_k - (J_linear + J_nonlinear(x_k))⁻¹ · (f_linear(x_k) + f_nonlinear(x_k))
```

The "linear" Jacobian is the cached PWL one; the nonlinear refresh function (`refresh_smooth_diodes`, `refresh_mosfets_level1`, etc.) only rebuilds the contributions from the nonlinear branches each iteration. The combined nonlinear refresh is what `simulate(...)` auto-enables when `pool.has_nonlinear_devices()` returns True.

The Newton loop is hardened with:
- **Line search globalization** (V4)
- **Levenberg-Marquardt trust region** (V5)
- **Pseudo-transient continuation** (V10)
- **Smart warm-start with V_F0 continuation** (V9)

These kick in only when needed — most tame circuits converge in 2-5 plain Newton iterations.

## 4. Event detection — bisecting commutations

Diodes and ideal switches change state when a current/voltage crosses zero. v2 detects the crossing within a sample step and **bisects** to find the exact `t_event`, then restarts the integration from that instant with the new switch state. This keeps the simulation accurate even when commutations happen between samples.

## YAML pathway

For circuits that don't need programmatic generation, you can author a YAML file and load it:

```python
loaded = p.load_yaml_file("examples/v2/buck.yaml")
res = p.simulate(loaded.builder, t_end=loaded.t_end, dt=loaded.dt,
                 switch_fn=loaded.switch_fn)
```

The YAML loader builds a `LoadedCircuit` containing the populated `CircuitBuilder`, default time window, and a `switch_fn` that combines all declared PWM helpers.

## How a tutorial walks through this

Every tutorial in this guide hits the same three checkpoints:

1. **Circuit description** — node list, components, switch driver.
2. **Run the transient** — call `simulate()` or the explicit pipeline.
3. **Look at the result** — print samples, plot, sanity-check against analytical expectations.

Start with [01 — RC charging](tutorials/01-rc-charging.md).
