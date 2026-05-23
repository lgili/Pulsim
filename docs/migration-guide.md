# Migration Guide — v1 → v2 (pulsim 1.0.0)

`pulsim` 1.0.0 ships only the v2 kernel. The legacy v1 surface
(``Circuit``, ``Simulator``, ``YamlParser``, ``Preset``, ``codegen``,
``fmu``, ``schematic``, ``templates``, ``robust=True``, …) was
retired during the 1.0 cycle.

This guide maps every v1 idiom you might have to its v2 equivalent.
If you're hitting an ``AttributeError: pulsim.Foo was a v1 symbol``
at runtime, look up ``Foo`` below.

## At a glance

| Aspect | v1 | v2 |
|---|---|---|
| Top-level import | `import pulsim as ps` | `import pulsim as p` (same alias works) |
| Build a circuit | `ps.Circuit()` + `ps.Resistor()` + `ps.add_component(...)` | `p.CircuitBuilder()` + `b.add_resistor(...)` |
| Run a transient | `ps.Simulator(ckt, opts).run_transient(x0)` | `p.simulate(b, t_end=..., dt=...)` |
| Load from YAML | `ps.YamlParser(opts).load("foo.yaml")` | `p.load_yaml("foo.yaml")` → returns a `CircuitBuilder` |
| AC sweep / Bode | `Simulator.run_ac_sweep(AcSweepOptions(...))` | `p.run_ac_sweep(b, …)` (MNA, fast) or `p.run_fra(b, …)` (swept-sine) |
| Closed-loop control | Hand-rolled callbacks | `p.MixedDomainBlockChain` with `PIController`, `PIDController`, etc. |
| Plot | matplotlib by hand | `p.scope(b, res, signals=[...])`, `p.plot_bode(...)` |
| Discovery | n/a | `p.catalog()`, `p.example("buck_open_loop")` |

## Top-level objects

### `pulsim.Circuit` → `pulsim.CircuitBuilder`

```python
# v1
ckt = ps.Circuit()
r1 = ckt.add_node("n1")          # node names were integer IDs returned by add_node
gnd = ckt.add_node("gnd")
ckt.add_component(ps.Resistor("R1", r1, gnd, 100.0))

# v2 — string node names; helpers per device kind
b = p.CircuitBuilder()
b.add_resistor("R1", "n1", "gnd", 100.0)
```

### `pulsim.Simulator` → `pulsim.simulate(...)`

```python
# v1
opts = ps.SimulationOptions(t_end=1e-3, dt=1e-5)
sim = ps.Simulator(ckt, opts)
res = sim.run_transient(ckt.initial_state())

# v2
res = p.simulate(b, t_end=1e-3, dt=1e-5)
```

For full control over the cache/solver/options pipeline, the
explicit form is still available:

```python
cache = p.PwlStateSpaceCache(b.graph, b.pool)
cache.build(dt=1e-5)
opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)
res  = p.run_transient(cache, b.graph, b.pool, opts,
                       switch_fn=lambda t: p.SwitchStateMask(0))
```

### `pulsim.YamlParser` → `pulsim.load_yaml(...)`

```python
# v1
parser = ps.YamlParser(ps.YamlParserOptions())
ckt, opts = parser.load("circuit.yaml")

# v2
b = p.load_yaml("circuit.yaml")
res = p.simulate(b, t_end=..., dt=...)
```

### `pulsim.Preset` / `pulsim.AdvancedOptions` → explicit `SimulationOptions`

No global presets in v2. Set what you need on `SimulationOptions`
directly. Solver kind, Newton tolerance, line-search globalization,
event-bisection thresholds — all explicit knobs.

## Device models

| v1 | v2 |
|---|---|
| `ps.Resistor("R", a, b, R)` | `b.add_resistor("R", a, b, R)` |
| `ps.Capacitor("C", a, b, C)` | `b.add_capacitor("C", a, b, C)` |
| `ps.Inductor("L", a, b, L)` | `b.add_inductor("L", a, b, L)` |
| `ps.VoltageSource("V", a, b, V)` | `b.add_voltage_source("V", a, b, V)` |
| `ps.CurrentSource("I", a, b, I)` | `b.add_current_source("I", a, b, I)` |
| `ps.Diode("D", a, k, ...)` | `b.add_diode("D", anode, cathode, g_on=..., g_off=..., V_th=...)` |
| `ps.MOSFET(...)` | `b.add_mosfet_with_body_diode(...)` or `b.add_mosfet_level1(...)` (Shichman-Hodges) |
| `ps.IGBT(...)` | `b.add_igbt_level1(...)` |
| `ps.VoltageControlledSwitch(...)` | gate via `p.SwitchStateMask` + `switch_fn(t)` |
| `ps.Transformer(...)` | `b.add_two_winding_transformer(...)` or `b.add_multi_winding_transformer(...)` |
| `ps.SineParams(...)` / `ps.PulseParams(...)` | `b.add_sine_voltage_source(...)` / `b.add_pulse_voltage_source(...)` |
| `ps.PWMVoltageSource(...)` | `b.add_pwm_voltage_source(...)` |

## Control surfaces

The hand-rolled v1 step-callback / `signal_evaluator.py` flows are
replaced by the `MixedDomainBlockChain` executor (runs in C++, no
Python interpreter cost per step):

```python
chain = p.MixedDomainBlockChain()
pi    = chain.add_block(p.PIController(kp=0.5, ki=1200.0))
chain.wire(source="vout", to=pi.input)
chain.wire(source=pi.output, to="duty")

res = p.run_transient_with_chain(
    cache, b.graph, b.pool, opts,
    switch_fn, chain, chain_dt=1e-5,
)
```

Available blocks: `PIController`, `PIDController`, `Comparator`,
`RateLimiter`, `FirstOrderLowPass`, `Clarke`/`Park`/inverse,
``ThyristorBlock``, ``FuseBlock``, ``FosterThermalNetwork``, …

## Analyses

| v1 | v2 |
|---|---|
| `Simulator.run_ac_sweep(...)` | `p.run_ac_sweep(b, f_start=..., f_stop=..., n_points=...)` (MNA, fast) |
| `Simulator.run_fra(...)` | `p.run_fra(b, ...)` — swept-sine FRA (closed-loop and nonlinear-friendly) |
| `Simulator.run_periodic_steady_state(...)` | `p.run_periodic(b, ...)` (Newton-based shooting) |
| `pulsim.sweep.run(...)` (Monte-Carlo / LHS / Cartesian) | `p.run_parameter_sweep(b_factory, samples, metrics)` |

## Features that did NOT migrate

| Feature | Status on 1.0.0 |
|---|---|
| `pulsim.codegen` (C99 codegen) | not in v2 |
| `pulsim.fmu` (FMI 2.0 CS export) | not in v2 |
| `pulsim.schematic` (ELK + netlistsvg auto-layout) | not in v2 |
| `pulsim.templates.{buck,boost,buckboost,…}` (converter auto-design) | not in v2 |
| `pulsim.Preset` / `AdvancedOptions` global presets | replaced by explicit `SimulationOptions` knobs |
| `compressor_load` + R600a/R134a refrigerants | not in v2 |
| MMC arm templates | not in v2 (single-arm helper deferred) |
| `RobustnessProfile` retry-loop | not in v2 (use `enable_nonlinear_refresh=True` + DC-OP strategies) |
| Single-phase induction motor (PSC) | not in v2 (DC motor, BLDC, PMSM, induction are) |

If you need any of these on v2, pin pulsim ``0.10.x`` or open an
issue on the tracker — the v2 architecture supports each of them as
a future increment, none was retired for technical reasons.

## See also

- ``examples/scripts/`` — 20 runnable v2 reference scripts.
- ``docs/tutorials/`` — six narrative tutorials.
- ``docs/api-reference.md`` — the v2 surface.
- ``docs/gotchas.md`` — every footgun we've hit so far.
