# Migration Guide — pre-1.0 → 1.0

Pulsim 1.0.0 shipped a brand-new kernel. The pre-1.0 API surface
(``Circuit``, ``Simulator``, ``YamlParser``, ``Preset``, ``codegen``,
``fmu``, ``schematic``, ``templates``, ``robust=True``, …) was
retired during the 1.0 cycle.

This guide maps every legacy idiom to its 1.0 equivalent. If you're
hitting an ``AttributeError: pulsim.Foo was a v1 symbol`` at runtime,
look up ``Foo`` below.

## At a glance

| Aspect | Pre-1.0 (`v1`) | 1.0 |
|---|---|---|
| Top-level import | `import pulsim as ps` | `import pulsim as p` (same alias works) |
| Build a circuit | `ps.Circuit()` + `ps.Resistor()` + `ps.add_component(...)` | `p.CircuitBuilder()` + `b.add_resistor(...)` |
| Run a transient | `ps.Simulator(ckt, opts).run_transient(x0)` | `p.simulate(b, t_end=..., dt=...)` |
| Load from YAML | `ps.YamlParser(opts).load("foo.yaml")` | `p.load_yaml_file("foo.yaml")` |
| AC sweep / Bode | `Simulator.run_ac_sweep(AcSweepOptions(...))` | `p.run_ac_sweep(b, …)` (MNA, fast) |
| Swept-sine FRA | `Simulator.run_fra(...)` | `p.run_fra(b, …)` (closed-loop friendly) |
| Closed-loop control | Hand-rolled callbacks / `signal_evaluator.py` | `p.MixedDomainBlockChain` + `PIController`, `PIDController`, … |
| Plot | matplotlib by hand | `p.scope(b, res, signals=[...])`, `p.plot_bode(...)` |
| Discovery | n/a | `p.catalog()`, `p.example("buck_open_loop")` |

The 1.0 surface is **flat** — there's no `pulsim.v2` namespace. All
device builders, analyses, and helpers live directly on the
`pulsim` module.

## Top-level objects

### `pulsim.Circuit` → `pulsim.CircuitBuilder`

```python
# Pre-1.0
ckt = ps.Circuit()
r1 = ckt.add_node("n1")          # node names were integer IDs returned by add_node
gnd = ckt.add_node("gnd")
ckt.add_component(ps.Resistor("R1", r1, gnd, 100.0))

# 1.0 — string node names; helpers per device kind
b = p.CircuitBuilder()
b.add_resistor("R1", "n1", "gnd", 100.0)
```

### `pulsim.Simulator` → `pulsim.simulate(...)`

```python
# Pre-1.0
opts = ps.SimulationOptions(t_end=1e-3, dt=1e-5)
sim = ps.Simulator(ckt, opts)
res = sim.run_transient(ckt.initial_state())

# 1.0
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

### `pulsim.YamlParser` → `pulsim.load_yaml_file(...)`

```python
# Pre-1.0
parser = ps.YamlParser(ps.YamlParserOptions())
ckt, opts = parser.load("circuit.yaml")

# 1.0
loaded = p.load_yaml_file("circuit.yaml")
b = loaded.builder
res = p.simulate(b, t_end=..., dt=...)
```

### `pulsim.Preset` / `pulsim.AdvancedOptions` → explicit `SimulationOptions`

No global presets in 1.0. Set what you need on `SimulationOptions`
directly. Solver kind, Newton tolerance, line-search globalization,
event-bisection thresholds — all explicit knobs.

## Device models

| Pre-1.0 | 1.0 |
|---|---|
| `ps.Resistor("R", a, b, R)` | `b.add_resistor("R", a, b, R)` |
| `ps.Capacitor("C", a, b, C)` | `b.add_capacitor("C", a, b, C)` |
| `ps.Inductor("L", a, b, L)` | `b.add_inductor("L", a, b, L)` |
| `ps.VoltageSource("V", a, b, V)` | `b.add_voltage_source("V", a, b, V)` |
| `ps.CurrentSource("I", a, b, I)` | `b.add_current_source("I", a, b, I)` |
| `ps.Diode("D", a, k, ...)` | `b.add_diode("D", anode, cathode, g_on=..., g_off=..., V_th=...)` |
| `ps.MOSFET(...)` | `b.add_mosfet_level1(...)` (Shichman-Hodges) or `b.add_mosfet_with_body_diode(...)` |
| `ps.IGBT(...)` | `b.add_igbt_level1(...)` |
| `ps.VoltageControlledSwitch(...)` | gate via `p.SwitchStateMask` + `switch_fn(t)` |
| `ps.Transformer(...)` | `b.add_two_winding_transformer(...)` or `b.add_multi_winding_transformer(...)` |
| `ps.SineParams(...)` | `b.add_sine_voltage_source(...)` |
| `ps.PulseParams(...)` | `b.add_pulse_voltage_source(...)` |
| `ps.PWMVoltageSource(...)` | `b.add_pwm_voltage_source(...)` |
| `ps.SaturableInductor(...)` | `b.add_saturable_inductor(...)` |

## Control surfaces

The hand-rolled pre-1.0 step-callback / `signal_evaluator.py` flows
are replaced by the `MixedDomainBlockChain` executor (runs in C++,
no Python interpreter cost per step):

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
`ThyristorBlock`, `FuseBlock`, `FosterThermalNetwork`, ...

## Analyses

| Pre-1.0 | 1.0 |
|---|---|
| `Simulator.run_ac_sweep(...)` | `p.run_ac_sweep(b, freqs=..., excite_fn=..., output_idx=...)` (swept-sine) or `p.run_mna_sweep(b, ...)` (linearised, faster) |
| `Simulator.run_fra(...)` | `p.run_fra(...)` — swept-sine FRA (closed-loop and nonlinear-friendly) |
| `Simulator.run_periodic_steady_state(...)` | not yet on 1.0 — file an issue if you need it |
| `pulsim.sweep.run(...)` (Monte-Carlo / LHS / Cartesian) | `p.sweep(builder_factory, ...)` and `p.monte_carlo(...)` |

## Features that did NOT migrate

| Feature | Status on 1.0.0 |
|---|---|
| `pulsim.codegen` (C99 codegen) | not on 1.0 |
| `pulsim.fmu` (FMI 2.0 CS export) | not on 1.0 |
| `pulsim.schematic` (ELK + netlistsvg auto-layout) | not on 1.0 |
| `pulsim.templates.{buck,boost,buckboost,…}` (converter auto-design) | not on 1.0 |
| `pulsim.Preset` / `AdvancedOptions` global presets | replaced by explicit `SimulationOptions` knobs |
| `compressor_load` + R600a/R134a refrigerants | not on 1.0 |
| MMC arm templates | not on 1.0 (single-arm helper deferred) |
| `RobustnessProfile` retry-loop | not on 1.0 (use `enable_nonlinear_refresh=True` + DC-OP strategies) |
| Single-phase induction motor (PSC) | not on 1.0 (DC motor, BLDC, PMSM, three-phase induction are) |

If you need any of these on 1.0, pin pulsim `0.10.x` or open an
issue on the tracker — the 1.0 architecture supports each of them
as a future increment, none was retired for technical reasons.

## A note on the `pulsim.v2` namespace

Versions `0.10.x` exposed a parallel `pulsim.v2` surface while the
new kernel was being built next to the legacy one. With the legacy
kernel retired and the new kernel becoming the default in 1.0.0,
the `v2` prefix was dropped: `pulsim.v2.CircuitBuilder` is now
`pulsim.CircuitBuilder`, `pulsim.v2.simulate` is `pulsim.simulate`,
and so on. Code that used the `v2` prefix during the transition
just needs:

```python
# Before
import pulsim.v2 as p
from pulsim.v2 import CircuitBuilder, simulate

# After
import pulsim as p
from pulsim import CircuitBuilder, simulate
```

The C++ side mirrors this: `pulsim::v2::Foo` is now `pulsim::Foo`,
headers moved from `pulsim/v2/<sub>/<file>.hpp` to
`pulsim/<sub>/<file>.hpp`, and the CMake alias renamed from
`pulsim::v2` to `pulsim::core`.

## See also

- `examples/scripts/` — 20 runnable reference scripts.
- `docs/tutorials/` — six narrative tutorials.
- `docs/api-reference.md` — the full surface in one page.
- `docs/gotchas.md` — every footgun we've hit so far.
