# Control graph: order, rates and delay

A `MixedDomainBlockChain` (and the `CxxBlockChain` it compiles to) is a
**control graph**: blocks connected by named signals, evaluated by the
kernel with no Python call per step. This page says in which order the
blocks run, when they run, and when their outputs reach the circuit.

## Order

Blocks run in the topological order of the *direct-feedthrough* graph:
a block runs after every block whose output it reads in the same
instant. Ties keep insertion order, so a chain written in order keeps
its order, and any insertion order gives the same numbers.

| Block | Inputs that order the schedule |
|---|---|
| every block except `DelayBlock` | all (their output is computed from this instant's input, the stateful ones included) |
| `DelayBlock` | none (its output is the input of `samples` evaluations ago) |

A cycle through feedthrough inputs is an **algebraic loop**: building
the graph raises `pulsim.ControlGraphBuildError` naming its blocks.
Insert a `DelayBlock` to break it. A block reading its own output and a
signal written by two blocks are build errors too.

`DelayBlock` runs in two phases: every block computes its output first,
then the delays shift in this instant's input.

### Migration: chains that relied on insertion order

Before 2.1 the chain ran in insertion order: a block inserted before the
block feeding it read last step's value (0 on the first step), a
one-step delay nobody wrote down. Building such a chain now warns once:

```
control graph: evaluation order differs from insertion order (each was a
one-step delay before the schedule): 'half' reads 'u' written by 'gain', ...
```

To keep the old delay, write it: feed the consumer through
`DelayBlock(samples=1)`. Otherwise the new (scheduled) numbers are the
intended ones.

## Rates

Without a `Rate` anywhere, a chain runs on every solver step with the
`dt` given to `make_step_observer` (the fixed engine only, as before).

With rates, each block belongs to a group that ticks at
`t_start + offset + k*period` (integer `k`, never an accumulated time):

```python
ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))            # default group
ch.add("pi", p.PIController(Kp=0.1, Ki=10),
       inputs=dict(setpoint=1.0, measured="out", dt="dt"), output="u")
ch.add("mon", p.FirstOrderLowPass(tau=1e-3),
       inputs=dict(input_value="out", dt="dt"), output="m",
       rate=p.Rate(5e-5))                                  # its own group
ch.drive_source(b, "u", ("v", "src", "gnd"))               # a ZOH source
r = p.simulate(b, 20e-3, step_observer=ch.make_step_observer(b))
ch.tick_counts                                             # [200, 400]
```

- A block's rate is its own `rate=`; a stateless block without one
  inherits the fastest rate among the blocks feeding it; otherwise the
  chain's `rate=`. A stateful block left without a rate is a build
  error.
- `dt` passed to a block is its group's period.
- Groups due at the same instant run in one pass of the schedule, so a
  fast group reading a slow group's signal sees that instant's value.
- Ticks fire for every `k` with an instant before `t_end`:
  `ceil((t_end - t_start - offset) / period)` of them, on `pwl` and on
  `trbdf2` alike. On `pwl` a tick reads the state at its own instant
  (the step grid; an off-grid instant is served at the first step
  after it); on `trbdf2` the engine lands a step on every tick.

## Compute delay

`Rate(period, compute_delay=...)` says when a group's outputs reach the
circuit (driven sources, switch masks); blocks always read the current
signals.

| `compute_delay` | Outputs take effect | Engines |
|---|---|---|
| `0` (default) | at the tick | all |
| a time `0 < Td < period` | at the first instant ≥ tick + Td | `pwl` (refused on `trbdf2`) |
| `"one_sample"` | at the next tick (a microcontroller writing the duty at the next carrier period) | all |

A `Rate` with an `offset` is refused on `trbdf2` too: its tick scheduler
lands steps at `t_start + k*T` only.

### Exact zero-order hold on `pwl`

By default the fixed engine averages a source jump into the step after
it (its trapezoidal history, first order after any discontinuity; see
[known limitations](known-limitations.md)). A driven source declares its
tick instants, so with `discontinuity_restart=True` the engine restarts
on each tick and the hold is exact:

| Discrete P loop, `v[10]` (closed form 0.643545722992) | error |
|---|---|
| `pwl`, `dt` = 1 µs | 6.148e-5 |
| `pwl`, `dt` = 0.1 µs | 6.138e-6 |
| `pwl`, `dt` = 1 µs, `discontinuity_restart=True` | 7.2e-9 |
| `pwl`, `dt` = 0.1 µs, `discontinuity_restart=True` | 7.3e-11 |
| `trbdf2`, `rtol=1e-10` | 7.8e-9 |
| `trbdf2`, `rtol=1e-12` | 3.7e-10 |

## Not yet

Continuous-rate blocks solved with the circuit, tick events on the
unified event queue, modulators and comparator state events, and graph
state in the solver snapshot (tasks 6–7 of `add-compiled-control-graph`).
