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

## Blocks

Every block of `pulsim.control` that a chain can hold runs in the kernel
graph. The ones below are ported operation for operation from their
Python class and are bit-identical to it (10⁴ random inputs each,
`test_control_graph_blocks.py`): `Comparator`, `RateLimiter`,
`SampleHold`, `TransferFunction`, `StateMachine`, `OpAmp`, `DelayBlock`,
`SignalMux`, `SignalDemux`, `LookupTable1D`, `MathBlock`,
`MovingAverageFilter`, `Differentiator`, `PIDController`, `Sum`,
`Integrator`, `FirstOrderLowPass`, `PwmGenerator`; and `PIController`,
`Gain`, `Subtract`, `Limiter` (kernel blocks, same numbers). Not yet
compared byte for byte: `SpaceVectorModulator`, the Clarke / Park
transforms and the PLL.

A block's initial state (`PIController(integrator_state=...)`,
`Comparator(state=True)`, `FirstOrderLowPass(state=...)`, a
`TransferFunction`'s history, ...) is the kernel block's initial state.
`chain.reset()` zeroes the graph: block states, signals, ticks and
recordings.

N-input blocks take a tuple: `Sum(weights=(1, -1, 0.5))` with
`inputs=dict(inputs=("channel:a", "channel:b", "vout"))`, and
`SignalMux` with `inputs=dict(selector=..., inputs=(...))`. `Sum` also
accepts `a` / `b` / `c`; its weights must match the inputs given.

Custom Python logic goes in `pulsim.PythonBlock(fn)`: `fn(**inputs)`
returns a float, or a tuple for a tuple of output names. It costs one
Python call per evaluation.

### Migration: the Python evaluator is gone

| Before (2.0) | Now |
|---|---|
| A chain holding a block the kernel did not map ran entirely on the Python evaluator, behind a warning. | `ControlGraphBuildError` naming the block; wrap it in `PythonBlock`. |
| `make_step_observer(..., use_kernel=False)` | Raises `DeprecationWarning`. Drop it; `record_channel` works on the kernel graph. |
| `record_channel(name)` before `make_step_observer` recorded nothing on the kernel path. | Recorded. |
| `PIDController` in a chain ran the kernel PID (derivative on the measurement, `tau_d = 0`): not the class's law. | The class's law (derivative on the error, `derivative_alpha`). |
| `Sum` ignored its `weights` (and a `c` input). | Weights applied; a count mismatch is a build error. |
| `Differentiator`, `MovingAverageFilter`, `MathBlock` (any chain holding one ran on the Python evaluator). | Kernel, same numbers. |
| `Integrator` ran the kernel law (no anti-windup hold, `gain*x*dt`): 9559 of 10⁴ random inputs differed from the class, by up to 2.3e-2. | The class's law. |
| `FirstOrderLowPass` with `dt = 0` passed the input through, and started from 0 whatever its `state=`. | Holds, as the class does; starts from `state`. |
| `PwmGenerator(phase=...)` read `phase` as a fraction of the period, carrier starting at `-phase*T` (the class: seconds, starting at `+phase`); `frequency <= 0` gave `duty > 0` instead of 0. | The class's law. |
| `SlidingModeObserver`, `FluxMRASObserver` ran on the Python evaluator. | `PythonBlock` adapters (same numbers, one Python call per evaluation). |
| A block's initial state (`integrator_state=`, `state=`) was ignored by the kernel `PIController`. | Honoured. |

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
  chain's `rate=`. A block left without a rate is a build error.
- A `PwmGenerator` in a rated chain needs its own `rate=` (on `pwl`,
  `Rate(dt)`: the carrier is compared every step); inheriting the
  controller's rate would sample the gate at the ticks, so it is a
  build error.
- On `pwl` a period below the solver step is a build error (a group
  ticks at most once per step). `controller_period=` with a rated
  chain is a `ValueError`: the groups set the ticks.
- A reused observer (a second `simulate()`) restarts its tick grid at
  the new `t_start`; block state carries over unless you call
  `chain.reset()`. With `resume_from=` the grid restarts at the resume
  instant (graph state in the snapshot is task 6).
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
circuit (driven sources, `make_pwm_switch_fn` / `make_multi_pwm_switch_fn`
gates); blocks, `chain.get` and the recordings always read the current
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
