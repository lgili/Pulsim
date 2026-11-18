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
`Gain`, `Subtract`, `Limiter` (kernel blocks, same numbers);
`SpaceVectorModulator`, `ClarkeTransform`, `InverseClarkeTransform`,
`ParkTransform`, `InverseParkTransform` and `PLL` (two or three outputs:
`theta_hat`, `omega_hat`, and `omega_hat / 2 pi`). Every law lives once,
in `core/include/pulsim/blockchain/block_laws.h`, which the kernel calls
and `export_c` copies.

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

## Label time: the closed-loop helpers and C blocks

`bind_pi_to_switch`, `bind_pi_to_duty_callable` and `add_c_block` build
control graphs (`MixedDomainBlockChain(label_time=True)`, one rate group
per loop / block). `simulate()` composes every graph, closed loop and C
block in one place above the engine dispatch (`_control_plan.py`); with
no `PythonBlock` in them the kernel runs them with no Python call (the
gate of `bind_pi_to_switch` is a C++ `CxxChainPwmGate`).

**The law.** The fixed engine calls an observer with `(t_n, x_{n-1})`
before step n (from `t_{n-1}` to `t_n`). A label-time group due at
`tau = t_start + k*T` runs at the first call whose **label** `t_n >=
tau - 1e-9*dt` (and `tau < t_end - 1e-9*dt`), reads `x_{n-1}`, and its
output drives step n. So tick 0 runs at `t_start` on `x0`, and tick
`k >= 1` samples `x(kT - dt)` and holds its output from `kT - dt`: the
whole grid after tick 0 is **one solver step early** (an advance, not a
delay; the first hold lasts `T - dt`). This is the protocol the helpers
had before 2.1; it is kept so every closed-loop number is unchanged. A
chain with ordinary `Rate`s (state time, G6) samples `x(kT)` instead.
On `trbdf2` both modes tick on the exact instants.

Closed form, P loop on the RC of the table above (`K_p = 2`, set-point
1 V, `T = 100 µs`), label time at step `h`: `w_0 = 0`,
`w_{k+1} = a_k w_k + (1 - a_k)·2(1 - w_k)`, `a_0 = e^{-(T-h)/τ}`,
`a_k = e^{-T/τ}`, `w_k = v(kT - h)`. `w_10 = 0.643457839998` at
`h = 1 µs`, `0.643536938647` at `h = 0.1 µs`; `h → 0` gives the state-time
`v[10] = 0.643545722992` (`one_sample`: 0.659695654069).

| C-block P loop on `pwl`, `w_10` | error |
|---|---|
| `dt` = 1 µs | 6.17e-5 |
| `dt` = 0.1 µs | 6.14e-6 |

First order: a label-time graph declares no jumps (as the helpers did
not), so the trapezoidal step averages each held jump (known
limitations); `discontinuity_restart=True` does not restart on them
(1.06e-4 / 1.05e-5). Use a state-time chain with `drive_source` for an
exact hold.

**Gate edges.** The gate of `bind_pi_to_switch` (and any
`make_pwm_gate`) declares its edges (`gate.next_edge_after(t)`: period
starts and the OFF threshold under the duties held now). `trbdf2` lands
on them; `pwl` splits at them with `discontinuity_restart=True,
enable_substep_state_correction=True`. Without the split `pwl` samples
the gate on its grid, so the duty moves in steps of `dt/T`. Closed-loop
buck against an independent reference (max |v(kT) - ref| / span,
`dt = T/20`): `pwl` 3.6e-2, with the split 1.4e-3, and 1.8e-5 for a
state-time chain with the same gate; `trbdf2` default 4.2e-5.

### Migration: helpers and C blocks (2.1)

| Before | Now |
|---|---|
| `bind_pi_to_switch` / `bind_pi_to_duty_callable` ran Python closures: per step a tick scheduler, the composed observer, the gate and `pwm_on` (4044 Python calls for 10 ticks over 1 ms). | A control graph: the PI in the kernel, the gate in C++. `measured="vout"` (a node name) makes no Python call; a `measured` / `setpoint` callable runs through `PythonBlock` with a `DeprecationWarning` (3 calls per tick). Closed-loop buck, 20 ms at `dt` = 0.1 µs: 2188 → 204 ns/step. Every result bit-identical. |
| `pi` (the `PIController` passed in) was updated in place. | The kernel block starts from `pi`'s state; `pi` is not updated. Read `loop.duty_history` / `error_history` (live views of the graph's recordings). |
| `bind_pi_to_switch(t_start=)` aligned the throttle. | Ignored: ticks align to the run's `t_start`. |
| `add_c_block`: a Python observer and `b_extra` per step (5014 calls / 1 ms); it also fired at `t_end` (11 fires where 10 are due), which changed the last sample of the run. | One graph per builder (reached through its C-block handles, so a `Circuit` carries it in its ControlDecls), one `CBlock` per block: `code=` / `lib=` run in C++ with no Python call (2041 → 168 ns/step), `fn=` through the adapter with a `DeprecationWarning`. No tick at `t_end`: only the run's final sample changes (t = `t_end` when the span is a multiple of the block `dt`). |
| `add_c_block` with a block `dt` below the solver step (no `sim_dt=`) ran every step with a wrong `dt`. | `ControlGraphBuildError` (pass `sim_dt=` to clamp, as before). |
| `simulate()` composed closed loops and C blocks in two places below the engine dispatch (pwl only). | One composition above the dispatch (`ControlPlan.for_fixed_step` / `for_variable_step`). |
| `SpaceVectorModulator`, the Clarke / Park transforms and `PLL` in a chain ran the kernel `add_svm` / `add_clarke` / `add_park` / `add_pll` laws: other roundings, `v_dc <= 0` gave duty 0.5, the PLL started at 0 Hz and wrapped to [-π, π); `PLL` needed three outputs. | Their classes' laws, bit-identical (G18): numbers change in FOC chains (rounding level; from the first tick for a PLL); `PLL` takes two or three outputs. |
| A C block cache under the system temp dir, shared by all users. | A per-user cache, mode 0700; a foreign or group/world-writable cached library is refused. |

## C export

`pulsim.export_c(chain, builder, outputs=[...])` renders a rated graph to
one C99 file (`CExport.source`; `.write(path)`): the law table, a state
struct, `<prefix>_init`, one `<prefix>_step_rate<g>(s, in, t)` per rate,
the rate table (`<prefix>_period`, `<prefix>_offset`), 64-bit tick
counters (`<prefix>_ticks`, `<prefix>_next_tick`) and, for a single-rate
graph, the `pulsim_cblock_step` shim, so the file loads back through
`add_c_block(lib=...)`. `in[k]` is `CExport.inputs[k]` (a node, or a
channel no block writes); the shim writes `outputs` to `out[]`. No
`malloc`: the shim's state is static, so a circuit takes one block of a
library (a second `add_c_block` of it on the same builder is refused);
across circuits each `add_c_block` resets the shared state.

Compiled with `-ffp-contract=off` (MSVC's default `/fp:precise` does not
contract) it is bit-identical to the kernel graph: every block kind over
10⁴ ticks, and a PI + limiter + PWM graph in a circuit through
`add_c_block`. Refused by name: `PythonBlock`, a `CBlock`, a graph with
no `Rate` (continuous time) and a rate with a `compute_delay`.
A multi-rate export splits the kernel's single pass into one function
per rate: when several rates are due at one instant, call them in rate
order (`step_rate0` first); that reproduces the kernel when no block
reads, at a shared instant, a signal a later-called rate writes. The shim
writes NaN to its outputs when it is wired (`n_in`, `n_out`) or clocked
(`dt`) unlike the export. For a second instance, compile the file again
under another name, or export it with another `prefix` and drive
`<prefix>_init` / `<prefix>_step_rate0` on your own state.

```python
ex = pulsim.export_c(chain, builder, outputs=["y"])
ex.write("ctl.c")   # cc -std=c99 -O2 -ffp-contract=off -shared -fPIC ctl.c -o ctl.so
pulsim.add_c_block(b, inputs=[("v", "out")], outputs=[("v", "sw", "gnd")],
                   dt=1e-6, lib="ctl.so")
```

C blocks compile into a per-user cache, mode 0700
(`~/Library/Caches/pulsim/cblock`, `~/.cache/pulsim/cblock` or
`$XDG_CACHE_HOME/pulsim/cblock`, `%LOCALAPPDATA%\pulsim\cblock`); a
cached library that another user owns or can write is refused, and so is
a `pulsim` or `cblock` directory that is not both yours and private. Windows
has no POSIX owner or mode: there the cache must sit inside
`%USERPROFILE%` (whose ACL is the user's), and a `LOCALAPPDATA` outside
it is refused.

## Other engines and resume

C blocks run on `trbdf2` (`engine="trbdf2"`; `auto` keeps routing them
to `pwl`): the engine lands its steps on the ticks, so a block samples
`v(kT)` and holds from `kT`, the state-time law (`v[10] =
0.643545722992`, met within 1e-7 at rtol 1e-9). The exact strategy
(`exact` / `dsed`) refuses them by name: it has no `b_extra` input.

A graph keeps its tick counters and block states across `simulate()`
calls on the same builder: a run that goes on from the previous `t_end`
(`resume_from=`) continues the tick grid; a run that starts earlier
restarts it. The C-block graph is compiled once per circuit layout.
`control_tick_counts` and `CBlockHandle.n_fires` count the last run's
ticks. Resume between ticks: a tick due exactly at the previous `t_end`
is dropped there (the horizon) and runs at the resumed start on the
snapshot state `x(t_end)`, where the uninterrupted `pwl` run used
`x(t_end - h)` (label time).

## Not yet

Continuous-rate blocks solved with the circuit, tick events on the
unified event queue, comparator state events, and graph
state in the solver snapshot (tasks 6–7 of `add-compiled-control-graph`).
