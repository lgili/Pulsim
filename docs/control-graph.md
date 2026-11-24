# Control graph: order, rates, delay and export

A `MixedDomainBlockChain` (and the `CxxBlockChain` it compiles to) is a
**control graph**: typed blocks connected by integer signal slots,
evaluated by the C++ kernel with zero Python calls per step (unless the
graph explicitly includes a `PythonBlock`). The closed-loop helpers
(`bind_pi_to_switch`, `bind_pi_to_duty_callable`), `add_c_block`, and
YAML controller chains all compile to this same kernel graph.

This guide documents:

- topological schedule order, algebraic-loop detection, and `DelayBlock`;
- `Rate` groups, rate inheritance, integer tick counters, and `control_tick`;
- `compute_delay` and exact tick delivery across `pwl`, `trbdf2`, and `dsed` / `exact`;
- circuit sinks (`drive_source`, `gate`, PWM modulators), snapshots, and resume;
- `PythonBlock` vs kernel blocks and the current status of continuous blocks;
- C99 code export (`pulsim.export_c`) and reloading via `add_c_block`;
- migration notes from pre-2.1 insertion-order chains and `use_kernel=False`.

## Order and algebraic loops

Blocks run in the topological order of the *direct-feedthrough* graph
computed by Kahn's algorithm with deterministic ties (insertion order):
a block runs after every block whose output it reads in the same
instant. Consequently, insertion order no longer changes simulation
results — a chain built in any order produces bit-identical output.

| Block | Inputs that order the schedule |
|---|---|
| every block except `DelayBlock` | all (their output depends on this instant's input, stateful blocks included) |
| `DelayBlock` | none (its output is the input from `samples` evaluations ago) |

Execution of a pass is **two-phase**:

1. Every block in the topological schedule computes its outputs
   (`DelayBlock` outputs its stored FIFO head).
2. `DelayBlock` instances shift in their new inputs for the next tick.

A strongly connected component (SCC, Tarjan) of direct-feedthrough edges
— or a block reading its own output — is an **algebraic loop**. Building
the graph raises `pulsim.ControlGraphBuildError` (a `ValueError`
subclass) naming the blocks in the cycle and suggesting a `DelayBlock`.
A signal written by two blocks raises `ControlGraphBuildError` as well.

```python
import pulsim as p

b = p.CircuitBuilder()
b.add_resistor("R", "a", "gnd", 1.0)

# Algebraic loop: g1 -> sum -> g2 -> g1
loop = p.MixedDomainBlockChain()
loop.add("g1", p.Gain(k=0.5), inputs=dict(x="channel:fb"), output="a")
loop.add("sum", p.Sum(), inputs=dict(a="channel:a", b=1.0), output="s")
loop.add("g2", p.Gain(k=0.5), inputs=dict(x="channel:s"), output="fb")

try:
    loop.make_step_observer(b, dt=1e-6)
except p.ControlGraphBuildError as exc:
    print("caught:", "algebraic loop" in str(exc), "DelayBlock" in str(exc))

# Break the cycle with DelayBlock(samples=1); insertion order does not matter.
broken = p.MixedDomainBlockChain()
broken.add("g2", p.Gain(k=0.5), inputs=dict(x="channel:s"), output="fb_raw")
broken.add("z", p.DelayBlock(samples=1), inputs=dict(x="channel:fb_raw"), output="fb")
broken.add("sum", p.Sum(), inputs=dict(a="channel:a", b=1.0), output="s")
broken.add("g1", p.Gain(k=0.5), inputs=dict(x="channel:fb"), output="a")

obs = broken.make_step_observer(b, dt=1e-6)
print("schedule:", list(broken._cxx_chain.schedule()))
```

```text
caught: True True
schedule: ['z', 'g1', 'sum', 'g2']
```

### Migration: chains that relied on insertion order

Before 2.1, `MixedDomainBlockChain` evaluated blocks in insertion order:
a block inserted before its producer read the previous step's value (or
`0.0` on the first step), creating an implicit one-step delay.

Building a chain whose topological schedule differs from insertion order
now emits a `UserWarning` once per build listing every reordered
consumer/producer pair (also inspectable via `chain._cxx_chain.reorders()`):

```text
control graph: evaluation order differs from insertion order (each was a
one-step delay before the schedule): 'half' reads 'u' written by 'gain', ...
Insert a DelayBlock to keep a delay.
```

- **To keep the old one-step delay**, make it explicit by routing the
  signal through `p.DelayBlock(samples=1)`.
- **Otherwise**, the topologically scheduled result is the intended
  zero-lag evaluation; reorder the `chain.add(...)` calls if you want to
  silence the warning.

## Rates and tick counters

Without a `Rate` anywhere, a chain runs on every solver step with the
`dt` given to `make_step_observer(builder, dt=...)` (fixed-step `pwl`
only).

With `p.Rate(period, offset=0.0, compute_delay=0.0)`, each block belongs
to a rate group that fires at exact integer-counted instants:

$$t_k = t_{\text{start}} + \text{offset} + k \cdot \text{period}, \qquad k = 0, 1, 2, \dots$$

Using an integer counter `k` avoids accumulated floating-point drift.
The `dt` input passed to every block in the group is `period` exactly.

### Rate assignment and inheritance

1. If a block passes `rate=p.Rate(...)`, it uses that rate.
2. Otherwise, if the block is **stateless** (`Gain`, `Subtract`, `Sum`,
   `MathBlock`, `Limiter`, `OpAmp`, `SignalMux`, `SignalDemux`,
   `LookupTable1D`, `ClarkeTransform`, `InverseClarkeTransform`,
   `ParkTransform`, `InverseParkTransform`, `SpaceVectorModulator`), it
   inherits the fastest rate among the blocks writing its inputs.
3. Otherwise it takes the chain's default `MixedDomainBlockChain(rate=...)`.
4. Any stateful block left without a rate raises `ControlGraphBuildError`
   naming the block.

Additional rate rules:

- **Shared instants:** when multiple groups are due at the same instant,
  they execute in one merged pass of the global topological schedule, so
  a fast group reading a slow group's signal sees that instant's value.
- **Rate transitions:** require no special block — each signal slot holds
  its latest value (slow-to-fast acts as zero-order hold; fast-to-slow
  samples at the slow tick).
- **`PwmGenerator` in a rated chain:** requires an explicit `rate=` (for
  example `Rate(dt)` on `pwl`); inheriting a slower controller rate
  would sample the carrier only at controller ticks, so it is a build
  error. For variable-step runs or analytic PWM edges, drive switches
  with `bind_pi_to_switch` or `chain.gate(...)` instead.
- **Period vs fixed step:** on `pwl`, a `period` smaller than `dt`
  raises `ControlGraphBuildError`. Passing `controller_period=` alongside
  a rated chain raises `ValueError` because the groups define the ticks.
- **Tick count:** ticks fire for every `k` strictly before `t_end`,
  giving `ceil((t_end - t_start - offset) / period)` ticks on `pwl`,
  `trbdf2`, and `dsed` / `exact`. Inspect per-group counts via
  `chain.tick_counts`, `res.control_tick_counts`, and total block
  evaluations via `chain._cxx_chain.n_block_evals`.

```python
import numpy as np
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "in", "gnd", 1.0)
b.add_resistor("R", "in", "out", 1e3)
b.add_capacitor("C", "out", "gnd", 1e-6)

ch = p.MixedDomainBlockChain(rate=p.Rate(1.0 / 2e3))       # 2 kHz default
ch.add("pi", p.PIController(Kp=0.1, Ki=10.0),
       inputs=dict(setpoint=1.0, measured="out", dt="dt"), output="u")
ch.add("scale", p.Gain(k=2.0),                             # stateless: inherits 2 kHz
       inputs=dict(x="channel:u"), output="u2")
ch.add("mon", p.FirstOrderLowPass(tau=1e-3),               # 20 kHz group
       inputs=dict(input_value="out", dt="dt"), output="m",
       rate=p.Rate(1.0 / 20e3))

res = p.simulate(b, 20e-3, engine="trbdf2",
                 step_observer=ch.make_step_observer(b))
print("tick_counts:", ch.tick_counts, "n_block_evals:", ch._cxx_chain.n_block_evals)
```

```text
tick_counts: [40, 400] n_block_evals: 480
```

## Compute delay and exact tick delivery

`Rate(period, offset=0.0, compute_delay=...)` controls when a group's
outputs take effect at circuit sinks (`drive_source`, `gate`,
`make_pwm_switch_fn` / `make_multi_pwm_switch_fn`). Downstream blocks in
the graph, `chain.get(...)`, and `record_channel(...)` always observe
the current computed signals.

| `compute_delay` | Outputs take effect at circuit sinks | Supported engines |
|---|---|---|
| `0` (default) | at the tick `t_k` | `pwl`, `trbdf2`, `dsed` / `exact` |
| `0 < Td < period` (fixed time) | at the first instant `>= t_k + Td` | `pwl` (refused on `trbdf2` as `UnsupportedError`, feature `compute_delay`) |
| `"one_sample"` | at the next tick `t_{k+1}` (models a microcontroller latching duty for the next carrier period) | `pwl`, `trbdf2`, `dsed` / `exact` |

A `Rate` with a non-zero `offset` (`0 <= offset < period`) ticks at
`t_start + offset + k*period` on both `pwl` and `trbdf2` (the unified
event queue lands a step on every tick).

### Engine delivery summary (`pwl`, `trbdf2`, `dsed` / `exact`)

| Engine | Tick delivery | Notes and limits |
|---|---|---|
| `pwl` | Ordinary `Rate` chains run in **state time** (tick at `t_k` reads `x(t_k)` on the step grid; off-grid ticks fire at the next grid step). `bind_pi_to_switch` and `add_c_block` default to **label time** unless `control_tick="state"` is passed. | With `control_tick="state"`, every rate group ticks from the kernel event queue. With `discontinuity_restart=True`, driven sources declare their jumps (`next_jump_after`) so the zero-order hold is second order. |
| `trbdf2` | Unified event queue lands a step at every `t_start + offset + k*period`. | Supports `offset`, `compute_delay=0` or `"one_sample"`, `drive_source`, `gate`, `bind_pi_to_switch`, and `add_c_block` (`engine="trbdf2"`). Refuses fixed-time `compute_delay`. |
| `dsed` / `exact` | Unified event queue lands ticks like gate edges; at a shared instant ticks run before gate edges so the gate sees updated duty immediately. | Supports `bind_pi_to_switch` and `controller_period` observers with exact `ceil` tick counts. Refuses `add_c_block` (`feature="c_blocks"`, no `b_extra`), every-step observers without a period, `t_start` with controllers, and snapshots (`resume_from`). |

### Exact zero-order hold (closed-form check)

Consider a discrete proportional loop ($K_p = 2$, $T_s = 100\,\mu\text{s}$,
set-point 1 V) driving an RC plant ($\tau = 1\text{ ms}$) through a
zero-order-hold controlled source (`drive_source`). With $a = e^{-0.1}$:

- **`compute_delay=0`:** $v[10] = \frac{2}{3}(1 - (a - 2(1 - a))^{10}) = 0.643545722992\text{ V}$.
- **`compute_delay="one_sample"`:** $v[10] = 0.659695654069\text{ V}$.

On `pwl`, pass `discontinuity_restart=True` so the trapezoidal integrator
restarts at each declared zero-order-hold jump instead of averaging it
over one step (first-order error without restart; second-order with it):

| Discrete P loop, `v[10]` (closed form `0.643545722992`) | error |
|---|---|
| `pwl`, `dt` = 1 µs | 6.148e-5 |
| `pwl`, `dt` = 0.1 µs | 6.138e-6 |
| `pwl`, `dt` = 1 µs, `discontinuity_restart=True` | 7.2e-9 |
| `pwl`, `dt` = 0.1 µs, `discontinuity_restart=True` | 7.3e-11 |
| `trbdf2`, `rtol=1e-10` | 7.8e-9 |
| `trbdf2`, `rtol=1e-12` | 3.7e-10 |

```python
TS = 1e-4

for delay in (0.0, "one_sample"):
    b = p.CircuitBuilder()
    ch = p.MixedDomainBlockChain(rate=p.Rate(TS, compute_delay=delay))
    ch.add("err", p.Subtract(), inputs=dict(a=1.0, b="v"), output="e")
    ch.add("kp", p.Gain(k=2.0), inputs=dict(x="channel:e"), output="u")
    ch.drive_source(b, "u", ("v", "u", "gnd"))
    b.add_resistor("R", "u", "v", 1e3)
    b.add_capacitor("C", "v", "gnd", 1e-6)

    res = p.simulate(b, 11 * TS, dt=1e-6, discontinuity_restart=True,
                     step_observer=ch.make_step_observer(b, dt=1e-6))
    t = np.asarray(res.times)
    v10 = float(res.v("v")[int(np.argmin(np.abs(t - 10 * TS)))])
    print(f"compute_delay={delay!r:12s} -> v[10] = {v10:.6f} V")
```

```text
compute_delay=0.0          -> v[10] = 0.643546 V
compute_delay='one_sample' -> v[10] = 0.659696 V
```

## Circuit sinks: sources, gates and modulators

A control graph drives the circuit through declared sinks:

- **`chain.drive_source(builder, channel, wire)`** — adds a controlled
  voltage `("v", n_pos, n_neg)` or current `("i", n_pos, n_neg)` source
  held constant between publication instants (zero-order hold). Runs in
  C++ (`CxxChainBExtra`) and declares its jump instants on `pwl`.
- **`chain.gate(builder, {channel: switch})`** — returns a native
  discrete gate schedule (`CxxChainSignalGate`, call after
  `make_step_observer`): the switch is ON while the published signal
  exceeds `0.5`. Its `next_edge_after(t)` reports the graph's
  publication instants, so `trbdf2` lands every transition on a tick and
  `pwl` with `discontinuity_restart=True, enable_substep_state_correction=True`
  splits at them.
- **PWM modulators (`bind_pi_to_switch` / `make_pwm_gate`)** — the
  graph writes a duty register; the native `CxxChainPwmGate` computes
  carrier start and OFF edges analytically (`next_edge_after(t)`).
  `trbdf2` lands those edges within 1 ppm of commanded duty and
  `n_gate_events` matches the analytic edge count; `pwl` splits at them
  when `discontinuity_restart=True, enable_substep_state_correction=True`.

## Label time vs `control_tick="state"`

`bind_pi_to_switch`, `bind_pi_to_duty_callable`, and `add_c_block` build
`MixedDomainBlockChain(label_time=True)` graphs.

**Why label time exists.** On `pwl`, the observer callback for step $n$
(advancing from $t_{n-1}$ to $t_n$) receives `(t_n, x_{n-1})`. A
label-time group due at $\tau = t_{\text{start}} + k T$ runs at the
first call whose **label** $t_n \ge \tau - 10^{-9}\Delta t$ (with
$\tau < t_{\text{end}} - 10^{-9}\Delta t$), reads $x_{n-1}$, and drives
step $n$. Thus tick 0 samples $x_0$ at $t_{\text{start}}$, while tick
$k \ge 1$ samples $x(kT - \Delta t)$ and holds from $kT - \Delta t$ (one
solver step early, matching pre-2.1 closed-loop numbers). On `trbdf2`
and `dsed`, both modes tick on the exact instants $kT$.

**`simulate(..., control_tick="state")`** (default `"label"`) switches
those label-time graphs to **state time** for that run:

- every rate group becomes a kernel event-queue tick source on `pwl`,
  sampling $x(kT)$ and holding from $kT$;
- `SimulationOptions.split_snap` narrows from `0.01 * dt` to `1e-6 * dt`,
  so with `discontinuity_restart=True, enable_substep_state_correction=True`,
  closed-loop `pwl+split` achieves second-order convergence against the
  sampled-data reference (for example, `cl_buck_pi` error at $T/20$ drops
  from `1.39e-3` to `8.35e-6`).

Closed form for the label-time RC P loop at step $h$ ($K_p = 2$,
set-point 1 V, $T = 100\,\mu\text{s}$): $w_0 = 0$,
$w_{k+1} = a_k w_k + (1 - a_k)\cdot 2(1 - w_k)$ with
$a_0 = e^{-(T - h)/\tau}$ and $a_k = e^{-T/\tau}$ gives
$w_{10} = 0.643457839998$ at $h = 1\,\mu\text{s}$ and
$0.643536938647$ at $h = 0.1\,\mu\text{s}$ (approaching the state-time
$v[10] = 0.643545722992$ as $h \to 0$).

## Kernel blocks, `PythonBlock`, and continuous-block status

### Zero-Python-call kernel block library

Every standard control block in `pulsim.control` executes in C++ with
zero Python calls in the step loop and shares one law table
(`core/include/pulsim/blockchain/block_laws.h`, bit-identical to the
Python classes on $10^4$ random inputs each):

- **Arithmetic & routing:** `Gain`, `Subtract`, `Sum`, `MathBlock`,
  `Limiter`, `OpAmp`, `LookupTable1D`, `SignalMux`, `SignalDemux`,
  `DelayBlock`.
- **Controllers & filters:** `PIController`, `PIDController`,
  `Integrator`, `Differentiator`, `FirstOrderLowPass`,
  `MovingAverageFilter`, `TransferFunction`, `RateLimiter`.
- **Logic, sampling & modulation:** `Comparator`, `SampleHold`,
  `StateMachine`, `PwmGenerator`, `SpaceVectorModulator`.
- **Three-phase transforms & sync:** `ClarkeTransform`,
  `InverseClarkeTransform`, `ParkTransform`, `InverseParkTransform`,
  `PLL` (2 or 3 outputs: `theta_hat`, `omega_hat`, and optional
  `omega_hat / (2*pi)`).

Initial states (`PIController(integrator_state=...)`,
`Comparator(state=True)`, `FirstOrderLowPass(state=...)`,
`TransferFunction` history, etc.) initialize the kernel block directly.
`chain.reset()` zeroes block states, signals, ticks, and recordings.
N-input blocks accept tuples: `Sum(weights=(1, -1, 0.5))` with
`inputs=dict(inputs=("channel:a", "channel:b", "vout"))` (or `a`/`b`/`c`
matching `len(weights)`), and `SignalMux` with
`inputs=dict(selector=..., inputs=(...))`.

### `PythonBlock` adapter for custom Python logic

An unknown block class is a `ControlGraphBuildError` — there is no
silent Python fallback. To run custom Python logic inside a graph, wrap
a callable `fn(**inputs)` (or an object with an `.update(**inputs)`
method) in `pulsim.PythonBlock(fn)`:

- Each evaluation acquires the GIL once per tick (not per solver step,
  when given a `Rate`).
- Every input is direct feedthrough, and the block counts as stateful
  (give it a `rate=` or a chain-level `rate=`).
- `PythonBlock` is flagged opaque: `export_c` refuses it by name.
- `SlidingModeObserver` and `FluxMRASObserver` (in `pulsim.observers`)
  automatically run through the `PythonBlock` adapter.

```python
b = p.CircuitBuilder()
b.add_voltage_source("Vin", "in", "gnd", 10.0)
b.add_switch("Q", "in", "sw", 1e2, 1e-9)
b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
b.add_inductor("L", "sw", "out", 200e-6)
b.add_capacitor("C", "out", "gnd", 47e-6)
b.add_resistor("R", "out", "gnd", 10.0)

def hysteresis_cmd(vout):
    return 1.0 if vout < 5.0 else 0.0

ch = p.MixedDomainBlockChain(rate=p.Rate(1e-5))
ch.add("cmd", p.PythonBlock(hysteresis_cmd),
       inputs=dict(vout="node:out"), output="on")
obs = ch.make_step_observer(b)
gate = ch.gate(b, {"on": "Q"})

res = p.simulate(b, 2e-3, engine="trbdf2", step_observer=obs, switch_fn=gate)
t = np.asarray(res.times)
v_avg = p.time_average(t, res.v("out"), t_from=1e-3)
print(f"PythonBlock + discrete gate: V_out avg = {v_avg:.2f} V, ticks = {ch.tick_counts}")
```

```text
PythonBlock + discrete gate: V_out avg = 5.08 V, ticks = [200]
```

### Continuous-rate blocks: implemented vs planned

Be explicit about what ships today versus what is planned under Group 7
of `add-compiled-control-graph`:

- **Implemented today:**
  - Discrete-rate graphs (`Rate(period, offset, compute_delay)`) running
    in the kernel with integer tick counters and exact event-queue
    delivery on `trbdf2`, `dsed`/`exact`, and `pwl` (`control_tick="state"`).
  - Unrated every-step chains on `pwl` (evaluated once per fixed `dt`
    step via `make_step_observer(builder, dt=dt)`).
  - Discrete sinks (`drive_source`, `gate`, `CxxChainPwmGate`), solver
    snapshots with pickle support (`pwl` and `trbdf2`), sweep state
    isolation, and C99 code export (`export_c`).
- **Planned (Group 7, not yet implemented):**
  - `Continuous` rate subgraphs compiled directly into the circuit
    solver's implicit residual
    ($\frac{d}{dt} x_c - f_c(x_c, y_{\text{probe}}) = 0$) with exact AD
    Jacobians (eliminating inter-step lag and letting the exact
    propagator advance linear compensators with the circuit).
  - Continuous-rate `Comparator` gates located as state events with
    hysteresis between solver ticks.
  - Piecewise-affine continuous blocks (`Limiter`, `LookupTable1D`) as
    affine regions with breakpoint state events.
- **Other current limits:** fixed-time `compute_delay` is refused on
  `trbdf2`; `dsed` / `exact` refuses C blocks (`b_extra`) and snapshots
  (`resume_from`).

## C99 export (`pulsim.export_c`)

`pulsim.export_c(chain, builder, outputs, *, prefix="pulsim_graph")`
renders a rated kernel graph to one self-contained C99 translation unit
(`CExport.source`; save with `CExport.write(path)`):

- **Contents:** `block_laws.h` copied verbatim (same arithmetic as the
  kernel), a `<prefix>_state` struct, `<prefix>_init`, one
  `<prefix>_step_rate<g>(s, in, t)` per rate group, `<prefix>_period[]`
  and `<prefix>_offset[]` rate tables, 64-bit tick counters
  (`<prefix>_ticks`, `<prefix>_next_tick`), `<prefix>_state_size`, and
  `<prefix>_get`.
- **Single-rate C-block shim:** for a single-rate graph, the export also
  emits `pulsim_cblock_init`, `pulsim_cblock_step`, `pulsim_cblock_save`,
  and `pulsim_cblock_restore` over a static state struct (zero `malloc`),
  so the compiled library loads directly back via
  `pulsim.add_c_block(..., lib="ctl.so")` and supports `resume_from`.
- **Strict C99 & bit identity:** the generated C compiles cleanly with
  `-std=c99 -Wall -Wextra -Werror -pedantic -ffp-contract=off` (MSVC
  `/fp:precise` already avoids FMA contraction) and contains no `malloc`.
  Compiled without FMA contraction, it reproduces the in-kernel graph
  bit for bit over $10^4$ ticks across all kernel block kinds.
- **Refusals (named `ControlGraphBuildError`):** `PythonBlock` (and
  Python library observers), any `CBlock` (no source to inline), an
  unrated chain (no discrete sample time), and any `Rate` with
  `compute_delay != 0`.
- **Multi-rate & static-state notes:** when multiple rates are due at
  one instant, call `<prefix>_step_rate0`, `step_rate1`, ... in rate
  order (exact when no earlier rate reads a signal written by a later
  rate at that instant). Because the single-rate shim holds static state,
  a second `add_c_block` of the same shared library on the same builder
  is refused, and concurrent runs of that library on multiple threads
  raise `RuntimeError`. For a second independent instance, compile a
  separate shared library or drive `<prefix>_step_rate0` on your own
  `<prefix>_state`. The shim writes `NaN` to `out[]` if `n_in`, `n_out`,
  or `dt` does not match the export.

```python
b = p.CircuitBuilder()
b.add_resistor("R", "out", "gnd", 10.0)

ctl = p.MixedDomainBlockChain(rate=p.Rate(1e-5))
ctl.add("pi", p.PIController(Kp=0.05, Ki=200.0, output_min=0.0, output_max=0.9),
        inputs=dict(setpoint=12.0, measured="out", dt="dt"), output="d_raw")
ctl.add("lim", p.Limiter(output_min=0.05, output_max=0.85),
        inputs=dict(x="channel:d_raw"), output="duty")

ex = p.export_c(ctl, b, outputs=["duty"], prefix="buck_ctl")
ex.write("buck_ctl.c")
print("inputs:", ex.inputs, "outputs:", ex.outputs,
      "has_malloc:", "malloc" in ex.source,
      "has_shim:", "pulsim_cblock_step" in ex.source)
```

```text
inputs: ['out'] outputs: ['duty'] has_malloc: False has_shim: True
```

Compile the exported file with strict C99 flags and no FMA contraction:

```bash
cc -std=c99 -Wall -Wextra -Werror -pedantic -ffp-contract=off \
   -O2 -shared -fPIC buck_ctl.c -o buck_ctl.so -lm
```

Then load it back into a Pulsim circuit through `add_c_block`:

<!-- doctest: skip reason="requires the compiled buck_ctl.so shared library built by the C compiler command above" -->
```python
p.add_c_block(
    b,
    inputs=[("v", "out")],
    outputs=[("v", "sw", "gnd")],
    dt=1e-5,
    lib="buck_ctl.so",
)
```

C blocks built via `code=` compile into a per-user cache (mode `0700`:
`~/Library/Caches/pulsim/cblock`, `$XDG_CACHE_HOME/pulsim/cblock` or
`~/.cache/pulsim/cblock`, `%LOCALAPPDATA%\pulsim\cblock` inside
`%USERPROFILE%`). A foreign or group/world-writable cache directory or
library is refused.

## Snapshots, resume, and parameter sweeps

- **Solver snapshots (`resume_from=`):** on `pwl` and `trbdf2`,
  `res.final_snapshot` captures every control graph's kernel state,
  signal slots, tick grid, recordings, `PythonBlock` `__dict__`, and
  C-block state (including `export_c` libraries via
  `pulsim_cblock_save` / `pulsim_cblock_restore`). Snapshots also
  **pickle** cleanly (`pickle.dumps` / `pickle.loads`) and restore into
  freshly built controllers with matching structure (mismatched graphs
  or opaque native libraries without save/restore are refused by name).
  On `pwl` (both label and state time), mid-period and on-tick resumes
  are bit-identical to the uninterrupted run.
- **Reusing a graph across runs:** a chain you pass directly keeps its
  block state across `simulate()` calls (restarting its tick grid when
  time goes back) unless you call `chain.reset()`. By contrast, a
  circuit's C blocks reset to their declared initial state on every
  fresh `simulate()` call so repeated runs of an immutable `Circuit` are
  bit-identical; only `resume_from=` continues them.
- **Sweeps (`sweep` / `monte_carlo`):** each sweep point restores
  control graphs to their pre-sweep state and deep-copies Python
  controller objects so points never leak integrator state into one
  another. Closures over uncopyable mutable state and graphs on
  `parallel=True` sweeps are refused by name (`feature="controller_state"`).

## Migration summary (2.0 → 2.1+)

### Evaluator and block changes

| Before (2.0) | Now |
|---|---|
| Chains ran in insertion order (silent one-step delay when a consumer preceded its producer). | Topological order (Kahn's algorithm); build warns once naming reordered pairs. Insert `DelayBlock(samples=1)` to keep an explicit delay. |
| A block without a kernel mapping fell back silently to the Python evaluator behind a warning. | `ControlGraphBuildError` naming the block; wrap custom Python logic in `pulsim.PythonBlock(fn)`. |
| `make_step_observer(..., use_kernel=False)` | Raises `DeprecationWarning`. Drop the argument; `record_channel` works on the kernel graph. |
| `record_channel(name)` before `make_step_observer` recorded nothing on the kernel path. | Recorded automatically. |
| `PIDController` in a chain ran a different kernel law (derivative on measurement, `tau_d = 0`). | Runs the class's law (derivative on error, `derivative_alpha`). |
| `Sum` ignored `weights` and a `c` input. | Weights applied; length mismatch is a `ControlGraphBuildError`. |
| `Integrator` ran a different kernel law without anti-windup hold. | Runs the class's law (bit-identical on $10^4$ inputs). |
| `FirstOrderLowPass` passed input through at `dt = 0` and ignored `state=`. | Holds at `dt = 0` and starts from `state=`. |
| `PwmGenerator(phase=...)` interpreted `phase` as a period fraction with opposite sign. | Interprets `phase` in seconds, matching the Python class. |
| `SpaceVectorModulator`, Clarke/Park transforms, and `PLL` used different kernel laws (`PLL` started at 0 Hz and wrapped to $[-\pi, \pi)$). | Unified `block_laws.h` laws bit-identical to the Python classes; `PLL` accepts 2 or 3 outputs. |
| `PIController` initial state (`integrator_state=`) was ignored by the kernel chain. | Honoured. |

### Closed-loop helpers and C blocks

| Before | Now |
|---|---|
| `bind_pi_to_switch` / `bind_pi_to_duty_callable` ran Python closures per step (~4000 Python calls / 1 ms). | Compiled kernel graph + C++ `CxxChainPwmGate`. Node-name `measured="vout"` makes 0 Python calls (buck: `2188 -> 204 ns/step`). Callable `measured`/`setpoint` runs via `PythonBlock` with a `DeprecationWarning`. |
| The `PIController` object passed to `bind_pi_to_switch` was mutated in place. | Kernel block starts from `pi`'s initial state; `pi` is not mutated. Read `loop.duty_history` / `loop.error_history`. |
| `add_c_block` ran a Python observer per step and fired an extra tick at `t_end`. | Compiled `CBlock` graph (`code=`/`lib=` make 0 Python calls; `fn=` emits a `DeprecationWarning`); no tick at `t_end`. Runs on `pwl` and `trbdf2`; refused by name on `dsed`/`exact`. |
| `add_c_block` with `dt < sim_dt` silently ran every step with a wrong `dt`. | `ControlGraphBuildError` (pass `sim_dt=` to clamp explicitly). |
| Shared temp-directory C-block compile cache. | Private per-user cache (`0700`). |
