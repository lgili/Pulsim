# `@fast_block` — PSIM/PLECS-style C block (via Numba)

`pulsim.fast_block` lets you write a control law in Python, decorate
it, and get back a Numba-JIT-compiled native callable suitable for
plugging into `step_observer=`. It's Pulsim's answer to **PSIM's
Custom C Block** and **PLECS's C-Script Block** — same workflow
(read inputs, mutate state, return scalar output), without the
runtime `cc` invocation, the cross-OS compiler-detection dance, or
the security risk of arbitrary native-code compilation.

> **TL;DR**
>
> ```python
> import pulsim as p
>
> @p.fast_block
> def pi(err, dt, Kp, Ki, state):
>     state[0] += Ki * dt * err
>     return Kp * err + state[0]
>
> state = pi.make_state()
> u = pi(error, DT, 0.5, 100.0, state)   # native-speed call
> ```

## Install

Numba is an **optional** dependency — install via

```bash
pip install pulsim[fast]
```

(installs `numba>=0.58`, which pulls ~80 MB of LLVM bits). Without
that extra, `pulsim` itself imports fine; only the moment you
*call* `@fast_block` do you get an `ImportError` with the install
hint.

Check at runtime:

```python
import sys, pulsim
fb = sys.modules["pulsim.fast_block"]
if fb.is_available():
    controller = pulsim.fast_block(my_pi)
else:
    controller = my_pi          # pure-Python fallback
```

## Authoring contract

```python
@pulsim.fast_block
def block_step(*scalar_inputs, state):
    # mutate state in-place
    state[0] = ...
    # return one scalar output
    return out
```

- **Last positional argument** must be a 1-D `float64` `numpy.ndarray`
  holding the block's persistent state. Mutate in place; return the
  scalar output.
- **Body** can use any Numba-supported Python: arithmetic, `np.*`
  reductions, control flow (`if/while/for`), `math.*`, `numpy`
  vector ops. No Python objects, no string formatting, no dicts of
  Python values. See [Numba's `@njit` reference](https://numba.readthedocs.io/en/stable/reference/pysupported.html).
- **First call triggers JIT compilation** (~0.3–1 s for typical
  control laws). Subsequent calls hit the LRU cache. Call
  `block.warm_up()` once at sim-build time if you need predictable
  first-step latency.

## API surface

```python
@pulsim.fast_block                       # bare form (n_states=1)
def pi(err, dt, Kp, Ki, state): ...

@pulsim.fast_block(n_states=3,           # parametrised form
                    cache=True,
                    parallel=False)
def lqr(x0, x1, x2, dt, state): ...
```

### `FastBlock` methods

| Method | Returns | Purpose |
|---|---|---|
| `block(*args)` | scalar | forward to JIT-compiled fn |
| `block.make_state()` | `np.ndarray(n_states, float64)` | zero-init state vector |
| `block.warm_up(*args)` | `None` | force JIT specialisation; no args = heuristic from declared signature |
| `block.py_func` | callable | the un-JIT'd Python original — useful for unit tests |

### Decorator kwargs

| kwarg | default | meaning |
|---|---|---|
| `n_states` | `1` | size of `make_state()` vector. Doesn't affect the compiled function. |
| `cache` | `True` | persist compiled code to `__pycache__` (instant on re-run). |
| `parallel` | `False` | enable Numba `parallel=True`. Default off — control loops are too small to amortise parallel launch overhead. |

## Worked example — buck CL with PI

```python
import pulsim as p

# 1. Plant.
b = p.CircuitBuilder()
b.add_voltage_source("Vin", "vin", "gnd", 48.0)
b.add_switch("HS", "vin", "sw", g_on=1e3, g_off=1e-9)
b.add_diode("D1", "gnd", "sw", g_on=1e3, g_off=1e-6, V_th=0.0)
b.add_inductor("L1", "sw", "vout", 100e-6)
b.add_capacitor("C1", "vout", "gnd", 100e-6)
b.add_resistor("R_load", "vout", "gnd", 4.0)

# 2. PI as a fast_block.
@p.fast_block
def pi_step(err, dt, Kp, Ki, state):
    state[0] += Ki * dt * err
    u = Kp * err + state[0]
    return min(1.0, max(0.0, u))    # saturate to [0, 1]

pi_state = pi_step.make_state()
pi_step.warm_up()                    # eat the LLVM cost up front

# 3. PWM + observer.
F_SW, V_REF, KP, KI = 100_000.0, 12.0, 0.02, 200.0
T_SW = 1.0 / F_SW
duty = [0.0]
last_update = [0.0]
vout_idx = b.node_id_of("vout")

def switch_fn(t):
    m = p.SwitchStateMask(b.graph.num_switches)
    m.set(0, (t % T_SW) / T_SW < duty[0])
    return m

def observer(t, x):
    if t - last_update[0] >= T_SW:
        v_out = float(x[vout_idx])
        duty[0] = float(pi_step(V_REF - v_out, T_SW, KP, KI, pi_state))
        last_update[0] = t

res = p.simulate(b, t_end=2e-3, dt=1e-7,
                   switch_fn=switch_fn, step_observer=observer)
```

Runs in ~90 ms wall-clock for 2 ms of simulated time; v_out
converges to the 12 V setpoint. The full runnable script is
[`examples/scripts/run_fast_block_pi_buck.py`](https://github.com/lgili/Pulsim/blob/main/examples/scripts/run_fast_block_pi_buck.py).

## When to use what

| Use case | What ships |
|---|---|
| Standard PI/PID/PLL/Clarke/Park controllers | [`pulsim.control`](https://github.com/lgili/Pulsim/blob/main/python/pulsim/control.py) blocks — 20+ pre-built, already JIT-compiled to C++ at chain build time. Use these first. |
| Custom control law, prototype-fast | `step_observer=` callable in plain Python. No new deps; perfectly fast for kHz-class loops. |
| Custom control law, **lots** of math per step (vector ops, lookup tables, state machines with hundreds of branches) | **`@fast_block`** — native LLVM-compiled speed without leaving Python. |
| MHz-class hot path inside the kernel | Add a new block type in `python/pulsim/blockchain.py::_compile_to_cxx` and `core/include/pulsim/blockchain/blocks.hpp`. |

## Limitations

- **Numba subset.** Strings, dicts, generators, custom exceptions
  (beyond `ValueError`/`AssertionError`) are out of scope. Stick to
  numerical Python.
- **First-call JIT cost.** ~0.3–1 s. Use `warm_up()` if predictable
  latency matters.
- **Python → Numba boundary.** Each call from the step observer pays
  ~1 µs crossing the FFI boundary. Fine for kHz control loops; not
  the right tool for MHz inner hot paths.
- **Module-name caveat.** `pulsim.fast_block` resolves to the
  decorator (re-exported at top level), not the submodule. For
  submodule helpers like `is_available`, use
  `sys.modules["pulsim.fast_block"]` or qualify the import:
  `from pulsim.fast_block import is_available`.

## Why Numba and not literal C

PSIM and PLECS write C blocks because their host runtimes are C++.
Pulsim is dual-language (C++ kernel + Python control), so the
analogous "fast-path" slot is *fast Python*. Numba's `@njit`
generates LLVM machine code that matches hand-written C on the
loop shapes power-electronics engineers actually write (10–50
ops, no Python objects).

Trade-off: you can't paste an existing C file directly into Pulsim
the way you can into PSIM. If that becomes a real ask we'll wire a
runtime-`cc` path in a follow-up release; for now, `@fast_block`
captures 95 % of the value with a tenth of the engineering cost.
