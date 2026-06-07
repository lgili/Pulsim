# Custom Code Blocks (C / C++ / Python)

A **C block** is a PSIM-/Simulink-style **sampled subsystem** you wire
into the circuit. It does three things, every block step:

1. **reads** a set of circuit signals (its *inputs*),
2. runs **your code** at a sample rate **you choose**,
3. **drives** signals back into the circuit (its *outputs*).

Reach for it when your logic doesn't map to a fixed component — digital
control laws, lookup tables, protection/interlocks, state machines,
soft-start sequencers, or a custom behavioural model. The step code can
be **Python**, **C**, or **C++** behind one interface.

Everything lives in `pulsim.c_block` and is re-exported on the top-level
package (`pulsim.add_c_block`, `pulsim.wire_c_blocks_from_yaml`,
`pulsim.CBlockHandle`, `pulsim.CBLOCK_ABI`).

!!! info "No kernel rebuild"
    A C block is pure-Python orchestration on top of Pulsim's existing
    per-step hooks (a throttled `step_observer` + `b_extra` residual
    injection). `add_c_block` registers the block on the builder and
    `simulate()` picks it up automatically — you just call `simulate`.

---

## Quick start

A proportional gain that reads `V(vout)` and drives a control node:

```python
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "vout", "gnd", 2.0)   # stand-in source
b.add_resistor("Rin", "vout", "gnd", 1e3)
b.add_resistor("Rout", "ctrl", "gnd", 1e3)

def step(t, dt, inp, out, state):
    out[0] = 3.0 * inp[0]            # control law

p.add_c_block(
    b,
    inputs =[("v", "vout")],          # read the vout node voltage
    outputs=[("v", "ctrl", "gnd")],   # drive a controlled voltage source
    dt=1e-4,                          # block sample time (100 µs)
    fn=step,
)

res = p.simulate(b, t_end=2e-3, dt=2e-6)
print(res.v("ctrl")[-1])              # ≈ 6.0  (= 3 · 2.0)
```

---

## At a glance

| Topic | API |
|---|---|
| Add a block | `add_c_block(builder, inputs, outputs, *, dt, fn= \| lib= \| code=)` |
| Input wires (read) | `("v", node)` node voltage · `("i", branch)` branch current |
| Output wires (drive) | `("v", n+, n-)` controlled V source · `("i", n+, n-)` controlled I source |
| Sample time | `dt` — fires at `t=0`, then every `dt`; zero-order hold between |
| Python | `fn=callable` |
| Compiled C/C++ | `lib="block.so"` (+ `symbol=`) |
| Inline C/C++ | `code="…", lang="c"\|"cpp"` |
| From YAML | `wire_c_blocks_from_yaml(loaded, spec)` |
| Inspect at runtime | the returned `CBlockHandle` (`.outputs`, `.state`, `.n_fires`) |

---

## The step interface

Whatever the language, the block implements the **same logical step**,
invoked once per `dt`:

```
step(t, dt, in[N], out[M], state)
```

| Argument | Meaning |
|---|---|
| `t` | current time (s) |
| `dt` | the block sample time (s) — *not* the simulation `dt` |
| `in` | the `N` input wires, sampled at this step (read-only) |
| `out` | write the `M` outputs here; held until the next block step |
| `state` | persists across steps — a dict (Python) or an opaque pointer (C/C++) |

`N` and `M` are simply `len(inputs)` and `len(outputs)` — you choose them.

---

## Writing the block

### Python

The step is a plain callable. Write `out` in place, or `return` a
length-`M` sequence.

```python
def step(t, dt, inp, out, state):
    # stateful example: a leaky integrator + gain
    state["z"] = 0.99 * state.get("z", 0.0) + inp[0] * dt
    out[0] = 2.0 * inp[0] + state["z"]

p.add_c_block(b, inputs=[("v", "vout")], outputs=[("v", "ctrl", "gnd")],
              dt=1e-4, fn=step, state={"z": 0.0})
```

For hot loops, a Numba-jitted function works too (see
`pulsim.fast_block`).

### Inline C / C++

`code` is the **body** of the step function. In scope: `in`, `out`,
`n_in`, `n_out`, `t`, `dt`, `state`. Pulsim wraps it in the ABI,
compiles it once with the system compiler, and caches the result (keyed
by source + flags), so repeat runs don't recompile.

```python
# C
p.add_c_block(b, inputs=[("v", "vout")], outputs=[("v", "ctrl", "gnd")],
              dt=1e-4, lang="c", code="out[0] = 0.5 * in[0];")

# C++ (math from <cmath> available)
p.add_c_block(b, inputs=[("v", "vout")], outputs=[("v", "ctrl", "gnd")],
              dt=1e-4, lang="cpp", code="out[0] = std::tanh(in[0]);")
```

`<math.h>`/`<cmath>`, `<string.h>`/`<cstring>`, `<stdlib.h>`/`<cstdlib>`
are pre-included. Extra headers/libraries:

```python
p.add_c_block(b, ..., lang="c", code="...",
              include_dirs=["/opt/mylib/include"],
              extra_compile_args=["-ffast-math"],
              extra_link_args=["-lm"])
```

### Precompiled C / C++ shared library

Ship a `.so`/`.dll`/`.dylib` that exports the [ABI](#the-c-abi). Useful
for larger blocks, shared IP, or code you don't want recompiled.

```c
/* my_block.c */
#include "pulsim_cblock.h"   /* or copy the prototypes from p.CBLOCK_ABI */
#include <stdlib.h>

void *pulsim_cblock_init(int n_in, int n_out) {
    return calloc(1, sizeof(double));          /* one state word */
}
void pulsim_cblock_term(void *s) { free(s); }

void pulsim_cblock_step(const double *in, int n_in,
                        double *out, int n_out,
                        double t, double dt, void **state) {
    double *acc = (double *)(*state);
    *acc += in[0] * dt;                        /* integrate */
    out[0] = 0.5 * in[0] + *acc;
}
```

Compile (note `-shared -fPIC`):

```bash
cc  -shared -fPIC -O2 my_block.c   -o my_block.so      # Linux/macOS (C)
c++ -shared -fPIC -O2 my_block.cpp -o my_block.so      # C++ (keep extern "C")
cl /LD my_block.c                                       # Windows (MSVC)
```

```python
p.add_c_block(b, inputs=[("v", "vout")], outputs=[("v", "ctrl", "gnd")],
              dt=1e-4, lib="my_block.so")
# multiple blocks in one library → pass symbol="my_other_step"
```

!!! tip "Same code, both compilers"
    The ABI is `extern "C"`, so the **same** `.c` file compiles as C or
    C++. In C++ keep the `extern "C"` linkage (the shipped header and the
    inline `lang="cpp"` template do this for you) so the symbols aren't
    name-mangled.

---

## The C ABI

The contract a compiled block must satisfy — available as the string
`pulsim.CBLOCK_ABI` and as the shipped header `pulsim/cblock_abi.h`:

```c
/* required */
void  pulsim_cblock_step(const double *in, int n_in,
                         double *out, int n_out,
                         double t, double dt, void **state);
/* optional */
void *pulsim_cblock_init(int n_in, int n_out);   /* allocate state */
void  pulsim_cblock_term(void *state);           /* free state */
```

- `in` / `out` — contiguous `double` buffers of length `n_in` / `n_out`.
  Read `in`, write `out`.
- `state` — an opaque pointer **you own**. Either allocate it in
  `pulsim_cblock_init` (its return value is passed back as `*state` to
  every `step`, and to `term`), or lazily inside `step` via
  `if (!*state) { *state = malloc(...); }`. `pulsim_cblock_term` runs
  when the block handle is garbage-collected.
- Symbol names are configurable via `symbol=` (default
  `pulsim_cblock_step`; `init`/`term` use the `…_init`/`…_term` suffixes).

---

## Wires — reading and driving the circuit

### Inputs (read)

| Wire | Reads |
|---|---|
| `("v", "node")` | the node voltage `V(node)` |
| `("i", "branch")` | the current through an inductor or source `branch` |

Input wires are resolved to state-vector locations when you call
`add_c_block`, so add the block **after** the circuit topology is built.

### Outputs (drive)

Each output wire creates **one controlled source**:

| Wire | Imposes |
|---|---|
| `("v", "n+", "n-")` | a voltage `V(n+) − V(n−) = out[k]` |
| `("i", "n+", "n-")` | a current `out[k]` flowing from `n+` to `n−` |

Outputs are injected through Pulsim's `b_extra` path, which is part of
the fixed-step (**PWL**) engine — see [Engines](#engines-and-limits).

---

## Sample time, ZOH, and the one-sample delay

`dt` is the block's **own** rate, independent of the simulation step.
The step fires at `t = 0` and at each subsequent multiple of `dt`;
between firings the outputs are **held constant** (zero-order hold) —
exactly how a discrete controller sees a continuous plant.

```python
# Block runs at 20 kHz while the sim integrates at 500 kHz:
p.add_c_block(b, ..., dt=50e-6, fn=step)   # sim with dt=2e-6
```

- `dt` must be ≥ the simulation `dt`; if smaller it is **clamped** to the
  sim step with a warning. Pass `sim_dt=` to `add_c_block` if you want
  that check to run at build time rather than at `simulate`.
- There is a **one-sample delay** in the feedback path: the output
  applied during a step was computed from the *previous* block step's
  inputs. This is inherent to any sampled block (same as PSIM/PLECS
  discrete C blocks). Algebraic (zero-delay feedthrough) blocks are not
  supported.

---

## State management

| Language | State |
|---|---|
| Python | the `state` dict argument; persists across steps. Seed it with `state={...}` on `add_c_block`. |
| Compiled C/C++ | the opaque `*state` pointer — allocate in `init` (or lazily), free in `term`. One instance per `add_c_block` call. |
| Inline C/C++ | works the same via `*state`. **Caveat:** a `static` variable in inline code is shared by every block compiled from *identical* source (they share one cached library) — use `*state` for per-instance state. |

---

## From YAML

Declare blocks declaratively and wire them after loading the circuit:

```python
loaded = p.load_yaml_file("drive.yaml")
p.wire_c_blocks_from_yaml(loaded, [
    {
        "name": "duty_ctrl",
        "inputs":  [["v", "vout"]],
        "outputs": [["v", "ctrl", "gnd"]],
        "dt": 1e-4,
        "lang": "c",
        "code": "out[0] = 0.5 * in[0];",
    },
])
res = p.simulate(loaded.builder, t_end=..., dt=2e-6)
```

`spec` may be a Python list of dicts (shown) or a YAML string. Per-block
fields:

| Field | Required | Notes |
|---|---|---|
| `inputs` | yes | list of `[kind, name]` wires |
| `outputs` | yes | list of `[kind, n+, n-]` wires |
| `dt` | yes | block sample time (s) |
| `code` + `lang` | one of | inline source |
| `file` + `lang` | one of | path to a source file (read as `code`) |
| `lib` | one of | path to a precompiled library |
| `symbol` | no | step symbol name |
| `name` | no | block / output-source name |
| `include_dirs`, `extra_compile_args`, `extra_link_args` | no | compiler passthrough |

(Python `fn` callables can't be serialised, so YAML blocks use
`code` / `file` / `lib`.)

---

## Worked example: a discrete PI controlling a converter

A PI compensator living entirely in a C block, regulating an LC output
to a voltage setpoint:

```python
b = p.CircuitBuilder()
b.add_inductor("L", "sw", "out", 10e-3)
b.add_capacitor("C", "out", "gnd", 10e-6)
b.add_resistor("R", "out", "gnd", 10.0)

setpoint, Kp, Ki, umin, umax = 5.0, 0.3, 200.0, 0.0, 20.0

def pi(t, dt, inp, out, st):
    e = setpoint - inp[0]                 # read V(out)
    integ = st.get("i", 0.0) + e * dt
    u = Kp * e + Ki * integ
    if   u > umax: u, integ = umax, (umax - Kp * e) / Ki   # anti-windup
    elif u < umin: u, integ = umin, (umin - Kp * e) / Ki
    st["i"] = integ
    out[0] = u                            # drive the control voltage

p.add_c_block(b, inputs=[("v", "out")], outputs=[("v", "sw", "gnd")],
              dt=50e-6, fn=pi, name="PI")

res = p.simulate(b, t_end=40e-3, dt=2e-6)   # V(out) → 5 V
```

The block reads `V(out)`, runs the discrete PI at 20 kHz, and drives the
control voltage back; Pulsim solves the plant and the controller
together.

---

## Engines and limits

- **Outputs require the PWL engine** (`engine="pwl"`, the default) — they
  are injected via `b_extra`, which the variable-step DSED engine does
  not consume. A block with **inputs only** (logging, an observer, a
  software trip) also runs under `engine="dsed"`.
- One-sample (ZOH) feedback delay; no algebraic feedthrough.
- Inline compilation needs a system C/C++ compiler; without one, use a
  precompiled `lib=`.

---

## Performance

- Python blocks pay one Python call per **block step** (not per sim
  step) — coarse `dt` keeps that cheap; ZOH skips the call between
  firings.
- Compiled `lib=` / `code=` blocks are one C call per block step; numpy
  buffers are passed by pointer (no copy).
- The inline cache means a given source compiles once per machine.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `provide exactly one of fn= / lib= / code=` | give a single step source |
| `no C/C++ compiler found` | install a compiler, or ship a precompiled `lib=` |
| `shared library … does not export 'pulsim_cblock_step'` | wrong `symbol=`, or missing `extern "C"` in C++ |
| output has the wrong sign | check the output wire orientation `("v", n+, n-)` |
| block never updates | `dt` larger than `t_end`; lower `dt` |
| `dt … < simulation dt … clamping` warning | block `dt` below the sim step — raise it |
| outputs ignored under `engine="dsed"` | outputs are PWL-only; use `engine="pwl"` |

---

## Reference

- `pulsim.add_c_block(builder, inputs, outputs, *, dt, fn=None, lib=None,
  code=None, lang="c", symbol="pulsim_cblock_step", include_dirs=None,
  extra_compile_args=None, extra_link_args=None, name="CBLK",
  state=None, sim_dt=None) -> CBlockHandle`
- `pulsim.wire_c_blocks_from_yaml(loaded, spec, *, sim_dt=None) -> list[CBlockHandle]`
- `pulsim.CBlockHandle` — `.outputs`, `.state`, `.n_fires`, `.dt_block`
- `pulsim.CBLOCK_ABI` — the C ABI as a string; header: `pulsim/cblock_abi.h`
