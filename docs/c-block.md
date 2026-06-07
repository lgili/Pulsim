# Custom Code Blocks (C / C++ / Python)

A **C block** is a PSIM-/Simulink-style sampled subsystem you wire into
the circuit: it **reads** a chosen set of circuit signals, runs **your
code** at a sample rate **you choose**, and **drives** signals back into
the circuit. Use it for digital control laws, lookup tables, protection
logic, state machines, or custom models that don't map to a fixed
component.

The step code can be **Python**, **C**, or **C++** — same interface. It
lives in `pulsim.c_block` and is re-exported as `pulsim.add_c_block`.

---

## At a glance

| | |
|---|---|
| Add a block | `p.add_c_block(builder, inputs, outputs, *, dt, fn=… \| lib=… \| code=…)` |
| Inputs (read) | `("v", node)` node voltage · `("i", branch)` branch current |
| Outputs (drive) | `("v", n+, n-)` controlled V source · `("i", n+, n-)` controlled I source |
| Sample time | `dt` — fires at `t=0` then every `dt`, zero-order hold between |
| Languages | `fn=` Python callable · `lib=` compiled `.so`/`.dll` · `code=`+`lang=` inline C/C++ |
| From YAML | `p.wire_c_blocks_from_yaml(loaded, spec)` |
| Engine | outputs use the **PWL** engine (`b_extra`); inputs-only also run under DSED |

The block rides Pulsim's per-step hooks (a throttled `step_observer` +
`b_extra` injection) and registers itself on the builder, so you just
call `simulate` — no manual wiring.

---

## 1. The step interface

Every block implements the same logical step, called once per `dt`:

```
step(t, dt, in[N], out[M], state)
```

- `in` — the N input wires, sampled now.
- `out` — write the M outputs here (held until the next block step).
- `state` — persists across steps (a dict in Python, an opaque pointer
  in C/C++).

### Python

```python
def step(t, dt, inp, out, state):
    state["acc"] = state.get("acc", 0.0) + inp[0] * dt   # integrator
    out[0] = 0.5 * inp[0] + state["acc"]

p.add_c_block(b,
    inputs =[("v", "vout")],
    outputs=[("v", "ctrl", "gnd")],
    dt=1e-4, fn=step)
```

### Inline C / C++

The `code` is the **body** of the step function; `in`, `out`, `t`, `dt`,
`n_in`, `n_out`, `state` are in scope. Pulsim compiles it to a cached
shared library with the system compiler.

```python
p.add_c_block(b, inputs=[("v", "vout")], outputs=[("v", "ctrl", "gnd")],
              dt=1e-4, lang="c", code="out[0] = 0.5 * in[0];")

p.add_c_block(b, inputs=[("v", "vout")], outputs=[("v", "ctrl", "gnd")],
              dt=1e-4, lang="cpp", code="out[0] = std::tanh(in[0]);")
```

### Precompiled C / C++ library

Compile against the ABI (`pulsim.CBLOCK_ABI`, or the shipped
`pulsim/cblock_abi.h`) and pass the library path. C++ must keep
`extern "C"` linkage.

```c
/* my_block.c  →  cc -shared -fPIC -O2 my_block.c -o my_block.so */
#include <stdlib.h>
void* pulsim_cblock_init(int n_in, int n_out) { return calloc(1, sizeof(double)); }
void  pulsim_cblock_term(void* s) { free(s); }
void  pulsim_cblock_step(const double* in, int n_in, double* out, int n_out,
                         double t, double dt, void** state) {
    double* acc = (double*)(*state);
    *acc += in[0] * dt;
    out[0] = 0.5 * in[0] + *acc;
}
```

```python
p.add_c_block(b, inputs=[("v", "vout")], outputs=[("v", "ctrl", "gnd")],
              dt=1e-4, lib="my_block.so")    # symbol= for a custom name
```

`init`/`term` are optional (allocate/free per-block state); omit them and
manage state lazily through `*state`.

---

## 2. Wires — reading and driving the circuit

**Inputs** sample the circuit each block step:

- `("v", "node")` → the node voltage,
- `("i", "branch")` → the current through an inductor or source branch.

**Outputs** each create one controlled source between two nodes:

- `("v", "n+", "n-")` → imposes a voltage `V(n+) − V(n−) = out[k]`,
- `("i", "n+", "n-")` → injects a current `out[k]` from `n+` to `n−`.

Outputs are injected through Pulsim's `b_extra` path, so they require the
fixed-step **PWL** engine. A block with **only inputs** (logging, an
observer) also runs under the variable-step **DSED** engine.

---

## 3. Sample time and zero-order hold

`dt` is the block's own rate. The step fires at `t = 0` and every `dt`
after; between firings the outputs are **held constant** (ZOH) — exactly
how a discrete controller sees the plant. `dt` must be ≥ the simulation
`dt` (it is clamped with a warning otherwise). Pass `sim_dt=` to
`add_c_block` if you want the clamp check at build time.

There is a **one-sample delay** in the feedback path (the output applied
at a step is computed from the previous block step's inputs) — inherent
to a sampled block, and the same as PSIM/PLECS discrete C blocks.

---

## 4. From YAML

```python
loaded = p.load_yaml_file("circuit.yaml")
p.wire_c_blocks_from_yaml(loaded, [
    {"inputs": [["v", "vout"]],
     "outputs": [["v", "ctrl", "gnd"]],
     "dt": 1e-4, "lang": "c", "code": "out[0] = 0.5*in[0];"},
])
res = p.simulate(loaded.builder, t_end=..., dt=2e-6)
```

Each entry takes `inputs`, `outputs`, `dt`, and the code as `code`+`lang`,
a source `file`+`lang`, or a precompiled `lib`. (Python `fn` callables
can't be serialised, so YAML blocks use `code`/`file`/`lib`.)

---

## Worked example: a PI controller in a C block

```python
b = p.CircuitBuilder()
b.add_inductor("L", "sw", "out", 10e-3)
b.add_capacitor("C", "out", "gnd", 10e-6)
b.add_resistor("R", "out", "gnd", 10.0)

setpoint, Kp, Ki = 5.0, 0.3, 200.0
def pi(t, dt, inp, out, st):
    e = setpoint - inp[0]
    st["i"] = st.get("i", 0.0) + e * dt
    out[0] = max(0.0, min(20.0, Kp * e + Ki * st["i"]))   # clamp

p.add_c_block(b, inputs=[("v", "out")], outputs=[("v", "sw", "gnd")],
              dt=50e-6, fn=pi, name="PI")
res = p.simulate(b, t_end=40e-3, dt=2e-6)
# V(out) converges to the 5 V setpoint.
```

The block reads `V(out)`, runs the discrete PI at 20 kHz, and drives the
control voltage back — Pulsim solves the plant and the controller
together.

---

## Security note

A compiled or inline C/C++ block executes **arbitrary native code** with
the same trust level as running your Python — there is no sandbox. Only
run blocks whose source/library you trust. Inline compilation uses your
own system compiler.

---

## Reference

- `pulsim.add_c_block`, `pulsim.wire_c_blocks_from_yaml`,
  `pulsim.CBlockHandle`, `pulsim.CBLOCK_ABI`.
- ABI header: `pulsim/cblock_abi.h`.
