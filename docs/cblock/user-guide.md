# C-Block User Guide

## Writing your first C-Block

A C-Block is a `.c` file that exports at most three functions and one integer.
The only **required** export is `pulsim_cblock_step`.

### Minimal stateless block

```c
/* gain3x.c */
#include "pulsim/v1/cblock_abi.h"

/* Required: declare ABI version so the loader can verify compatibility. */
PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;

/* Required: compute outputs from inputs. */
PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx,       /* opaque state pointer (NULL if no init) */
    double t,                    /* simulation time [s] */
    double dt,                   /* timestep [s] */
    const double* in,            /* input array, length == n_inputs */
    double* out)                 /* output array, length == n_outputs */
{
    (void)ctx; (void)t; (void)dt;
    out[0] = 3.0 * in[0];
    return 0;   /* 0 = success; non-zero triggers CBlockRuntimeError */
}
```

Compile and run:

```python
from pulsim.cblock import compile_cblock, CBlockLibrary

lib = compile_cblock("gain3x.c", name="gain3x")
blk = CBlockLibrary(lib, n_inputs=1, n_outputs=1)
assert blk.step(0.0, 1e-6, [5.0]) == [15.0]
```

## Persistent state with `pulsim_cblock_init`

When your block needs to remember state across time steps (filters, integrators,
controllers), allocate it in `pulsim_cblock_init` and free it in
`pulsim_cblock_destroy`.

```c
/* iir_filter.c — first-order low-pass */
#include "pulsim/v1/cblock_abi.h"
#include <math.h>
#include <stdlib.h>

typedef struct { double y_prev; double alpha; } FilterState;

PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;

PULSIM_CBLOCK_EXPORT int pulsim_cblock_init(
    void** ctx_out, const PulsimCBlockInfo* info)
{
    (void)info;
    FilterState* s = (FilterState*)malloc(sizeof(FilterState));
    if (!s) return -1;
    s->y_prev = 0.0;
    s->alpha  = 0.0;   /* updated each step using dt */
    *ctx_out = s;
    return 0;
}

PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx, double t, double dt,
    const double* in, double* out)
{
    (void)t;
    FilterState* s = (FilterState*)ctx;
    double fc = 100.0;                    /* cut-off frequency [Hz] */
    double tau = 1.0 / (2.0 * 3.14159265358979323846 * fc);
    double alpha = dt / (tau + dt);       /* bilinear approximation */
    s->y_prev = alpha * in[0] + (1.0 - alpha) * s->y_prev;
    out[0] = s->y_prev;
    return 0;
}

PULSIM_CBLOCK_EXPORT void pulsim_cblock_destroy(PulsimCBlockCtx* ctx)
{
    free(ctx);
}
```

The Python side is identical — Pulsim calls `init` automatically on load and
`destroy` on `CBlockLibrary.reset()` or `__del__`.

## Multi-output blocks and SIGNAL_DEMUX

A block with `n_outputs > 1` produces a vector each step. If you need to route
individual outputs to different downstream blocks, use `SIGNAL_DEMUX`.

```c
/* split.c — 1 input → [2×input, 3×input] */
PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx, double t, double dt,
    const double* in, double* out)
{
    (void)ctx; (void)t; (void)dt;
    out[0] = 2.0 * in[0];
    out[1] = 3.0 * in[0];
    return 0;
}
```

```python
from pulsim.cblock import compile_cblock, CBlockLibrary

lib = compile_cblock("split.c", name="split")
blk = CBlockLibrary(lib, n_inputs=1, n_outputs=2)
result = blk.step(0.0, 0.0, [5.0])
assert result == [10.0, 15.0]
```

In a YAML netlist, connect the block to a `SIGNAL_DEMUX` to fan out individual
outputs to different wires.

## Compiling and loading

### `compile_cblock()`

```python
from pulsim.cblock import compile_cblock

lib_path = compile_cblock(
    "my_block.c",           # path to .c file (or inline C source as str)
    name="my_block",        # stem of output filename
    extra_cflags=["-lm"],   # any extra flags (libraries, include paths, …)
    compiler=None,          # None → auto-detect; or pass "/usr/bin/gcc"
)
```

`compile_cblock` adds `-O2 -shared -fPIC -std=c11 -Wall -Wextra` automatically
on POSIX. Raises `CBlockCompileError` if compilation fails (message includes
stdout/stderr).

### `CBlockLibrary`

```python
from pulsim.cblock import CBlockLibrary

# As a context manager (recommended):
with CBlockLibrary(lib_path, n_inputs=1, n_outputs=1) as blk:
    out = blk.step(t=0.0, dt=1e-6, inputs=[3.0])

# Or keep open:
blk = CBlockLibrary(lib_path, n_inputs=2, n_outputs=1, name="pid")
blk.reset()   # re-runs init, clears state
```

## Python callable fallback (`PythonCBlock`)

No compiler? Use a Python function with the same interface:

```python
from pulsim.cblock import PythonCBlock

def my_gain(ctx, t, dt, inputs):
    return [4.0 * inputs[0]]

blk = PythonCBlock(fn=my_gain, n_inputs=1, n_outputs=1, name="gain4")
assert blk.step(0.0, 0.0, [2.5]) == [10.0]
```

The `ctx` argument is a plain `dict` — store any state you need:

```python
def accumulator(ctx, t, dt, inputs):
    ctx["total"] = ctx.get("total", 0.0) + inputs[0] * dt
    return [ctx["total"]]

blk = PythonCBlock(fn=accumulator, n_inputs=1, n_outputs=1)
blk.step(0.0, 0.001, [1.0])   # total = 0.001
blk.step(0.001, 0.001, [1.0]) # total = 0.002
blk.reset()                    # ctx cleared → total = 0
```

## YAML netlist integration

```yaml
schema: pulsim-v1
version: 1

components:
  - type: c_block
    name: GAIN
    nodes: [in_wire, out_wire]
    n_inputs: 1
    n_outputs: 1
    source: gain3x.c          # compile at simulation start

  - type: c_block
    name: LUT
    nodes: [vds, id, loss]
    n_inputs: 2
    n_outputs: 2
    lib_path: prebuilt/efficiency_map.so   # use pre-compiled library
    extra_cflags: ["-lm"]
```

Exactly one of `source` or `lib_path` must be present. Omitting both is valid
only when the block is instantiated programmatically with a `python_fn`.

## Debugging C-Block code

**Print to stderr** — Pulsim does not capture stderr, so `fprintf(stderr, ...)` prints
to the terminal immediately:

```c
fprintf(stderr, "[my_block] t=%.6f in[0]=%.4f\n", t, in[0]);
```

**Return non-zero on error** — any non-zero return from `pulsim_cblock_step`
triggers `CBlockRuntimeError` with the return code, simulation time, and step
index. Use this to signal NaN / overflow / missing data.

**GDB attach** — because the block runs in-process, a normal `gdb python3`
session can set breakpoints in your C source if compiled with `-g`:

```python
lib_path = compile_cblock("my_block.c", name="my_block", extra_cflags=["-g", "-O0"])
```

## Security and trust model

!!! warning "Only load libraries you trust"
    `CBlockLibrary` calls arbitrary machine code with the full privileges of
    the Python interpreter. Never load `.so`/`.dll` files from untrusted sources.
    In production or multi-tenant environments, consider sandboxing (containers,
    seccomp) at the process level — Pulsim provides no sandboxing of its own.
