# C-Block — Custom Computation Blocks

C-Blocks let you inject **arbitrary signal-domain logic** into a Pulsim circuit without modifying the simulator's C++ core. You write a small C function (or a Python callable), the runtime loads it as a shared library, and it becomes a first-class block in the signal-evaluation graph — with the same wire-based connectivity as any built-in block.

## Why use a C-Block?

| Scenario | Recommendation |
|---|---|
| Standard filter, gain, integrator | Use built-in blocks (`PI_CONTROLLER`, `TRANSFER_FUNCTION`, …) |
| Lookup table, custom non-linearity | **C-Block** — direct C code, no overhead |
| Algorithm you already have in C | **C-Block** — just add the ABI wrapper |
| Rapid prototyping, no compiler | **`PythonCBlock`** — Python callable, zero compilation |
| Full device model (electrical ports) | C++ extension in `core/` (different path) |

## Quick-start (5 lines)

```python
from pulsim.cblock import compile_cblock, CBlockLibrary

lib = compile_cblock("gain3x.c", name="gain3x")
blk = CBlockLibrary(lib, n_inputs=1, n_outputs=1)
print(blk.step(0.0, 1e-6, [2.0]))   # → [6.0]
```

Where `gain3x.c` is:

```c
#include "pulsim/v1/cblock_abi.h"

PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;

PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx, double t, double dt,
    const double* in, double* out)
{
    (void)ctx; (void)t; (void)dt;
    out[0] = 3.0 * in[0];
    return 0;
}
```

## Two paths

```
Your logic
    │
    ├─── C source (.c)  ─── compile_cblock() ─── CBlockLibrary  ─┐
    │                                                               ├─ SignalEvaluator
    └─── Python callable ──── PythonCBlock ──────────────────────┘
```

**`CBlockLibrary`** loads a compiled shared library (`.so` / `.dylib` / `.dll`).
Use it for performance-critical code, code you already have in C, or anything that
needs `malloc` / file I/O.

**`PythonCBlock`** wraps any Python function.
Use it for prototyping: no compiler, no build step, same API.

## Using in a YAML netlist

```yaml
schema: pulsim-v1
version: 1

components:
  - type: c_block
    name: MY_GAIN
    nodes: [sig_in, sig_out]
    n_inputs: 1
    n_outputs: 1
    source: path/to/gain3x.c
```

## Next steps

- [Writing your first C-Block →](user-guide.md)
- [ABI reference →](abi-reference.md)
- [Python API reference →](api-reference.md)
- [Annotated examples →](examples.md)
