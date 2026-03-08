# CBlock ABI Reference

The C ABI is defined in `core/include/pulsim/v1/cblock_abi.h`. Include it in
every C source file that implements a C-Block:

```c
#include "pulsim/v1/cblock_abi.h"
```

## Symbol overview

| Symbol | Type | Required | Description |
|---|---|---|---|
| `pulsim_cblock_abi_version` | `int` | **Yes** | ABI version integer |
| `pulsim_cblock_init` | function | No | Alllocate per-block state |
| `pulsim_cblock_step` | function | **Yes** | Compute outputs from inputs |
| `pulsim_cblock_destroy` | function | No | Free per-block state |

## ABI version

```c
#define PULSIM_CBLOCK_ABI_VERSION 1

PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;
```

Always set this to `PULSIM_CBLOCK_ABI_VERSION`. The loader reads the value at
runtime and raises `CBlockABIError` if it differs from the expected version.

| ABI version | Pulsim release | Changes |
|---|---|---|
| 1 | 0.1 | Initial version |

## `PULSIM_CBLOCK_EXPORT`

Platform-appropriate visibility macro. Apply to every exported symbol:

```c
PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;
PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(...) { ... }
```

| Platform | Expands to |
|---|---|
| GCC / Clang (Linux, macOS) | `__attribute__((visibility("default")))` |
| MSVC / Windows | `__declspec(dllexport)` |
| Other | *(empty)* |

On GCC/Clang you should also compile with `-fvisibility=hidden` to keep
internal symbols private and avoid accidental name collisions with symbols in
other loaded libraries. `PULSIM_CBLOCK_EXPORT` then explicitly makes only the
ABI symbols public.

On MSVC you can alternatively use a `.def` file to list the exported names.

## `PulsimCBlockCtx`

```c
typedef struct PulsimCBlockCtx PulsimCBlockCtx;
```

Opaque handle. Allocate on the heap inside `pulsim_cblock_init`; free inside
`pulsim_cblock_destroy`. The simulator never inspects the contents.

**Ownership rules:**

- `pulsim_cblock_init` allocates and sets `*ctx_out`. The caller takes ownership.
- `pulsim_cblock_destroy` receives ownership and must free.
- Between `init` and `destroy`, the pointer is passed by value to `step` — do
  not free or re-allocate it there.

## `PulsimCBlockInfo`

```c
typedef struct {
    int         abi_version;  /* PULSIM_CBLOCK_ABI_VERSION at call time */
    int         n_inputs;     /* number of scalar input channels         */
    int         n_outputs;    /* number of scalar output channels        */
    const char* name;         /* block name from netlist; may be NULL    */
} PulsimCBlockInfo;
```

Passed read-only to `pulsim_cblock_init`. All pointers (including `name`) are
valid only for the duration of the `init` call. Copy `name` if you need it later.

## `pulsim_cblock_init`

```c
int pulsim_cblock_init(
    PulsimCBlockCtx**       ctx_out,
    const PulsimCBlockInfo* info
);
```

**Optional.** Called once before the first `step`. Use it to allocate state.

- Set `*ctx_out` to the allocated context, or leave it `NULL` if no state is needed.
- Return `0` on success. Any nonzero return aborts the simulation with an error.

## `pulsim_cblock_step`

```c
int pulsim_cblock_step(
    PulsimCBlockCtx* ctx,
    double           t,
    double           dt,
    const double*    in,
    double*          out
);
```

**Required.** Called once per accepted simulation timestep.

| Parameter | Description |
|---|---|
| `ctx` | Opaque state from `init`; `NULL` if `init` was not exported |
| `t` | Current simulation time [s] |
| `dt` | Elapsed time since previous accepted step [s]; `0.0` at the first step |
| `in` | Input array; `n_inputs` elements (read-only) |
| `out` | Output array; `n_outputs` elements (write-only) |

**Return codes:**

| Value | Meaning |
|---|---|
| `0` | Success; `out` values accepted |
| Nonzero | Error; Python side raises `CBlockRuntimeError(return_code, t, step_index)` |

## `pulsim_cblock_destroy`

```c
void pulsim_cblock_destroy(PulsimCBlockCtx* ctx);
```

**Optional.** Called once after the simulation ends (even on error). Free any
memory allocated during `init`. If `init` was not exported, `destroy` is never
called.

## Memory ownership rules

1. The simulator owns `in[]` — do not retain a pointer to it after `step` returns.
2. The user owns `ctx` — allocate in `init`, free in `destroy`, nowhere else.
3. `out[]` is owned by the simulator — write your results there; do not free it.
4. `PulsimCBlockInfo` pointers (including `name`) are valid only during `init`.

## Platform notes

### Symbol visibility (`-fvisibility=hidden`)

Compile with:

```sh
gcc -shared -fPIC -fvisibility=hidden -o my_block.so my_block.c
```

This hides all symbols except those marked `PULSIM_CBLOCK_EXPORT`, preventing
name clashes with other loaded libraries.

`compile_cblock()` does *not* add `-fvisibility=hidden` automatically because it
may interfere with libraries that depend on default visibility. Add it via
`extra_cflags` if needed.

### MSVC `.def` files

On Windows with MSVC you can use a module-definition file instead of
`__declspec(dllexport)`:

```
; my_block.def
EXPORTS
    pulsim_cblock_abi_version
    pulsim_cblock_step
```

Pass `/DEF:my_block.def` in `extra_cflags` when calling `compile_cblock`.

### `extern "C"` for C++ implementations

The header wraps all declarations in `extern "C" { ... }` when compiled as C++,
so you can write the implementation in C++ as long as you `#include` the header
and mark the exported functions `extern "C"`:

```cpp
#include "pulsim/v1/cblock_abi.h"  // pulls in extern "C" wrapper
#include <cmath>

PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;

PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx, double t, double dt,
    const double* in, double* out)
{
    out[0] = std::sin(in[0]);
    return 0;
}
```
