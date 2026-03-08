## Context

The Pulsim simulation engine evaluates circuits in two coupled domains:

1. **Electrical domain** — Modified Nodal Analysis (MNA).  Components stamp
   conductance matrices.  The Newton–Raphson solver iterates per timestep.
2. **Signal domain** — `SignalEvaluator` DAG, evaluated once per accepted
   step, after the MNA solution is accepted.  Inputs come from probes; outputs
   drive PWM duty cycles and similar controllable sources.

The C-Block lives entirely in **domain 2**.  It is called once per accepted
timestep (not per Newton iteration), receives signal values, and returns new
signal values.  This is identical to the PSIM "C DLL Block" execution model
and is the correct scope — control code rarely needs Newton-iteration granularity.

## Goals / Non-Goals

**Goals:**
- Let users write C functions (or Python callables) as signal-domain blocks.
- Support persistent state between steps (filter memory, PID integrator, etc.).
- Work cross-platform: Linux (.so), macOS (.dylib), Windows (.dll).
- No recompilation of Pulsim itself.
- Full algebraic-loop detection (existing DAG machinery).
- Graceful degradation when C compiler is absent (Python callable fallback).
- Error codes from user C code propagate as structured simulation diagnostics.

**Non-Goals:**
- C-Block stamping MNA matrices directly (requires CRTP device, breaking ABI).
- GPU/FPGA execution of C-Block code.
- Hot-reload during a running simulation.
- Sandboxing / security isolation (same posture as PSIM/PLECS: trusted code).
- C++ blocks (ABI is C to avoid mangling; C++ callers wrap with `extern "C"`).

## Decisions

### Decision 1: Stable versioned C ABI via a single header

The user-facing ABI is declared in `core/include/pulsim/v1/cblock_abi.h`:

```c
#pragma once
#define PULSIM_CBLOCK_ABI_VERSION 1

typedef struct PulsimCBlockCtx PulsimCBlockCtx;

typedef struct {
    int abi_version;   /* Set to PULSIM_CBLOCK_ABI_VERSION */
    int n_inputs;
    int n_outputs;
    const char* name;  /* NULL or block name string */
} PulsimCBlockInfo;

/* Optional – called once before simulation starts.
 * *ctx_out receives an opaque pointer the block allocates for state.
 * Return 0 on success; non-zero aborts with a diagnostic. */
typedef int (*pulsim_cblock_init_fn)
    (PulsimCBlockCtx** ctx_out, const PulsimCBlockInfo* info);

/* Required – called once per accepted timestep.
 * t  : current time [s]
 * dt : last accepted step size [s]
 * in : array of n_inputs doubles (read-only)
 * out: array of n_outputs doubles (pre-zeroed; write results here)
 * Return 0 to continue; non-zero aborts simulation. */
typedef int (*pulsim_cblock_step_fn)
    (PulsimCBlockCtx* ctx, double t, double dt,
     const double* in, double* out);

/* Optional – called once after simulation ends or on error.
 * Must free any memory allocated in init. */
typedef void (*pulsim_cblock_destroy_fn)(PulsimCBlockCtx* ctx);
```

The shared library must export, at minimum, the symbol `pulsim_cblock_step`.
`pulsim_cblock_init` and `pulsim_cblock_destroy` are optional (defaults to
no-op).  A `pulsim_cblock_abi_version` symbol (returns int) allows the loader
to reject incompatible future versions gracefully.

**Why not matching PSIM's `simuser(t, dt, in, out)` exactly?**
PSIM's signature lacks a context pointer, forcing users to use global variables.
Multiple instances of the same block then share state — a correctness hazard.
Using `void* ctx` (the common C idiom, also used by SQLite, FFTW, etc.) fixes this.

### Decision 2: Python-side compilation via `subprocess` + system compiler

`compile_cblock(source: str | Path, ...)` calls the system C compiler
(`cc`/`gcc`/`clang` on POSIX; `cl.exe` on Windows) as a subprocess to build a
shared library into a user-controlled output directory (default: `tempfile`).

**Rationale:**
- Zero new C++ build-system dependencies.
- Python's `ctypes` stdlib loads the resulting `.so`/`.dylib`/`.dll`.
- Mirrors the PSIM/PLECS workflow where the user presses "Compile" and the IDE
  calls the system compiler.
- `compile_cblock` raises `CBlockCompileError` with the full compiler stderr
  if compilation fails — identical in spirit to a compile error in the IDE.

**Compiler resolution order** (highest to lowest priority):
1. `compiler=` keyword argument passed directly to `compile_cblock()` — used as-is,
   no detection performed.  Allows pinning a specific toolchain per block.
2. Environment variable `PULSIM_CC` — process-wide override, useful for CI.
3. On POSIX: `cc`, then `gcc`, then `clang` (checked via `shutil.which`).
4. On Windows: MSVC `cl.exe` via VS Developer Command Prompt; fallback to
   `mingw-w64` `gcc.exe` if on PATH.
5. If none found: `CBlockCompileError` with actionable message telling the user
   to install a compiler, set `PULSIM_CC`, or pass `compiler=` explicitly.

This design matches PLECS, which has a "Compiler path" field in Preferences that
maps to option 1/2 above.

**Compile flags applied by default:**
```
-O2 -shared -fPIC -std=c11 -Wall -Wextra
```
User may pass `extra_cflags: list[str]` to append additional flags.

### Decision 3: Python-callable fallback (`PythonCBlock`)

For rapid prototyping and CI environments without a C toolchain, the user may
pass any Python callable as the step function:

```python
def my_gain(ctx, t, dt, inputs):
    return [inputs[0] * 2.5]

block = ps.PythonCBlock(fn=my_gain, n_inputs=1, n_outputs=1)
```

`PythonCBlock` and `CBlockLibrary` share the same interface:
```
.step(t, dt, inputs) -> list[float]
.reset()
.n_inputs / .n_outputs
```
This allows `SignalEvaluator` to use them interchangeably.

### Decision 4: `SignalEvaluator` integration — no C++ changes required

`"C_BLOCK"` is added to `SIGNAL_TYPES` and `_OUTPUT_PIN_NAMES` in
`signal_evaluator.py`.  During `build()`, each `C_BLOCK` component initialises
its `CBlockLibrary` or `PythonCBlock` into `self._controllers`.  During `step()`,
the block calls `.step(t, dt, inputs)` and stores the output vector.

For blocks with `n_outputs > 1`, a `SIGNAL_DEMUX` must be wired downstream.
The evaluator exposes the full output vector as a `list[float]` stored in
`_state[comp_id]`; the demux extracts individual outputs by index.

### Decision 5: YAML netlist schema

```yaml
components:
  - id: cb1
    name: MY_CONTROLLER
    type: C_BLOCK
    parameters:
      n_inputs: 2
      n_outputs: 1
      # Exactly one of: lib_path or source must be provided
      lib_path: path/to/controller.so    # pre-compiled shared library
      # source: path/to/controller.c     # compile at load time
      extra_cflags: []                   # optional extra compiler flags
    pins:
      - { index: 0, name: IN0, x: 0, y: 0 }
      - { index: 1, name: IN1, x: 0, y: 0 }
      - { index: 2, name: OUT, x: 0, y: 0 }
```

`lib_path` and `source` are mutually exclusive.  The YAML parser validates this
and rejects circuits where both or neither are specified.

### Decision 6: Security posture

C-Block code executes in-process with full privileges, identical to PSIM and
PLECS.  This is a deliberate design choice:
- The audience is professional engineers working with their own trusted code.
- Sandboxing (seccomp, WASM, etc.) would add significant complexity and latency.
- The documentation SHALL prominently warn users not to load untrusted libraries.

## Platform Matrix

| Platform       | Library extension | Default compiler     | Status       |
|----------------|-------------------|----------------------|--------------|
| Linux x86_64   | `.so`             | `cc` / `gcc` / `clang` | Primary     |
| macOS arm64    | `.dylib`          | `clang` (XCode CLI)  | Primary      |
| macOS x86_64   | `.dylib`          | `clang` (XCode CLI)  | Primary      |
| Windows x64    | `.dll`            | MSVC `cl.exe`        | Best-effort  |
| Windows x64    | `.dll`            | `mingw-w64 gcc`      | Best-effort  |

CI gates run on Linux and macOS.  Windows is tested in best-effort optional CI.

## Risks / Trade-offs

| Risk | Severity | Mitigation |
|------|----------|------------|
| User C code crashes (segfault, stack overflow) — kills Python process | High | Document clearly; advanced users can run in subprocess isolation (not provided by default) |
| Compiler not found in Docker/minimal CI images | Medium | `PythonCBlock` fallback; `skip` guard in test if `PULSIM_CC` not set |
| ABI binary compatibility across future Pulsim versions | Medium | `PULSIM_CBLOCK_ABI_VERSION` let us version-check at load time |
| Algebraic loop with C_BLOCK | Low | Existing DAG cycle detection catches it |
| Shared library leaks between test runs | Low | `CBlockLibrary.__del__` calls `_dlclose()` via ctypes |

## Migration Plan

No migration needed.  This is a purely additive change.

## Open Questions

- Should `compile_cblock` optionally embed the source in the `.so` debug section
  for reproducibility in benchmark archives?  → Defer to a follow-on spec.
- Should the YAML schema support `python_callable` (dotted import path) as a
  third alternative to `lib_path` / `source`?  → Yes, planned for Phase 2.
