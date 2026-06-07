## Context

Pulsim's transient loop already exposes three per-step hooks (see
`core/include/pulsim/solver/run_transient.hpp` and
`python/pulsim/__init__.py::simulate`):

- `step_observer(t, x)` — read the state vector `x` at the start of each
  step (works for PWL **and** DSED).
- `b_extra_fn(t) -> Vector` — add a contribution to the MNA RHS each step
  (**PWL only**). This is how a value is injected back into the circuit:
  a source assembled with value 0 becomes time-varying when its RHS row
  is set from `b_extra`.
- `switch_fn(t) -> SwitchStateMask` — controlled-switch state.

`simulate()` already **composes** lists of observers / b_extra fns (the
`closed_loops=` path), so a c-block can add its own without disturbing
user-supplied callbacks. Reading signals uses `builder.node_id_of(name)`
(node voltage) and the branch-id helpers (currents). There is no
kernel-level sample-time/sub-rate concept; the established pattern is
throttling inside `step_observer` (e.g. `bind_pi_to_switch(freq=...)`).

The kernel currently has **no** ctypes/cffi/dlopen path — runtime loading
of user C/C++ is new.

## Goals / Non-Goals

**Goals**
- One block abstraction, three languages (Python / C / C++), unified ABI.
- User picks N inputs, M outputs, and the block sample time.
- Inputs read node V / branch I; outputs drive controlled V/I sources.
- Rides the existing PWL callback path — minimal kernel risk for v1.
- Reproducible (YAML) and visual (GUI emits the same representation).

**Non-Goals (v1)**
- DSED output injection (DSED has no `b_extra`; inputs-only is fine).
- Sandboxing user native code (trust boundary documented).
- Algebraic/zero-sample-time blocks or implicit feedthrough solved inside
  the Newton loop (the block is **sampled**: outputs are ZOH from the
  previous block step, so there is a one-sample delay — standard for
  PSIM/PLECS discrete C blocks).
- In-kernel C++ block compilation into `_pulsim.so`.

## Decisions

### D1 — Execution model: orchestrate the existing PWL hooks
`add_c_block(builder, ...)` returns a handle and registers, via the
`simulate()` composition path:
1. a **step_observer** that, when `t - t_last >= dt_block - eps`, reads
   the N inputs from `x`, calls the user `step`, stores the M outputs,
   and advances `t_last`; otherwise no-op (ZOH);
2. a **b_extra_fn** that writes the held outputs into the RHS rows of the
   controlled sources every sim step.

No C++ change for v1. (A later native fast path can mirror the
`NativePwm2Switch` pattern; out of scope here.)

### D2 — Unified step ABI
Logical signature (all languages):
`step(t, dt_block, inputs[N] -> outputs[M], state)`.

- **Python**: `def step(t, dt, inp, out, state): ...` where `inp`/`out`
  are length-N/M float buffers (numpy views) and `state` is a dict
  persisted across calls. (Or `return out_list` for a pure-functional
  style.)
- **C/C++ ABI** (`extern "C"`):
  ```c
  void pulsim_cblock_step(const double* in, int n_in,
                          double* out, int n_out,
                          double t, double dt, void** state);
  /* optional */
  void* pulsim_cblock_init(int n_in, int n_out);   /* alloc state */
  void  pulsim_cblock_term(void* state);           /* free state  */
  ```
  `state` is an opaque pointer the block owns (alloc in `init`, freed in
  `term`); `**state` lets a single function lazily allocate too.

### D3 — C/C++ delivery (BOTH)
- **Shared library**: `lib="path/to/block.so"` → `ctypes.CDLL`, resolve
  the symbols, marshal numpy↔`double*`. C++ uses `extern "C"`.
- **Inline source**: `code="...", lang="c"|"cpp"` → wrap in a template
  carrying the ABI, write to a temp dir, compile with `cc`/`c++`
  (`-shared -fPIC -O2`), cache the `.so` by **hash(source+flags+compiler)**
  so rebuilds are skipped, then load via the shared-library path. Missing
  compiler → clear error pointing to the shared-lib option.

### D4 — Wire specs
- **Input** wire: `("v", "node")` (node voltage) or `("i", "branch")`
  (branch current; resolved via the existing inductor/source branch-id
  helpers). Resolved to a state index once at `add_c_block` time.
- **Output** wire: `("v", "n+", "n-")` inserts a controlled **voltage**
  source; `("i", "n+", "n-")` inserts a controlled **current** source.
  The block writes `out[k]`; the b_extra_fn injects it into that source's
  RHS row.

### D5 — Sample time / ZOH
`dt_block` (seconds). The block fires at `t = 0` and every `dt_block`
thereafter (nearest sim step at-or-after the boundary); outputs are held
constant between firings. `dt_block` defaults to the sim `dt`. A warning
is emitted if `dt_block < dt` (clamped to `dt`).

### D6 — Surfaces
- **Python**: `add_c_block(builder, inputs, outputs, *, dt, code=…|lib=…|fn=…, lang=…, name=…)`.
- **YAML**: a `c_block` node mirroring the Python kwargs (code may be
  inline or a file path).
- **GUI**: a PulsimGUI node with N input / M output pins; it serialises
  to the YAML/Python `c_block`. Implemented in PulsimGUI (separate repo).

## Alternatives considered

- **Numba `fast_block`** (exists): Python→LLVM, but not C/C++ and not a
  wired multi-IO sampled block. Complementary, not a substitute.
- **Extend `MixedDomainBlockChain`** with a user C++ block type: requires
  recompiling `_pulsim.so` per user block — not viable for end users.
- **Inline-source only** (no shared-lib): simplest UX but forces a
  compiler on every machine and can't ship precompiled IP. Rejected in
  favour of BOTH.

## Risks / Trade-offs

- **Arbitrary native code execution** → documented trust boundary; inline
  compile uses the user's own toolchain; no sandbox (out of scope).
- **One-sample (ZOH) delay** in the feedback path → inherent to sampled
  blocks; documented; matches PSIM/PLECS. Algebraic-loop blocks are a
  non-goal.
- **Compiler portability** (flags/paths across macOS/Linux/Windows) →
  mitigated by the shared-lib path always being available and by caching;
  inline compile is best-effort with a clear fallback message.
- **ctypes marshaling overhead** per block step → negligible at typical
  controller sample rates; the shared-lib call is one C call per firing.

## Migration Plan

Purely additive. No existing API changes. Existing `step_observer` /
`b_extra_fn` users are unaffected (the c-block composes alongside them).

## Open Questions

- Multi-rate scheduling of *several* c-blocks at different `dt_block`:
  v1 supports each block at its own rate independently; a shared
  discrete-time scheduler is a possible later optimisation.
- Should inline C++ allow `#include` of user headers / extra link flags?
  v1 exposes `include_dirs` / `extra_compile_args` / `extra_link_args`
  passthrough; broader build integration deferred.
