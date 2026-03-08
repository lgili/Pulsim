# C-Block Examples

All five examples live under `examples/cblock/` and are fully self-contained
runnable scripts. Each demonstrates a progressively more advanced use case.

Run any example from the repository root:

```sh
python examples/cblock/01_passthrough_gain/01_passthrough_gain.py
```

---

## Example 1 — Passthrough gain

**Files:** `examples/cblock/01_passthrough_gain/`

| File | Purpose |
|---|---|
| `gain_block.c` | Stateless C-Block that multiplies its input by 3× |
| `01_passthrough_gain.py` | Compile, load, wire into a circuit, assert output |

### What it demonstrates

- The minimal C-Block skeleton: one required export (`pulsim_cblock_step`),
  one required integer (`pulsim_cblock_abi_version`), no state.
- Using `compile_cblock()` to compile on the fly and `CBlockLibrary` to load.
- Wiring the block inside a `SignalEvaluator` graph with a `CONSTANT` source.

### Key code

```c
/* gain_block.c */
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

```python
lib = compile_cblock(C_SOURCE / "gain_block.c", name="gain_block")
blk = CBlockLibrary(lib, n_inputs=1, n_outputs=1)
assert blk.step(0.0, 1e-6, [2.0]) == [6.0]
```

### Design notes

Calling `(void)ctx` suppresses the "unused parameter" warning from `-Wall`.
For truly stateless blocks this is the correct pattern: no `init`, no `destroy`,
context is always `NULL`.

---

## Example 2 — First-order IIR filter

**Files:** `examples/cblock/02_first_order_filter/`

| File | Purpose |
|---|---|
| `iir_filter.c` | First-order low-pass filter with persistent state |
| `02_first_order_filter.py` | Run step response, verify 95% of final value at 3τ |

### What it demonstrates

- The `pulsim_cblock_init` / `pulsim_cblock_destroy` lifecycle: allocating a
  heap struct for state and freeing it cleanly.
- Computing `alpha` from `dt` each step — correct for variable-timestep solvers.
- Using `blk.reset()` to repeat the simulation with fresh initial conditions.

### Key code

```c
typedef struct { double y_prev; } State;

int pulsim_cblock_init(void** ctx_out, const PulsimCBlockInfo* info) {
    State* s = malloc(sizeof(State));
    s->y_prev = 0.0;
    *ctx_out = s;
    return 0;
}

int pulsim_cblock_step(
    PulsimCBlockCtx* ctx, double t, double dt,
    const double* in, double* out)
{
    State* s = (State*)ctx;
    double fc = 100.0;                          /* [Hz] */
    double tau = 1.0 / (2.0 * M_PI * fc);
    double alpha = dt / (tau + dt);
    s->y_prev = alpha * in[0] + (1.0 - alpha) * s->y_prev;
    out[0] = s->y_prev;
    return 0;
}

void pulsim_cblock_destroy(PulsimCBlockCtx* ctx) { free(ctx); }
```

### Design notes

The `alpha = dt / (tau + dt)` formula is a bilinear (Tustin) approximation of
the continuous-time pole. It converges to the exact RC behaviour as `dt → 0`
and is stable for any positive `dt`. This is preferable to the Euler approximation
`alpha = 1 - exp(-dt/tau)` only because it is cheaper to compute (no `exp`).

---

## Example 3 — PI controller in a closed loop

**Files:** `examples/cblock/03_pi_controller_closed_loop/`

| File | Purpose |
|---|---|
| `pi_controller.c` | Clamp-output PI controller with two state variables |
| `03_pi_controller_closed_loop.py` | Close a voltage-regulation loop; compare C PI to the built-in `PI_CONTROLLER` block |

### What it demonstrates

- A 2-input C-Block: `[error, feedforward]` inputs, `[duty_cycle]` output.
- Output clamping inside the C function to keep duty in [0, 1].
- Numeric equivalence verification between a custom C implementation and Pulsim's
  built-in `PI_CONTROLLER` — useful for regression testing of ported algorithms.

### Key code

```c
typedef struct {
    double integral;
    double t_prev;
} PIState;

int pulsim_cblock_step(
    PulsimCBlockCtx* ctx, double t, double dt,
    const double* in, double* out)
{
    PIState* s = (PIState*)ctx;
    double error = in[0];
    double kp = 0.5, ki = 50.0;
    s->integral += error * dt;
    double u = kp * error + ki * s->integral;
    /* clamp to [0, 1] */
    out[0] = u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u);
    return 0;
}
```

### Design notes

Anti-windup (clamping the integral when output saturates) is omitted for
brevity but is straightforward to add:

```c
if (u >= 0.0 && u <= 1.0) {
    s->integral += error * dt;   /* only integrate when not saturated */
}
```

---

## Example 4 — Lookup table with file I/O

**Files:** `examples/cblock/04_lookup_table_efficiency/`

| File | Purpose |
|---|---|
| `efficiency_map.c` | 2D bilinear interpolation from a CSV lookup table |
| `efficiency_map.csv` | 3×3 grid: Vds [V] × Id [A] → loss [W] |
| `04_lookup_table_efficiency.py` | Set environment variable, compile, run operating points |

### What it demonstrates

- File I/O inside `pulsim_cblock_init`: reading a CSV at startup, storing the
  table on the heap, performing bilinear interpolation each step.
- Using `PULSIM_CBLOCK_CSV_PATH` as a runtime-configurable path (environment
  variable set before instantiating `CBlockLibrary`).
- Multi-output block: outputs are `[loss_W, efficiency]`.

### Key code

```python
os.environ["PULSIM_CBLOCK_CSV_PATH"] = str(CSV_PATH)
lib = compile_cblock(C_SOURCE, name="efficiency_map", extra_cflags=["-lm"])
blk = CBlockLibrary(lib, n_inputs=2, n_outputs=2)

# Exact grid point
out = blk.step(0.0, 0.0, [30.0, 10.0])  # Vds=30V, Id=10A
assert abs(out[0] - 1.50) < 1e-9        # loss_W
assert 0.0 <= out[1] <= 1.0             # efficiency in [0, 1]
```

### Design notes

Passing the CSV path via an environment variable rather than a hardcoded string
keeps the C source portable and avoids recompilation when the data file changes.
Alternatively, pass it through `PulsimCBlockInfo.name` (encode the path there)
or use extra `n_inputs` as a side-channel — but the environment variable pattern
is the simplest.

---

## Example 5 — Python callable, no compiler

**File:** `examples/cblock/05_python_callable_no_compiler.py`

### What it demonstrates

- `PythonCBlock` as a drop-in replacement for `CBlockLibrary` — identical `step`
  and `reset` API, no compilation, no shared library.
- Stateless logic (sigmoid with gain) as a pure function.
- Using the same block in both prototyping and production: swap `PythonCBlock`
  for `CBlockLibrary` once the C version is ready.

### Key code

```python
import math
from pulsim.cblock import PythonCBlock

def sigmoid_gain(ctx, t, dt, inputs):
    gain = 4.0
    x = gain * inputs[0]
    return [1.0 / (1.0 + math.exp(-x))]

blk = PythonCBlock(fn=sigmoid_gain, n_inputs=1, n_outputs=1, name="sigmoid")
assert abs(blk.step(0.0, 0.0, [0.0])[0] - 0.5) < 1e-12   # midpoint = 0.5
```

### Design notes

`PythonCBlock` is optimised for convenience, not throughput. For an inner loop
running at tens of kHz simulation time, compile the function to C eventually.
`PythonCBlock` is ideal for:

- Rapid prototyping / algorithm exploration.
- Unit tests that don't require a compiler.
- Logic that's inherently Python (e.g. calls into `scipy`, `numpy`, or Pandas).
