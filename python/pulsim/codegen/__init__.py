"""Real-time C99 code generation from Pulsim Circuit / LinearSystem.

`add-realtime-code-generation` Phase 1+2+3+8: takes a `Circuit` plus a
fixed step `dt`, computes the discrete-time state-space matrices via
matrix-exponential discretization, and emits a self-contained
`model.c / model.h` pair plus a small `model_test.c` runner that
verifies the generated code reproduces the Pulsim-native transient
within tolerance.

Usage::

    import pulsim
    from pulsim.codegen import generate

    ckt = ...  # existing Circuit
    summary = generate(
        ckt,
        dt=1e-6,
        out_dir="gen/buck",
        target="c99",
    )
    print(summary.rom_estimate_bytes)
    print(summary.stability_radius)

The generated `model.c` exposes:

```c
typedef struct PulsimModel { /* state vector */ } PulsimModel;
void pulsim_model_init(PulsimModel*);
void pulsim_model_step(PulsimModel*, const float* u, float* y);
```

That signature is target-agnostic; downstream you compile against
your platform's toolchain (gcc, arm-gcc, xilinx-gcc...) and link into
your control loop. The PIL test harness wraps `pulsim_model_step` and
diffs against Pulsim native to confirm parity.
"""

from .generator import (
    CodegenSummary,
    generate,
    discretize_state_space,
    stability_radius,
)

__all__ = [
    "CodegenSummary",
    "generate",
    "discretize_state_space",
    "stability_radius",
]
