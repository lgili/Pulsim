"""Example 05 — C block with stateful context (optional init/destroy).

This example shows the full ABI surface: a C block with an ``init`` callback
that allocates state, a ``step`` callback that accumulates an integral, and a
``destroy`` callback that frees memory.

Requires gcc / clang (or PULSIM_CC).  Falls back to a PythonCBlock equivalent
if no compiler is available.

Run::

    PYTHONPATH=python python examples/cblock/05_stateful_c_block.py
"""

import sys
import textwrap

from pulsim.cblock import CBlockCompileError, CBlockLibrary, PythonCBlock, compile_cblock, detect_compiler

# C source: integrator with heap-allocated state ------------------------------
SOURCE = textwrap.dedent("""
    #include <stdlib.h>
    #include <string.h>

    int pulsim_cblock_abi_version = 1;

    typedef struct { double sum; } Ctx;

    int pulsim_cblock_init(void** ctx_out, const void* info)
    {
        (void)info;
        Ctx* c = (Ctx*)calloc(1, sizeof(Ctx));
        if (!c) return -1;
        *ctx_out = c;
        return 0;
    }

    int pulsim_cblock_step(
        void* ctx, double t, double dt,
        const double* in, double* out)
    {
        (void)t;
        Ctx* c = (Ctx*)ctx;
        if (c) c->sum += in[0] * dt;
        out[0] = c ? c->sum : 0.0;
        return 0;
    }

    void pulsim_cblock_destroy(void* ctx)
    {
        free(ctx);
    }
""")

DT = 100e-6
N_STEPS = 10


def run_python_fallback() -> None:
    """Pure-Python integrator using PythonCBlock."""
    print("Running Python fallback integrator:")

    def integrator(ctx, t, dt, inputs):
        ctx.setdefault("sum", 0.0)
        ctx["sum"] += inputs[0] * dt
        return [ctx["sum"]]

    blk = PythonCBlock(integrator, n_inputs=1, n_outputs=1, name="integrator")
    for i in range(N_STEPS):
        t = i * DT
        dt = DT if i > 0 else 0.0
        out = blk.step(t, dt, [1.0])
        print(f"  t={t*1e3:.2f} ms  integral={out[0]:.6f}")


def run_c_integrator(cc: str) -> None:
    """C integrator compiled from SOURCE."""
    print(f"Compiling C integrator with: {cc}")
    try:
        lib_path = compile_cblock(SOURCE, name="integrator_c_example", compiler=cc)
    except CBlockCompileError as e:
        print(f"Compilation failed:\n{e.stderr_output}")
        run_python_fallback()
        return

    print(f"Loaded: {lib_path}\n")
    with CBlockLibrary(lib_path, n_inputs=1, n_outputs=1, name="integrator") as blk:
        for i in range(N_STEPS):
            t = i * DT
            dt = DT if i > 0 else 0.0
            out = blk.step(t, dt, [1.0])
            print(f"  t={t*1e3:.2f} ms  integral={out[0]:.6f}")

    print(f"\nFinal integral ≈ {out[0]:.6f}  (expected ~{(N_STEPS - 1) * DT:.6f})")


cc = detect_compiler()
if cc is None:
    run_python_fallback()
else:
    run_c_integrator(cc)
