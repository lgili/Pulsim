"""Example 04 — compile and load a C gain block (requires a C compiler).

Demonstrates the full workflow:
  1. Write a minimal C CBlock source string.
  2. Compile it to a shared library with ``compile_cblock``.
  3. Load it with ``CBlockLibrary``.
  4. Call ``step`` in a loop.

To run this example you need gcc / clang (or set PULSIM_CC).

Run::

    PYTHONPATH=python python examples/cblock/04_compile_and_load_c_block.py
"""

import sys
import textwrap

from pulsim.cblock import CBlockCompileError, CBlockLibrary, compile_cblock, detect_compiler

# Check that a compiler is available before trying ----------------------------
cc = detect_compiler()
if cc is None:
    print("No C compiler found.  Set PULSIM_CC or install gcc/clang.")
    sys.exit(0)

print(f"Using compiler: {cc}")

# Minimal CBlock source: y = 2 * x -------------------------------------------
source = textwrap.dedent("""
    int pulsim_cblock_abi_version = 1;

    int pulsim_cblock_step(
        void* ctx, double t, double dt,
        const double* in, double* out)
    {
        (void)ctx; (void)t; (void)dt;
        out[0] = 2.0 * in[0];
        return 0;
    }
""")

# Compile --------------------------------------------------------------------
try:
    lib_path = compile_cblock(source, name="gain2_c_example", compiler=cc)
    print(f"Compiled to: {lib_path}")
except CBlockCompileError as e:
    print(f"Compilation failed:\n{e.stderr_output}")
    sys.exit(1)

# Load and run ----------------------------------------------------------------
with CBlockLibrary(lib_path, n_inputs=1, n_outputs=1) as blk:
    for i in range(5):
        t = i * 1e-4
        out = blk.step(t, 1e-4 if i > 0 else 0.0, [float(i)])
        print(f"t={t:.4f}s  in={float(i):.1f}  out={out[0]:.1f}")

print("Done.")
