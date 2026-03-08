"""Example 01 — Python gain block via PythonCBlock.

Demonstrates using PythonCBlock to double an input signal without any
C compiler, running inside a SignalEvaluator pipeline.

Run::

    PYTHONPATH=python python examples/cblock/01_python_gain.py
"""

from pulsim.cblock import PythonCBlock

# --- Define a gain-of-2 block in Python ------------------------------------

def gain2(ctx, t, dt, inputs):
    """Double the first input signal."""
    return [2.0 * inputs[0]]


blk = PythonCBlock(gain2, n_inputs=1, n_outputs=1, name="gain2")

# Simulate 5 steps
for i in range(5):
    t = i * 1e-4
    dt = 1e-4 if i > 0 else 0.0
    out = blk.step(t, dt, [float(i)])
    print(f"t={t:.4f}s  in={float(i):.1f}  out={out[0]:.1f}")

# Expected output:
# t=0.0000s  in=0.0  out=0.0
# t=0.0001s  in=1.0  out=2.0
# t=0.0002s  in=2.0  out=4.0
# t=0.0003s  in=3.0  out=6.0
# t=0.0004s  in=4.0  out=8.0
