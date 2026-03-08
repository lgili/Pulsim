"""05_python_callable_no_compiler.py — Custom block using a Python callable.

No C compiler required. Demonstrates PythonCBlock, which wraps any Python
function in the same interface as CBlockLibrary so it can be wired into a
SignalEvaluator graph transparently.

The block computes:  out = sigmoid(gain * in)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build" / "python"))

from pulsim.cblock import PythonCBlock


def sigmoid_gain(ctx, t: float, dt: float, inputs: list[float]) -> list[float]:
    """1-input, 1-output sigmoid with gain=4."""
    gain = 4.0
    x = gain * inputs[0]
    y = 1.0 / (1.0 + math.exp(-x))
    return [y]


def main() -> None:
    # Create the block — no compiler, no .c file, no shared library
    blk = PythonCBlock(fn=sigmoid_gain, n_inputs=1, n_outputs=1, name="sigmoid")

    print("Sigmoid(gain=4) block:")
    print(f"{'Input':>8}  {'Output':>10}  {'Expected':>12}")

    test_cases = [
        (-2.0, 1.0 / (1.0 + math.exp(8.0))),    # far negative → ~0
        (-0.5, 1.0 / (1.0 + math.exp(2.0))),
        (0.0,  0.5),                                 # midpoint is exactly 0.5
        (0.5,  1.0 / (1.0 + math.exp(-2.0))),
        (2.0,  1.0 / (1.0 + math.exp(-8.0))),    # far positive → ~1
    ]

    for x, expected in test_cases:
        (y,) = blk.step(t=0.0, dt=0.0, inputs=[x])
        print(f"{x:>8.2f}  {y:>10.6f}  {expected:>12.6f}")
        assert abs(y - expected) < 1e-12, f"Mismatch at x={x}: {y} != {expected}"

    # PythonCBlock is usable in place of CBlockLibrary in a SignalEvaluator graph
    # (shown here with direct step calls, but the interface is identical)
    print("\nPythonCBlock interface identical to CBlockLibrary ✓")
    print("No compiler required ✓")


if __name__ == "__main__":
    main()
