"""01_passthrough_gain.py — Simplest possible C-Block: multiply input by a gain.

Demonstrates:
- Compiling a C source file with compile_cblock()
- Loading the compiled library with CBlockLibrary
- Wiring the block into a SignalEvaluator pipeline
- Running a short simulation and printing the output
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure pulsim is importable (adjust if your build tree differs)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build" / "python"))

from pulsim.cblock import CBlockLibrary, compile_cblock, detect_compiler
from pulsim.signal_evaluator import SignalEvaluator

HERE = Path(__file__).parent
C_SOURCE = HERE / "gain_block.c"
INCLUDE_DIR = HERE.parents[2] / "core" / "include"


def build_circuit(lib_path: Path) -> dict:
    """Build a minimal signal-flow circuit: CONSTANT(2.0) → C_BLOCK(gain=3) → result."""
    return {
        "components": [
            {
                "id": "src", "name": "SRC", "type": "CONSTANT",
                "parameters": {"value": 2.0},
                "pins": [{"index": 0, "name": "OUT", "x": 0, "y": 0}],
            },
            {
                "id": "gain", "name": "GAIN", "type": "C_BLOCK",
                "parameters": {
                    "n_inputs": 1, "n_outputs": 1,
                    "lib_path": str(lib_path),
                },
                "pins": [
                    {"index": 0, "name": "IN0", "x": 0, "y": 0},
                    {"index": 1, "name": "OUT",  "x": 0, "y": 0},
                ],
            },
        ],
        "wires": [
            {
                "start_connection": {"component_id": "src",  "pin_index": 0},
                "end_connection":   {"component_id": "gain", "pin_index": 0},
            },
        ],
        "node_map": {}, "node_aliases": {},
    }


def main() -> None:
    cc = detect_compiler()
    if cc is None:
        print("No C compiler found. Set PULSIM_CC or install gcc/clang.")
        sys.exit(1)

    print(f"Compiling {C_SOURCE.name} with {cc} ...")
    lib_path = compile_cblock(
        C_SOURCE,
        name="gain_block",
        extra_cflags=[f"-I{INCLUDE_DIR}"],
        compiler=cc,
    )
    print(f"  → {lib_path}")

    ev = SignalEvaluator(build_circuit(lib_path))
    ev.build()

    print("\nSimulation output (CONSTANT=2.0, gain=3.0, expected=6.0):")
    for i in range(5):
        t = i * 1e-4
        state = ev.step(t)
        print(f"  t={t:.4f}  gain_out={state['gain']:.4f}")

    assert state["gain"] == 6.0, f"Unexpected output: {state['gain']}"
    print("\nAll outputs correct ✓")


if __name__ == "__main__":
    main()
