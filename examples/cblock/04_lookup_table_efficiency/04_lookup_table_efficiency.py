"""04_lookup_table_efficiency.py — MOSFET switching-loss via C lookup table.

Demonstrates:
- File I/O in pulsim_cblock_init (CSV → 2-D table loaded at startup)
- Complex state allocation in the C block
- Multi-output block (loss_W, efficiency)
- Integration with a SignalEvaluator pipeline

The efficiency_map.c block loads efficiency_map.csv (in the same directory)
via the PULSIM_CBLOCK_CSV_PATH environment variable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build" / "python"))

from pulsim.cblock import CBlockLibrary, compile_cblock, detect_compiler

HERE = Path(__file__).parent
C_SOURCE = HERE / "efficiency_map.c"
CSV_PATH = HERE / "efficiency_map.csv"
INCLUDE_DIR = HERE.parents[2] / "core" / "include"


def main() -> None:
    cc = detect_compiler()
    if cc is None:
        print("No C compiler found. Set PULSIM_CC or install gcc/clang.")
        sys.exit(1)

    print(f"Compiling {C_SOURCE.name} ...")
    lib_path = compile_cblock(
        C_SOURCE,
        name="efficiency_map",
        extra_cflags=[f"-I{INCLUDE_DIR}", "-lm"],
        compiler=cc,
    )
    print(f"  → {lib_path}\n")

    # Point the C init function to the CSV file
    os.environ["PULSIM_CBLOCK_CSV_PATH"] = str(CSV_PATH)

    blk = CBlockLibrary(lib_path, n_inputs=2, n_outputs=2, name="efficiency_map")

    test_points = [
        (10.0, 1.0),   # Vds=10V, Id=1A  → table: loss=0.05W
        (20.0, 5.0),   # Vds=20V, Id=5A  → table: loss=0.45W
        (30.0, 10.0),  # Vds=30V, Id=10A → table: loss=1.50W
        (15.0, 3.0),   # Interpolated
    ]

    print(f"{'Vds (V)':>10}  {'Id (A)':>8}  {'Loss (W)':>10}  {'Efficiency':>12}")
    for vds, id_a in test_points:
        out = blk.step(0.0, 0.0, [vds, id_a])
        loss_w, eff = out[0], out[1]
        print(f"{vds:>10.1f}  {id_a:>8.1f}  {loss_w:>10.4f}  {eff:>12.4f}")

    # Verify exact table values (exact lookup points)
    out = blk.step(0.0, 0.0, [10.0, 1.0])
    assert abs(out[0] - 0.05) < 1e-9, f"Expected 0.05, got {out[0]}"

    out = blk.step(0.0, 0.0, [30.0, 10.0])
    assert abs(out[0] - 1.50) < 1e-9, f"Expected 1.50, got {out[0]}"

    # Efficiency must be in (0, 1]
    for vds, id_a in test_points:
        out = blk.step(0.0, 0.0, [vds, id_a])
        assert 0.0 <= out[1] <= 1.0, f"Efficiency out of range: {out[1]}"

    print("\nAll lookup values and efficiency bounds verified ✓")


if __name__ == "__main__":
    main()
