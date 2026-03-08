"""02_first_order_filter.py — First-order IIR low-pass filter via C-Block.

Demonstrates:
- pulsim_cblock_init / pulsim_cblock_destroy lifecycle (state allocation)
- State persistence across steps
- Filter frequency response verification: step input should reach ~63% of
  final value within one time constant (tau = 1 / (2*pi*100) ≈ 1.59 ms)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build" / "python"))

from pulsim.cblock import CBlockLibrary, compile_cblock, detect_compiler

HERE = Path(__file__).parent
C_SOURCE = HERE / "iir_filter.c"
INCLUDE_DIR = HERE.parents[2] / "core" / "include"

CUTOFF_HZ = 100.0
TWO_PI = 6.28318530718


def main() -> None:
    cc = detect_compiler()
    if cc is None:
        print("No C compiler found. Set PULSIM_CC or install gcc/clang.")
        sys.exit(1)

    print(f"Compiling {C_SOURCE.name} (fc={CUTOFF_HZ} Hz) ...")
    lib_path = compile_cblock(
        C_SOURCE,
        name="iir_filter",
        extra_cflags=[f"-I{INCLUDE_DIR}", "-lm"],
        compiler=cc,
    )
    print(f"  → {lib_path}\n")

    tau = 1.0 / (TWO_PI * CUTOFF_HZ)          # ≈ 1.59 ms
    dt  = 1e-4                                  # 100 µs per step
    n_steps = int(3 * tau / dt) + 1            # simulate 3 time constants

    blk = CBlockLibrary(lib_path, n_inputs=1, n_outputs=1, name="iir")

    x_in = 1.0   # unit step input
    print(f"Step response (tau ≈ {tau*1e3:.2f} ms, dt={dt*1e6:.0f} µs):")
    step_63_idx = None
    for i in range(n_steps):
        t = i * dt
        out = blk.step(t, dt if i > 0 else 0.0, [x_in])
        y = out[0]
        if step_63_idx is None and y >= 0.63 * x_in:
            step_63_idx = i
        if i % max(1, n_steps // 10) == 0:
            print(f"  t={t*1e3:.2f} ms  y={y:.4f}")

    if step_63_idx is not None:
        print(f"\n63% threshold reached at step {step_63_idx} "
              f"(t ≈ {step_63_idx * dt * 1e3:.2f} ms, expected ≈ {tau*1e3:.2f} ms)")
    print(f"Final value: {y:.6f}  (expected → 1.0 asymptotically)")
    assert y > 0.95, f"Filter output too low after 3τ: {y:.4f}"
    print("Filter response verified ✓")


if __name__ == "__main__":
    main()
