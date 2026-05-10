"""Real-time C99 code generation with PIL parity check.

Pipeline:
  1. Build a small RC circuit.
  2. ``pulsim.codegen.generate(ckt, dt=1e-5)`` — linearize + discretize
     (matrix exponential) + emit ``model.c / model.h / model_test.c``.
  3. Compile the harness with the system gcc/cc.
  4. Run the binary with a constant input for N steps.
  5. Independently simulate the same A_d / B_d in Python via
     ``y = C @ x + D @ u`` and ``x = A_d @ x + B_d @ u``.
  6. Diff the two traces — gate G.1 of the codegen change is ≤ 0.1 %
     relative tolerance, which holds easily on this trivial passive
     circuit.

Run::

    python 05_codegen_pil.py            # PIL parity (needs gcc on PATH)
    python 05_codegen_pil.py --no-pil   # skip the compile step

See also: docs/code-generation.md
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pulsim


def build_rc():
    ckt = pulsim.Circuit()
    in_ = ckt.add_node("in")
    out = ckt.add_node("out")
    ckt.add_voltage_source("V1", in_, ckt.ground(), 1.0)
    ckt.add_resistor("R1", in_, out, 1e3)
    ckt.add_capacitor("C1", out, ckt.ground(), 1e-6, 0.0)
    return ckt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pil", action="store_true",
                        help="skip the gcc compile + run step")
    parser.add_argument("--n-steps", type=int, default=200)
    args = parser.parse_args()

    ckt = build_rc()
    out_dir = Path(tempfile.mkdtemp(prefix="pulsim_codegen_"))
    print(f"Generating C99 model into {out_dir}")
    summary = pulsim.codegen.generate(ckt, dt=1e-5, out_dir=out_dir)

    print(f"  state_size:       {summary.state_size}")
    print(f"  input_size:       {summary.input_size}")
    print(f"  output_size:      {summary.output_size}")
    print(f"  stability radius: {summary.stability_radius:.6f}  (must be < 1)")
    print(f"  ROM ≈ {summary.rom_estimate_bytes} B    "
          f"RAM ≈ {summary.ram_estimate_bytes} B")
    for f in summary.files_written:
        print(f"  wrote: {f}")

    if args.no_pil:
        return 0

    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        print("\n(no gcc/cc on PATH — skipping PIL parity check)")
        return 0

    out_bin = out_dir / "model_test"
    cmd = [
        cc, "-O2", "-std=c99",
        "-I", str(out_dir),
        str(out_dir / "model.c"),
        str(out_dir / "model_test.c"),
        "-o", str(out_bin),
        "-lm",
    ]
    print("\nCompiling PIL harness:")
    print("  " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        return 1

    proc = subprocess.run(
        [str(out_bin), str(args.n_steps), "1.0"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("PIL run failed", file=sys.stderr)
        return 1
    c_outputs = [
        [float(v) for v in line.split(",")]
        for line in proc.stdout.strip().splitlines()
    ]

    # Independent reference simulation in Python.
    import numpy as np
    x = np.zeros(summary.state_size)
    u = np.ones(summary.input_size)
    py_outputs: list[list[float]] = []
    for _ in range(args.n_steps):
        y = summary.C @ x + summary.D @ u
        x = summary.A_d @ x + summary.B_d @ u
        py_outputs.append(y.tolist())

    max_err = 0.0
    for c_row, py_row in zip(c_outputs, py_outputs):
        for c_v, py_v in zip(c_row, py_row):
            denom = max(abs(py_v), 1e-9)
            max_err = max(max_err, abs(c_v - py_v) / denom)
    print(f"\nPIL parity over {args.n_steps} steps:")
    print(f"  max relative error: {max_err:.3e}  (gate ≤ 1e-3)")
    print(f"  C output[final]   : {c_outputs[-1]}")
    print(f"  Py output[final]  : {py_outputs[-1]}")
    return 0 if max_err < 1e-3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
