#!/usr/bin/env python3
"""A self-heating resistor written OUTSIDE Pulsim, as an external device.

   V (10 V DC) ── X (self-heating resistor) ── gnd

The device law lives in ``examples/external_devices/self_heating_resistor.c``
(plain C against the stable header ``pulsim.get_include()``); this script
compiles it with the system C compiler, loads it with
``CircuitBuilder.add_external_device`` (the host checks the model's contract
and hand-written Jacobian there) and runs it on the fixed-step engine.

ILLUSTRATION: R_0 = 100 Ohm, alpha = 3.9e-3 /K, R_th = 50 K/W, C_th = 0.02 J/K
are round numbers (a 1 s thermal time constant), not a datasheet part.

Expected (closed form). With T_amb = T_0 the steady temperature rise solves
    theta (1 + alpha theta) = R_th V^2 / R_0,
    theta = (-1 + sqrt(1 + 4 alpha R_th V^2 / R_0)) / (2 alpha) = 42.842 K,
so the settled current is V / (R_0 (1 + alpha theta)) = 85.684 mA, down from
V / R_0 = 100 mA cold. After 20 s (over 20 time constants of 0.875 s) the run sits on
that root within the engine's accuracy.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import pulsim as p

SRC = Path(__file__).resolve().parent.parent / "external_devices" / "self_heating_resistor.c"
V, R0, ALPHA, R_TH = 10.0, 100.0, 3.9e-3, 50.0


def build_model(out_dir: Path) -> Path:
    """Compile the C model into a shared library next to nothing else."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        sys.exit("no C compiler found: this example builds its model with one")
    lib = out_dir / ("self_heating_resistor" + (".dll" if sys.platform == "win32" else ".so"))
    cmd = [cc, "-std=c99", "-O2", "-shared", "-fPIC", f"-I{p.get_include()}",
           str(SRC), "-o", str(lib)]
    subprocess.run(cmd, check=True)
    return lib


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lib = build_model(Path(tmp))
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", V)
        b.add_external_device("X", ["in", "gnd"], str(lib), "self_heating_resistor",
                              {"R_0": (R0, "Ohm"), "alpha": ALPHA, "R_th": R_TH})
        res = p.simulate(b, t_end=20.0, dt=2e-3)
        x = np.asarray(res.states)
        # theta is the device's internal unknown, carried by its branch "X"
        k_theta = b.pool.branch_var_id_for_inductor(b.branch_id_of("X"), b.graph)
        theta = x[:, k_theta]
        i = V / (R0 * (1.0 + ALPHA * theta))

    exact = (-1.0 + math.sqrt(1.0 + 4.0 * ALPHA * R_TH * V**2 / R0)) / (2.0 * ALPHA)
    print(f"cold current: {i[0] * 1e3:.4f} mA (V/R_0 = {V / R0 * 1e3:.4f} mA)")
    print(f"settled temperature rise: {theta[-1]:.6f} K (closed form {exact:.6f} K)")
    print(f"settled current: {i[-1] * 1e3:.5f} mA")
    print(f"rise error: {abs(theta[-1] - exact):.3e} K")


if __name__ == "__main__":
    main()
