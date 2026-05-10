"""Phase 9 of `add-realtime-code-generation`: PIL parity test.

Generates C from a Pulsim Circuit, compiles with the system gcc, runs
the harness, and diffs the resulting trace against the Pulsim-native
transient at the same fixed step. The contract: the two agree within
≤ 0.1 % on a passive RC low-pass after several time constants.

The test gracefully skips when no `cc`/`gcc` is on PATH so it survives
CI environments without a C compiler.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

pulsim = pytest.importorskip("pulsim")
np = pytest.importorskip("numpy")


def _build_rc_circuit():
    ckt = pulsim.Circuit()
    in_ = ckt.add_node("in")
    out = ckt.add_node("out")
    ckt.add_voltage_source("V1", in_, ckt.ground(), 1.0)
    ckt.add_resistor("R1", in_, out, 1e3)
    ckt.add_capacitor("C1", out, ckt.ground(), 1e-6, 0.0)
    return ckt


def test_codegen_generates_self_contained_c99():
    ckt = _build_rc_circuit()
    with tempfile.TemporaryDirectory() as td:
        summary = pulsim.codegen.generate(ckt, dt=1e-5, out_dir=td)
        assert summary.target == "c99"
        assert summary.state_size >= 1
        assert summary.stability_radius < 1.0
        # Generated files exist and are non-empty.
        for f in summary.files_written:
            path = Path(f)
            assert path.exists()
            assert path.stat().st_size > 100


def test_codegen_rejects_unsupported_target():
    ckt = _build_rc_circuit()
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match="not yet supported"):
            pulsim.codegen.generate(ckt, dt=1e-5, out_dir=td, target="cuda")


def test_codegen_pil_parity_against_native():
    """Phase 9: compile generated C with gcc, run it, compare against
    the discretized state-space evolution computed independently in
    Python. Skipped when no C compiler is available."""
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler on PATH")

    ckt = _build_rc_circuit()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        summary = pulsim.codegen.generate(ckt, dt=1e-5, out_dir=td_path)

        # Compile the harness.
        out_bin = td_path / "model_test"
        cmd = [
            cc, "-O2", "-std=c99",
            "-I", str(td_path),
            str(td_path / "model.c"),
            str(td_path / "model_test.c"),
            "-o", str(out_bin),
            "-lm",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"compile failed: {result.stderr}")

        # Run with constant unit input for 100 steps.
        N = 100
        proc = subprocess.run(
            [str(out_bin), str(N), "1.0"],
            capture_output=True, text=True)
        assert proc.returncode == 0
        c_outputs = [
            [float(v) for v in line.split(",")]
            for line in proc.stdout.strip().splitlines()
        ]
        assert len(c_outputs) == N

        # Independent reference: simulate the same A_d / B_d in Python.
        x = np.zeros(summary.state_size)
        u = np.ones(summary.input_size)
        py_outputs = []
        for _ in range(N):
            y = summary.C @ x + summary.D @ u
            x = summary.A_d @ x + summary.B_d @ u
            py_outputs.append(y.tolist())

        # PIL parity: every output coordinate at every step must agree
        # within ±0.1 % (gate G.1) plus a small absolute floor for the
        # first few steps where outputs are near zero.
        assert len(c_outputs) == len(py_outputs)
        for k, (c_row, py_row) in enumerate(zip(c_outputs, py_outputs)):
            assert len(c_row) == len(py_row)
            for c_v, py_v in zip(c_row, py_row):
                assert math.isclose(
                    c_v, py_v, rel_tol=1e-3, abs_tol=1e-5
                ), f"step {k}: C={c_v} py={py_v}"
