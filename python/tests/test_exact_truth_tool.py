"""The exact-truth tool (benchmarks/tools/exact_truth.py) against closed forms.

The tool never imports pulsim; this test does not either. Each case solves a
small netlist exactly (mpmath, 30 digits) and compares it with a closed form
written from the circuit's own equations: RC / RL / RLC steps, a sine-driven
RC, a PWM buck's periodic orbit in DCM (the diode turn-off by root finding),
an ideal diode feeding R-L (the extinction angle), charge conservation when
an ideal switch parallels two capacitors, and an ideal switch closing a DC
source onto an inductor (the matrix-exponential path). The tool's values are
compared after conversion to float: both sides agree to ~1e-20, so 1e-12
relative is a pure float64 bound. Energy balance (sources = dissipation +
stored-energy change + impulsive loss) closes to ~1e-20 or better.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

mp = pytest.importorskip("mpmath")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "tools"))
import exact_truth as et  # noqa: E402

CASES = ["RC step", "RL step", "RLC step", "RC sine", "buck DCM orbit", "ideal half-wave RL",
         "capacitor jump", "ideal L ramp (expm path)"]


@pytest.mark.parametrize("case", CASES)
def test_closed_form(case):
    rows = et.validate(dps=30, names=[case])
    assert rows
    for name, what, got, ref, rd in rows:
        if what == "energy balance":
            assert abs(float(got)) < 1e-18, (name, float(got))
        else:
            assert float(got) == pytest.approx(float(ref), rel=1e-12, abs=1e-30), (name, what)


def test_unsupported_by_name():
    net = et._net("shock", [et._Vdc("V", "a", 1.0),
                            dict(type="diode", name="D1", nodes=["a", 0], **{"is": 1e-14})],
                  1e-3, 1e-6)
    with pytest.raises(et.Unsupported, match="Shockley"):
        et.Circuit(net)
