"""The exact-truth tool (benchmarks/tools/exact_truth.py) against closed forms.

The tool never imports pulsim; this test does not either. Each case solves a
small netlist exactly (mpmath, 30 digits) and compares it with a closed form
written from the circuit's own equations: RC / RL / RLC steps, a sine-driven
RC, a PWM buck's periodic orbit in DCM (the diode turn-off by root finding),
an ideal diode feeding R-L (the extinction angle), charge conservation when
an ideal switch parallels two capacitors, the impulsive loss when one closes
a source onto a capacitor, and an ideal switch closing a DC
source onto an inductor and an RC driven by a finite pulse edge (the
matrix-exponential path). The tool's values are
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
         "capacitor jump", "source jump", "ideal L ramp (expm path)", "RC ramp edge"]


@pytest.mark.parametrize("case", CASES)
def test_closed_form(case):
    rows = et.validate(dps=30, names=[case])
    assert rows
    for name, what, got, ref, rd in rows:
        if what == "energy balance":
            assert abs(float(got)) < 1e-18, (name, float(got))
        else:
            assert float(got) == pytest.approx(float(ref), rel=1e-12, abs=1e-30), (name, what)


def test_truth_csv_round_trip(tmp_path):
    """The stored truth reads back with numpy and matches the RC closed form."""
    import gzip
    import math

    np = pytest.importorskip("numpy")
    yaml = pytest.importorskip("yaml")
    net = et._net("rc_csv", [et._Vdc("V", "in", 10.0), et._R("R", "in", "out", 1e3),
                             dict(type="capacitor", name="C", nodes=["out", 0], value=1e-6)],
                  2e-3, 1e-4, ["V(out)"])
    src = tmp_path / "rc_csv.yaml"
    src.write_text(yaml.safe_dump(net), encoding="utf-8")
    rec = et.truth(str(src), dps=(20, 30), out_dir=tmp_path)
    assert rec["supported"] and rec["n_commutations"] == 0
    with gzip.open(tmp_path / "rc_csv.csv.gz", "rt", encoding="utf-8") as f:
        assert f.readline().strip() == "time,V(out)"
        d = np.loadtxt(f, delimiter=",", ndmin=2)
    assert np.all(np.diff(d[:, 0]) > 0)
    tau = float(et._g(1e3)) * 1e-6
    ref = [10.0 * -math.expm1(-t / tau) for t in d[:, 0]]
    assert d[:, 1] == pytest.approx(ref, rel=1e-12, abs=1e-12)


def test_unsupported_by_name():
    net = et._net("shock", [et._Vdc("V", "a", 1.0),
                            dict(type="diode", name="D1", nodes=["a", 0], **{"is": 1e-14})],
                  1e-3, 1e-6)
    with pytest.raises(et.Unsupported, match="Shockley"):
        et.Circuit(net)
