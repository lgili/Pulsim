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
np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "tools"))
import exact_truth as et  # noqa: E402
import metrics  # noqa: E402 - pure numpy


@pytest.fixture(scope="session", autouse=True)
def _pulsim_is_this_checkout():
    """Overrides the conftest's check: this tool never imports pulsim, so
    these tests run without a build (or with a stale installed pulsim)."""

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


# ------------------------------------------------- the ``exact`` measure --
# A switch S (gate: 10 us period, 4 us on, ideal edges on the 1 us grid) ties
# n to a 10 V source; V(n) is algebraic and jumps at every edge.
DT, T_END = 1e-6, 40e-6


@pytest.fixture(scope="module")
def jump_sol():
    net = et._net("jump", [
        et._Vdc("V", "a", 10.0),
        dict(type="voltage_source", name="Vg", nodes=["g", 0],
             waveform=dict(type="pulse", v_initial=0.0, v_pulse=10.0, t_delay=0.0, t_width=4e-6,
                           period=10e-6)),
        dict(type="vcswitch", name="S", nodes=["g", "a", "n"], v_threshold=5.0, g_on=1e3, g_off=1e-9),
        et._R("R", "n", "out", 10.0),
        dict(type="capacitor", name="C", nodes=["out", 0], value=1e-6)],
        T_END, DT, ["V(n)", "V(out)"])
    return et.Solver(et.Circuit(net), 30).run()


def _truth_trace(sol, ts):
    """The truth itself (mpmath ``sample``, not the float evaluator) at ts."""
    v = sol.sample(["V(n)"], [mp.mpf(float(x)) for x in ts])["V(n)"]
    return np.array([float(x) for x in v])


def _score(sol, t, y, delta=1e-3 * DT):
    cand, gap = sol.at_float(["V(n)"], t, delta, gaps=True)
    return metrics.err_exact(t, y, cand["V(n)"], span=10.0, gap=gap)



def test_exact_measure_has_no_floor(jump_sol):
    """The truth graded on any grid scores its evaluation error (~1e-16);
    the interp gate's floor on the same uniform grid is orders larger."""
    rng = np.random.default_rng(7)
    grids = {"uniform dt (edges on samples)": np.linspace(0.0, T_END, 41),
             "0.37 dt": np.arange(0.0, T_END, 0.37 * DT),
             "random": np.sort(np.r_[0.0, rng.uniform(0.0, T_END, 300)])}
    t2, y2 = et._floats(jump_sol, ["V(n)"], et.out_grid(jump_sol, DT / 4))
    for name, t in grids.items():
        assert _score(jump_sol, t, _truth_trace(jump_sol, t)) < 1e-12, name
    # two-sided (dsed style: left limit at the event, right one ulp after)
    assert _score(jump_sol, t2, y2["V(n)"]) < 1e-12
    t = grids["uniform dt (edges on samples)"]
    floor = metrics.err_rms(t, _truth_trace(jump_sol, t), (t2, y2["V(n)"]), abs_scale=1e-2, exclude_t0=True)
    assert floor > 1e-3


def test_exact_measure_one_sided_limits(jump_sol):
    """A sample on an event is graded against its own side; a value that is
    neither limit, or a switch later than the snap window, is an error."""
    te = 4e-6                                       # the first turn-off edge
    c = jump_sol.at_float(["V(n)"], [te], DT)["V(n)"][0]
    left, right = c[0], c[1]
    assert abs(left - right) > 1.0                  # V(n) jumps here
    t = np.linspace(0.0, T_END, 41)
    y = _truth_trace(jump_sol, t)
    k = int(np.argmin(abs(t - te)))
    for v in (left, right):                          # either one-sided value is right
        y[k] = v
        assert _score(jump_sol, t, y) < 1e-12
    y[k] = 0.5 * (left + right)                      # neither: one sample, weight dt
    want = abs(left - right) / 2 * np.sqrt(DT / (T_END - DT)) / 10.0
    assert _score(jump_sol, t, y) == pytest.approx(want, rel=1e-9)
    # dsed style: left at t_e, right one ulp later
    t2 = np.sort(np.r_[t, np.nextafter(te, 1.0)])
    y2 = _truth_trace(jump_sol, t2)
    y2[k], y2[k + 1] = left, right
    assert _score(jump_sol, t2, y2) < 1e-12
    y2[k], y2[k + 1] = right, left                   # swapped sides: graded on the side's value
    assert _score(jump_sol, t2, y2) < 1e-12          # both lie within the snap window
    # a switch 0.5 dt late: the pre-edge branch continued past the window is wrong
    t3 = np.sort(np.r_[t, te + 0.5 * DT])
    late = jump_sol.at_float(["V(n)"], [te + 0.5 * DT], DT)["V(n)"][0]
    y3 = _truth_trace(jump_sol, t3)
    y3[k + 1] = late[0]
    assert _score(jump_sol, t3, y3) > 1e-2
    # inside the snap window the switch is charged its true L2 cost,
    # jump^2 x lateness, not hidden and not the sample's whole weight
    for late_dt in (1e-4, 9e-4):
        t4 = np.sort(np.r_[t, te + late_dt * DT])
        y4 = _truth_trace(jump_sol, t4)
        b = jump_sol.at_float(["V(n)"], [te + late_dt * DT], DT)["V(n)"][0]
        y4[k + 1] = b[0]
        want = abs(b[0] - b[1]) * np.sqrt(late_dt * DT / (T_END - DT)) / 10.0
        assert _score(jump_sol, t4, y4) == pytest.approx(want, rel=1e-6), late_dt
    with pytest.raises(ValueError):                  # time order is checked
        _score(jump_sol, t[::-1].copy(), y[::-1].copy())
    y5 = _truth_trace(jump_sol, t)
    y5[7] = np.nan                                   # a NaN sample is an infinite error
    assert _score(jump_sol, t, y5) == np.inf


def test_err_exact_time_weights():
    """Trapezoid weights on the trace's own grid; the first sample excluded."""
    t = np.array([0.0, 1.0, 1.5, 4.0, 4.1, 10.0])
    cand = np.zeros((t.size, 1))
    assert metrics.err_exact(t, np.full(t.size, 3.0), cand, span=2.0) == pytest.approx(1.5)
    y = np.zeros(t.size)
    y[3] = 1.0                                       # weight (4.1 - 1.5) / 2 of a 9 s window
    assert metrics.err_exact(t, y, cand, span=1.0) == pytest.approx(np.sqrt(1.3 / 9.0))
    y = np.zeros(t.size)
    y[0] = 1e9                                       # the initial sample is not graded
    assert metrics.err_exact(t, y, cand, span=1.0) == 0.0
    nan2 = np.c_[np.zeros(t.size), np.full(t.size, np.nan)]
    assert metrics.err_exact(t, np.full(t.size, 1.0), nan2, span=1.0) == pytest.approx(1.0)


def test_err_exact_timing_charge_survives_a_bad_first_sample():
    """A non-finite initial sample (not graded) must not switch the charge off."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    cand = np.array([[0.0, np.nan], [0.0, 1.0], [0.0, np.nan], [0.0, np.nan]])
    gap = np.array([[0.0, np.nan], [0.0, 0.5], [0.0, np.nan], [0.0, np.nan]])
    y = np.array([np.nan, 1.0, 0.0, 0.0])            # sample 1 matches a branch 0.5 away
    assert metrics.err_exact(t, y, cand, span=1.0, gap=gap) == pytest.approx(np.sqrt(0.5 / 2.0))
