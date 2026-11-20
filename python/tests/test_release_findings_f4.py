"""Release findings, group F, fourth session: the output rule's promise.

The rule promises that linear interpolation between stored samples stays within
``output_rtol·range`` (floored at ``atol``) of the exact trajectory. The stored samples
are the RECOVERED vector (node voltages, branch currents), not only the state: a node
behind a high impedance reads a state's chord error times that impedance.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p
from pulsim._result_views import with_right_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "tools"))
import metrics  # noqa: E402
import v1_reader  # noqa: E402

LL06 = "circuits/switching/buck/ll06_buck_vcswitch.yaml"


def _no_dt(case, **kw):
    b, sw, _ = case.build()
    args = dict(engine="exact", t_start=case.t_start, **kw)
    if sw is not None:
        args["switch_fn"] = sw
    if not case.uic:
        args["initial_state"] = case._dc_start(b, sw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, case.t_end, **args)


def _ratio(r, flow, col, rtol):
    """The rule's worst chord against the exact flow (the same run at a smaller
    output_rtol, same events), events and the step leaving them excluded."""
    rr, ff = with_right_limits(r), with_right_limits(flow)
    excl = [(e, np.nextafter(e, np.inf)) for e in np.asarray(r.event_times, dtype=float)]
    return metrics.rule_ratio_flow(np.asarray(rr.times), np.asarray(v1_reader.observable(rr, col)),
                                   np.asarray(ff.times), np.asarray(v1_reader.observable(ff, col)),
                                   rtol=rtol, atol=1e-9, exclude=excl)


def test_f4_a_high_impedance_node_is_held_to_its_own_tolerance():
    """ll06: with switch and diode both off V(sw) = (48·g_s − I(L1))/(g_s + g_d), 5 kΩ: the
    rule held I(L1) to 0.82 of its tolerance and V(sw) read 596 × its own (32.76 V against
    33.05 V across a 5.5 ns chord). Now every stored column is within the promise."""
    case = v1_reader.load(ROOT / "benchmarks" / LL06)
    r = _no_dt(case, output_rtol=1e-5)
    flow = _no_dt(case, output_rtol=5e-7)
    assert np.array_equal(np.asarray(r.event_times), np.asarray(flow.event_times))
    assert flow.diagnostics["output_grid"]["n_output_capped"] == 0
    for col in ("V(sw)", "V(n_l)", "V(out)", "I(L1)"):
        assert _ratio(r, flow, col, 1e-5) <= 1.0, col


def test_f4_rule_ratio_flow_reads_each_chord_as_one_smooth_piece():
    """metrics.rule_ratio_flow: the flow at a chord's midpoint comes from the dense samples
    inside that chord and its two ends only (a source corner or a jump next to it never
    enters the stencil: the global reading put 5e6 on a 1 ns pulse edge)."""
    t = np.linspace(0.0, 1.0, 11)
    y = np.exp(-3 * t)
    td = np.linspace(0.0, 1.0, 1001)
    r = metrics.rule_ratio_flow(t, y, td, np.exp(-3 * td), rtol=1.0, atol=0.0)
    worst = np.max(np.abs(0.5 * (y[:-1] + y[1:]) - np.exp(-3 * 0.5 * (t[:-1] + t[1:]))))
    assert abs(r * (1.0 - np.exp(-3.0)) - worst) < 1e-9


def test_f4_rule_ratio_flow_never_extrapolates_one_short_dense_piece():
    """One dense sample 1.8 ns into a 985 ns chord (ll09 after its 24 us edge): the quadratic
    through it and the chord's ends extrapolated the fast piece's curvature over the long one
    (5.8x on a chord within 0.02 of its tolerance); the dense run's own chord is read instead."""
    t = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])            # a straight flow ...
    td = np.array([0.0, 1.8e-3, 1.0])   # ... whose dense sample sits a little above it
    yd = np.array([0.0, 1.8e-3 + 1e-6, 1.0])
    r = metrics.rule_ratio_flow(t, y, td, yd, rtol=1e-3, atol=0.0)
    assert r < 1e-3


CLAMP = "circuits/diodes/diode_clamp_overvoltage.yaml"


def test_f8_a_source_corner_is_always_a_stored_sample():
    """D-1 (session 7): the no-edge gate poll landed at 2.5199999999999966e-3, 3.4e-18 s short
    of the pulse's fall corner at 2.52e-3: at_corner was false, the landing counted as
    transparent and the rule deferred it, so the stored line 1.02 ms -> 2.533 ms ran straight
    across the corner (9.64x its tolerance at 2.5162 ms). A corner is a kink of the output:
    it is stored whatever lands on its instant, and the referee reads every line."""
    import output_rule_referee as F
    case = v1_reader.load(ROOT / "benchmarks" / CLAMP)
    r = _no_dt(case, output_rtol=1e-5)
    t = np.asarray(r.times)
    for c in (1e-3, 1.02e-3, 2.52e-3, 2.54e-3):
        assert np.min(np.abs(t - c)) <= 1e-12, c
    rec = F.check(CLAMP)
    assert rec["comparable"], rec
    assert rec["ratio"] <= 1.0, rec


def test_f8_the_referee_fails_a_line_it_cannot_read():
    """The referee reads every stored line at reference instants inside it: a line with no
    instant inside, or a NaN, is a harness failure (session 6: the dense flow shared the
    clamp's 11 samples and read 0 inside the violating line)."""
    import output_rule_referee as F
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 1.0])
    with pytest.raises(F.RefereeError, match="without a reference instant"):
        F.lines_ratio(t, y, np.array([0.5]), np.array([0.5]), 1e-3)
    with pytest.raises(F.RefereeError, match="NaN"):
        F.lines_ratio(t, np.array([0.0, np.nan, 1.0]), np.array([0.5, 1.5]), np.array([0.5, 1.0]), 1e-3)
    r, at = F.lines_ratio(t, y, np.array([0.5, 1.5]), np.array([0.5, 1.0 + 2e-3]), 1e-3)
    assert r == pytest.approx(2.0) and at == 1.5
