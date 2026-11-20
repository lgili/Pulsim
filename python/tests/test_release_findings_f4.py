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
