"""Release findings, group F, third session: the exact engine's leftovers.

F-X-6  a trial mode's partial projection survived the settle under the output rule
       (forward converter, I(L_o) +0.585 A in 0.2 ps).
F-X-5  the exact step's event scan missed a double crossing (active-clamp forward).
F-X-4  the rehearsal harness reads a short trace; the output rule's bar is its promise.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

ROOT = Path(__file__).resolve().parents[2]


def _case(rel):
    sys.path[:0] = [str(ROOT / "benchmarks" / "tools")]
    try:
        import v1_reader
    finally:
        del sys.path[0]
    return v1_reader.load(ROOT / "benchmarks" / rel)


def _no_dt(case, **kw):
    """The netlist's run without its dt (what the release route runs)."""
    b, sw, _ = case.build()
    args = dict(engine="exact", t_start=case.t_start, **kw)
    if sw is not None:
        args["switch_fn"] = sw
    if not case.uic:
        args["initial_state"] = case._dc_start(b, sw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, case.t_end, **args)


FWD = "circuits/switching/isolated/forward_converter.yaml"


@pytest.mark.parametrize("rt", [None, 1e-4, 1e-5, 1e-6])
def test_fx6_forward_converter_keeps_the_inductor_current_at_the_switch_off(rt):
    """mpmath truth (benchmarks/truth/forward_converter.json, exact_truth.py): I(L_o) is
    5.620787 A across the 48 us switch-off (D_fw takes it 5e-14 s later) and the full-run
    mean of V(out) is 6.6073535537897 V. Before the fix the output rule's run read 6.2063 A
    right after the event and its mean was 0.131 V off (1e-4 / 1e-5 / 1e-6 alike)."""
    truth = json.loads((ROOT / "benchmarks/truth/forward_converter.json").read_text(encoding="utf-8"))
    r = _no_dt(_case(FWD), **({} if rt is None else {"output_rtol": rt}))
    t = np.asarray(r.times)
    k = np.searchsorted(t, 4.80000553e-5)
    assert abs(float(np.asarray(r.i("L_o"))[k]) - 5.620787) < 1e-5
    mean = float(np.trapezoid(np.asarray(r.v("out")), t) / (t[-1] - t[0]))
    # the samples' linear interpolation: 1.9e-3 V on the rule-free grid, ~0.17·rtol·range
    # under the rule (1.1e-5 V at 1e-5); the defect moved it by 0.131 V
    tol = 1e-2 if rt is None else 100 * rt
    assert abs(mean - float(truth["windows"]["full"]["V(out)"]["mean"])) < tol


ACF = "circuits/switching/isolated/active_clamp_forward.yaml"
# mpmath (exact_truth.py at fa4b5c71, dps 30): D_fw OFF then ON again 77 ns later
ACF_TRUTH = (1.66354841887779847e-5, 1.67126164738377463e-5)


@pytest.mark.parametrize("rt", [None, 1e-4, 1e-5, 1e-6])
def test_fx5_a_double_crossing_inside_one_scan_interval_is_found(rt):
    """Both sign changes of D_fw's predicate fell inside one exact step's scan interval
    (no under-damped mode: the D3 scan is the step itself), so the end signs agreed and
    no event was seen: rule off, D_fw stayed ON at -1.4e-2 A (silently, 2.8 us from the
    nearest event); under the rule the stored samples broke the diode law
    (DiodeModeInconsistentError). The certificate halves an interval whose ends it
    cannot prove root-free."""
    r = _no_dt(_case(ACF), **({} if rt is None else {"output_rtol": rt}))
    et = np.asarray(r.event_times)
    for t in ACF_TRUTH:
        assert np.min(np.abs(et - t)) < 1e-12


def _tools():
    sys.path[:0] = [str(ROOT / "benchmarks" / "tools")]
    try:
        import metrics
    finally:
        del sys.path[0]
    return metrics


def test_fx4_harness_compares_a_two_sample_trace():
    """A DC circuit under the output rule stores its two ends; dropping the first sample
    left one, a NaN err_rms and 'undersampled'. The short trace is compared whole, from
    the reference's own first kept sample (its sample 0 is an initial state, 0 V here)."""
    m = _tools()
    ref = (np.array([0.0, 0.0, 1e-6, 2e-6]), np.array([0.0, 5.0, 5.0, 5.0]))
    kw = dict(exclude_t0=True, abs_scale=5e-3)
    assert m.err_rms([0.0, 2e-6], [5.0, 5.0], ref, **kw) == 0.0
    assert m.err_max([0.0, 2e-6], [5.0, 5.0], ref, **kw) == 0.0
    with pytest.raises(ValueError, match="2 samples"):
        m.err_rms([1e-6], [5.0], ref, **kw)


def test_fx4_rule_ratio_reads_the_chord_against_the_promise():
    """exp(-t) sampled every 0.1: the worst chord midpoint error is about h^2/8 = 1.25e-3."""
    m = _tools()
    t = np.linspace(0.0, 1.0, 11)
    r = m.rule_ratio(t, np.exp(-t), lambda s: np.exp(-s), rtol=1e-3, atol=0.0)
    span = 1.0 - np.exp(-1.0)
    assert 0.9 < r * 1e-3 * span / 1.25e-3 < 1.0   # h^2/8 * f''(xi), f'' <= 1 on the first interval
    assert m.rule_ratio(t, np.exp(-t), lambda s: np.exp(-s), rtol=1e-2, atol=0.0) < 1.0
