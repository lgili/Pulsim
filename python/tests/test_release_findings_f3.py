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


class _NanGate:
    """A gate whose oracle answers NaN (a 0/0 in a user's period arithmetic)."""

    def __init__(self, n, i):
        self.n, self.i = n, i

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        m.set(self.i, (t % 1e-5) < 5e-6)
        return m

    def next_edge_after(self, t):
        return float("nan")


@pytest.mark.parametrize("integrator", [None, "rk45", "bdf2"])
def test_require_after_refuses_nan_by_name(integrator):
    """events::require_after let NaN through: a NaN pending event compares false everywhere,
    so the queue never asked again and the gate's edges were silently never landed."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_switch("S", "a", "o", 1e3, 1e-9)
    b.add_resistor("R", "o", "gnd", 10.0)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    gate = _NanGate(b.graph.num_switches, b.switch_index_of("S"))
    with pytest.raises(p.UnsupportedError, match=r"returned nan .*NaN is not an instant"):
        p.simulate(b, t_end=1e-4, engine="dsed", switch_fn=gate,
                   **({} if integrator is None else {"integrator": integrator}))


def _lone_inductor(i0):
    """1 mH whose only onward path is a diode reverse-biased by 10 V (D: a -> p); 10 Ohm
    across the source sets the circuit's current scale (1 A)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "p", "gnd", 10.0)
    b.add_resistor("R", "p", "gnd", 10.0)
    b.add_diode("D", "a", "p", 1000.0, 1e-9)
    b.add_inductor("L", "a", "gnd", 1e-3, i0)
    return b


def _exact(b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=1e-5, engine="exact", **kw)


def test_start_projection_of_a_given_state_is_reported():
    """F-S-4's leftover: 1 A given in L, no mode keeps it (the diode would conduct in
    reverse), the start projection takes it to 0 A -- it used to do so silently. Now it is
    a reinit record at t0 with the jump's energy, 1/2 L i^2 = 0.5 mJ (the energy-weighted
    distance of the jump, as the dependent-state records), and on_impulse='error' refuses it."""
    r = _exact(_lone_inductor(1.0))
    assert abs(float(np.asarray(r.i("L"))[0])) < 1e-7   # the g_off leakage, 1e-8 A
    recs = [x for x in r.reinit_records if x.kind == "start_projection"]
    assert len(recs) == 1 and recs[0].t == 0.0
    assert recs[0].dW == pytest.approx(0.5e-3, rel=1e-6)
    assert recs[0].projection_gap == pytest.approx(1.0, rel=1e-6)
    with pytest.raises(p.UnsupportedError, match="not a state of the start mode"):
        _exact(_lone_inductor(1.0), on_impulse="error")


def test_a_start_that_needs_no_projection_writes_no_record():
    for b in (_lone_inductor(0.0), _lone_inductor(None)):
        r = _exact(b, on_impulse="error")
        assert not list(r.reinit_records)


def test_a_precharged_capacitor_inrush_is_reported_not_refused():
    """10 V through an ON diode (1 mOhm) into 1 nF given 8 V: the 1 ps inrush is the start's
    physics; the run is not refused, the jump is reported: 1/2 C (2 V)^2 = 2 nJ (the
    ideal-limit loss, source work included: C dv V - (1/2 C 10^2 - 1/2 C 8^2))."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "p", "gnd", 10.0)
    b.add_diode("D", "p", "c", 1000.0, 1e-9)
    b.add_capacitor("C", "c", "gnd", 1e-9, 8.0)
    b.add_resistor("R", "c", "gnd", 1e3)
    r = _exact(b)
    assert float(np.asarray(r.v("c"))[0]) == pytest.approx(10.0, abs=1e-3)
    recs = [x for x in r.reinit_records if x.kind == "start_projection"]
    assert len(recs) == 1
    assert recs[0].dW == pytest.approx(2e-9, rel=1e-3)


@pytest.mark.xfail(strict=True, reason=(
    "open (group F, item 6): the parasitic cluster's cut is 1/(1e-6 * t_res) with "
    "t_res = next time event - last time event; with no gate edge the next time event is "
    "the run's end, so a 0.1 ns RC is projected at the start of a 1 ms run and resolved in a "
    "10 us one. It should depend on the mode's own spectrum (a gap in its decay rates), not "
    "on t_end; that moves every oracle-less run's settle decisions (resolved_scale_ is read by "
    "settled_ on the default path), so it is a separate change"))
def test_parasitic_cut_does_not_depend_on_t_end():
    def b():
        c = p.CircuitBuilder()
        c.add_voltage_source("V", "a", "gnd", 10.0)
        c.add_diode("D", "a", "m", 1000.0, 1e-9)
        c.add_resistor("R", "m", "c", 0.1)
        c.add_capacitor("C", "c", "gnd", 1e-9)
        c.add_resistor("RL", "c", "gnd", 1e3)
        return c
    v0 = []
    for te in (1e-5, 1e-3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = p.simulate(b(), te, engine="exact", output_rtol=1e-5)
        v0.append(float(np.asarray(r.v("c"))[0]))   # 0.0 V vs 9.999 V today
    assert v0[0] == pytest.approx(v0[1], abs=1e-6)
