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
sys.path.insert(0, str(Path(__file__).resolve().parent))
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


def test_f8_ten_bridges_rows_at_rounding_do_not_cut_the_samples():
    """D-2 (session 7): ten paralleled bridges capped EVERY interval (409 600 samples in
    0.1 ms): a 1.5e11 /s mode carrying the state's own rounding (1e-13 in modal units) on a
    node behind 1/g_off read 7.6e-5 V against a 4e-9 tolerance, above the old floor
    Σ|P||z| (3.4e-13, P after its cancellation). The floor is now ε·Σ|terms| of the recovery
    (|O||V||z|); the rows it certifies are counted, never silent."""
    from test_exact_route import _paralleled_bridges
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(_paralleled_bridges(10), t_end=1e-4, engine="exact", output_rtol=1e-5)
    og = r.diagnostics["output_grid"]
    assert og["n_output_capped"] == 0 and og["output_budget_t"] is None
    assert len(r.times) < 3000          # 926 at session 7
    assert og["n_output_floored"] > 0
    assert og["output_floor_max"] > 0.0
    fc = np.asarray(og["output_floor_by_column"])
    assert fc.shape == (np.asarray(r.states).shape[1],)
    assert fc.max() == og["output_floor_max"]


def test_f8_a_balanced_neutral_does_not_spend_the_budget():
    """D-3 (session 7): three_phase_balanced_rl_load spent the whole 2e6-sample budget by
    4.53 ms (8 038 samples before 075feac9's per-row floor; 5 153 rule-free)."""
    case = v1_reader.load(ROOT / "benchmarks" / "circuits/three_phase/three_phase_balanced_rl_load.yaml")
    r = _no_dt(case, output_rtol=1e-5)
    og = r.diagnostics["output_grid"]
    assert og["output_budget_t"] is None and og["n_output_capped"] == 0
    assert len(r.times) < 12_000        # 5 290 at session 7


def test_fx7_a_jump_of_an_output_is_stored_on_both_sides():
    """F-X-7 (release): with the rule on, the bus v(p) - v(n) of the three-phase overlap
    bridge (branch form) read by plain interpolation of ``res.times`` at 4001 instants was
    34.18 V off of a 173 V range at t = 0.12 ms (39.56 V on the release tip), 22.36 V with
    output_rtol 1e-7, 0.0107 V with the rule off (truth: pwl + restart + split at dt 1e-7,
    itself good to ~6e-3 V). Cause: the sample stored at an event is its LEFT limit; the
    bus jumps 132.6 -> 173.1 V at the commutation and the rule's next sample was 4 us away
    (rule off, a fast sample 1e-10 s after the event hid the ramp). Read two-sided
    (``with_right_limits``) the same run was 0.0107 V: the chord bound held, the stored
    trace did not show the jump. Under the rule the solution 2 ps after such an event is
    now a stored sample."""
    from tests.oracles import lcp_pulsim_bridge as B

    def run(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return p.simulate(B.FAMILIES["bridge3_overlap"].builder(0.0, form="branch"), t_end=0.02, **kw)

    def bus(x):
        return np.asarray(x.v("p")) - np.asarray(x.v("n"))
    q = run(engine="pwl", dt=1e-7, discontinuity_restart=True, enable_substep_state_correction=True)
    off = run(engine="exact")
    tg = np.linspace(0.0, 0.02, 4001)
    bq = np.interp(tg, q.times, bus(q))
    assert np.abs(np.interp(tg, off.times, bus(off)) - bq).max() <= 3e-2          # 0.0107
    t_off, ev = np.asarray(off.times), np.asarray(off.event_times)
    # not next to a jump (the rule moves an event by picoseconds: 2.2e-12 s at 15 ms)
    away = np.abs(t_off[:, None] - ev[None, :]).min(axis=1) > 1e-9
    for rt in (1e-5, 1e-7):
        r = run(engine="exact", output_rtol=rt)
        np.testing.assert_allclose(r.event_times, ev, rtol=0, atol=1e-10)
        # every commutation: a sample at the event and one 2 ps later
        t = np.asarray(r.times)
        k = np.searchsorted(t, np.asarray(r.event_times))
        assert np.all(t[k] == r.event_times) and np.all(t[k + 1] - t[k] < 1e-11)
        assert np.abs(np.interp(tg, t, bus(r)) - bq).max() <= 3e-2             # was 34.18 / 22.36
        # the promise, per stored column, against the engine's own exact samples (the rule-off
        # run's instants): worst 0.9971 of a tolerance at 1e-5, 0.44 at 1e-7. (The bus is a
        # difference of two columns: its bound is the sum of theirs.)
        og = r.diagnostics["output_grid"]
        assert not og["n_output_capped"] and not og["n_output_unbounded"] and og["output_budget_t"] is None
        X, Xo = np.asarray(r.states), np.asarray(off.states)
        for j in range(X.shape[1]):
            rng = max(float(np.ptp(Xo[:, j])), 1e-3 * float(np.abs(Xo[:, j]).max()))
            tol = max(rt * rng, 1e-9, float(og["output_floor_by_column"][j]))
            e = np.abs(np.interp(t_off[away], t, X[:, j]) - Xo[away, j])
            assert e.max() <= 1.001 * tol, (rt, j, e.max(), tol)
