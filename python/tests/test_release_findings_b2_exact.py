"""Release finding F-B2-1: the exact engine's window integrals were chords.

On the buck of ``test_recorder_device_readings._buck`` the exact engine's
STATES at the window ends agree with an independent mpmath truth to 6e-15
(relative), but its recorder window integrals over [A, B] were 1.1e-4 to
2.0e-4 off: the recorder integrated each accepted step as the straight line
between its ends, and the exact engine's steps are its 10 us output steps.
Now each step is integrated in closed form along its mode
(``core/include/pulsim/dsed/segment_integrals.hpp``), so the statistics are
the trajectory's and do not depend on the output grid; ``output_rtol`` is no
longer dropped with recorders (it shapes the stored grid only); a piece with
no closed form warns.

Truth (no pulsim): two mpmath derivations that agree to 18 digits on the
integrals (``benchmarks/tools/exact_truth.py`` of fa4b5c71, and the hand
derivation x = [iL, vC] per mode, x(t) = e^{Mt}(x0 + xp) - xp, the diode's
turn-off by bisection). The energy ledgers need no truth: they close.
Notes: openspec/changes/update-number-changing-release/fixes/
exact-buck-integrals.md.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_recorder_device_readings as m  # noqa: E402

# mpmath, 40+ digits (hand derivation == exact_truth.py)
TRUTH_STATES = {m.A: (1.8962389778368713475, 7.8493711689425254904),
                m.B: (3.4379292124049610102, 9.5072665802156611277)}
TRUTH_INT = {"i(L)": 0.0012814829296360187342,
             "v(out)": 0.0063245198776164365313,
             "p(R)": 0.012186147898259251171}
# the ledger's devices: every element's absorbed power but C's (a stored-
# sample energy); p(V) is minus the power the source delivers
LEDGER = ["p(V)", "p(S1)", "p(D1)", "p(L)", "p(R)"]


def _run(t_end, make=m._buck, **kw):
    b, fn, _, _, _ = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, switch_fn=fn, engine="dsed", **kw)


def _windows(**kw):
    r = _run(1e-3, recorders=[
        p.Recorder("w", list(TRUTH_INT), keep="none", windows=[(m.A, m.B)])], **kw)
    return r, {q: r.recorded["w"].windows[q]["integral"][0] for q in TRUTH_INT}


def test_exact_engine_window_integrals_match_truth():
    # control: the states are exact. A RuntimeError so a state regression is
    # never read as an integral failure.
    for t, (il, vc) in TRUTH_STATES.items():
        r = _run(t)
        e = max(abs(r.i("L")[-1] / il - 1), abs(r.v("out")[-1] / vc - 1))
        if not (abs(r.times[-1] - t) < 1e-15 and e < 1e-12):
            raise RuntimeError(f"exact-engine state at {t} off by {e:.2e}")
    r, got = _windows(output_rtol=1e-10)
    errs = {q: abs(got[q] / v - 1) for q, v in TRUTH_INT.items()}
    # was 1.08e-4 / 1.60e-4 / 2.02e-4 (the 10 us chords)
    assert max(errs.values()) < 1e-11, errs
    cf = r.recorder_feed["closed_form"]
    assert cf["n_line"] == 0 and cf["n_closed"] > 0, cf


def test_window_statistics_do_not_depend_on_the_output_grid():
    """Default grid, dt = 1e-7 and output_rtol = 1e-8: the recorders'
    statistics are the trajectory's, not the grid's."""
    _, base = _windows()
    for kw in (dict(dt=1e-7), dict(output_rtol=1e-8)):
        _, got = _windows(**kw)
        for q in TRUTH_INT:
            assert got[q] == pytest.approx(base[q], rel=1e-12, abs=0), (kw, q)


def test_output_rtol_is_not_dropped_with_recorders():
    r, _ = _windows(output_rtol=1e-6)
    assert "output_grid" in r.diagnostics       # the rule ran
    assert r.recorder_feed["mode"] == "in-loop"


def _ledger(make, t_end, extra=(), **kw):
    """Σ ∫p (absorbed, every element but the capacitors) over [0, t_end]
    from one in-loop recorder, the capacitors' ΔE from the stored states."""
    probes = list(LEDGER) + list(extra)
    r = _run(t_end, make=make, recorders=[
        p.Recorder("e", probes, keep="none", windows=[(0.0, t_end)])], **kw)
    w = r.recorded["e"].windows
    return r, {q: w[q]["integral"][0] for q in probes}


def test_buck_energy_ledger_closes_on_the_default_grid():
    t_end = 1e-3
    r, e = _ledger(m._buck, t_end)
    vc = np.asarray(r.v("out"))
    de_c = 0.5 * 10e-6 * (vc[-1] ** 2 - vc[0] ** 2)
    # the inductor's ∫p(L) is its ΔE: two independent readings of one number
    il = np.asarray(r.i("L"))
    assert e["p(L)"] == pytest.approx(0.5 * 100e-6 * (il[-1] ** 2 - il[0] ** 2), rel=1e-11)
    total = sum(e.values()) + de_c
    assert abs(total) <= 1e-11 * abs(e["p(V)"]), (total, e)


def _boost_dcm():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 12.0)
    b.add_inductor("L", "in", "sw", 20e-6)
    b.add_switch("S1", "sw", "gnd", 1e3, 1e-9)
    b.add_diode("D1", "sw", "out", 1e3, 1e-9)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 50.0)
    fn = p.make_pwm_switch_fn(50e3, 0.3, b.switch_index_of("S1"), b.graph.num_switches)
    return b, fn, None, None, None


def test_boost_dcm_energy_ledger_closes_through_diode_commutations():
    t_end = 0.4e-3
    r, e = _ledger(_boost_dcm, t_end)
    assert len(r.event_times) > 40     # gate edges and diode turn-offs (DCM)
    vc = np.asarray(r.v("out"))
    total = sum(e.values()) + 0.5 * 10e-6 * (vc[-1] ** 2 - vc[0] ** 2)
    assert abs(total) <= 1e-11 * abs(e["p(V)"]), (total, e)
    assert r.recorder_feed["closed_form"]["n_line"] == 0


def test_a_state_jump_is_the_loss_port_s_impulse_not_the_window_s():
    """Two capacitors joined by an ideal switch at 1 us: the jump's energy
    ½·C1C2/(C1+C2)·ΔV² is booked by the loss port (E_impulse); the window
    integrals of the resistor and the switch run on both sides of it (the
    switch's ∫p over the jump instant is zero: no window integral sees an
    impulse). Ledger: ΔE_C + ∫p(R) + ∫p(S) + E_impulse = 0."""
    c1, c2, v1, v2, rr = 1e-6, 3e-6, 10.0, 2.0, 20.0
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", c1, v1)
    b.add_capacitor("C2", "y", "gnd", c2, v2)
    b.add_resistor("R", "y", "gnd", rr)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    k = b.switch_index_of("S")

    def fn(t, k=k, n=b.graph.num_switches):
        mk = p.SwitchStateMask(n)
        mk.set(k, t >= 1e-6)
        return mk
    tab = p.LossTable(v_axis=[10.0], i_axis=[1.0], tj_axis=[25.0],
                      energy=np.full((1, 1, 1), 0.0))
    t_end = 60e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.set_device_losses("S", Tj=25.0, E_on=tab, E_off=tab)
        r = p.simulate(b, t_end=t_end, engine="dsed", switch_fn=fn, auto_regularize=False,
                       recorders=[p.Recorder("e", ["p(R)", "p(S)"], keep="none",
                                             windows=[(0.0, t_end)])])
    e_imp = r.device_energy("S")["E_impulse"]
    v2s = v2 * np.exp(-1e-6 / (rr * c2))   # C2 discharges through R until the switch closes
    assert e_imp == pytest.approx(0.5 * c1 * c2 / (c1 + c2) * (v1 - v2s) ** 2, rel=1e-8)
    x, y = np.asarray(r.v("x")), np.asarray(r.v("y"))
    de = 0.5 * c1 * (x[-1] ** 2 - v1 ** 2) + 0.5 * c2 * (y[-1] ** 2 - v2 ** 2)
    w = r.recorded["e"].windows
    e_r = w["p(R)"]["integral"][0]
    e_s = w["p(S)"]["integral"][0]   # the open switch's g_off leak before 1 us: 6.4e-14 J
    assert de + e_r + e_s + e_imp == pytest.approx(0.0, abs=1e-11 * e_r)
    assert r.recorder_feed["closed_form"]["n_line"] == 0


def test_a_piece_with_no_closed_form_is_the_line_and_warns():
    b, fn, _, _, _ = m._buck()
    with pytest.warns(RuntimeWarning, match="no closed form"):
        r = p.simulate(b, 1e-3, switch_fn=fn, engine="dsed", recorders=[
            p.Recorder("w", ["abs(i(L))"], keep="none", windows=[(m.A, m.B)])])
    cf = r.recorder_feed["closed_form"]
    assert cf["n_line"] > 0 and "abs(i(L))" in cf["why"]
