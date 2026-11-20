"""Release findings, group F: the exact engine's output rule (``output_rtol``).

F-6-7  the trajectory must not depend on the output grid (mpmath truth, event states).
F-3-6 / F-6-6  the step counters count integrator steps under the rule too.
F-X-1  ten paralleled bridges under the release profile run to 20 ms.
F-X-3  a 500-state RC ladder with the rule costs about what it costs without it.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

sys.path.insert(0, str(Path(__file__).parent))
from test_exact_route import _paralleled_bridges, _rc_with_b_extra  # noqa: E402

TRUTH_IL = 0.12972284951463489   # i(L0a)(1e-4), mpmath 50 digits (test_exact_route)


def _run(b, t_end, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, engine="exact", **kw)


def _kw(rt):
    return {} if rt is None else {"output_rtol": rt}


@pytest.mark.parametrize("rt", [None, 1e-4, 1e-5, 1e-8])
def test_f67_ten_bridges_first_mode_is_the_truth_on_every_grid(rt):
    """F-6-7: 2.6e-10 off on the default grid; 6.6e-8 at 1e-5 and 2.1e-6 at 1e-8 when every
    output sample advanced the state (x_{k+1} = Φ(δ_k)·x_k)."""
    r = _run(_paralleled_bridges(10), 1e-4, **_kw(rt))
    il = float(np.asarray(r.i("L0a"))[-1])
    assert abs(il - TRUTH_IL) <= 1e-9
    if rt is not None:
        assert r.diagnostics["output_grid"]["n_output_samples"] == len(r.times) - r.n_accept - 1


def _event_rows(r):
    t = np.asarray(r.times)
    et = np.asarray(r.event_times)
    return et, np.asarray(r.states)[np.searchsorted(t, et)], np.asarray(r.states)[-1]


def test_f67_event_states_do_not_depend_on_the_output_grid():
    """Ten bridges over 3 ms (dozens of diode events): the event instants and the states there
    are the same bits whatever the rule samples in between. (Not the rule-off run's: the rule
    gates the event projection and the root rule, test_exact_output_grid.)"""
    ref = _event_rows(_run(_paralleled_bridges(10), 3e-3, output_rtol=1e-4))
    assert len(ref[0]) > 10
    for rt in (1e-5, 1e-6):
        got = _event_rows(_run(_paralleled_bridges(10), 3e-3, output_rtol=rt))
        assert np.array_equal(got[0], ref[0])
        assert np.array_equal(got[1], ref[1])
        assert np.array_equal(got[2], ref[2])


def test_f67_samples_inside_a_step_are_the_closed_form():
    """The rule's samples are the exact solution: RC charging, closed form at every sample."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_resistor("R", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    r = _run(b, 5e-3, output_rtol=1e-6)
    t = np.asarray(r.times)
    assert len(t) > 50
    v = np.asarray(r.v("o"))
    assert np.max(np.abs(v - 5.0 * (1.0 - np.exp(-t / 1e-3)))) <= 1e-12 * 5.0


@pytest.mark.parametrize("rt", [None, 1e-5])
def test_f36_numeric_segment_counts_its_integrator_steps(rt):
    """F-3-6 / F-6-6: a tier-3 (numeric) segment's DOPRI5 steps are n_rk45_steps on the event
    integrator too; output samples are not steps."""
    b, fn = _rc_with_b_extra()
    r = _run(b, 1e-3, b_extra_fn=fn, allow_implicit_segments=True, **_kw(rt))
    assert r.n_exact_steps == 0
    assert r.n_rk45_steps + r.n_bdf2_steps > 0
    assert np.asarray(r.v("o"))[-1] == pytest.approx(5.0 * (1 - np.exp(-1.0)), rel=1e-5)


def test_f67_chords_from_rest_are_within_the_rule():
    """RC charging from 0 V: every chord between consecutive samples is within output_rtol
    of the closed form at its midpoint, the first one (state at rest, floor tolerance)
    included."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_resistor("R", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    rt = 1e-6
    r = _run(b, 5e-3, output_rtol=rt)
    t = np.asarray(r.times)
    v = np.asarray(r.v("o"))
    tm = 0.5 * (t[:-1] + t[1:])
    err = np.abs(0.5 * (v[:-1] + v[1:]) - 5.0 * (1.0 - np.exp(-tm / 1e-3)))
    assert err.max() <= rt * 5.0, (err.argmax(), err.max())


def test_f36_bdf2_reports_no_rk45_steps():
    """F-3-6's DOPRI5 count is the exact engine's: a BDF2 run's steps are not RK45 ones."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_resistor("R", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    r = p.simulate(b, t_end=1e-3, engine="dsed", integrator="bdf2")
    assert r.n_accept > 0 and r.n_exact_steps == 0
    assert r.n_rk45_steps == 0


def test_f36_exact_steps_are_steps_not_samples():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_resistor("R", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    off = _run(b, 5e-3)
    on = _run(b, 5e-3, output_rtol=1e-6)
    assert on.n_exact_steps == off.n_exact_steps and on.n_rk45_steps == off.n_rk45_steps == 0
    assert on.diagnostics["output_grid"]["n_output_samples"] == len(on.times) - len(off.times)


def _boost(R_load):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L1", "vin", "sw", 100e-6)
    b.add_switch("Q1", "sw", "gnd", g_on=1e3, g_off=1e-9)
    b.add_diode("D1", "sw", "out", 1e3, 1e-9)
    b.add_capacitor("C1", "out", "gnd", 47e-6)
    b.add_resistor("RL", "out", "gnd", R_load)
    q1 = b.switch_index_of("Q1")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(q1, (t % 1e-5) < 0.5e-5)
        return m
    return b, pwm


def _recovery(R_load, **kw):
    b, pwm = _boost(R_load)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 3e-3, switch_fn=pwm, **kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = p.device_loss_summary(b, r, switch_fn=pwm, diode_specs={"D1": {"Q_rr": 100e-9}})
    d = next(e for e in s if e["name"] == "D1")
    return d, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize("R_load", [20.0, 500.0])
def test_f58_recovery_energy_on_the_exact_rule_grid(R_load):
    """F-5-8: on the exact engine with the output rule V_R is read on the samples' step lines
    from the right limit (the rule bounds their chords), not by the fixed engine's midpoint
    pairing: CCM E_sw 6.775e-4 J (warned, 60 turn-offs 'unresolved') -> 6.948e-4 J against
    pwl at dt = 10 ns 6.9495e-4 J; DCM 9.934e-4 against 9.939e-4."""
    d_ex, w_ex = _recovery(R_load, engine="exact", output_rtol=1e-5)
    d_ref, _ = _recovery(R_load, engine="pwl", dt=1e-8)
    assert not w_ex, w_ex
    assert d_ex["n_turn_off_events"] == d_ref["n_turn_off_events"]
    assert d_ex["n_turn_off_events_unresolved"] == 0
    assert d_ex["E_sw_total"] == pytest.approx(d_ref["E_sw_total"], rel=1e-3)


def _branch_bridges(n):
    """_paralleled_bridges with branch-form diodes (the release's add_diode default):
    R_on = 1/g_on, the same g_off."""
    orig = p.CircuitBuilder.add_diode

    def branch(self, name, a, k, g_on=None, g_off=None, V_th=0.0, V_f=0.0):
        return orig(self, name, a, k, g_off=g_off, V_th=V_th, V_f=V_f, form="branch", R_on=1.0 / g_on)
    p.CircuitBuilder.add_diode = branch
    try:
        return _paralleled_bridges(n)
    finally:
        p.CircuitBuilder.add_diode = orig


@pytest.mark.xfail(strict=True, reason=(
    "release finding F-X-1: under output_rtol the event projection keeps every mode but the "
    "parasitic cluster (1/(1e-6 * 1 us) = 1e12 /s); ten branch-form bridges keep their "
    "~2e-10 s common mode (L / R_g), and after Dt0c turns OFF at 19.96896 ms the b-phase "
    "bottom diodes chatter (Db0b located OFF 18 times in 43 ns, intervals shrinking ~0.85x: "
    "a Zeno approach to a sliding mode) until the settle leaves one OFF 0.73 V forward. "
    "The old projection (100/dt_max) removes that mode and the run passes; it is not the "
    "fix (it projects ll06's real 100 ns transient, 1.1e-2 off the truth). Needs a "
    "sliding-mode rule in the exact engine's event decisions (zero_band_on_ / settle)"))
def test_fx1_ten_branch_form_bridges_run_under_the_rule():
    r = _run(_branch_bridges(10), 0.02, output_rtol=1e-5)
    assert r.times[-1] == pytest.approx(0.02)


def _ladder(n):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "n0", "gnd", 10.0)
    for k in range(n):
        b.add_resistor(f"R{k}", f"n{k}", f"n{k + 1}", 10.0)
        b.add_capacitor(f"C{k}", f"n{k + 1}", "gnd", 1e-9)
    return b


def test_fx3_ladder_far_nodes_are_held_to_the_absolute_floor():
    """F-X-3: a 100-section RC ladder, 2e-5 s. The far nodes sit at ~0 V; their tolerance was
    rtol*atol = 1e-14 while the chord bound sums volts of modes that cancel there: every step
    was cut to that floor, 197 755 samples. With atol an absolute floor (max(rtol*range, atol))
    they are within tolerance: no cap, no warning, nothing left out of the check. The state
    carried is the rule-free run's."""
    off = _run(_ladder(100), 2e-5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        on = p.simulate(_ladder(100), t_end=2e-5, engine="exact", output_rtol=1e-5)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    og = on.diagnostics["output_grid"]
    assert og["n_output_capped"] == 0 and og["n_output_unbounded"] == 0
    assert "n_output_uncertified" not in og
    assert og["n_output_samples"] < 10000
    assert np.array_equal(np.asarray(on.states)[-1], np.asarray(off.states)[-1])
