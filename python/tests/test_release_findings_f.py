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
        # every stored point is a rule sample or a step end (ends the chord covers are not stored)
        assert len(r.times) <= r.n_accept + 1 + r.diagnostics["output_grid"]["n_output_samples"]


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
    assert len(on.times) <= len(off.times) + on.diagnostics["output_grid"]["n_output_samples"]


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


def test_fx1_ten_branch_form_bridges_run_under_the_rule():
    """F-X-1: refused at 19.969 ms (Db8b OFF 0.73 V forward after Db0b chattered 18 times in
    43 ns), and only for t_end = 0.02: t_end = 0.0205 / 0.025 / 0.03 ran. The parasitic cut is
    1/(1e-6 t_res), t_res = next time event - last time event; with no gate oracle the next is
    t_end and every dt_max/10 poll landing counted as the last, so t_res was the time LEFT in the
    run (31 us there) and the common mode L/R_g = 2e-10 s left the cluster near the end. A poll
    landing that changes nothing is not a time event: the rule's run takes the rule-free run's
    events (111, one at 19.96895575516 ms, no chatter) to 1e-12 s and its end state to 1e-12 of
    each state's peak (measured 1e-12 V on the bus)."""
    on = _run(_branch_bridges(10), 0.02, output_rtol=1e-5)
    off = _run(_branch_bridges(10), 0.02)
    assert on.times[-1] == pytest.approx(0.02)
    eon, _, fon = _event_rows(on)
    eoff, _, foff = _event_rows(off)
    assert len(eon) == len(eoff) == 111
    assert np.abs(eon - eoff).max() <= 1e-12
    scale = np.abs(np.asarray(off.states)).max(axis=0) + 1e-12
    assert (np.abs(fon - foff) / scale).max() <= 1e-11


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


def test_fx4_a_dc_circuit_stores_what_the_tolerance_needs():
    """F-X-4: a DC source across a resistor (the electrothermal benchmark's netlist), 2 s, no dt:
    the exact engine steps 1 us (the poll of a gate that names no edge) and stored every step end,
    2 000 002 samples, with the rule as without it. Under the rule a step end the chord from the
    last stored sample covers is not stored: a constant is two samples, still exactly 5 V. The
    steps (n_exact_steps) are the rule-free run's."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 5.0)
    b.add_resistor("R", "vin", "gnd", 1.0)
    off = _run(b, 0.2)
    on = _run(b, 0.2, output_rtol=1e-5)
    assert on.n_exact_steps == off.n_exact_steps
    assert len(on.times) <= 4
    assert on.times[0] == 0.0 and on.times[-1] == pytest.approx(0.2)
    assert np.all(np.asarray(on.v("vin")) == 5.0)


def test_fx4_deferred_ends_keep_the_rule_on_an_rc():
    """The deferred step ends do not loosen the rule: RC charging over many steps, every chord
    between stored samples within output_rtol of the closed form at its midpoint, and far fewer
    samples than steps once the charge flattens."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_resistor("R", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    rt = 1e-5
    r = _run(b, 2e-2, output_rtol=rt)
    t = np.asarray(r.times)
    v = np.asarray(r.v("o"))
    assert len(t) < r.n_exact_steps
    tm = 0.5 * (t[:-1] + t[1:])
    err = np.abs(0.5 * (v[:-1] + v[1:]) - 5.0 * (1.0 - np.exp(-tm / 1e-3)))
    assert err.max() <= rt * 5.0, (err.argmax(), err.max())


def test_fx4_thermal_kpi_is_exact_on_any_sample_grid():
    """The benchmark KPI integrates the thermal RC exactly per sample interval (P linear between
    samples): on the two samples of the exact engine's DC run and on the 1 004 of the trbdf2
    route it is the closed form 25 + 125 (1 - e^{-t/0.5}) (the explicit Euler it replaces read
    the 2 s interval as its step: unstable)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bench_kpi", Path(__file__).resolve().parents[2] / "benchmarks" / "kpi" / "__init__.py")
    kpi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kpi)
    truth = 25.0 + 125.0 * (1.0 - np.exp(-2.0 / 0.5))
    for t in (np.array([0.0, 2.0]), np.linspace(0.0, 2.0, 1004)):
        r = kpi.compute_junction_temperature(np.full(len(t), 25.0), t, 5.0, 0.1, 25.0)
        assert r["t_j_final_c"] == pytest.approx(truth, rel=1e-13)


class _StuckGate:
    """F-T-2: next_edge_after returns t itself (a float round-off in a user gate did it)."""

    def __init__(self, n, i):
        self.n, self.i = n, i

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        m.set(self.i, (t % 1e-5) < 5e-6)
        return m

    def next_edge_after(self, t):
        return t


@pytest.mark.parametrize("integrator", [None, "rk45", "bdf2"])
def test_ft2_a_gate_oracle_that_does_not_advance_is_refused_by_name(integrator):
    """F-T-2: the exact engine spun at 100 % CPU forever. One rule (events::require_after) where
    an oracle is asked (the event queue: exact and trbdf2's native oracles; the auto and BDF2
    schedulers): a refusal naming next_edge_after and the instant. trbdf2 never asks a Python
    gate's oracle (it locates a Python gate's edges by bisection), so it runs this gate."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_switch("S", "a", "o", 1e3, 1e-9)
    b.add_resistor("R", "o", "gnd", 10.0)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    gate = _StuckGate(b.graph.num_switches, b.switch_index_of("S"))
    with pytest.raises(p.UnsupportedError, match=r"next_edge_after\(t\) returned .* not a later instant"):
        p.simulate(b, t_end=1e-4, engine="dsed", switch_fn=gate,
                   **({} if integrator is None else {"integrator": integrator}))
