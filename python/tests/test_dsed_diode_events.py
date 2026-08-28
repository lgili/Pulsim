"""v2.0 Phase 3, item 1: the dsed engine commutates PWL diodes.

Before this change the dsed engine had no diode — only a resistor
whose state the user pinned through switch_fn: a reverse-biased
series diode conducted backwards (−10.909 V where pwl blocks at
−1e-06 V), and a buck's freewheel diode froze OFF, settling v_out at
0.59 V where 12 V is correct.

Now every PWL diode gets two auto-derived event predicates on its
branch voltage — reconstructed from the reduced state through the
algebraic recovery map — armed by the diode's own bit: turn-ON
(g = v_D − V_th) while OFF, turn-OFF (g = v_D, the i_D zero-cross)
while ON. Firing flips the bit inside the scheduler, and a zero-time
cascade settles diodes the mask change instantaneously forward-biases
(the freewheel diode at gate-off). The census and the decision rule
are the SAME code the pwl engine uses (DiodeEventState,
SwitchedDiode::decide_next_state), so the engines cannot drift on
what "conducting" means.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def _series_diode(vin):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", vin)
    b.add_resistor("R1", "vin", "na", 10.0)
    b.add_diode("D", "na", "vout", 1e3, 1e-9, 0.7)
    b.add_capacitor("C", "vout", "gnd", 10e-6)
    b.add_resistor("Rl", "vout", "gnd", 100.0)
    return b


def _run_dsed(b, t_end, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, engine="dsed", **kw)


def _run_pwl(b, t_end, dt, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, dt=dt, **kw)


def _time_avg(t, v):
    t = np.asarray(t)
    v = np.asarray(v)
    return float(np.trapezoid(v, t) / (t[-1] - t[0]))


def test_the_backwards_conduction_is_gone():
    """THE bug: reverse-biased, the diode must block. It conducted
    backwards (−10.909 V) for as long as the dsed engine existed."""
    rd = _run_dsed(_series_diode(-12.0), 2e-3)
    assert abs(np.asarray(rd.states)[-1][0]) < 1e-3

    # And forward it still conducts — no choice of frozen bit ever
    # satisfied both polarities.
    rd2 = _run_dsed(_series_diode(+12.0), 2e-3)
    assert np.asarray(rd2.states)[-1][0] == pytest.approx(10.909,
                                                           abs=0.05)


def test_halfwave_rectifier_commutates_and_is_sharper_than_pwl():
    """A 60 Hz half-wave rectifier: two commutations per cycle. The
    event-located answer must not overshoot the source peak — the
    pwl engine at dt = 1e-6 overshoots to 10.46 V from trapezoidal
    commutation ringing and only converges to 10.000 at dt = 1e-8.
    Event location gets it exactly, which is the entire argument for
    an event-driven engine."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0, 60.0)
    b.add_diode("D", "ac", "vout", 1e3, 1e-9, 0.7)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("Rl", "vout", "gnd", 200.0)

    rd = _run_dsed(b, 5e-2)
    v = np.asarray(rd.states)[:, 0]
    assert rd.n_events >= 6            # ≥ 2 commutations × 3 cycles
    assert v.max() == pytest.approx(10.0, abs=1e-2)
    assert v.max() <= 10.0 + 1e-6      # no commutation overshoot
    assert v.min() >= -1e-6            # never negative: it rectifies


def test_fullwave_bridge_cascade_matches_pwl():
    """Two diodes commutate together at every zero crossing — the
    zero-time cascade. Compare TIME averages (dsed samples cluster
    near events, so raw means are biased by construction)."""
    def bridge():
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("Vac", "a", "bn", 0.0, 10.0, 60.0)
        b.add_resistor("Rg", "bn", "gnd", 1e-3)
        for n, (an, ca) in {"D1": ("a", "p"), "D2": ("bn", "p"),
                             "D3": ("gnd", "a"),
                             "D4": ("gnd", "bn")}.items():
            b.add_diode(n, an, ca, 1e3, 1e-9, 0.7)
        b.add_capacitor("C", "p", "gnd", 47e-6)
        b.add_resistor("Rl", "p", "gnd", 200.0)
        return b

    rd = _run_dsed(bridge(), 5e-2)
    rp = _run_pwl(bridge(), 5e-2, 1e-7)
    avg_d = _time_avg(rd.times, np.asarray(rd.states)[:, 0])
    avg_p = _time_avg(rp.times, rp.v("p"))
    assert avg_d == pytest.approx(avg_p, rel=1e-3)


def test_buck_ccm_with_a_real_diode():
    """The 20x error: the frozen freewheel diode settled v_out at
    0.59 V. With commutation it must land where the pwl engine does."""
    b = p.CircuitBuilder()
    p.add_buck(b, V_in=24.0, L=100e-6, C=100e-6, R_load=2.0,
                f_sw=100e3)
    T = 1e-5

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(0, (t % T) < 0.5 * T)
        return m

    rd = _run_dsed(b, 5e-3, switch_fn=pwm)
    sd = np.asarray(rd.states)
    tail = sd[int(0.9 * len(sd)):]
    assert float(np.mean(tail[:, 0])) == pytest.approx(12.0, rel=0.02)
    assert float(np.mean(tail[:, 1])) == pytest.approx(6.0, rel=0.05)
    assert rd.n_events > 500           # gate edges + commutations


def test_dcm_fails_fast_and_names_the_cause():
    """Discontinuous conduction is NOT yet supported: the idle mode's
    L·g_off time constant (~1e-13 s) grinds an explicit integrator to
    a halt. That used to burn the 10M-step cap for 7 seconds and die
    with a generic message; it must now fail in well under a second
    naming the mechanism and the engine that handles it."""
    import time

    b = p.CircuitBuilder()
    p.add_buck(b, V_in=24.0, L=100e-6, C=100e-6, R_load=50.0,
                f_sw=100e3)
    T = 1e-5

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(0, (t % T) < 0.3 * T)
        return m

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError) as exc:
        _run_dsed(b, 5e-3, switch_fn=pwm)
    elapsed = time.perf_counter() - t0
    msg = str(exc.value)
    assert "DISCONTINUOUS" in msg
    assert "engine='pwl'" in msg
    assert elapsed < 2.0, elapsed


def test_explicit_bdf2_with_diodes_is_refused():
    """Predicates live on the RK45 scheduler only. 'auto' routes
    there silently; an EXPLICIT bdf2 request must refuse rather than
    silently freeze the diode bits."""
    with pytest.raises(ValueError, match="bdf2"):
        _run_dsed(_series_diode(12.0), 1e-3, integrator="bdf2")
