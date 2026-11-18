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
    assert abs(np.asarray(rd.v("vout"))[-1]) < 1e-3

    # And forward it still conducts — no choice of frozen bit ever
    # satisfied both polarities.
    rd2 = _run_dsed(_series_diode(+12.0), 2e-3)
    assert np.asarray(rd2.v("vout"))[-1] == pytest.approx(10.909,
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
    v = np.asarray(rd.v("vout"))
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
    avg_d = _time_avg(rd.times, np.asarray(rd.v("p")))
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
    vo = np.asarray(rd.v("vout"))
    il = np.asarray(rd.i("buck__L"))
    assert float(np.mean(vo[int(0.9 * len(vo)):])) == pytest.approx(12.0, rel=0.02)
    assert float(np.mean(il[int(0.9 * len(il)):])) == pytest.approx(6.0, rel=0.05)
    assert rd.n_events > 500           # gate edges + commutations


def test_dcm_runs_exactly_and_matches_pwl():
    """Phase-3 item 2's gate, and the history matters: this test
    REFUSED discontinuous conduction when item 1 landed — the idle
    mode's L·g_off time constant (~1e-13 s) pinned the explicit
    integrator at h ≈ 2e-10 s and 10M steps covered 0.3 ms in
    7 seconds. The consistent-reinitialization projection alone did
    NOT fix it (measured: the state sat exactly at the idle
    equilibrium and still ground — an explicit method is STABILITY-
    limited, not accuracy-limited). What fixes it is the pairing:
    projection puts the state on the slow manifold, and exact LTI
    stepping (closed-form e^{At} per mode) removes the stability
    limit entirely. The whole 5 ms run now takes ~0.02 s and lands
    within 0.1% of the pwl engine at dt = 1e-8."""
    import time

    def mk():
        b = p.CircuitBuilder()
        p.add_buck(b, V_in=24.0, L=100e-6, C=100e-6, R_load=50.0,
                    f_sw=100e3)
        return b

    T = 1e-5

    def pwm_for(b):
        def pwm(t):
            m = p.SwitchStateMask(b.graph.num_switches)
            m.set(0, (t % T) < 0.3 * T)
            return m
        return pwm

    b = mk()
    t0 = time.perf_counter()
    rd = _run_dsed(b, 5e-3, switch_fn=pwm_for(b))
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, elapsed          # was a 10M-step grind

    td = np.asarray(rd.times)
    vo = np.asarray(rd.v("vout"))
    i0 = int(np.searchsorted(td, 4e-3))
    vo_d = float(np.trapezoid(vo[i0:], td[i0:])
                 / (td[-1] - td[i0]))

    b2 = mk()
    rp = _run_pwl(b2, 5e-3, 1e-8, switch_fn=pwm_for(b2))
    tp = np.asarray(rp.times)
    vp = np.asarray(rp.v("vout"))
    j0 = int(np.searchsorted(tp, 4e-3))
    vo_p = float(np.trapezoid(vp[j0:], tp[j0:]) / (tp[-1] - tp[j0]))

    # DCM raises the output above CCM's D·V_in = 7.2 V; both engines
    # must agree on where it lands.
    assert vo_p > 8.0
    assert vo_d == pytest.approx(vo_p, rel=1e-3)
    # And the current stays at the physical g_off leakage level when
    # the diode is off — never ampere-scale negative.
    assert np.asarray(rd.i("buck__L")).min() > -1e-3


def _rectifier_bank(n_bridge, ls=0.2e-3):
    """n three-phase diode bridges (100 V, 50 Hz, bridge k shifted 6k
    degrees), each phase behind `ls`, each bridge into its own
    200 uF || 50 ohm (probes/bank60_cost.py of
    add-complementarity-mode-selection)."""
    b = p.CircuitBuilder()
    for k in range(n_bridge):
        for ph, ang in (("a", 0.0), ("b", -120.0), ("c", 120.0)):
            b.add_sine_voltage_source(f"V{k}{ph}", f"s{k}{ph}", "gnd", 0.0,
                                      100.0, 50.0, np.deg2rad(ang + 6.0 * k))
            b.add_inductor(f"L{k}{ph}", f"s{k}{ph}", f"x{k}{ph}", ls)
            b.add_diode(f"Dt{k}{ph}", f"x{k}{ph}", f"p{k}", 1e3, 1e-9, 0.0)
            b.add_diode(f"Db{k}{ph}", f"n{k}", f"x{k}{ph}", 1e3, 1e-9, 0.0)
        b.add_capacitor(f"C{k}", f"p{k}", f"n{k}", 200e-6)
        b.add_resistor(f"R{k}", f"p{k}", f"n{k}", 50.0)
        b.add_resistor(f"Rg{k}", f"n{k}", "gnd", 1e6)
    return b


@pytest.mark.parametrize("mode", [None, "lcp"])
def test_a_signal_returning_below_its_threshold_is_not_an_event(mode):
    """Five bridges behind line inductors. From 3.67 ms to 5 ms the
    bottom rail of each bridge moves from phase b to phase c carrying
    only the 1 Mohm tie's 50 uA. The outgoing diode goes OFF, and the
    projection's rounding in its line inductor (~1e-14 A through
    g_off = 1 nS) leaves its signal at +2e-5 V, relaxing to its negative
    value in L*g_off = 2e-13 s. The locator read that return as a
    turn-ON crossing and re-decided the same mode until the chatter
    guard threw ("Db2b is chattering at t = 4.333e-3 s"), default and
    mode_selection='lcp' alike.

    Reference: pwl, which is first order here (its end-value error
    halves with dt: 1.1e-2, 5.4e-3, 2.7e-3 V at 1, 0.5, 0.25 us), so the
    two finest runs are extrapolated; dsed is 3.4e-4 V from that."""
    n, t_end = 5, 6e-3

    def v_end(r):
        return np.array([r.v(f"p{k}")[-1] - r.v(f"n{k}")[-1] for k in range(n)])

    coarse = v_end(_run_pwl(_rectifier_bank(n), t_end, 5e-7))
    fine = v_end(_run_pwl(_rectifier_bank(n), t_end, 2.5e-7))
    ref = 2.0 * fine - coarse

    kw = {"mode_selection": mode} if mode else {}
    rd = _run_dsed(_rectifier_bank(n), t_end, **kw)
    assert v_end(rd) == pytest.approx(ref, abs=2e-3)
    # one event per commutation (15 measured), not a burst per diode
    assert rd.n_events < 30


@pytest.mark.parametrize("mode", [None, "lcp"])
def test_a_signal_drifting_out_of_the_zero_band_is_still_an_event(mode):
    """The one-sided rule above skips a start past the threshold only beyond the zero band
    (1e-12 x the largest node voltage). A single-phase bridge whose DC bus floats on the g_off
    leaks: at 5.103 ms the series pair D1/D4 turns off with i = 0, D4 goes OFF and D1 is left ON
    at v_D = -1.4e-14 V. Its current then drifts negative at leak scale; a crossing from inside
    the band is read both ways, so D1 goes OFF at 5.109 ms. Skipping every start past zero held
    D1 ON at -1.9e-7 A until 10 ms and moved the bus common mode: mean v(p) 43.38 V against
    47.62 V. Reference: pwl at 0.1 us, mean v(p) over the run within 6e-3 V."""
    def bridge():
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 100.0, 50.0)
        for nm, an, ca in (("D1", "a", "p"), ("D3", "gnd", "p"), ("D2", "n", "a"), ("D4", "n", "gnd")):
            b.add_diode(nm, an, ca, 1e3, 1e-9, 0.0)
        b.add_capacitor("C", "p", "n", 1e-3)
        b.add_resistor("R", "p", "n", 100.0)
        return b

    t_end = 0.012
    kw = {"mode_selection": mode} if mode else {}
    b = bridge()
    r = _run_dsed(b, t_end, **kw)
    on = np.asarray(r.switch_states)[:, b.switch_index_of("D1")].astype(bool)
    i_d1 = 1e3 * (np.asarray(r.v("a")) - np.asarray(r.v("p")))
    assert i_d1[on].min() > -1e-9                                  # was -1.9e-7 A
    rp = _run_pwl(bridge(), t_end, 1e-7)
    mean = np.trapezoid(np.asarray(r.v("p")), r.times) / t_end
    mean_ref = np.trapezoid(np.asarray(rp.v("p")), rp.times) / t_end
    assert mean == pytest.approx(mean_ref, abs=2e-2)               # was 4.2 V off


def test_explicit_bdf2_with_diodes_is_refused():
    """Predicates live on the RK45 scheduler only. 'auto' routes
    there silently; an EXPLICIT bdf2 request must refuse rather than
    silently freeze the diode bits."""
    with pytest.raises(ValueError, match="bdf2"):
        _run_dsed(_series_diode(12.0), 1e-3, integrator="bdf2")
