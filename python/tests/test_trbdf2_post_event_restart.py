"""TR-BDF2 restart after an event (update-trbdf2-post-event-restart).

After every gate edge the variable-step engine collapsed its step and
climbed back: the companion history still held the PRE-edge derivative
(inductor voltage, capacitor current), so the first post-edge step had
an O(h) error the controller rightly refused, and the post-edge memory
stored the accepted step, which can only shrink. On the 24 V buck of
the core review (numerics/p6_ratchet.py, p5_trbdf2_edges.py) the
rising-edge step was 8.26e-13 s for the whole 400-period run.

Counters, not wall time: accepted steps, solves and the first step
after each edge.
"""

import numpy as np
import pytest
from scipy.linalg import expm

import pulsim as p

VIN, L, C, GON, GOFF, T, T0 = 24.0, 100e-6, 47e-6, 1e3, 1e-9, 1e-5, 2e-6
NPER = 400
PERIODS = (1, 2, 3, 5, 10, 50, 100, 399)


def _buck(R, D, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", C)
    b.add_resistor("RL", "vout", "gnd", R)
    n = b.graph.num_switches
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        ph = (t - T0) / T
        fr = ph - np.floor(ph + 1e-9)
        m.set(k, bool(t >= T0 - 1e-13 and fr < D - 1e-9))
        return m
    return p.simulate(b, t_end=T0 + NPER * T, engine="trbdf2",
                      switch_fn=fn, **kw)


def _ccm_exact():
    """x = [i_L, v_C] at T0 + NPER*T: CCM buck, piecewise LTI, expm."""
    def prop(gs, gd, x, dt):
        k = 1 / (gs + gd)
        M = np.zeros((3, 3))
        M[:2, :2] = [[-k / L, -1 / L], [1 / C, -1 / (1.0 * C)]]
        M[0, 2] = gs * VIN * k / L
        E = expm(M * dt)
        return E[:2, :2] @ x + E[:2, 2]
    x = np.zeros(2)
    for _ in range(NPER):
        x = prop(GON, GOFF, x, 0.5 * T)
        x = prop(GOFF, GON, x, 0.5 * T)
    return x


def _post_edge_step(t, te):
    k = int(np.argmin(np.abs(t - te)))
    assert abs(t[k] - te) < 1e-12, "edge not landed"
    if t[k + 1] - te < 1e-9:  # the committed edge probe (h_probe)
        k += 1
    return t[k + 1] - t[k]


@pytest.fixture(scope="module")
def ccm():
    return _buck(1.0, 0.5)


@pytest.fixture(scope="module")
def dcm():
    return _buck(50.0, 0.3)


def test_ccm_post_edge_step_and_steps_per_period(ccm):
    """The first step after every edge is sized by a true estimate: above
    1e-8 s at every listed period (8.26e-13 s on the rising edge before
    this change) and at most 8 accepted steps per period (18 before)."""
    t = np.asarray(ccm.times)
    for n in PERIODS:
        for te in (T0 + n * T, T0 + (n + 0.5) * T):
            assert _post_edge_step(t, te) > 1e-8
    st = ccm._trbdf2_stats
    assert st["n_accept"] <= 8 * NPER
    assert st["n_derivative_refreshes"] >= st["n_gate_events"]
    assert st["h_post_event_min"] > 1e-8
    assert st["n_refresh_probes"] == 0   # the edge cascade's probe serves


def test_ccm_error_not_above_twice_the_previous(ccm):
    """End-of-run error against the expm reference: at most twice the
    value before this change (9.84e-4, max of the i_L and v_C errors at
    the default tolerances, probes/trbdf2_baseline.json)."""
    xe = _ccm_exact()
    err = max(abs(np.asarray(ccm.i("L"))[-1] - xe[0]),
              abs(np.asarray(ccm.v("vout"))[-1] - xe[1]))
    assert err < 2 * 9.84e-4


def test_dcm_no_step_below_1e_11(dcm):
    """No integration step below 1e-11 s. Runs of accepted steps between
    4e-14 and 1e-12 s before the post-event restart; 2,606 below 1e-11 s
    before the filtered SDIRK restart on piecewise-linear circuits
    (1,806 of them in the tau ~ 2e-13 s L*g_off mode after each diode
    turn-off, which the raw estimate read as error at any h).

    The only shorter intervals are the two samples recorded AT an event by
    design: the committed gate-edge probe (one per edge) and the post-event
    record of a diode event (one per event). Each starts at an event
    instant; structurally exact, so their count is bounded exactly. The
    run's first nanosecond (h_init) is left out."""
    t = np.asarray(dcm.times)
    k = np.flatnonzero(np.diff(t) < 1e-11)
    k = k[t[k] > 1e-9]   # the start: h_init and the first diode turn-on
    edges = np.concatenate([T0 + np.arange(NPER + 1) * T,
                            T0 + (np.arange(NPER) + 0.3) * T])
    ev = np.asarray([e.t_estimated for e in dcm.commutation_events])
    inst = np.sort(np.concatenate([edges, ev]))
    j = np.clip(np.searchsorted(inst, t[k]), 1, len(inst) - 1)
    dist = np.minimum(np.abs(inst[j] - t[k]), np.abs(inst[j - 1] - t[k]))
    # 5e-13: well above the landing's h_min noise (4e-14) and below the
    # record after a turn-off (>= 9.6e-11 s here), so a short step after
    # the record does not pass as "at the event".
    assert dist.max(initial=0.0) < 5e-13   # every short interval is at an event
    assert len(k) <= len(inst)
    assert dcm._trbdf2_stats["n_diode_events"] > NPER // 2   # the DCM turn-offs ran


def _mosfet_half_bridge(engine, **kw):
    T = 1e-5
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", 48.0)
    b.add_mosfet_level1("M1", "vdc", "sw", "g1", K=20.0, V_T=3.0)
    b.add_mosfet_level1("M2", "sw", "gnd", "g2", K=20.0, V_T=3.0)
    b.add_pulse_voltage_source("Vg1", "g1", "sw", 0.0, 12.0, 0.2e-6, 4.6e-6, T)
    b.add_pulse_voltage_source("Vg2", "g2", "gnd", 0.0, 12.0, 5.2e-6, 4.6e-6, T)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("RL", "out", "gnd", 5.0)
    return p.simulate(b, t_end=50 * T, engine=engine, **kw)


def _time_mean(res, node):
    t = np.asarray(res.times)
    return float(np.trapezoid(np.asarray(res.v(node)), t) / (t[-1] - t[0]))


def test_body_diode_events_restart_with_an_l_stable_step():
    """One SDIRK(2,2) step after each diode event: TR-BDF2's first step
    read the pre-event derivative of the body-diode commutation. The
    output mean was 5.7 mV off the fixed engine at T/4000 (2.0e-4 V
    with the SDIRK step), in 1555 steps and 772 rejects (1280, 173)."""
    ref = _time_mean(_mosfet_half_bridge("pwl", dt=1e-5 / 4000), "out")
    res = _mosfet_half_bridge("trbdf2")
    st = res._trbdf2_stats
    assert st["n_sdirk_steps"] > 0
    # 2.36e-4 V and 101 rejects with the restart size remembered per
    # diode transition (240 rejects before: 215 of them the restart
    # step taken at the pre-event h and rejected once).
    assert abs(_time_mean(res, "out") - ref) < 3e-4
    assert st["n_reject"] < 200


def test_diode_event_is_recorded_where_it_happens():
    """ll08_buck_mosfet: the SDIRK restart step after a diode event may be
    long (500 ns), and V(sw) jumps at the event. With no sample between
    the event and that step's end the record ramped across it (strict
    error 0.00231 -> 0.0252, edge instants 0.48 us late). The cascade's
    probe under the post-event bits is recorded right after the event."""
    res = _mosfet_half_bridge("trbdf2")
    t = np.asarray(res.times)
    ev = np.array([e.t_estimated for e in res.commutation_events])
    assert ev.size > 100
    assert np.all(np.diff(t) > 0)
    nxt = t[np.searchsorted(t, ev, side="right")]
    assert (nxt - ev).max() < 1e-9


def _tick_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "sw", 100.0, 1e-6)
    b.add_diode("D", "gnd", "sw", 100.0, 1e-6)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _edges_sampled(r, k_sw):
    """(rising with a post-edge sample within 1 ns, rising, falling with
    one, falling) of switch column k_sw."""
    t = np.asarray(r.times)
    s = np.asarray(r.switch_states)[:, k_sw]
    k = np.flatnonzero(s[1:] != s[:-1])
    near = (t[k + 1] - t[k]) < 1e-9
    up = s[k + 1]
    return (int(np.sum(near & up)), int(np.sum(up)),
            int(np.sum(near & ~up)), int(np.sum(~up)))


def test_a_rising_edge_on_a_controller_tick_gets_its_post_edge_sample():
    """A closed loop's period start is both a tick and a rising edge. The
    tick now fires before the edge probe is committed; before, the commit
    was refused there and 0 of 199 rising edges had a sample just after the
    edge (V(sw) ramped across the first post-edge step). Every edge, both
    directions, structurally."""
    Tp = 1e-5
    b = _tick_buck()
    pi = p.PIController(Kp=0.02, Ki=40.0, output_min=0.1, output_max=0.9)
    loop = p.bind_pi_to_switch(b, pi=pi,
                               measured=lambda x, i=b.node_id_of("out"): x[i],
                               setpoint=5.0, switch="S", freq=1 / Tp)
    r = p.simulate(b, 2e-3, engine="trbdf2", switch_fn=loop.switch_fn,
                   step_observer=loop.step_observer)
    up_s, up, dn_s, dn = _edges_sampled(r, b.switch_index_of("S"))
    assert up > 150 and dn > 150
    assert (up_s, dn_s) == (up, dn)
    assert r._trbdf2_stats["n_ctrl_ticks"] == 200


def test_the_pulse_cap_follows_the_recent_pulses():
    """An opaque gate caps the step at twice the narrowest pulse it has
    shown. That was a run-wide minimum: one 1 ns pulse at start-up held
    every later step at 2 ns (cl_boost_pi: 519,783 steps in 0.3 ms). It now
    follows each gate bit's last pulses, lifted between predicted toggles of
    a periodic direction; the quiet gap before the 10 ns PWM pulses keeps the
    run's minimum, so none of them is stepped over (the cap that lapsed
    there missed 4 of 99). The
    1 ns pulse sits where the start-up steps are short enough to see it
    (gate_pulse_min proves it was seen)."""
    Tp = 1e-5
    b = _tick_buck()
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        on = (1e-8 <= t < 1.1e-8) or (t >= 1e-5 and (t % Tp) < 1e-3 * Tp)
        m.set(k, bool(on))
        return m
    with pytest.warns(UserWarning, match="narrower than the spacing"):
        r = p.simulate(b, 1e-3, engine="trbdf2", switch_fn=fn)
    assert r._trbdf2_stats["gate_pulse_min"] < 2e-9
    t = np.asarray(r.times)
    assert np.diff(t[t > 0.5e-3]).max() > 1e-7      # 2e-9 before
    up_s, up, dn_s, dn = _edges_sampled(r, k)
    assert (up, dn) == (100, 100)                    # every edge landed
    assert r._trbdf2_stats["n_accept"] < 20_000
    # Once the rises prove periodic, the long phase is not capped: the
    # steady state costs what the same PWM with declared edges costs (the
    # run-wide 1 ns minimum: 2 ns steps, 250,000 in this window).
    ref = p.simulate(b, 1e-3, engine="trbdf2", switch_fn=p.make_pwm_switch_fn(
        1 / Tp, 1e-3, k, b.graph.num_switches))
    tr = np.asarray(ref.times)
    assert np.sum(t > 0.5e-3) <= 2 * np.sum(tr > 0.5e-3)


def test_a_gate_that_gets_faster_loses_no_pulse():
    """A 100 ns pulse every third period, then every period (pulse skipping
    resuming). The rises prove periodic at 3T and the cap lifts between
    them; the new pulses come where none is predicted, and only the gate
    samples inside the lifted step can see them: they stay half a pulse
    apart. Every edge, structurally."""
    Tp, w = 1e-5, 1e-7
    b = _tick_buck()
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        n = int(t // Tp)
        m.set(k, bool(n >= 1 and t - n * Tp < w
                      and (n % 3 == 0 or n * Tp >= 3e-4 - 1e-12)))
        return m
    r = p.simulate(b, 1e-3, engine="trbdf2", switch_fn=fn)
    # From the second pulse on: a pulse before any is seen can be stepped
    # over by any opaque-gate run (nothing to cap by yet).
    want = sum(1 for n in range(4, 100) if n % 3 == 0 or n * Tp >= 3e-4 - 1e-12)
    t = np.asarray(r.times)
    s = np.asarray(r.switch_states)[:, k]
    j = np.flatnonzero(s[1:] != s[:-1])
    j = j[t[j] > 3.5 * Tp]
    assert (int(np.sum(s[j + 1])), int(np.sum(~s[j + 1]))) == (want, want)
