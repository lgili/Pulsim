"""Variable-step engine robustness: opaque gates, held diodes, floating DC.

Three defects of ``engine='auto'`` / ``'trbdf2'`` found while polishing
2.0, each reproduced on the base of this change and pinned here against
the fixed-step engine:

E1  An opaque Python ``switch_fn`` was sampled only at t and t + h, and
    the edge bisection assumed ONE edge in [t, t + h]. On a long run the
    default ceiling (span/1000 = 20 µs on 20 ms) is two 100 kHz periods,
    so a step's ends fell on the same phase: a boost took 112 gate events
    instead of 3999 and ended at 590 V against 48.04 V, with no warning
    about the gates.

E2  The diode crossing detector read each step's START signal from the
    committed solve, which after a zero-time cascade is a solve under the
    PRE-cascade bits. A V_th = 0 bridge then ran half-wave (318.79 V
    against 322.06 V, D1/D3 never conducting) and a 1 mH line bridge held
    two diodes OFF while forward-biased by 45 V.

E3  A bridge whose DC side floats aborted with "factorize failed
    (numerically singular) ... dt=6e-14": the zero-time probe stamps
    470 µF as 1.6e10 S against the 4e-9 S the OFF diodes give the DC
    pair. ``engine='pwl'`` (dt = 1e-6) ran it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

T_SW = 1e-5    # 100 kHz


def _py_pwm(bit, n, duty, t_on=0.0):
    """A plain Python PWM: the engine sees it only where it calls it."""
    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(bit, t >= t_on and (t % T_SW) < duty * T_SW)
        return m
    return fn


def _run(b, t_end, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = p.simulate(b, t_end=t_end, **kw)
    msgs = [str(w.message) for w in caught]
    return r, msgs


def _avg(r, node, t_from):
    return p.time_average(np.asarray(r.times), r.v(node), t_from=t_from)


# ---------------------------------------------------------------------
# E1 — opaque switch_fn on long runs
# ---------------------------------------------------------------------

def _boost():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_inductor("L", "vin", "sw", 220e-6)
    b.add_mosfet("Q1", "sw", "gnd")
    b.add_diode("D1", "sw", "out", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 30.0)
    return b, _py_pwm(b.switch_index_of("Q1"), b.graph.num_switches, 0.5)


def test_opaque_pwm_on_a_long_run_lands_every_edge():
    """20 ms of a 100 kHz boost from a Python PWM. Before: 112 gate
    events, steps of 20 µs, 589.97 V. The fixed engine gives 48.0380 V
    (dt = 2e-8, last 5 ms); this engine 48.0335 V."""
    b, pwm = _boost()
    r, msgs = _run(b, 20e-3, switch_fn=pwm)
    st = r._trbdf2_stats
    assert r.engine_used == "trbdf2"
    assert st["n_gate_events"] == 3999
    assert st["gate_pulse_min"] == pytest.approx(5e-6, rel=1e-6)
    assert st["n_gate_undersampled"] == 0
    assert float(np.diff(np.asarray(r.times)).max()) <= 2 * 5e-6 * 1.001
    assert _avg(r, "out", 15e-3) == pytest.approx(48.038, abs=0.02)
    assert not [m for m in msgs if "gate pulse" in m], msgs


def test_opaque_three_phase_spwm_matches_the_fixed_engine():
    """10 kHz SPWM with 1 µs dead time behind a Python wrapper, 20 ms:
    1600 mask changes (counted by sampling the wrapper every 10 ns; the
    grid-exact SPWM puts the last edge exactly on t_end) and
    I_a 17.702 A rms against 17.700 A at dt = 1e-7."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
    v = p.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd",
                              out_a="a", out_b="b", out_c="c")
    for ph in "abc":
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
    spwm = p.make_three_phase_spwm_fn(
        10e3, 50.0, 0.8, p.ThreePhaseLegIndices(*v.switch_indices),
        b.graph.num_switches, 1e-6)
    r, _ = _run(b, 20e-3, switch_fn=lambda t: spwm(t))
    t = np.asarray(r.times)
    assert r._trbdf2_stats["n_gate_events"] == 1600
    assert p.time_rms(t, r.i("La"), t_from=15e-3) == pytest.approx(
        17.700, abs=0.01)


def test_a_step_spanning_several_edges_lands_on_the_first():
    """engine='trbdf2' with a step ceiling of 3.5 periods: every edge is
    still landed (199 over 1 ms at D = 0.3) and the steps are capped at
    twice the narrowest pulse (3 µs) once it is known. The output
    averages 7.4911 V against 7.4907 V at dt = 1e-8 and 5e-9."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "x", 1e3, 1e-9)
    b.add_resistor("R", "x", "y", 10.0)
    b.add_capacitor("C", "y", "gnd", 1e-6)
    b.add_resistor("RL", "y", "gnd", 100.0)
    r, _ = _run(b, 1e-3, engine="trbdf2", dt=3.5 * T_SW,
                switch_fn=_py_pwm(0, b.graph.num_switches, 0.3))
    st = r._trbdf2_stats
    assert st["n_gate_events"] == 199
    assert st["gate_pulse_min"] == pytest.approx(3e-6, rel=1e-6)
    assert _avg(r, "y", 0.5e-3) == pytest.approx(7.4907, abs=2e-3)


def test_pulses_that_start_late_are_found_and_the_gap_confessed():
    """The gate is idle for 5 ms, then a 100 kHz, D = 0.25 PWM starts.
    By then the step is at its 20 µs ceiling, two periods: the fixed
    sampling fractions alone phase-locked to the PWM and saw 2 of 3000
    edges (0.0037 V against 7.134 V). A sample at a walking fraction
    finds the train; the pulses before that are confessed."""
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 10.0)
        b.add_switch("S", "in", "x", 1e3, 1e-9)
        b.add_resistor("R", "x", "y", 10.0)
        b.add_capacitor("C", "y", "gnd", 1e-6)
        b.add_resistor("RL", "y", "gnd", 100.0)
        return b, _py_pwm(0, b.graph.num_switches, 0.25, t_on=5e-3)

    b, pwm = build()
    r, msgs = _run(b, 20e-3, switch_fn=pwm)
    st = r._trbdf2_stats
    assert 2900 <= st["n_gate_events"] <= 3000
    assert st["n_gate_undersampled"] >= 1
    assert any("gate pulse of 2.5e-06 s" in m for m in msgs), msgs
    b2, pwm2 = build()
    ref = p.simulate(b2, t_end=20e-3, dt=1e-7, switch_fn=pwm2)
    assert _avg(r, "y", 15e-3) == pytest.approx(_avg(ref, "y", 15e-3),
                                                 abs=0.01)


# ---------------------------------------------------------------------
# E2 — diode commutations on the variable-step engine
# ---------------------------------------------------------------------

BRIDGE = ("BR__D1", "BR__D2", "BR__D3", "BR__D4")


def _bridge(V_th, l_line=0.0):
    """230 V rms behind 0.1 ohm (+ optional line inductance), bridge,
    470 µF || 1 kOhm."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "bb", 0.0, 325.27, 50.0)
    if l_line > 0.0:
        b.add_resistor("Rs", "s", "sr", 0.1)
        b.add_inductor("Lline", "sr", "a", l_line)
    else:
        b.add_resistor("Rs", "s", "a", 0.1)
    p.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="bb", dc_pos="dc",
                           dc_neg="gnd", V_th=V_th)
    b.add_capacitor("C", "dc", "gnd", 470e-6)
    b.add_resistor("RL", "dc", "gnd", 1e3)
    return b


def _diode_report(r):
    """(per-diode time-averaged current over 60-100 ms, peak, and the
    RuntimeWarnings reading res.i raised)."""
    t = np.asarray(r.times)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cur = {d: np.asarray(r.i(d)) for d in BRIDGE}
    avg = {d: p.time_average(t, i, t_from=0.06) for d, i in cur.items()}
    pk = {d: float(i[t > 0.06].max()) for d, i in cur.items()}
    return avg, pk, [str(w.message) for w in caught
                     if issubclass(w.category, RuntimeWarning)]


def test_v_th_0_bridge_runs_full_wave():
    """Before: 318.79 V, BR__D1 / BR__D3 never conducting, "OFF while
    forward-biased" on 52 samples. Now 322.057 V against 322.059 V
    (dt = 1e-6), every diode carrying 0.1606 A against 0.1610 A, peaks
    7.95 A against 7.92 A (a chatter-guard step the length of the
    controller's h had carried D1 OFF to 1.93 V of forward bias: an
    18.9 A turn-on)."""
    r, msgs = _run(_bridge(0.0), 0.1)
    ref, _ = _run(_bridge(0.0), 0.1, dt=1e-6)
    assert _avg(r, "dc", 0.08) == pytest.approx(_avg(ref, "dc", 0.08),
                                                abs=0.02)
    avg, pk, rule = _diode_report(r)
    avg_ref, pk_ref, _ = _diode_report(ref)
    assert not rule, rule
    assert not [m for m in msgs if "commutation rule forbids" in m]
    for d in BRIDGE:
        assert avg[d] == pytest.approx(avg_ref[d], rel=0.01), d
        assert pk[d] <= 1.02 * pk_ref[d], (d, pk[d], pk_ref[d])


def test_line_inductance_bridge_holds_no_diode_against_its_rule():
    """V_th = 0.7 with 1 mH of line: the engine used to hold BR__D2 /
    BR__D4 OFF while forward-biased by 45.1 V at t = 75.92 ms. Now no
    sample breaks the rule and V_dc agrees with dt = 1e-6 to 0.04 V
    (321.807 / 321.845 V)."""
    r, msgs = _run(_bridge(0.7, l_line=1e-3), 0.1)
    ref, _ = _run(_bridge(0.7, l_line=1e-3), 0.1, dt=1e-6)
    _, _, rule = _diode_report(r)
    assert not rule, rule
    assert not [m for m in msgs if "commutation rule forbids" in m]
    assert _avg(r, "dc", 0.08) == pytest.approx(_avg(ref, "dc", 0.08),
                                                abs=0.05)


@pytest.mark.parametrize("v_th, ref_v", [(0.7, 321.865), (0.0, 321.870)])
def test_line_inductance_bridge_completes_a_long_run(v_th, ref_v):
    """Past 0.3 s the V_th = 0.7 / 1 mH bridge ground at h_min on one
    commutation (all bridge bits OFF, a 1 ps L*g_off mode the LTE read
    as error at every h) and aborted on max_steps after 73 s; V_th = 0
    did the same at 0.146 s. Now 0.5 s takes about 5k steps and lands
    on dt = 1e-6 (ref_v, measured: 321.865 / 321.870 V) to 0.04 V."""
    r, _ = _run(_bridge(v_th, l_line=1e-3), 0.5)
    assert r._trbdf2_stats["n_accept"] < 20_000
    assert _avg(r, "dc", 0.48) == pytest.approx(ref_v, abs=0.05)


def test_v_th_0_line_inductance_bridge_is_not_slow():
    """V_th = 0 behind 1 mH: 122,597 steps / 64k chatter breaks for
    0.1 s on the previous revision; now about 1.5k, on dt = 1e-6's
    321.850 V to 0.003 V."""
    r, _ = _run(_bridge(0.0, l_line=1e-3), 0.1)
    assert r._trbdf2_stats["n_accept"] < 5000
    assert _avg(r, "dc", 0.08) == pytest.approx(321.850, abs=0.05)


def _snubbed_boost():
    """24 V boost, 220 µH, Q1 g_on 50 S at 100 kHz D = 0.5, 100 nF
    across D1 (V_th 0.7), 100 µF || 30 ohm."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_inductor("L", "vin", "sw", 220e-6)
    b.add_switch("Q1", "sw", "gnd", 50.0, 1e-9)
    b.add_diode("D1", "sw", "out", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cs", "sw", "out", 100e-9)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 30.0)
    return b, _py_pwm(b.switch_index_of("Q1"), b.graph.num_switches, 0.5)


def test_snubbed_boost_tracks_the_fixed_engine():
    """The task's headline E2c case. engine='auto' without dt ended at
    27.59 V at 3 ms on the base of this change; engine='pwl' at
    dt = 1e-8 ends at 54.48 V (measured; too slow to rerun here)."""
    b, pwm = _snubbed_boost()
    r, _ = _run(b, 3e-3, switch_fn=pwm)
    assert r.engine_used == "trbdf2"
    assert float(r.v("out")[-1]) == pytest.approx(54.48, abs=0.1)


def test_sliding_mode_warning_names_the_engine_that_was_asked():
    """The boundary-ride warning said engine='auto' when engine='trbdf2'
    had been requested by name."""
    b, pwm = _snubbed_boost()
    _, msgs = _run(b, 6e-3, engine="trbdf2", switch_fn=pwm)
    ride = [m for m in msgs if "conduction boundary" in m]
    assert ride and all(m.startswith("simulate(engine='trbdf2')")
                        for m in ride), msgs


# ---------------------------------------------------------------------
# E3 — a floating DC side
# ---------------------------------------------------------------------

def _floating_dc_bridge():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 325.0, 50.0)
    b.add_resistor("Rs", "ac", "acx", 1e-3)
    p.add_bridge_rectifier(b, "BR", ac_a="acx", ac_b="gnd", dc_pos="dp",
                           dc_neg="dn")
    b.add_capacitor("C", "dp", "dn", 470e-6)
    b.add_resistor("RL", "dp", "dn", 1e3)
    return b


def test_floating_dc_side_runs_without_dt():
    """Before: SimulationAborted / RuntimeError "factorize failed
    (numerically singular) for mask 0b0000 N=4 (dt=6e-14) ... node dn".
    Now the probe step grows to 6e-10 s once and the run gives
    321.851 V across the rails against 321.852 V at dt = 1e-6."""
    r, msgs = _run(_floating_dc_bridge(), 0.06)
    ref, _ = _run(_floating_dc_bridge(), 0.06, dt=1e-6)
    assert r.engine_used == "trbdf2"
    assert 0.0 < r._trbdf2_stats["probe_floor_used"] <= 0.06 * 1e-6
    # The floor is evidence of a floating subnet: named, with the fix.
    flt = [m for m in msgs if "reach the rest of the circuit" in m]
    assert len(flt) == 1 and "node(s) dp, dn" in flt[0], msgs
    assert "ground a rail" in flt[0]

    def vdc(res):
        t = np.asarray(res.times)
        return p.time_average(t, res.v("dp") - res.v("dn"), t_from=0.04)

    assert vdc(r) == pytest.approx(vdc(ref), abs=0.01)


def test_floating_dc_side_leakage_is_not_a_commutation():
    """At t_end = 0.04 an ON bridge diode carrying g_off-scale leakage
    (-1.6e-7 A) was re-landed on every step: 318,068 steps, 317,101
    diode events. Now about 1.1k steps."""
    r, _ = _run(_floating_dc_bridge(), 0.04)
    assert r._trbdf2_stats["n_accept"] < 5000


def test_grounded_dc_side_raises_no_floating_warning():
    """The control: the same bridge with dc_neg grounded needs no probe
    floor and so names no floating subnet."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 325.0, 50.0)
    b.add_resistor("Rs", "ac", "acx", 1e-3)
    b.add_resistor("Rb", "acb", "gnd", 1e-3)
    p.add_bridge_rectifier(b, "BR", ac_a="acx", ac_b="acb", dc_pos="dp",
                           dc_neg="gnd")
    b.add_capacitor("C", "dp", "gnd", 470e-6)
    b.add_resistor("RL", "dp", "gnd", 1e3)
    r, msgs = _run(b, 0.02)
    assert r._trbdf2_stats["probe_floor_used"] == 0.0
    assert not [m for m in msgs if "reach the rest of the circuit" in m]


def test_floating_subnet_finder():
    """Union-find over the non-switch branches: the DC pair of a
    floating bridge is named, a buck's switch node (tied by its
    inductor) is not."""
    assert p._floating_subnets(_floating_dc_bridge()) == [["dp", "dn"]]
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 12.0)
    b.add_mosfet("Q", "in", "sw")
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "o", 1e-4)
    b.add_resistor("R", "o", "gnd", 1.0)
    assert p._floating_subnets(b) == []
    # Nodes between series switches and an ideal transformer's primary
    # hold no stored energy: not floating rails (were all named).
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_diode("D", "a", "b", 1e3, 1e-9)
    b.add_mosfet("M", "b", "c")
    b.add_mosfet_with_body_diode("MB", "c", "d")
    b.add_igbt("Q", "d", "e")
    b.add_switch("S", "e", "f", 1e3, 1e-9)
    b.add_ideal_transformer("T", "f", "gnd", "g", "h", 2.0)
    b.add_resistor("R", "h", "gnd", 1.0)
    assert p._floating_subnets(b) == []


# ---------------------------------------------------------------------
# E1 limits — what the gate sampling can and cannot promise
# ---------------------------------------------------------------------

def _one_shot(t_on, width):
    """10 V through a 100 S switch, 1 ohm into 1 µF || 1 kOhm; the
    switch closes ONCE, for `width` seconds from `t_on`."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "x", 100.0, 1e-9)
    b.add_resistor("R", "x", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-6)
    b.add_resistor("RL", "c", "gnd", 1e3)
    n, s = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(s, t_on <= t < t_on + width)
        return m
    return b, fn


@pytest.mark.parametrize("k", range(0, 20, 3))
def test_one_shot_pulse_a_quarter_of_the_ceiling_wide_is_landed(k):
    """20 ms run: the default ceiling is 20 µs and switch_fn is sampled
    every 5 µs, so a 5 µs one-shot pulse is landed at every phase
    (v_C peaks at 9.92 V)."""
    b, fn = _one_shot(10e-3 + k * 0.37e-6 * 7.3, 5e-6)
    r, _ = _run(b, 20e-3, switch_fn=fn)
    assert r._trbdf2_stats["n_gate_events"] == 2
    assert float(r.v("c").max()) > 9.0


def test_one_shot_pulse_narrower_than_a_quarter_of_the_ceiling_needs_dt():
    """The documented limit: an isolated 2 µs pulse under the 20 µs
    default ceiling falls between samples at some phases (11 of 20
    measured) and NOTHING warns, because nothing saw it. dt = 4 µs (a
    1 µs sampling spacing) lands it: 8.6149 V peak against 8.6154 V on
    engine='pwl' at dt = 1e-7."""
    t_on = 10e-3 + 2 * 0.37e-6 * 7.3      # a phase that misses it
    b, fn = _one_shot(t_on, 2e-6)
    r, msgs = _run(b, 20e-3, switch_fn=fn)
    assert r._trbdf2_stats["n_gate_events"] == 0
    assert float(r.v("c").max()) < 1e-3
    assert not [m for m in msgs if "gate pulse" in m], msgs
    b, fn = _one_shot(t_on, 2e-6)
    r, _ = _run(b, 20e-3, engine="trbdf2", dt=4e-6, switch_fn=fn)
    assert r._trbdf2_stats["n_gate_events"] == 2
    assert float(r.v("c").max()) == pytest.approx(8.6154, abs=0.01)


def test_smooth_spwm_over_many_cycles_raises_no_pulse_warning():
    """0.2 s of a 10 kHz SPWM VSI: every one of its 16000 edges is
    landed and each pulse is sampled at a spacing capped by the
    previous one. Compared with the run-wide largest spacing (12.5 µs,
    from before the first pulse) 22 gradually-narrowing pulses used to
    count as undersampled and raise the pulse warning."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
    v = p.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd",
                              out_a="a", out_b="b", out_c="c")
    for ph in "abc":
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
    spwm = p.make_three_phase_spwm_fn(
        10e3, 50.0, 0.8, p.ThreePhaseLegIndices(*v.switch_indices),
        b.graph.num_switches, 1e-6)
    r, msgs = _run(b, 0.2, switch_fn=lambda t: spwm(t))
    st = r._trbdf2_stats
    assert st["n_gate_events"] == 16000
    assert st["n_gate_undersampled"] == 0
    assert not [m for m in msgs if "gate pulse" in m], msgs


def test_held_diode_warning_names_a_floating_subnet():
    """The floating bridge with a 1 mH DC choke learns no probe floor,
    but a missed commutation swings dq/dn to a 76.7 kV common-mode
    sample: the held-diode warning names the subnet and the cure (with
    dc_neg grounded no diode is held)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0, 50.0)
    b.add_resistor("Rs", "s", "a", 1e-3)
    p.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="gnd", dc_pos="dp",
                           dc_neg="dn")
    b.add_inductor("Lf", "dp", "dq", 1e-3)
    b.add_capacitor("C", "dq", "dn", 470e-6)
    b.add_resistor("RL", "dq", "dn", 1e3)
    _, msgs = _run(b, 0.06)
    held = [m for m in msgs if "commutation rule forbids" in m]
    assert held and "Node(s) dp, dn, dq reach the rest" in held[0], msgs
