"""add-complementarity-mode-selection tasks 6 and 4: ``mode_selection='lcp'`` and the default guard on dsed.

'lcp' replaces the settle: the LCP on the diode ports decides the mode at t = 0 and at every event
(DECISIONS.md #13). The default keeps the settle and judges each settled mode; an inconsistent one
is decided by the LCP instead and recorded in ``diagnostics['mode_guard']`` (DECISIONS.md #15).
Since the number-changing release the default ('auto') IS 'lcp' on dsed for a circuit with a PWL
diode; the guarded settle these tests compare against is ``mode_selection='legacy'``.

References: trbdf2 at rtol 1e-10 (probes/guard_survey.out, probes/extra_circuits.out) or the
prototype's exact LCP simulation (test_mode_selection_today.py). The bridges compare p - n: their
DC side floats and the common mode is a gauge the engines set differently (design D10).
"""
import warnings

import numpy as np
import pytest

import pulsim as p
import pulsim._dsed_dispatch as DD
from tests.oracles import lcp_pulsim_bridge as B

T = 1e-5
GON, GOFF = 1e3, 1e-9


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _boost(R, vth=0.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", GON, GOFF)
    b.add_diode("D1", "sw", "out", GON, GOFF, vth)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", R)
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"), b.graph.num_switches, 0.0)


def _buck(L=100e-6, vth=0.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF, vth)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("R", "vout", "gnd", 50.0)
    return b, p.make_pwm_switch_fn(1 / T, 0.3, b.switch_index_of("S"), b.graph.num_switches, 0.0)


def _half_wave_rl():
    """benchmarks/circuits/diodes/diode_half_wave_rl_load (a v1 netlist the 2.0 loader refuses)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "in", "gnd", 0.0, 10.0, 50.0)
    b.add_diode("D1", "in", "out", 200.0, 1e-6)
    b.add_resistor("R1", "out", "nl", 10.0)
    b.add_inductor("L1", "nl", "gnd", 1e-3)
    return b


def _sim(b, t_end, fn=None, guard=True, **kw):
    """guard=False: the settle alone, with BOTH guards off (the kernel's
    MODE_GUARD and the stored-sample diode-law check, which would name
    the wrong mode instead of returning it)."""
    if fn is not None:
        kw["switch_fn"] = fn
    # legacy subject: "the default" here is the guarded settle (pre-release default)
    kw.setdefault("mode_selection", "legacy")
    kw.setdefault("output_rtol", 0)  # release: the subject is the engine, not the default output rule's samples
    saved = DD.MODE_GUARD, DD.DIODE_LAW_REL
    DD.MODE_GUARD = guard
    if not guard:
        DD.DIODE_LAW_REL = 0.0
    try:
        return p.simulate(b, t_end=t_end, engine="dsed", **kw)
    finally:
        DD.MODE_GUARD, DD.DIODE_LAW_REL = saved


def _end(r, node):
    if "-" in node:
        a, c = node.split("-")
        return float(np.asarray(r.v(a))[-1] - np.asarray(r.v(c))[-1])
    return float(np.asarray(r.v(node))[-1])


def _tw_mean(r, node):
    t, v = np.asarray(r.times), np.asarray(r.v(node))
    return float(np.trapezoid(v, t) / (t[-1] - t[0]))


def test_lcp_buck_at_rest_stays_at_zero():
    b, _ = _buck()
    r = _sim(b, 1e-6, lambda t: p.SwitchStateMask(b.graph.num_switches), mode_selection="lcp")
    assert abs(_end(r, "sw")) < 1e-9
    assert r.diagnostics["mode_selection"]["lcp_count"] >= 1


def test_lcp_dcm_buck_4000_periods_matches_the_prototype():
    b, fn = _buck()
    r = _sim(b, 4000 * T, fn, mode_selection="lcp")
    assert _end(r, "vout") == pytest.approx(8.996246971827961, abs=4e-7)   # measured +1.2e-7
    assert r.diagnostics["mode_selection"]["max_pivots_per_event"] <= 2


# trbdf2 rtol 1e-10, 400 T; the settle and 'lcp' agree to the last digit here (no wrong event)
L_DELTA_REF = {(1e-5, 1): 9.100180227, (1e-5, -1): 9.100245895, (1e-4, 1): 9.099884742,
               (1e-4, -1): 9.100541421, (1e-2, 1): 9.067580214, (1e-2, -1): 9.133251029}


@pytest.mark.parametrize("delta,sign", list(L_DELTA_REF))
def test_lcp_dcm_buck_with_perturbed_inductance_completes(delta, sign):
    b, fn = _buck(L=100e-6 * (1 + sign * delta))
    r = _sim(b, 400 * T, fn, mode_selection="lcp")
    assert _end(r, "vout") == pytest.approx(L_DELTA_REF[(delta, sign)], abs=3e-6)   # measured +1.1e-6
    assert r.n_events == 1178


# R -> (trbdf2 rtol 1e-10 end v(out), events settle-only, events 'lcp')
BOOST = {20.0: (23.786198209, 852, 856), 50.0: (23.806428021, 969, 978), 500.0: (44.721664927, 1133, 1157)}


@pytest.mark.parametrize("R", list(BOOST))
def test_lcp_and_guard_fix_the_zero_threshold_boost(R):
    """With V_th = 0 the settle kept D1 ON through the DCM current zero (-0.86 / -1.07 / -1.18 A
    at R = 20 / 50 / 500 ohm), and with the guard off it refused there (DECISIONS #26). Since
    exact engine session 11 decisions at a located diode event are read on the candidate mode's slow manifold (its parasitic transient settled first),
    so the settle alone, the guarded default and 'lcp' take the same events to the last bit,
    with no repair."""
    ref, ev_settle, ev_lcp = BOOST[R]
    b, fn = _boost(R)
    r_set = _sim(b, 400 * T, fn, guard=False)
    r_lcp = _sim(b, 400 * T, fn, mode_selection="lcp")
    r_def = _sim(b, 400 * T, fn)
    assert _end(r_lcp, "out") == pytest.approx(ref, abs=2e-5)          # measured -1.5e-6 .. -1.24e-5
    assert r_lcp.n_events == ev_lcp
    assert np.array_equal(np.asarray(r_lcp.states), np.asarray(r_def.states))
    assert np.array_equal(np.asarray(r_set.states), np.asarray(r_def.states))
    assert r_def.diagnostics["mode_guard"]["repairs"] == 0


def test_half_wave_rl_rectifier_matches_the_exact_truth():
    """Against the exact truth (truth2 benchmarks/tools/exact_truth.py, diode_half_wave_rl_load,
    50 digits): D1 ON from t = 0, OFF at 10.0999172, 30.0999172, 50.0999172 ms, ON at 20 and
    40 ms + 1 us; mean v(out) 3.18069264315 V. The settle once held D1 OFF at t = 0 (1.06 V
    low; fixed by the significance ladder) and then kept it ON at the 30.0999 ms turn-off: the
    OFF probe read the root's residual (5e-15 A through g_off: +5e-9 V for ~1 ns), which is the
    located root's own error, now part of the ladder's bound. The settle alone (no guard, no
    law check) decides every event; the guard repairs nothing."""
    b = _half_wave_rl()
    truth_off = [10.0999172e-3, 30.0999172e-3, 50.0999172e-3]
    truth_on = [20.000001e-3, 40.000001e-3]
    for guard in (False, True):
        r = _sim(b, 60e-3, guard=guard)
        assert _tw_mean(r, "out") == pytest.approx(3.18069264315096, abs=1e-6)   # measured -1.1e-8
        et, ty = np.asarray(r.event_times), np.asarray(r.event_types)
        assert len(et) == 5
        assert et[ty == 2] == pytest.approx(truth_off, abs=1e-9)
        assert et[ty == 1] == pytest.approx(truth_on, abs=1e-9)
        on = np.asarray(r.switch_states)[:, b.switch_index_of("D1")].astype(bool)
        vd = np.asarray(r.v("in")) - np.asarray(r.v("out"))
        assert vd[~on].max() < 1e-3
    assert r.diagnostics["mode_guard"]["repairs"] == 0
    r_lcp = _sim(b, 60e-3, mode_selection="lcp")
    assert np.array_equal(np.asarray(r_lcp.states), np.asarray(r.states))


def test_lcp_double_crossing_with_a_band_switches_at_the_exact_instants():
    """test_double_crossing_inside_one_step_is_seen's tank at V_th = 1e-3: the correct conduction
    windows are 1.2536-1.8877 us and 7.537-8.170 us. (V_th = 0, where the turn-off inside the step
    after the turn-on used to be missed by the locator, DECISIONS.md #16 and #19: see
    test_double_crossing_inside_one_step_is_seen.)"""
    b = p.CircuitBuilder()
    b.add_capacitor("C", "t", "gnd", 100e-9)
    b.add_inductor("L", "t", "gnd", 10e-6, i0=-1.0)
    b.add_diode("D", "t", "k", GON, GOFF, 1e-3)
    b.add_resistor("R", "k", "r", 1e3)
    b.add_voltage_source("Vr", "r", "gnd", 9.5)
    r = _sim(b, 10e-6, mode_selection="lcp")
    t = np.asarray(r.times)
    on = np.asarray(r.switch_states)[:, b.switch_index_of("D")].astype(int)
    edges = t[np.flatnonzero(np.diff(on))]          # a row holds the mode of the interval ending at it
    assert edges == pytest.approx([1.2536e-6, 1.8877e-6, 7.5374e-6, 8.1703e-6], abs=1e-10)


# family, V_th, t_end, node, trbdf2 rtol 1e-9 end, tolerance (measured |error| x ~3)
BRIDGES = [
    ("bridge1_vf07", 0.0, 0.2, "p-n", 93.838054911, 3e-6),
    ("bridge1_vf0", 0.0, 0.2, "p-n", 95.171116006, 3e-6),
    ("bridge1_vf0", 1e-2, 0.2, "p-n", 95.171116116, 3e-6),
    ("bridge3_cfilter", 0.0, 0.2, "p-n", 171.787833603, 1e-7),
    ("bridge3_cfilter", 1e-2, 0.2, "p-n", 171.787833603, 1e-7),
]


@pytest.mark.parametrize("fam,vth,t_end,node,ref,tol", BRIDGES)
def test_lcp_bridges_match_trbdf2(fam, vth, t_end, node, ref, tol):
    b = B.FAMILIES[fam].builder(vth)
    r = _sim(b, t_end, mode_selection="lcp")
    assert _end(r, node) == pytest.approx(ref, abs=tol)
    ms = r.diagnostics["mode_selection"]
    assert ms["max_pivots_per_event"] <= 2 * (4 if fam.startswith("bridge1") else 6)   # 2k (D9)
    if fam == "bridge3_cfilter":
        assert ms["leak_pins"] > 0     # the floating DC bus is held by one diode's leak current


def test_lcp_series_lc_tank_into_a_diode():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_inductor("L", "in", "a", 100e-6)
    b.add_diode("D", "a", "out", GON, GOFF)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    r = _sim(b, 1e-4, mode_selection="lcp")
    r_ref = p.simulate(b, t_end=1e-4, engine="trbdf2", rtol=1e-10)
    assert _end(r, "out") == pytest.approx(_end(r_ref, "out"), abs=1e-5)
    assert _end(r, "out") > 19.0       # the tank overshoots to ~2 V_in and the diode holds it


def test_lcp_hysteresis_half_wave_rectifier():
    """V_th = 0.5 V is a hysteresis band (V_f = 0): the LCP holds the diode OFF inside it and
    releases it at the threshold; the run equals the default's and trbdf2's."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 50.0)
    b.add_diode("D", "in", "out", GON, GOFF, 0.5)
    b.add_resistor("R", "out", "gnd", 100.0)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    r, r0 = _sim(b, 0.04, mode_selection="lcp"), _sim(b, 0.04)
    assert np.array_equal(np.asarray(r.states), np.asarray(r0.states))
    assert _end(r, "out") == pytest.approx(2.345320201, abs=1e-5)       # trbdf2 rtol 1e-10
    assert r.diagnostics["mode_selection"]["hysteresis_released"] >= 1


def test_lcp_banded_dcm_buck_equals_the_default():
    b, fn = _buck(vth=1e-2)
    r0, r1 = _sim(b, 400 * T, fn), _sim(b, 400 * T, fn, mode_selection="lcp")
    assert np.array_equal(np.asarray(r0.times), np.asarray(r1.times))
    assert np.array_equal(np.asarray(r0.states), np.asarray(r1.states))
    assert r1.diagnostics["mode_selection"]["lcp_count"] > r1.n_events      # the LCP decided


@pytest.mark.parametrize("fam,vth,t_end", [("dcm_buck", 0.0, 400 * T), ("bridge3_cfilter", 0.0, 0.06),
                                           ("boost_500", 1e-3, 100 * T)])
def test_guard_leaves_consistent_runs_bit_identical(fam, vth, t_end):
    b = B.FAMILIES[fam].builder(vth)
    sws = [a[0] for k, *a in B.FAMILIES[fam].parts if k == "S"]
    fn = None
    if sws:
        fn = p.make_pwm_switch_fn(1 / T, 0.5 if fam.startswith("boost") else 0.3,
                                  b.switch_index_of(sws[0]), b.graph.num_switches, 0.0)
    r_off, r_on = _sim(b, t_end, fn, guard=False), _sim(b, t_end, fn)
    assert np.array_equal(np.asarray(r_off.times), np.asarray(r_on.times))
    assert np.array_equal(np.asarray(r_off.states), np.asarray(r_on.states))
    g = r_on.diagnostics["mode_guard"]
    assert g["repairs"] == 0 and g["checks"] == r_on.n_events + 1
    g_off = r_off.diagnostics["mode_guard"]                 # judged, nothing to refuse
    assert (g_off["checks"], g_off["repairs"]) == (g["checks"], 0)


def _floating_bridge(source):
    """A four-diode bridge whose AC side floats (1 Mohm to ground) and whose DC bus is tied to it
    by 1 Mohm and the g_off leaks only: the bus common mode is an island with a 1 uS pivot."""
    b = p.CircuitBuilder()
    if source == "sine":
        b.add_sine_voltage_source("V", "s", "r", 0.0, 50.0, 50.0)
    else:
        b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, 50.0)
        b.add_ideal_transformer("T1", "a", "gnd", "s", "r", 0.5)
    b.add_resistor("Rr", "r", "gnd", 1e6)
    for name, an, ca in (("D1", "s", "p"), ("D2", "n", "s"), ("D3", "r", "p"), ("D4", "n", "r")):
        b.add_diode(name, an, ca, GON, GOFF)
    b.add_capacitor("C", "p", "n", 100e-6)
    b.add_resistor("R", "p", "n", 100.0)
    b.add_resistor("Rn", "n", "r", 1e6)
    return b, {"D1": ("s", "p"), "D2": ("n", "s"), "D3": ("r", "p"), "D4": ("n", "r")}


@pytest.mark.parametrize("source", ["sine", "transformer"])
@pytest.mark.parametrize("h_e_rel", [1e-7, 1e-9, 1e-10])
def test_lcp_bridge_on_a_floating_source_is_not_decided_by_rounding(source, h_e_rel, monkeypatch):
    """DECISIONS.md #21: 'lcp' ended at 11.726610 V with D2 and D3 never ON and D2 blocking
    +49.86 V, and the answer flipped with H_E_REL (wrong at 1e-10, 1e-9, 1e-7). The port model's
    history term C*v_C/h_e (1.6e6 A) left 2e-10 A of rounding on the island's 1 uS pivot: 2e-4 V
    on D2's gap at its turn-on root, where h_e moves it 3e-5 V. Reference: pwl at dt = 1e-7,
    31.876241232 V; the default (settle + guard) 31.876241228 V."""
    monkeypatch.setattr(DD, "H_E_REL", h_e_rel)
    b, pairs = _floating_bridge(source)
    r = _sim(b, 0.04, mode_selection="lcp")
    assert _end(r, "p-n") == pytest.approx(31.876241232, abs=2e-8)     # measured -4e-9
    sw = np.asarray(r.switch_states).astype(bool)
    for d, (a, c) in pairs.items():
        on = sw[:, b.switch_index_of(d)]
        v = np.asarray(r.v(a)) - np.asarray(r.v(c))
        assert on.any(), d                                              # D2 and D3 never conducted
        assert v[~on].max() < 1e-9, d                                   # D2 blocked +49.86 V
    if h_e_rel == DD.H_E_REL == 1e-7:
        # Not bit-identical to the default since F-4-3 (group C): at D4's turn-off (5.98 ms) the
        # LCP decides D1 OFF (reverse-biased, -3.6e-6 V or less, no tie) and the band-tie release
        # no longer reopens it; the default keeps D1 ON for a few samples at -7.8e-11 A.
        r0 = _sim(b, 0.04)
        assert _end(r, "p-n") == pytest.approx(_end(r0, "p-n"), rel=1e-12)


def _bridge_bank(n, ls=0.2e-3):
    """n three-phase diode bridges, each on its own 200 uF || 50 ohm bus behind 0.2 mH line
    inductors, 100 V 50 Hz, the bridges 6 deg apart."""
    b = p.CircuitBuilder()
    for k in range(n):
        for ph, ang in (("a", 0.0), ("b", -120.0), ("c", 120.0)):
            b.add_sine_voltage_source(f"V{k}{ph}", f"s{k}{ph}", "gnd", 0.0, 100.0, 50.0,
                                      np.deg2rad(ang + 6.0 * k))
            b.add_inductor(f"L{k}{ph}", f"s{k}{ph}", f"x{k}{ph}", ls)
            b.add_diode(f"Dt{k}{ph}", f"x{k}{ph}", f"p{k}", 1e3, 1e-9, 0.0)
            b.add_diode(f"Db{k}{ph}", f"n{k}", f"x{k}{ph}", 1e3, 1e-9, 0.0)
        b.add_capacitor(f"C{k}", f"p{k}", f"n{k}", 200e-6)
        b.add_resistor(f"R{k}", f"p{k}", f"n{k}", 50.0)
        b.add_resistor(f"Rg{k}", f"n{k}", "gnd", 1e6)
    return b


@pytest.mark.parametrize("mode", [None, "lcp"])
def test_a_widened_pass_does_not_release_a_diode_before_its_own_turn_off(mode):
    """DECISIONS #24. Two bridge banks over 20 ms (h_e = 2 ns): at a commutation the widened
    pass (1e4*h_e = 20 us) saw a conducting diode's current reach zero inside the window, read
    it as a contradiction, released the diode and cut it 11 us early, 0.128 V off pwl. The
    widening is now capped at half the decided mode's next event: within 1e-3 V of pwl
    extrapolated in dt (measured 3.3e-4 V), and the cap fires."""
    t_end, n = 0.02, 2
    tg = np.linspace(0.0, t_end, 2001)

    def buses(r):
        return [np.interp(tg, r.times, np.asarray(r.v(f"p{k}")) - np.asarray(r.v(f"n{k}")))
                for k in range(n)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b2 = (buses(p.simulate(_bridge_bank(n), t_end=t_end, engine="pwl", dt=dt))
                 for dt in (5e-7, 2.5e-7))
        # None: the guarded settle (mode_selection='legacy'; 'auto' is 'lcp' since the release)
        kw = {"mode_selection": "legacy" if mode is None else mode}
        r = p.simulate(_bridge_bank(n), t_end=t_end, engine="dsed", output_rtol=0, **kw)
    ref = [2 * y - x for x, y in zip(a, b2)]
    dev = max(np.abs(v - e).max() for v, e in zip(buses(r), ref))
    assert dev <= 1e-3, dev
    d = r.diagnostics
    sel = d["mode_selection"] if mode == "lcp" else d["mode_guard"]["lcp"]
    assert sel["capped"] > 0


def _paralleled_bank(n):
    b = p.CircuitBuilder()
    for k in range(n):
        for ph, ang in (("a", 0.0), ("b", -120.0), ("c", 120.0)):
            b.add_sine_voltage_source(f"V{k}{ph}", f"s{k}{ph}", "gnd", 0.0, 100.0, 50.0,
                                      np.deg2rad(ang + 6.0 * k))
            b.add_inductor(f"L{k}{ph}", f"s{k}{ph}", f"x{k}{ph}", 0.2e-3)
            b.add_diode(f"Dt{k}{ph}", f"x{k}{ph}", "p", 1e3, 1e-9, 0.0)
            b.add_diode(f"Db{k}{ph}", "n", f"x{k}{ph}", 1e3, 1e-9, 0.0)
    b.add_capacitor("C", "p", "n", 2e-3)
    b.add_resistor("R", "p", "n", 5.0)
    b.add_resistor("Rg", "n", "gnd", 1e6)
    return b


@pytest.mark.parametrize("n", [3, 5])
def test_paralleled_bridges_run_their_cluster_modes_on_the_pade_tier(n):
    """add-exact-propagator-strategy D2 (exact engine 7): identical paralleled bridges reach
    modes whose eigenpairs miss the defect bar (a degenerate cluster near -5 /s: defect 3.3e-7
    with three banks, backward error 3.8 u). Those modes take the Pade tier, accepted on its
    own estimated error (two scalings, a u-perturbed matrix; max over h_max/2^k): measured
    2.8e-11 (three) and 4.1e-11 (five), the true error of one such mode against a 60-digit
    exponential 0.9e-11 / 1.7e-11. Bus voltage against pwl+split at dt = 1e-7 (dt/2 moves it
    1.1e-4 V with three banks, 1.3e-6 V with five): 1.1e-4 V, the same offset two banks show
    on eigenpairs alone (test_a_settle_that_hits_its_round_cap_is_decided_by_the_lcp)."""
    t_end = 0.02
    tg = np.linspace(0.0, t_end, 2001)

    def bus(r):
        return np.interp(tg, r.times, np.asarray(r.v("p")) - np.asarray(r.v("n")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(_paralleled_bank(n), t_end=t_end, engine="dsed", output_rtol=0)
        ref = p.simulate(_paralleled_bank(n), t_end=t_end, engine="pwl", dt=1e-7,
                         discontinuity_restart=True, enable_substep_state_correction=True)
    rows = r.exact["masks"]
    # which modes miss the eigenpair bar depends on the eigensolver's basis inside the
    # cluster; whatever their tier, each stated its bar
    assert rows and all(m["tier"] in (1, 2) for m in rows)
    assert all(m["defect"] <= 1e-10 for m in rows if m["tier"] == 1)
    assert all(m["pade_defect"] <= 1e-10 for m in rows if m["tier"] == 2)
    assert np.abs(bus(r) - bus(ref)).max() <= 1e-3


@pytest.mark.parametrize("cap", [64, 1])
def test_a_settle_that_hits_its_round_cap_is_decided_by_the_lcp(cap, monkeypatch):
    """DECISIONS #25. Bridges paralleled on one 2 mF || 5 ohm bus: at rest every diode
    pair fights over the bus, and on macOS five of them made the settle raise "did not
    settle after 64 rounds" at t = 1.8e-26 s (on Linux / Windows the same fight settles:
    0 caps, a rounding outcome). With the guard on, such an event is decided by the LCP
    from the held state. The engine's own cap at 1 round (the test hook) throws at any
    event where a diode flips, so the give-up is forced on any platform; either way the
    result is within 1e-3 V of pwl extrapolated in dt (measured 1.1e-4 V), and the forced
    cap is counted. Five bridges again since the cluster modes of identical banks take the
    Pade tier (exact engine 7; slices 5-6 had cut it to two while those modes were refused)."""
    monkeypatch.setattr(DD, "SETTLE_ROUND_CAP", cap)
    t_end, n = 0.02, 5
    tg = np.linspace(0.0, t_end, 2001)

    def bank():
        return _paralleled_bank(n)

    def bus(r):
        return np.interp(tg, r.times, np.asarray(r.v("p")) - np.asarray(r.v("n")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b2 = (bus(p.simulate(bank(), t_end=t_end, engine="pwl", dt=dt)) for dt in (5e-7, 2.5e-7))
        # legacy subject: the settle's round cap (mode_selection='legacy'; 'auto' is 'lcp')
        r = p.simulate(bank(), t_end=t_end, engine="dsed", mode_selection="legacy", output_rtol=0)
    assert np.abs(bus(r) - (2 * b2 - a)).max() <= 1e-3
    if cap == 1:
        assert r.diagnostics["mode_guard"]["settle_caps"] > 0


def _island_bridge(r_tie, c_bus):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "r", 0.0, 50.0, 50.0)
    b.add_resistor("Rr", "r", "gnd", r_tie)
    for name, an, ca in (("D1", "s", "p"), ("D2", "n", "s"), ("D3", "r", "p"), ("D4", "n", "r")):
        b.add_diode(name, an, ca, GON, GOFF)
    b.add_capacitor("C", "p", "n", c_bus)
    b.add_resistor("R", "p", "n", 100.0)
    b.add_resistor("Rn", "n", "r", r_tie)
    return b


@pytest.mark.parametrize("h_e_rel", [1e-10, 1e-5])
def test_port_factor_cancellation_stays_below_the_tie_tolerance(h_e_rel, monkeypatch):
    """DECISIONS #27. The factor G + C/h cancels on a floating island's pivot: the island gap
    carries eps*C*|dv/dt|/G_tie of rounding, independent of h_e (the 1/h of the cancellation
    and the h of the increment cancel). 2 mF on a 1 uS island at 1.6e4 V/s: 7.0e-9 V, under the
    tie tolerance 1e-9*vscale = 5e-8 V, so no mode is decided by rounding at any h_e: 'lcp'
    ends 6.1e-9 V from pwl at H_E_REL 1e-10 and 1e-5 alike. Where the bound exceeds the
    tolerance the factor's pivot check refuses by name (below)."""
    monkeypatch.setattr(DD, "H_E_REL", h_e_rel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = p.simulate(_island_bridge(1e6, 2e-3), t_end=0.04, engine="pwl", dt=1e-7)
    r = _sim(_island_bridge(1e6, 2e-3), 0.04, mode_selection="lcp")
    assert _end(r, "p-n") == pytest.approx(_end(ref, "p-n"), abs=2e-8)


@pytest.mark.parametrize("engine", ["dsed", "exact"])
def test_port_factor_beyond_the_bound_refuses_instead_of_rounding(engine):
    """The same bridge on 1 Tohm ties (bound 7e-3 V, 1.4e5 x the tie tolerance): the port
    factor is numerically singular at the island's node and the run stops with the typed
    refusal naming that node; no run returns a mode chosen by rounding (DECISIONS #27, the
    sweep). 'exact' and 'dsed' refuse alike."""
    with pytest.raises(p.UnsupportedError, match="numerically singular") as ei:
        p.simulate(_island_bridge(1e12, 2e-3), t_end=0.04, engine=engine, mode_selection="lcp")
    e = ei.value
    assert (e.engine, e.feature, e.kind) == ("dsed", "floating_island", "Node")
    assert e.device in {"s", "r", "p", "n"}
