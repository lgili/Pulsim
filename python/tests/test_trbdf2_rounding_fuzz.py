"""engine='trbdf2' under a rounding fuzz: no decision may be made by noise.

Each circuit is run with every element value perturbed by a random
relative 1e-15 (a few ulps) — the size of the difference between two
platforms' libm or two Eigen versions. A result that moves by more than
the tolerance under that perturbation, or an event count that changes,
means a decision was read from round-off. Counts are compared between
runs on the same platform, never pinned.
"""

import numpy as np
import pytest

import pulsim as p

FUZZ = 1e-15


def _bridge(d):
    """The floating-bus bridge of test_trbdf2_error_control (1 MΩ tie)."""
    gon, goff = 1e3, 1e-9
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0 * (1 + d[0]), 50.0)
    b.add_resistor("Rs", "s", "a", 0.5 * (1 + d[1]))
    b.add_inductor("Ls", "a", "ac1", 1e-3 * (1 + d[2]))
    b.add_diode("D1", "ac1", "dp", gon, goff)
    b.add_diode("D2", "gnd", "dp", gon, goff)
    b.add_diode("D3", "dn", "ac1", gon, goff)
    b.add_diode("D4", "dn", "gnd", gon, goff)
    b.add_capacitor("C", "dp", "dn", 470e-6 * (1 + d[3]))
    b.add_resistor("RL", "dp", "dn", 100.0 * (1 + d[4]))
    b.add_resistor("Rtie", "dn", "gnd", 1e6 * (1 + d[5]))
    return b


def test_floating_bus_bridge_finishes_under_rounding():
    """Before: 9 of these 20 runs ground to max_steps (1e7), at 4.45,
    13.2 or 33.5 ms, with D2 / D4 flipping every 1e-13 s. The
    simultaneous D1 / D4 turn-off at the first peak is a tie that
    rounding decides; the landing after it put the floating bus 83 V up
    (its common mode is round-off at a 3e-13 s step: 2C/h = 2.8e9 S
    against the 1e-6 S tie) and turned D4 back ON from that noise.
    NOT yet independent of rounding: the event count (7-10), the step
    count (1231-1254) and the bus level (outcomes up to 0.19 V apart)
    still vary between these runs (KNOWN_LIMITATIONS); only the grind
    is pinned."""
    rng = np.random.default_rng(20261117)
    for k in range(20):
        d = np.zeros(6) if k == 0 else rng.uniform(-1.0, 1.0, 6) * FUZZ
        n = [0]

        def cont(*_):
            n[0] += 1
            return n[0] < 100_000

        res = p.simulate(_bridge(d), t_end=40e-3, engine="trbdf2",
                         rtol=2e-5, atol=2e-8, should_continue=cont)
        assert float(np.asarray(res.times)[-1]) == pytest.approx(40e-3), k
        assert res._trbdf2_stats["n_accept"] < 2000, k


def _dcm_buck(d):
    """The design workflow's DCM buck (test_design_steady_and_batch)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0 * (1 + d[0]))
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6 * (1 + d[1]))
    b.add_capacitor("C", "vout", "gnd", 47e-6 * (1 + d[2]))
    b.add_resistor("RL", "vout", "gnd", 50.0 * (1 + d[3]))
    sf = p.make_pwm_switch_fn(1e5, 0.3, b.switch_index_of("S"),
                              b.graph.num_switches)
    return b, sf


def test_dcm_buck_turn_off_never_records_a_kilovolt():
    """Before: node sw at 3854 V for 4e-13 s after a diode turn-off at
    0.4855 ms (a 24 V circuit). The landing stopped 1e-4 h past the
    current zero and left -7.8 uA in the inductor with only g_off to go
    through (v = i/g_off); the committed femtosecond steps that followed
    recorded the L*g_off mode. A turn-off is now bracketed until that
    voltage is inside the run's tolerance."""
    rng = np.random.default_rng(4855)
    for k in range(8):
        d = np.zeros(4) if k == 0 else rng.uniform(-1.0, 1.0, 4) * FUZZ
        b, sf = _dcm_buck(d)
        res = p.simulate(b, t_end=0.64e-3, switch_fn=sf, engine="trbdf2")
        v = np.abs(np.asarray(res.v("sw")))
        assert v.max() < 25.0, (k, float(v.max()))


CORE = dict(N_p=25, N_s=13, Ae=76e-6, le=72e-3, lg=0.5e-3, B_sat=0.35)


def _snubbed_flyback(d):
    """test_saturable_transformer::test_engines_agree_below_saturation."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0 * (1 + d[0]))
    b.add_switch("Q1", "sw", "gnd", 1e3, 1e-9)
    b.add_resistor("Rsn", "sw", "sn1", 47.0 * (1 + d[1]))
    b.add_capacitor("Csn", "sn1", "gnd", 2.2e-9 * (1 + d[2]))
    b.add_saturable_transformer("T1", "vin", "sw", "sec_neg", "sec_anode",
                                L_leak_p=5e-6 * (1 + d[3]), **CORE)
    b.add_diode("D1", "sec_anode", "vout", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout", "sec_neg", 100e-6 * (1 + d[4]))
    b.add_resistor("R_L", "vout", "sec_neg", 5.0 * (1 + d[5]))
    b.add_resistor("Rgnd", "sec_neg", "gnd", 1e-6)
    q, n = b.switch_index_of("Q1"), b.graph.num_switches

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(q, (t % 10e-6) / 10e-6 < 0.4)
        return m
    return b, sf


def test_saturable_flyback_never_stalls_under_rounding():
    """Before: 2 of 22 fuzzed runs aborted at h_min (0.350 and 0.370 ms,
    "Newton failed"), as Linux CI did at 0.34 ms, right after a D1
    turn-off landing 1.3e-13 s long: the flux residual's 2/h*lambda
    (1.4e12) put its round-off above the 1e-9 V step tolerance."""
    rng = np.random.default_rng(99)
    steps, vend = [], []
    for k in range(6):
        d = np.zeros(6) if k == 0 else rng.uniform(-1.0, 1.0, 6) * FUZZ
        b, sf = _snubbed_flyback(d)
        res = p.simulate(b, t_end=0.4e-3, switch_fn=sf, engine="trbdf2",
                         rtol=1e-6, atol=1e-9)
        steps.append(res._trbdf2_stats["n_accept"])
        vend.append(float(np.asarray(res.v("vout"))[-1]))
    assert max(steps) / min(steps) - 1.0 < 0.02, steps
    assert max(vend) - min(vend) < 1e-6 * abs(vend[0]), vend


def test_sepic_probe_inside_its_round_off_is_retaken():
    """The SEPIC's S1 turn-off at 9.8 us was probed at span*1e-12 = 2e-15 s,
    where 2C/h = 1e10 S against the 2e-6 S of two OFF devices makes the
    coupling-capacitor subnet's common mode undecidable. Held OFF by the
    round-off bound, the rectifier let the committed step drain both
    inductors through g_off: V(out) was 0.0013 V at 12 us instead of
    0.23 V, and 17.3033 V at 2 ms instead of 17.3157 V (pwl dt/8:
    17.3149 V). The probe is now retaken longer until it can decide."""
    import copy
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "benchmarks" / "tools"))
    pytest.importorskip("yaml")
    import v1_reader

    base = v1_reader.load(str(Path(__file__).resolve().parents[2]
                              / "benchmarks/circuits/switching/buck_boost/sepic.yaml"))
    rng = np.random.default_rng(98)
    for k in range(4):
        case = copy.copy(base)
        case._components = copy.deepcopy(base._components)
        if k:
            for c in case._components:
                if isinstance(c.get("value"), (int, float)) and c["value"]:
                    c["value"] *= 1 + rng.uniform(-1.0, 1.0) * FUZZ
        res = case.run("trbdf2", rtol=1e-8, atol=1e-12)
        t, v = np.asarray(res.times), np.asarray(res.v("out"))
        assert np.interp(12e-6, t, v) > 0.2, k
        assert abs(v[-1] - 17.3157) < 1e-4, (k, float(v[-1]))
