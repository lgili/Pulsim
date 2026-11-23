"""One hand-checked circuit per v1 component type of the reader's mapping
table (restore-reference-benchmark-harness design decision 1, task 1.2).

Each truth is derived here from the circuit's own values (Ohm's law, a
closed-form transient, or a scalar root of KCL found by bisection); none
is copied from a comment in a benchmark file. A v1 type the reader cannot
map is refused by name.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

yaml = pytest.importorskip("yaml")

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path[:0] = [str(BENCH), str(BENCH / "tools")]

import v1_reader  # noqa: E402

# The fixed engine starts a stored state with zero history current (KNOWN_LIMITATIONS: the
# first step is not consistent), which lags the whole trace by dt/2: a transient truth is
# compared within scale * dt / tau (2x that lag). A 1 % error in R or C moves the trace ~10x more.

V_T_2_0 = 0.025852       # add_shockley_diode's default thermal voltage
G_MIN_2_0 = 1e-12        # add_shockley_diode's default G_min


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _case(tmp_path, comps, tstop, dt, uic=True):
    doc = {"schema": "pulsim-v1", "version": 1, "benchmark": {"id": "unit"},
           "simulation": {"tstop": tstop, "dt": dt, "uic": uic}, "components": comps}
    path = tmp_path / "unit.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return v1_reader.load(path)


def _dc(name, node, v):
    return {"type": "voltage_source", "name": name, "nodes": [node, 0],
            "waveform": {"type": "dc", "value": v}}


def _r(name, a, b, v):
    return {"type": "resistor", "name": name, "nodes": [a, b], "value": v}


def _root(f, lo, hi):
    """Bisection to the last bit of a sign change of f on [lo, hi]."""
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if (f(mid) > 0) == (flo > 0):
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------- resistor + dc source ---

def test_bridge_resistive_pure_from_its_own_values():
    case = v1_reader.load(BENCH / "circuits/linear/bridge_resistive_pure.yaml")
    comps = {c["name"]: c for c in case._components}
    v = comps["Vsrc"]["waveform"]["value"]
    r1, r2, r3, r4 = (comps[f"R{k}"]["value"] for k in (1, 2, 3, 4))
    va, vb = v * r2 / (r1 + r2), v * r4 / (r3 + r4)
    assert (round(va, 4), round(vb, 4)) == (6.6667, 4.0)       # the spec's numbers
    r = case.run("pwl")
    assert abs(r.v("a")[-1] - va) < 1e-6 and abs(r.v("b")[-1] - vb) < 1e-6


# ------------------------------------------------- capacitor, inductor ---

def test_capacitor_initial_condition_discharges_through_r(tmp_path):
    R, C, v0 = 1e3, 1e-6, 5.0
    case = _case(tmp_path, [_r("R", "a", 0, R),
                            {"type": "capacitor", "name": "C", "nodes": ["a", 0], "value": C,
                             "ic": v0}], 2e-3, 1e-7)
    r = case.run("pwl")
    t = np.asarray(r.times)
    np.testing.assert_allclose(r.v("a"), v0 * np.exp(-t / (R * C)), rtol=0, atol=v0 * 1e-7 / (R * C))


def test_inductor_initial_condition_decays_through_r(tmp_path):
    # i_L flows a -> gnd inside L, so it returns gnd -> a through R: v(a) = -R i_L
    R, L, i0 = 10.0, 10e-3, 1.0
    case = _case(tmp_path, [_r("R", "a", 0, R),
                            {"type": "inductor", "name": "L", "nodes": ["a", 0], "value": L,
                             "ic": i0}], 2e-3, 1e-7)
    r = case.run("pwl")
    t = np.asarray(r.times)
    np.testing.assert_allclose(r.v("a")[1:], -R * i0 * np.exp(-t[1:] * R / L), rtol=0,
                               atol=R * i0 * 1e-7 * R / L)


def test_snubber_rc_is_r_in_series_with_c(tmp_path):
    V, R, C = 10.0, 100.0, 1e-6
    case = _case(tmp_path, [_dc("V", "a", V),
                            {"type": "snubber_rc", "name": "S", "nodes": ["a", 0],
                             "resistance": R, "capacitance": C, "ic": 0.0}], 5e-4, 1e-8)
    r = case.run("pwl")
    t = np.asarray(r.times)
    # charging current V/R e^{-t/RC}: the mid node sits at V - iR = V (1 - e^{-t/RC})
    np.testing.assert_allclose(r.v("S__mid")[1:], V * (1 - np.exp(-t[1:] / (R * C))),
                               rtol=0, atol=V * 1e-8 / (R * C))


# ------------------------------------------------------- sources ---

def test_sine_source_through_a_divider(tmp_path):
    off, amp, f, ph = 1.0, 2.0, 1e3, 0.5
    case = _case(tmp_path, [{"type": "voltage_source", "name": "V", "nodes": ["s", 0],
                             "waveform": {"type": "sine", "offset": off, "amplitude": amp,
                                          "frequency": f, "phase": ph}},
                            _r("R1", "s", "m", 1e3), _r("R2", "m", 0, 1e3)], 2e-3, 1e-6)
    r = case.run("pwl")
    t = np.asarray(r.times)
    # [1:]: the first stored sample of an algebraic node is not solved (KNOWN_LIMITATIONS)
    np.testing.assert_allclose(np.asarray(r.v("m"))[1:],
                               0.5 * (off + amp * np.sin(2 * np.pi * f * t[1:] + ph)), rtol=0, atol=1e-9)


def test_pulse_source_is_the_spice_trapezoid(tmp_path):
    v0, v1, td, tr, tf, pw, per = 0.0, 5.0, 10e-6, 1e-6, 1e-6, 20e-6, 50e-6
    case = _case(tmp_path, [{"type": "voltage_source", "name": "V", "nodes": ["s", 0],
                             "waveform": {"type": "pulse", "v_initial": v0, "v_pulse": v1,
                                          "t_delay": td, "t_rise": tr, "t_fall": tf,
                                          "t_width": pw, "period": per}},
                            _r("R", "s", 0, 1e3)], 2e-4, 1e-7)
    r = case.run("pwl")
    t = np.asarray(r.times)

    def pulse(x):
        if x < td:
            return v0
        u = (x - td) % per
        if u < tr:
            return v0 + (v1 - v0) * u / tr
        if u < tr + pw:
            return v1
        if u < tr + pw + tf:
            return v1 - (v1 - v0) * (u - tr - pw) / tf
        return v0

    np.testing.assert_allclose(r.v("s"), [pulse(x) for x in t], rtol=0, atol=1e-6)


def test_pwm_source_phase_fraction_and_dead_time(tmp_path):
    # high while (t + phase*T) mod T < duty*T - dead_time (the reader's documented semantics)
    f, duty, ph, dead = 10e3, 0.3, 0.25, 2e-6
    T = 1 / f
    case = _case(tmp_path, [{"type": "voltage_source", "name": "V", "nodes": ["s", 0],
                             "waveform": {"type": "pwm", "v_high": 5.0, "v_low": 0.0,
                                          "frequency": f, "duty": duty, "phase": ph,
                                          "dead_time": dead}},
                            _r("R", "s", 0, 1e3)], 5e-4, 1e-7)
    r = case.run("pwl")
    t = np.asarray(r.times)
    u = (t + ph * T) % T
    t_on = duty * T - dead
    far = (np.minimum(np.abs(u - t_on), T - np.abs(u - t_on)) > 2e-7) & (np.minimum(u, T - u) > 2e-7)
    far[0] = False      # the unsolved first sample of an algebraic node (KNOWN_LIMITATIONS)
    np.testing.assert_allclose(np.asarray(r.v("s"))[far], np.where(u < t_on, 5.0, 0.0)[far],
                               rtol=0, atol=1e-9)
    assert far.sum() > 0.9 * t.size


def test_current_source_injects_into_its_first_node(tmp_path):
    # the ngspice pairing `I1 0 out DC 1m` (current_source_rc.cir): into `out`
    i_src, R, C = 1e-3, 5e3, 1e-6
    case = _case(tmp_path, [{"type": "current_source", "name": "I", "nodes": ["out", 0], "value": i_src},
                            _r("R", "out", 0, R),
                            {"type": "capacitor", "name": "C", "nodes": ["out", 0], "value": C}],
                 2e-2, 1e-6)
    r = case.run("pwl")
    t = np.asarray(r.times)
    np.testing.assert_allclose(r.v("out"), i_src * R * (1 - np.exp(-t / (R * C))), rtol=0,
                               atol=i_src * R * 1e-6 / (R * C))


# ------------------------------------------------------ diodes, switches ---

@pytest.mark.parametrize("forward", [True, False], ids=["forward", "reverse"])
def test_binary_diode_is_g_on_or_g_off(tmp_path, forward):
    V, R, g_on, g_off = 10.0, 1e3, 50.0, 1e-6          # not the reader's defaults (1e3 / 1e-9)
    nodes = ["a", 0] if forward else [0, "a"]
    case = _case(tmp_path, [_dc("V", "s", V), _r("R", "s", "a", R),
                            {"type": "diode", "name": "D", "nodes": nodes,
                             "g_on": g_on, "g_off": g_off}], 1e-5, 1e-6)
    r_d = 1 / (g_on if forward else g_off)
    assert case.run("pwl").v("a")[-1] == pytest.approx(V * r_d / (R + r_d), rel=1e-9)


def test_diode_with_is_is_the_shockley_law(tmp_path):
    # v1's kernel ignored `is` (binary diode); the reader keeps the file's intent (DECISIONS 14)
    V, R, i_s, n = 5.0, 1e3, 1e-12, 1.5
    case = _case(tmp_path, [_dc("V", "s", V), _r("R", "s", "a", R),
                            {"type": "diode", "name": "D", "nodes": ["a", 0], "is": i_s, "n": n}],
                 1e-5, 1e-6)
    vd = _root(lambda x: (V - x) / R - i_s * math.expm1(x / (n * V_T_2_0)) - G_MIN_2_0 * x, 0.0, V)
    assert 0.7 < vd < 1.2
    assert case.run("pwl").v("a")[-1] == pytest.approx(vd, abs=1e-6)


@pytest.mark.parametrize("closed", [True, False])
def test_static_switch_follows_initial_closed(tmp_path, closed):
    V, R, g_on, g_off = 10.0, 1e3, 1e4, 1e-8           # not the defaults (1e6 / 1e-12)
    case = _case(tmp_path, [_dc("V", "s", V), _r("R", "s", "a", R),
                            {"type": "switch", "name": "S", "nodes": ["a", 0],
                             "initial_closed": closed, "g_on": g_on, "g_off": g_off}], 1e-5, 1e-6)
    r_s = 1 / (g_on if closed else g_off)
    assert case.run("pwl").v("a")[-1] == pytest.approx(V * r_s / (R + r_s), rel=1e-9)


def test_vcswitch_follows_its_control_node_through_its_threshold(tmp_path):
    # control: a SPICE pulse 0 -> 5 V with 20 us ramps; at v_threshold 4 V (not the 2.5 V default)
    # the switch is ON from td + 0.8 tr to td + tr + pw + 0.2 tf in every period
    V, R, g_on, g_off, thr = 10.0, 1e3, 500.0, 1e-7, 4.0
    td, tr, pw, tf, per = 10e-6, 20e-6, 20e-6, 20e-6, 100e-6
    case = _case(tmp_path, [_dc("V", "vin", V),
                            {"type": "voltage_source", "name": "Vg", "nodes": ["ctrl", 0],
                             "waveform": {"type": "pulse", "v_initial": 0.0, "v_pulse": 5.0,
                                          "t_delay": td, "t_rise": tr, "t_fall": tf,
                                          "t_width": pw, "period": per}},
                            {"type": "vcswitch", "name": "S", "nodes": ["ctrl", "vin", "a"],
                             "v_threshold": thr, "g_on": g_on, "g_off": g_off},
                            _r("R", "a", 0, R)], 3e-4, 1e-7)
    r = case.run("pwl")
    t = np.asarray(r.times)
    x = thr / 5.0
    t_on, t_off = td + tr * x, td + tr + pw + tf * (1 - x)       # 26 us, 54 us
    u = t % per
    on = (u >= t_on) & (u < t_off)
    far = (np.abs(u - t_on) > 2e-7) & (np.abs(u - t_off) > 2e-7)
    far[0] = False
    want = np.where(on, V * R / (R + 1 / g_on), V * R / (R + 1 / g_off))
    np.testing.assert_allclose(np.asarray(r.v("a"))[far], want[far], rtol=1e-9, atol=1e-12)
    assert on[far].any() and (~on[far]).any()


# -------------------------------------------------- transistors ---

@pytest.mark.xfail(strict=True, reason="DECISIONS 14 R1: the reader maps K = kp; the v1 law is "
                   "id = kp (vov vds - vds^2/2)(1 + lambda vds), i.e. K = kp/2 (2x the current)")
def test_nmos_triode_dc_point_is_the_v1_law():
    case = v1_reader.load(BENCH / "circuits/switching/basic/mosfet_nmos_dc.yaml")
    c = {x["name"]: x for x in case._components}
    vdd, rl = c["Vdd"]["waveform"]["value"], c["Rload"]["value"]
    vov = c["Vgate"]["waveform"]["value"] - c["M1"]["vth"]
    kp, lam = c["M1"]["kp"], c["M1"]["lambda"]

    def kcl(vds):   # resistor current minus v1's triode current (vds < vov)
        return (vdd - vds) / rl - kp * (vov * vds - 0.5 * vds * vds) * (1 + lam * vds)

    vds = _root(kcl, 0.0, vov)
    assert vds == pytest.approx(0.0502, abs=1e-4)          # the file's comment, re-derived
    assert case.run("pwl").v("drain")[-1] == pytest.approx(vds, abs=1e-6)


def test_igbt_conducts_above_its_saturation_voltage(tmp_path):
    # mapped to add_igbt_level1: I_C = sigma(kappa (V_GE - V_T)) smoothmax0(V_CE - V_CE_sat) g_on,
    # smoothmax0(x) = (x + sqrt(x^2 + v_knee^2)) / 2, kappa 10, v_knee 0.01 (2.0 defaults)
    # gate 0.3 V above V_T: the sigmoid is 0.95, so vth is exercised (defaults: 5 V, 1.5 V, 1e4 S)
    V, R, g_on, vsat, vth, vg = 20.0, 1.0, 200.0, 1.2, 4.0, 4.3
    case = _case(tmp_path, [_dc("V", "s", V), _dc("Vg", "g", vg), _r("R", "s", "c", R),
                            {"type": "igbt", "name": "Q", "nodes": ["g", "c", 0], "vth": vth,
                             "g_on": g_on, "g_off": 1e-9, "v_ce_sat": vsat}], 1e-5, 1e-6)
    alpha = 1 / (1 + math.exp(-10 * (vg - vth)))

    def kcl(v):
        x = v - vsat
        return (V - v) / R - alpha * g_on * 0.5 * (x + math.hypot(x, 0.01))

    vce = _root(kcl, 0.0, V)
    assert vce == pytest.approx(vsat + (V - vsat) / (1 + R * g_on * alpha), abs=1e-3)   # ideal knee
    assert case.run("pwl").v("c")[-1] == pytest.approx(vce, abs=1e-6)


# --------------------------------------------------------- magnetics ---

def test_ideal_transformer_turns_ratio_is_np_over_ns(tmp_path):
    a, V, R = 0.5, 3.0, 100.0           # N_p/N_s = 0.5: v_s = v_p / a = 2 v_p
    case = _case(tmp_path, [_dc("V", "p", V),
                            {"type": "transformer", "name": "T", "nodes": ["p", 0, "s", 0],
                             "turns_ratio": a},
                            _r("R", "s", 0, R)], 1e-5, 1e-6)
    assert case.run("pwl").v("s")[-1] == pytest.approx(V / a, rel=1e-9)


def test_coupled_inductor_open_secondary_is_m_over_l1(tmp_path):
    l1, l2, k, f = 1e-3, 4e-3, 0.9, 1e3
    case = _case(tmp_path, [{"type": "voltage_source", "name": "V", "nodes": ["p", 0],
                             "waveform": {"type": "sine", "offset": 0.0, "amplitude": 1.0,
                                          "frequency": f}},
                            {"type": "coupled_inductor", "name": "K", "nodes": ["p", 0, "s", 0],
                             "l1": l1, "l2": l2, "coupling": k},
                            _r("R", "s", 0, 1e9)], 2e-3, 1e-6)
    r = case.run("pwl")
    m_over_l1 = k * math.sqrt(l1 * l2) / l1           # v_s = M di_p/dt = (M / L1) v_p, i_s ~ 0
    np.testing.assert_allclose(np.asarray(r.v("s"))[1:], m_over_l1 * np.asarray(r.v("p"))[1:],
                               rtol=0, atol=1e-6)


# --------------------------------------------------------- closed loop ---

def test_pi_controller_through_pwm_generator_regulates_the_average(tmp_path):
    # Vin -S- x -R- out, C || RL at out; PI (ref 5 V, fb out) sets S's duty once per period.
    # An integrating loop drives the sampled error to zero: v(out) at each period start -> 5 V,
    # and the period average stays within the ripple of it.
    case = _case(tmp_path, [_dc("Vin", "vin", 12.0), _dc("Vref", "ref", 5.0),
                            {"type": "switch", "name": "S", "nodes": ["vin", "x"],
                             "g_on": 1e3, "g_off": 1e-9},
                            _r("R", "x", "out", 10.0),
                            {"type": "capacitor", "name": "C", "nodes": ["out", 0], "value": 100e-6},
                            _r("RL", "out", 0, 20.0),
                            {"type": "pi_controller", "name": "PI", "nodes": ["ref", "out", "u"],
                             "kp": 0.05, "ki": 50.0, "output_min": 0.0, "output_max": 1.0},
                            {"type": "pwm_generator", "name": "PWM", "nodes": ["u"],
                             "duty_from_channel": "PI", "frequency": 10e3,
                             "target_component": "S"}], 0.1, 1e-6)
    r = case.run("pwl")
    t, v = np.asarray(r.times), np.asarray(r.v("out"))
    tail = t >= 0.09
    starts = np.isclose((t[tail] * 10e3) % 1, 0, atol=1e-6) | np.isclose((t[tail] * 10e3) % 1, 1, atol=1e-6)
    ripple = v[tail].max() - v[tail].min()
    assert 0 < ripple < 0.2
    assert np.abs(v[tail][starts] - 5.0).max() < 0.02
    assert abs(np.trapezoid(v[tail], t[tail]) / (t[tail][-1] - t[tail][0]) - 5.0) < ripple


# ------------------------------------------------------------ refusals ---

@pytest.mark.parametrize("comp, name", [
    ({"type": "saturable_inductor", "name": "Ls", "nodes": ["a", 0], "inductance": 1e-3,
      "saturation_current": 1.0, "saturation_inductance": 1e-4, "saturation_exponent": 2.0},
     "saturable_inductor"),
    ({"type": "mosfet", "name": "M", "nodes": ["a", "a", 0], "vth": 2.0, "kp": 0.1, "lambda": 0.0,
      "is_nmos": False}, "p-channel mosfet"),
    ({"type": "mosfet", "name": "M", "nodes": ["a", "a", 0], "vth": 2.0, "kp": 0.1, "lambda": 0.0,
      "loss": {"eon": 1e-6}}, "mosfet loss/thermal"),
    ({"type": "pmos", "name": "M", "nodes": ["a", "a", 0]}, "component type 'pmos'"),
    ({"type": "lookup_table", "name": "LUT", "nodes": ["a"], "x": [0, 1], "y": [0, 1]},
     "component type 'lookup_table'"),
    ({"type": "pi_controller", "name": "PI", "nodes": ["a", "a", "u"], "kp": 1.0, "ki": 1.0},
     "pi_controller PI does not drive a switch"),
], ids=["saturable_inductor", "pmos_flag", "mosfet_loss", "pmos_type", "lookup_table", "lone_pi"])
def test_an_unmappable_component_is_refused_by_name(tmp_path, comp, name):
    with pytest.raises(v1_reader.Unmapped, match=name.replace("(", r"\(")):
        _case(tmp_path, [_dc("V", "a", 1.0), _r("R", "a", 0, 1.0), comp], 1e-5, 1e-6)
