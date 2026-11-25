"""Release findings, group C: the new mode selection (``mode_selection='auto'``,
the complementarity / LCP selector). Each finding reproduced with the option set
explicitly on the PR line; notes and traces in
openspec/changes/update-number-changing-release/fixes/group-c.md.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f()


# ---------------------------------------------------------------------------
# F-2-3 / F-5-9: a bridge diode held OFF while forward-biased (fixed engine, 'auto')
# ---------------------------------------------------------------------------

def _bridge(floating):
    """325 V / 50 Hz bridge, 1 mOhm source, 470 uF || 1 kOhm; the AC side floats
    (F-2-3, V_th = 0) or the DC side floats (F-5-9, V_th = 0.7)."""
    b = p.CircuitBuilder()
    if floating:
        b.add_sine_voltage_source("Vac", "s", "gnd", v_dc=0.0, v_amplitude=325.0,
                                  frequency=50.0, phase=0.0)
        b.add_resistor("Rs", "s", "a", 1e-3)
        p.topology.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="gnd", dc_pos="p",
                                        dc_neg="n", V_th=0.7)
        b.add_capacitor("C", "p", "n", 470e-6)
        b.add_resistor("RL", "p", "n", 1000.0)
    else:
        b.add_sine_voltage_source("Vac", "s", "b", v_dc=0.0, v_amplitude=325.0,
                                  frequency=50.0, phase=0.0)
        b.add_resistor("Rs", "s", "a", 1e-3)
        p.topology.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="b", dc_pos="p",
                                        dc_neg="gnd", V_th=0.0)
        b.add_capacitor("C", "p", "gnd", 470e-6)
        b.add_resistor("RL", "p", "gnd", 1000.0)
    return b


def _forward_biased_off(b, r):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p.device_loss_summary(b, r)
    return [str(x.message) for x in w if "OFF while forward-biased" in str(x.message)]


@pytest.mark.parametrize("floating", [False, True], ids=["F-2-3", "F-5-9"])
def test_the_lcp_step_stores_the_state_it_decided_on(floating):
    """The LCP decides on the islands' KCL-corrected solve (a floating AC side /
    DC bus held by 1e-9 S leaks); the step stored the raw solve, whose common mode
    is the factor's rounding (1.6 mV, 13 mV): a decided-OFF diode read forward-biased.
    Truth for the all-OFF samples of the AC-floating bridge: KCL of the island
    {s, a, b} over the four equal leaks, v_a + v_b = v_p (a closed form)."""
    b = _bridge(floating)
    r = _quiet(lambda: p.simulate(b, 0.06, engine="pwl", dt=1e-6, mode_selection="auto"))
    assert _forward_biased_off(b, r) == []
    if not floating:
        bits = np.asarray([list(s) for s in r.switch_states])
        off = ~bits.any(axis=1)
        va, vb, vp = (np.asarray(r.v(n))[off] for n in ("a", "b", "p"))
        assert off.sum() > 1000
        assert np.max(np.abs(va + vb - vp)) <= 1e-9 * 325.0


# ---------------------------------------------------------------------------
# F-4-1 / F-3-2: one classification of not-posed circuits, by the user's name
# ---------------------------------------------------------------------------

def _ideal_loop(clamp):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    if clamp:
        b.add_voltage_source("Vc", "k", "gnd", 3.0)
        b.add_diode("D", "a", "k", form="branch", R_on=0.0, g_off=0.0)
    else:
        b.add_diode("D", "a", "gnd", form="branch", R_on=0.0, g_off=0.0)
    b.add_resistor("R", "a", "gnd", 1.0)
    return b


@pytest.mark.parametrize("mode", ["legacy", "auto"])
@pytest.mark.parametrize("engine", ["dsed", "pwl"])
@pytest.mark.parametrize("clamp", [False, True])
def test_an_ideal_voltage_loop_is_named_by_every_selector(engine, mode, clamp):
    kw = {"dt": 1e-7} if engine == "pwl" else {}
    with pytest.raises(p.UnsupportedError) as ei:
        _quiet(lambda: p.simulate(_ideal_loop(clamp), 1e-6, engine=engine,
                                  mode_selection=mode, **kw))
    assert (ei.value.feature, ei.value.device, ei.value.kind) == (
        "ideal_voltage_loop", "D", "Diode")
    assert "add the missing path" not in str(ei.value)


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_a_refusal_device_is_the_user_name(mode):
    """F-3-2 (the parked test of group G): an ideal diode across a source."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_diode("D", "a", "gnd", form="branch")
    b.add_resistor("R", "a", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-6)
    with pytest.raises(p.UnsupportedError) as ei:
        _quiet(lambda: p.simulate(b, 1e-4, dt=1e-6, engine="pwl", mode_selection=mode))
    assert (ei.value.feature, ei.value.device, ei.value.kind) == (
        "ideal_voltage_loop", "D", "Diode")


def test_a_floating_node_is_named_under_the_lcp_too():
    """F-4-1's rule on pwl: the LCP's solves go through the settle's structural
    pass, so a node behind two OFF ideal diodes is `floating_node` (it was a
    singular factor and an abort after 64 sub-steps)."""
    b = p.CircuitBuilder()
    b.add_current_source("I", "gnd", "a", 1.0)
    b.add_diode("D", "b", "a", form="branch", R_on=0.0, g_off=0.0)
    b.add_diode("D2", "gnd", "b", form="branch", R_on=0.0, g_off=0.0)
    for engine, kw in (("pwl", {"dt": 1e-6}), ("dsed", {})):
        for mode in ("legacy", "auto"):
            with pytest.raises(p.UnsupportedError) as ei:
                _quiet(lambda: p.simulate(b, 1e-5, engine=engine, mode_selection=mode, **kw))
            # `a` (behind D) and `b` (between D and D2) both float; which of the two
            # the factorization meets first is its pivot order (MSVC's differs), so
            # the refusal names D alone or D with D2. D is on both nodes.
            named = {d.strip() for d in ei.value.device.split(",")}
            assert ei.value.feature == "floating_node", (engine, mode)
            assert "D" in named and named <= {"D", "D2"}, (engine, mode, ei.value.device)


# ---------------------------------------------------------------------------
# F-4-2 / F-4-3: the phase-shifted full bridge under 'auto' (exact engine)
# ---------------------------------------------------------------------------

def _psfb(tmp_path, devices, coss=False):
    """examples/phase_shift_full_bridge.yaml, every switch / diode `devices`
    (a YAML fragment), optionally with C_oss / C_j across them."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    s = (root / "examples" / "phase_shift_full_bridge.yaml").read_text(encoding="utf-8")
    s2 = s.replace("      g_on: 1.0e3\n      g_off: 1.0e-9\n", devices)
    assert s2 != s
    if coss:
        caps = [("HS_A", "vbus", "mid_a", 300e-12), ("LS_A", "mid_a", "gnd", 300e-12),
                ("HS_B", "vbus", "mid_b", 300e-12), ("LS_B", "mid_b", "gnd", 300e-12),
                ("D1", "sec_pos", "rect_pos", 100e-12), ("D2", "sec_neg", "rect_pos", 100e-12),
                ("D3", "rect_neg", "sec_pos", 100e-12), ("D4", "rect_neg", "sec_neg", 100e-12)]
        add = "".join(f"    - type: capacitor\n      name: C_{n}\n      from: {a}\n      to: {c}\n"
                      f"      C: {v!r}\n" for n, a, c, v in caps)
        s2 = s2.replace("      R: 1.0e-3\n\nsimulation:", "      R: 1.0e-3\n" + add + "\nsimulation:")
    f = tmp_path / "psfb.yaml"
    f.write_text(s2, encoding="utf-8")
    lc = p.load_yaml_file(str(f))
    si = lc.builder.switch_index_of
    fn = p.make_phase_shift_full_bridge_fn(100e3, np.pi / 2, si("HS_A"), si("LS_A"), si("HS_B"),
                                           si("LS_B"), len(lc.switch_names), 100e-9)
    return lc, fn


def _vout_avg(lc, fn, t_end, mode):
    r = _quiet(lambda: p.simulate(lc, t_end=t_end, engine="exact", switch_fn=fn, rtol=1e-9,
                                  atol=1e-12, mode_selection=mode))
    t = np.asarray(r.times)
    assert t[-1] == pytest.approx(t_end, rel=1e-12)
    return p.time_average(t, np.asarray(r.v("vout")), t_from=0.9 * t_end)


def test_f_4_2_branch_form_psfb_at_a_micro_ohm_runs_under_auto(tmp_path):
    """R_on = 1e-6: the hybrid-coordinate row of an ON port cancelled to rounding
    (H_jj = -1.3e-7 S) and Lemke ended on a ray at 339 us. Truth: the
    conductance twin at g_on = 1e6 (24.071424 V) and the settle (24.071411 V)."""
    lc, fn = _psfb(tmp_path, "      form: branch\n      R_on: 1.0e-6\n      g_off: 1.0e-9\n")
    assert _vout_avg(lc, fn, 2e-3, "auto") == pytest.approx(24.0714, rel=1e-5)


def test_f_4_3_ideal_psfb_with_coss_runs_under_auto(tmp_path):
    """Ideal devices + C_oss / C_j: 'auto' left rectifier D1 ON at -2.2 A from
    14.99 us. Truth: the conductance twins at R_on = 1e-3 / 1e-4 (1.035708 /
    1.035833 V, converging to 1.03585 at R_on -> 0) and the settle (1.035847)."""
    lc, fn = _psfb(tmp_path, "      form: branch\n      R_on: 0.0\n      g_off: 1.0e-9\n", coss=True)
    assert _vout_avg(lc, fn, 3e-5, "auto") == pytest.approx(1.035847, rel=2e-5)


# ---------------------------------------------------------------------------
# F-5-3: a 1 ps gate pulse on the exact engine
# ---------------------------------------------------------------------------

def _buck(d):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "x", 100e-6)
    b.add_resistor("RLs", "x", "vout", 0.01)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 1.0)
    return b, p.make_pwm_switch_fn(1e5, d, b.switch_index_of("S"), b.graph.num_switches, 0.0)


@pytest.mark.parametrize("d", [1e-7, 3e-5, 1e-4])
def test_f_5_3_a_short_pulse_is_not_dropped_by_the_lcp(d):
    """At the pulse's falling edge the inductor's 2.4e-7 A (1 ps at 24 V / 100 uH)
    is a tie for the selector (tol_i ~ C_max·tol_w/h_e = 1.1 A at h_e = 1 ps) and a
    pin; the pin judge's horizons (100·h_e, 1e4·h_e) skip the g_off transient that
    dumps the current, so D1 OFF was judged consistent. Truth: the settle, and the
    ratio 0.987·D·Vin both selectors give at D = 1e-4."""
    out = {}
    for mode in ("legacy", "auto"):
        b, fn = _buck(d)
        r = _quiet(lambda: p.simulate(b, 100e-5, engine="dsed", switch_fn=fn, mode_selection=mode))
        out[mode] = float(np.asarray(r.v("vout"))[-1])
    assert out["auto"] == pytest.approx(out["legacy"], rel=1e-3)
    assert out["auto"] / (d * 24.0) == pytest.approx(0.987, rel=2e-2)


# ---------------------------------------------------------------------------
# Side observation (from group A): the F-3-3 flyback on the exact engine
# ---------------------------------------------------------------------------

def _flyback_vout(mode):
    from .test_saturable_transformer import _flyback
    b = _flyback(False, 1.0)
    pwm = p.make_pwm_switch_fn(frequency=100e3, duty=0.5, switch_idx=0,
                               num_switches=b.graph.num_switches, phase=0.0)
    r = _quiet(lambda: p.simulate(b, 1e-3, engine="dsed", switch_fn=pwm, mode_selection=mode))
    t = np.asarray(r.times)
    return p.time_average(t, np.asarray(r.v("vout")) - np.asarray(r.v("sec_neg")), t_from=8e-4)


@pytest.mark.parametrize("mode", [
    "auto",
    pytest.param("legacy", marks=pytest.mark.xfail(strict=True, reason=(
        "default-path defect of today's default selector, fixed by the release's flip "
        "to 'auto': at every gate-off the settle keeps the secondary diode OFF (v_D = 0 at "
        "t+: the secondary current is a state at 0) and the forward bias exists only inside "
        "the 1e-14 s R_off transient that collapses the magnetising current, which no "
        "settle decision reads (i_p 2.06 A -> 1e-7 A, v_out 0 V; guard: 0 repairs)"))),
])
def test_the_flyback_secondary_conducts_on_the_exact_engine(mode):
    """Truth: the restart fixed step (group A's fix) and the exact engine under 'auto'
    agree on the mean output 16.36 V (bucket-3 F-3-3: 16.365 V over 0.8..1 ms)."""
    assert _flyback_vout(mode) == pytest.approx(16.36, rel=2e-3)
