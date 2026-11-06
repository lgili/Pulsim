"""Post-hoc diode conduction state: the ON stamp is a pure conductance.

A PWL ``SwitchedDiode`` is ``g_on`` or ``g_off`` between its terminals —
no voltage offset in the stamp — and ``V_th`` only gates the OFF→ON
commutation (``switched_diode.hpp``: OFF→ON iff ``v_D >= V_th``, ON→OFF
iff ``g_on·v_D <= 0``). A conducting diode therefore sits at
``i / g_on`` (48 A → 48 mV), far INSIDE ``0 < v_D < V_th``.

``result.i()``, ``device_loss_summary`` and ``device_thermal_summary``
each rebuilt the state as ``v_D > V_th``. Measured at 82c567b1 on the
capacitor-input bridge below (``engine='pwl'``, ``dt = 1 µs``):

    V_th = 0.0:  BR__D1 i_peak 47.99 A, duty 0.106, P_avg 0.0975 W
    V_th = 0.7:  BR__D1 i_peak 0.0000 A, duty 0.000, P_avg 3.9e-5 W

with V_dc identical (312.38 V) and no warning — a zero in a loss table.
On ``engine='auto'`` the same rule invented phantoms instead: the
summary reported a 700.0 A peak (``g_on · V_th`` on an event landing
stored with the pre-flip mask) while ``result.i()`` said 0.0017 A, and
at ``V_th = 0`` a 3417 A peak on a diode the solver had stamped OFF.

These tests pin the shared reconstruction (``pulsim._diode_state``)
against what the result itself can prove: Kirchhoff's current law on
the DC side, the state variables (source and inductor currents), and
agreement across engines and across ``V_th``.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import _diode_state
from pulsim.losses import _time_fraction

T_END = 0.06
C_DC = 470e-6
R_LOAD = 1000.0
DIODES = ("BR__D1", "BR__D2", "BR__D3", "BR__D4")


def _bridge(V_th: float) -> p.CircuitBuilder:
    """325 V / 50 Hz, 1 mΩ source, bridge (helper defaults g_on=1e3,
    g_off=1e-9), 470 µF || 1 kΩ. The AC side floats so the DC rail can
    be grounded — with ``dc_neg`` floating instead, ``engine='auto'``
    cannot factor the all-OFF mask at its 6e-14 s first step."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "b", v_dc=0.0, v_amplitude=325.0,
                              frequency=50.0, phase=0.0)
    b.add_resistor("Rs", "s", "a", 1e-3)
    p.topology.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="b",
                                    dc_pos="p", dc_neg="gnd", V_th=V_th)
    b.add_capacitor("C", "p", "gnd", C_DC)
    b.add_resistor("RL", "p", "gnd", R_LOAD)
    return b


@functools.lru_cache(maxsize=None)
def _run_bridge(V_th: float, engine: str):
    b = _bridge(V_th)
    kw = dict(engine="pwl", dt=1e-6) if engine == "pwl" else {}
    with warnings.catch_warnings():
        # The fixed engine's V_th=0 mask-cycle notice and the auto
        # engine's sliding-mode notice are about the SIMULATION, not
        # the reporting under test.
        warnings.simplefilter("ignore")
        r = p.simulate(b, T_END, **kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = p.device_loss_summary(b, r)
        currents = {d: np.asarray(r.i(d)) for d in DIODES}
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    return b, r, summary, currents, msgs


def _entry(summary, name):
    return next(e for e in summary if e["name"] == name)


ENGINES = ("pwl", "auto")
VTHS = (0.7, 0.0)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("V_th", VTHS)
def test_bridge_diode_charge_closes_against_the_dc_side(V_th, engine):
    """∫(i_D1 + i_D2) dt must equal the charge the DC side absorbed —
    ∫ v_p/R_L dt + C·Δv_p — computed from node voltages alone; the
    return pair D3 + D4 must carry the same charge back. Fixed engine
    measured at 3e-12 relative; the auto engine's coarse adaptive grid
    integrates to 1.6e-4 (trapezoid over 1.1k samples), hence 1e-3."""
    _, r, _, cur, _ = _run_bridge(V_th, engine)
    t = np.asarray(r.times)
    v_p = np.asarray(r.v("p"))
    q_dc = np.trapezoid(v_p / R_LOAD, t) + C_DC * (v_p[-1] - v_p[0])
    q_in = np.trapezoid(cur["BR__D1"] + cur["BR__D2"], t)
    q_ret = np.trapezoid(cur["BR__D3"] + cur["BR__D4"], t)
    tol = 1e-9 if engine == "pwl" else 1e-3
    assert q_dc > 0.1
    assert abs(q_in - q_dc) <= tol * q_dc, (q_in, q_dc)
    assert abs(q_ret - q_dc) <= tol * q_dc, (q_ret, q_dc)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("V_th", VTHS)
def test_bridge_diode_currents_match_the_source_state(V_th, engine):
    """The source current is a STATE variable — not reconstructed. At
    every sample the AC-side diodes must carry exactly it: −i_Vac =
    i_D1 − i_D4 (node a) = i_D2 ... with the sign of the AC half-wave.
    This is also what makes the V_th = 0.7 turn-on spike (below) a
    property of the simulation, not of the reporting."""
    _, r, _, cur, _ = _run_bridge(V_th, engine)
    i_src = -np.asarray(r.i("Vac"))
    err = np.abs(i_src - (cur["BR__D1"] - cur["BR__D4"]))
    assert float(err.max()) <= 1e-6 * float(np.abs(i_src).max())


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("V_th", VTHS)
def test_bridge_summary_reports_conduction_and_agrees_with_result_i(
        V_th, engine):
    _, r, summary, cur, _ = _run_bridge(V_th, engine)
    for name in DIODES:
        e = _entry(summary, name)
        assert e["duty_conducting"] > 0.02, (name, e)
        assert e["P_avg"] > 1e-3, (name, e)
        assert e["i_avg"] > 0.05, (name, e)
        # result.i() and the summary come from one reconstruction.
        assert e["i_peak"] == pytest.approx(
            float(np.abs(cur[name]).max()), rel=1e-12)
    for e in summary:
        if "i_rms" in e:
            assert e["i_rms"] <= e["i_peak"] * (1 + 1e-12), e


@pytest.mark.parametrize("engine", ENGINES)
def test_bridge_v_th_does_not_change_the_reported_charge(engine):
    """V_th = 0.7 and V_th = 0 simulate the same DC side (V_dc agrees
    to 0.01 V), so each diode's average current must agree. Measured:
    0.2 % (pwl). 1 % leaves room for the start-up cycle, where V_th
    delays the first turn-on. Duty differs physically by the time v_D
    takes to rise through V_th each half-cycle (measured 0.0026 pwl,
    0.0083 auto incl. the auto engine's own V_th=0 defect) — 0.015.

    The peaks and RMS are NOT compared: at V_th = 0.7 every turn-on
    collapses the diode voltage from 0.7 V to i/g_on into a ~3 mΩ loop,
    a ~345 A spike the V_th = 0 circuit does not have (the source-state
    test above shows it is in the simulation, not the reporting)."""
    _, r7, s7, _, _ = _run_bridge(0.7, engine)
    _, r0, s0, _, _ = _run_bridge(0.0, engine)
    for name in DIODES:
        e7, e0 = _entry(s7, name), _entry(s0, name)
        if engine == "pwl":
            assert e7["i_avg"] == pytest.approx(e0["i_avg"], rel=1e-2)
        assert abs(e7["duty_conducting"] - e0["duty_conducting"]) < 0.015


@pytest.mark.parametrize("V_th", VTHS)
def test_bridge_duty_agrees_across_engines_or_warns(V_th):
    """Time-weighted duty must not depend on how the engine clusters
    its samples. Measured at V_th = 0.7: pwl 0.1030 vs auto 0.1021
    (the old sample-weighted auto figure was 0.0027). At V_th = 0 the
    auto engine holds D1/D3 OFF while forward-biased by up to 3.4 V
    for ~1.3 ms (a missed commutation — see _diode_state) and its
    duty legitimately differs; that run must then SAY so."""
    _, _, s_pwl, _, _ = _run_bridge(V_th, "pwl")
    _, _, s_auto, _, msgs = _run_bridge(V_th, "auto")
    for name in DIODES:
        d_pwl = _entry(s_pwl, name)["duty_conducting"]
        d_auto = _entry(s_auto, name)["duty_conducting"]
        if abs(d_pwl - d_auto) > 0.005:
            assert any("commutation rule forbids" in m for m in msgs), (
                name, d_pwl, d_auto, msgs)


def test_auto_engine_v_th_threshold_landing_is_not_a_phantom():
    """82c567b1 reported i_peak = 700.0017 A (= g_on · V_th) for
    BR__D1 on engine='auto' while result.i() said 0.0017 A: an OFF→ON
    event landing is stored with the pre-flip (OFF) mask at v_D just
    past V_th. The reported peak must now be a current the source
    actually delivered."""
    _, r, summary, _, _ = _run_bridge(0.7, "auto")
    i_src_peak = float(np.abs(np.asarray(r.i("Vac"))).max())
    for name in ("BR__D1", "BR__D3"):
        assert _entry(summary, name)["i_peak"] <= i_src_peak * (1 + 1e-9)


def test_the_v_th_0_engine_defect_is_reported_not_hidden():
    """engine='auto' at V_th = 0 stores D1 OFF with v_D up to +3.4 V
    (the crossing signal was already positive when the step began, so
    no sign change was seen). The old rule turned that into a 3417 A
    phantom peak. Now the current follows what the solver stamped and
    a RuntimeWarning names the diode."""
    _, r, summary, _, msgs = _run_bridge(0.0, "auto")
    i_src_peak = float(np.abs(np.asarray(r.i("Vac"))).max())
    assert _entry(summary, "BR__D1")["i_peak"] <= i_src_peak * (1 + 1e-9)
    if any(_diode_state.diode_state_traces(_bridge(0.0), r)[bid]
           .n_rule_violations for bid in (2, 4)):
        assert any("BR__D1" in m and "commutation rule forbids" in m
                   for m in msgs), msgs


# ---------------------------------------------------------------------
# Three-phase VSI freewheel diodes (V_f is passed as V_th)
# ---------------------------------------------------------------------

def _vsi(V_f: float):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "bp", "bn", 300.0)
    b.add_resistor("Rgnd", "bn", "gnd", 1e-3)
    vsi = p.add_three_phase_vsi(b, "INV", vdc_pos="bp", vdc_neg="bn",
                                out_a="ua", out_b="ub", out_c="uc",
                                R_on=0.05, V_f=V_f)
    b.add_inductor("Lload", "ua", "mid", 2e-3)
    b.add_resistor("Rload", "mid", "bn", 5.0)
    b.add_resistor("Rb", "ub", "bn", 1e6)
    b.add_resistor("Rc", "uc", "bn", 1e6)
    hs = vsi.high_side_switch_indices[0]
    n_sw = b.graph.num_switches

    def chop(t, fsw=5e3, duty=0.5):
        m = p.SwitchStateMask(n_sw)
        if ((t * fsw) % 1.0) < duty:
            m.set(hs, True)
        return m
    return b, vsi, chop


@functools.lru_cache(maxsize=None)
def _run_vsi(V_f: float, engine: str):
    b, vsi, chop = _vsi(V_f)
    kw = dict(engine="pwl", dt=2e-6) if engine == "pwl" else {}
    r = p.simulate(b, 4e-3, switch_fn=chop, **kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = p.device_loss_summary(b, r, switch_fn=chop)
    return b, vsi, r, summary, [str(w.message) for w in caught]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("V_f", (0.7, 0.0))
def test_vsi_freewheel_diode_carries_the_inductor_current(V_f, engine):
    """With the high side open, the low-side diode is the inductor's
    only path: i_Dla must equal the inductor STATE current there, and
    its duty must match the 50 % off-time."""
    b, vsi, r, summary, msgs = _run_vsi(V_f, engine)
    assert not msgs, msgs
    t = np.asarray(r.times)
    i_l = np.asarray(r.i("Lload"))
    i_dl = np.asarray(r.i(vsi.diode_names[1]))
    m = t >= 2e-3
    assert float(i_l[m].min()) > 20.0
    # Every sample where the diode is reported conducting at all must
    # carry the full inductor current (HS open, LS never gated).
    sel = m & (np.abs(i_dl) > 1e-3)
    assert sel.sum() > 10
    assert np.allclose(i_dl[sel], i_l[sel], rtol=1e-6, atol=1e-6)
    e = _entry(summary, vsi.diode_names[1])
    assert 0.45 < e["duty_conducting"] < 0.55, e
    assert e["P_avg"] > 1.0, e      # ~R_d · i² · 0.5 ≈ 0.05 · 30² / 2
    hs = _entry(summary, "INV__HSa")
    assert 0.48 < hs["duty_closed"] < 0.52, hs


@pytest.mark.parametrize("engine", ENGINES)
def test_vsi_v_f_is_a_threshold_not_a_drop(engine):
    """V_f = 0.7 vs 0: the ON diode is 1/R_d either way, so the
    freewheel current and its conduction loss agree (measured identical
    to 1e-6 on pwl; the threshold only shifts the turn-on instant by
    the time ua needs to swing 0.7 V)."""
    *_, s7, _ = _run_vsi(0.7, engine)
    *_, s0, _ = _run_vsi(0.0, engine)
    e7 = _entry(s7, "INV__Dla")
    e0 = _entry(s0, "INV__Dla")
    assert e7["i_rms"] == pytest.approx(e0["i_rms"], rel=1e-3)
    assert e7["P_avg"] == pytest.approx(e0["P_avg"], rel=2e-3)


# ---------------------------------------------------------------------
# Single diode, and the case the result cannot decide
# ---------------------------------------------------------------------

def _half_wave(V_th: float, with_vcvs: bool = False):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "a", "gnd", v_dc=0.0, v_amplitude=10.0,
                              frequency=1e3, phase=0.0)
    b.add_diode("D1", "a", "k", 1e3, 1e-9, V_th=V_th)
    b.add_resistor("RL", "k", "gnd", 10.0)
    if with_vcvs:
        # A device whose terminal currents the reconstruction does not
        # model: KCL is switched off for the whole circuit.
        b.add_vcvs("E1", "k", "gnd", "e", "gnd", 1.0)
        b.add_resistor("Re", "e", "gnd", 1e3)
    return b


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("V_th", VTHS)
def test_single_diode_current_is_the_load_current(V_th, engine):
    """Series diode + resistor: i_D must BE the resistor current,
    sample by sample, whatever V_th is (82c567b1: zero at V_th=0.7)."""
    b = _half_wave(V_th)
    kw = dict(engine="pwl", dt=1e-6) if engine == "pwl" else {}
    r = p.simulate(b, 5e-3, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        i_d = np.asarray(r.i("D1"))
        s = p.device_loss_summary(b, r)
    i_r = np.asarray(r.v("k")) / 10.0
    assert np.allclose(i_d, i_r, rtol=1e-6, atol=1e-6)
    e = _entry(s, "D1")
    assert e["i_peak"] > 0.9, e
    assert 0.35 < e["duty_conducting"] < 0.55, e


def test_undecidable_band_samples_warn_and_name_the_diode():
    """With a device whose currents are not reconstructed, no node
    balance is trusted; the band samples fall back to the kernel's
    hysteresis (inherit the previous state) and that must be LOUD."""
    b = _half_wave(0.7, with_vcvs=True)
    r = p.simulate(b, 5e-3, engine="pwl", dt=1e-6)
    with pytest.warns(RuntimeWarning, match=r"diode 'D1'.*vcvs"):
        p.device_loss_summary(b, r)
    with pytest.warns(RuntimeWarning, match=r"diode 'D1'"):
        r.i("D1")


def test_v_th_zero_without_kcl_is_still_exact_and_quiet():
    """V_th = 0 has no band: the voltage alone decides, so even with
    KCL disabled there is nothing to warn about."""
    b = _half_wave(0.0, with_vcvs=True)
    r = p.simulate(b, 5e-3, engine="pwl", dt=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        i_d = np.asarray(r.i("D1"))
    assert np.allclose(i_d, np.asarray(r.v("k")) / 10.0,
                       rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# Reverse recovery, switch duty on the adaptive grid, time weighting
# ---------------------------------------------------------------------

def _buck(V_th: float):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_switch("Q1", "vin", "sw", g_on=1e3, g_off=1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=V_th)
    b.add_inductor("L1", "sw", "out", 100e-6)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 2.0)
    T = 1 / 50e3
    q1 = b.switch_index_of("Q1")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(q1, (t % T) < 0.5 * T)
        return m
    return b, pwm, T


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("V_th", VTHS)
def test_buck_reverse_recovery_uses_the_blocked_voltage(V_th, engine):
    """Hard-switched freewheel diode: every turn-off is followed by the
    full 48 V across it, so E_rr = Q_rr · 48 V per event. 82c567b1
    evaluated |v_D| AT the interpolated current zero (~1 mV) and
    reported E_sw_total = 0.0 on pwl, and found no turn-off at all when
    V_th = 0.7 (the fixed engine records events from v_D − V_th sign
    changes, which a conducting diode with V_th > 0 never makes)."""
    b, pwm, T = _buck(V_th)
    kw = dict(engine="pwl", dt=T / 200) if engine == "pwl" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 2e-3, switch_fn=pwm, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        s = p.device_loss_summary(b, r, switch_fn=pwm,
                                  diode_specs={"D1": {"Q_rr": 100e-9}})
    d = _entry(s, "D1")
    n = d["n_turn_off_events"]
    assert 98 <= n <= 100, d
    assert d["E_sw_total"] == pytest.approx(n * 100e-9 * 48.0, rel=0.01)
    assert 0.49 < d["duty_conducting"] < 0.51, d


def test_trbdf2_switch_mask_is_sampled_at_the_step_midpoint():
    """engine='auto' commits a gate-edge landing with the PRE-edge mask
    (switch_fn is evaluated at t + h/2). Re-evaluating switch_fn at the
    sample time read the post-edge state: 82c567b1 reported Q1
    duty_closed = 0.4545 for a 50 % PWM and raised the closed-loop
    desync warning on a stateless switch_fn."""
    b, pwm, _ = _buck(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 2e-3, switch_fn=pwm)
    assert r.engine_used == "trbdf2"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = p.device_loss_summary(b, r, switch_fn=pwm)
        i_q = np.asarray(r.i("Q1"))
    q = _entry(s, "Q1")
    assert q["duty_closed"] == pytest.approx(0.5, abs=2e-3)
    # KCL at the switch node: i_Q1 = i_L + i_D (diode gnd→sw).
    i_d = np.asarray(r.i("D1"))
    i_l = np.asarray(r.i("L1"))
    assert np.allclose(i_q + i_d, i_l, rtol=1e-6, atol=1e-6)


def test_time_fraction_weights_by_time_not_samples():
    """Ten samples clustered in the first 1 % of the run must not
    outvote the long stride after them."""
    t = np.concatenate([np.linspace(0.0, 0.01, 10), [1.0]])
    m = np.array([True] * 10 + [False])
    assert _time_fraction(m, t) == pytest.approx(0.01 + 0.99 / 2)
    assert m.mean() > 0.9
    tu = np.linspace(0.0, 1.0, 101)
    mu = tu < 0.25
    assert _time_fraction(mu, tu) == pytest.approx(mu.mean(), abs=0.01)
