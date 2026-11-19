"""add-branch-form-ideal-switch 6.1: the LCP decides ideal diodes from their
crossing signals and reproduces the ideal-loop release (DECISIONS #12).

An ideal diode (R_on = 0) is a port in OFF coordinates: the selector reads the
current i_s it would carry and the gap w = V_f - v it would block, on the
site's network with the diode cut out. A loop of sources and closed ideal
elements shorts the port (Z_jj = 0): OFF when the loop reverse-biases it, no
mode when it forward-biases it. A dependent structure held by the diode is
released when its constraint force (i_s ON, w OFF) changes sign.

Four closed forms, pwl with the SDIRK x3 restart and the event split, each run
under the default settle and under mode_selection='lcp':

* clamp: 1 A, 1 kHz sine current into C || R || ideal diode. The diode holds
  v = 0 (a capacitor loop; force i_s = I(t)) until I(t) changes sign at T/2;
  then RC driven by the sine until v returns to 0.
* cut-set: 10 V sine (phase -pi/4) -> ideal diode -> 10 ohm + 5 mH. OFF, the
  diode holds i_L = 0 (force w = -v_s) until v_s changes sign; ON, the RL
  current of the classical half-wave rectifier until its extinction angle.
* RL buck in CCM: 24 V, ideal switch and freewheeling diode, 1 mH + 10 ohm,
  50 kHz, D = 0.5, from rest. At every turn-on the source, the closed switch
  and the ON diode form an ideal voltage loop that reverse-biases the diode:
  the default settle releases it structurally (counted), the LCP finds OFF
  directly (no structural release).
* leg: a half-bridge of ideal switches with ideal antiparallel diodes at
  V_f = 0 into R + L. Each diode sits across a closed switch, shorted (its
  gap is the loop's residual, zero or a few ulps); it stays OFF (no refusal)
  and the load current is the RL response to the +-V square wave. Under
  'lcp' the shorted ports are held OFF before Lemke (counted: shorted_held);
  that the hold is independent of the residual's SIGN is shown by the C++
  selector case, which injects +-1e-13 (this run cannot choose the sign).
"""
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

import pulsim as p

F = 1e3
W = 2 * np.pi * F
T = 1 / F
KW = {"engine": "pwl", "discontinuity_restart": True,
      "enable_substep_state_correction": True}


def _sim(b, mode, **kw):
    # Legacy subject: mode None is "the default settle" the LCP is compared with; the
    # release default mode_selection='auto' picks 'lcp' here, so the settle is named
    extra = {"mode_selection": mode or "legacy"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, **KW, **kw, **extra)


def _clamp(N, mode, periods=3):
    I0, C, R = 1.0, 10e-6, 100.0
    b = p.CircuitBuilder()
    b.add_sine_current_source("I", "n", "gnd", 0.0, I0, F)
    b.add_diode("D", "n", "gnd", form="branch", R_on=0.0, g_off=0.0)
    b.add_capacitor("C", "n", "gnd", C, 0.0)
    b.add_resistor("R", "n", "gnd", R)
    r = _sim(b, mode, t_end=periods * T, dt=T / N)
    t, v = np.asarray(r.times), np.asarray(r.v("n"))
    tau = R * C
    A, ph = I0 * R / np.hypot(1.0, W * tau), np.arctan(W * tau)

    def vp(tt):
        return A * np.sin(W * tt - ph)

    t1 = T / 2
    d = brentq(lambda s: vp(t1 + s) - vp(t1) * np.exp(-s / tau), 1e-9, 0.99 * T)
    vc = np.zeros_like(t)
    for k in range(periods):
        a = k * T + t1
        m = (t >= a) & (t <= a + d)
        vc[m] = vp(t[m] - k * T) - vp(t1) * np.exp(-(t[m] - a) / tau)
    return r, v, vc


def _cutset(N, mode, periods=3):
    V0, R, L = 10.0, 10.0, 5e-3
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", 0.0, V0, F, -np.pi / 4)
    b.add_diode("D", "s", "k", form="branch", R_on=0.0, g_off=0.0)
    b.add_resistor("R", "k", "m", R)
    b.add_inductor("L", "m", "gnd", L, 0.0)
    r = _sim(b, mode, t_end=periods * T, dt=T / N)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    Z, phi = np.hypot(R, W * L), np.arctan2(W * L, R)

    def ion(th):
        return V0 / Z * (np.sin(th - phi) + np.sin(phi) * np.exp(-th / np.tan(phi)))

    beta = brentq(ion, np.pi / 2, 2 * np.pi - 1e-9)   # extinction angle
    ic = np.zeros_like(t)
    for k in range(periods):
        a = k * T + (np.pi / 4) / W
        m = (t >= a) & (t <= a + beta / W)
        ic[m] = ion(W * (t[m] - a))
    return r, i, ic


def _rl_buck(N, mode, periods=20):
    V, L, R, Tb, D = 24.0, 1e-3, 10.0, 20e-6, 0.5
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V)
    b.add_ideal_switch("S", "vin", "sw", 0.0, 0.0)
    b.add_diode("D", "gnd", "sw", form="branch", R_on=0.0, g_off=0.0)
    b.add_inductor("L", "sw", "out", L, 0.0)
    b.add_resistor("R", "out", "gnd", R)
    fn = p.make_pwm_switch_fn(1 / Tb, D, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    r = _sim(b, mode, t_end=periods * Tb, dt=Tb / N, switch_fn=fn)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    tau, ic, i0 = L / R, np.empty_like(t), 0.0
    for k in range(periods):
        a = k * Tb
        on = (t >= a) & (t <= a + D * Tb)
        ic[on] = V / R + (i0 - V / R) * np.exp(-(t[on] - a) / tau)
        i1 = V / R + (i0 - V / R) * np.exp(-D * Tb / tau)
        off = (t > a + D * Tb) & (t <= a + Tb)
        ic[off] = i1 * np.exp(-(t[off] - a - D * Tb) / tau)
        i0 = i1 * np.exp(-(1 - D) * Tb / tau)
    return r, i, ic


def _leg(N, mode, periods=10):
    # half-bridge leg, ideal switches with ideal antiparallel diodes at
    # V_f = 0: each diode sits across a closed switch, shorted, and must stay
    # OFF (the switches carry both directions); the load sees the +-V square wave
    V, L, R, Tb = 200.0, 1e-3, 5.0, 1e-4
    b = p.CircuitBuilder()
    b.add_voltage_source("Vp", "p", "gnd", V)
    b.add_voltage_source("Vn", "gnd", "n", V)
    b.add_ideal_switch("S1", "p", "a", 0.0, 0.0)
    b.add_ideal_switch("S2", "a", "n", 0.0, 0.0)
    b.add_diode("D1", "a", "p", form="branch", R_on=0.0, g_off=0.0)
    b.add_diode("D2", "n", "a", form="branch", R_on=0.0, g_off=0.0)
    b.add_inductor("L", "a", "m", L, 0.0)
    b.add_resistor("R", "m", "gnd", R)
    ns = b.graph.num_switches
    fn = p.make_combined_switch_fn(ns, [
        p.make_pwm_switch_fn(1 / Tb, 0.5, b.switch_index_of("S1"), ns, 0.0),
        p.make_pwm_switch_fn(1 / Tb, 0.5, b.switch_index_of("S2"), ns, Tb / 2)])
    r = _sim(b, mode, t_end=periods * Tb, dt=Tb / N, switch_fn=fn)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    tau, ic, i0 = L / R, np.empty_like(t), 0.0
    for k in range(2 * periods):
        a, v = k * Tb / 2, (V if k % 2 == 0 else -V)
        m = (t >= a) & (t <= a + Tb / 2)
        ic[m] = v / R + (i0 - v / R) * np.exp(-(t[m] - a) / tau)
        i0 = v / R + (i0 - v / R) * np.exp(-Tb / 2 / tau)
    return r, i, ic


# measured max |error| at the coarser N (both selections give the same run);
# the bound leaves a factor 1.5 for platform rounding
CASES = {
    "clamp": (_clamp, 500, 3.30e-4),
    "cutset": (_cutset, 500, 5.34e-6),
    "rl_buck": (_rl_buck, 500, 6.64e-9),
    "leg": (_leg, 100, 1.58e-5),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def runs(request):
    make, N, err = CASES[request.param]
    out = {(mode, n): make(n, mode) for mode in (None, "lcp") for n in (N, 2 * N)}
    return request.param, N, err, out


def test_closed_form_second_order_under_both_selections(runs):
    name, N, bound, out = runs
    for mode in (None, "lcp"):
        e1 = np.abs(out[(mode, N)][1] - out[(mode, N)][2]).max()
        e2 = np.abs(out[(mode, 2 * N)][1] - out[(mode, 2 * N)][2]).max()
        assert e1 <= 3 * bound, (name, mode, e1)
        assert np.log2(e1 / e2) >= 1.85, (name, mode, e1, e2)


def test_lcp_takes_the_same_modes_and_releases_the_loop_itself(runs):
    name, N, _, out = runs
    r0, y0, _ = out[(None, N)]
    r1, y1, _ = out[("lcp", N)]
    # the waveforms carry the modes (no exact switch-state compare: a tie
    # broken by round-off would flip it on another platform)
    assert np.abs(y1 - y0).max() <= 1e-9 * max(1.0, np.abs(y0).max())
    ms = r1.diagnostics["mode_selection"]
    assert ms["inconsistent"] == 0 and ms["errors"] == 0
    # consistent entries of the dependent structures: nothing re-initialised
    assert len(r0.reinit_records) == 0 and len(r1.reinit_records) == 0
    if name == "rl_buck":
        assert r0.counters["ideal_loop_releases"] > 0      # the settle's structural release
        assert r1.counters["ideal_loop_releases"] == 0     # the LCP decided OFF itself
    if name == "leg":
        # every gate edge poses the LCP with a diode across the closed switch:
        # shorted (Z_jj = 0 structurally), held OFF, never released
        assert ms["shorted_held"] > 0 and ms["shorted_released"] == 0


@pytest.mark.parametrize("split", [False, True])
def test_a_flip_forced_by_a_gate_edge_is_stamped_at_the_edge(split):
    # the restart samples the gate at the step start: the freewheeling diode
    # turns ON in its first stage, at the gate-off edge (was stamped one dt late)
    V, L, R, Tb, N = 24.0, 1e-3, 10.0, 20e-6, 100
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V)
    b.add_ideal_switch("S", "vin", "sw", 0.0, 0.0)
    b.add_diode("D", "gnd", "sw", form="branch", R_on=0.0, g_off=0.0)
    b.add_inductor("L", "sw", "out", L, 0.0)
    b.add_resistor("R", "out", "gnd", R)
    fn = p.make_pwm_switch_fn(1 / Tb, 0.5, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, engine="pwl", t_end=3 * Tb, dt=Tb / N, switch_fn=fn,
                       discontinuity_restart=True, enable_substep_state_correction=split)
    on = [e.t_estimated for e in r.commutation_events if e.new_state]
    assert len(on) == 3
    for k, t in enumerate(on):
        assert abs(t - (k + 0.5) * Tb) <= 1e-3 * Tb / N, (k, t)


@pytest.mark.parametrize("mode", [None, "lcp"])
def test_a_loop_that_forward_biases_an_ideal_diode_is_refused_by_name(mode):
    # the LCP has no mode (Z_jj = 0, q_j < 0); the settle releases the diode
    # and sees it turn back ON: both refuse with the same feature
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    b.add_resistor("R", "a", "gnd", 10.0)
    b.add_diode("D", "a", "gnd", form="branch", R_on=0.0, g_off=0.0)
    with pytest.raises(p.UnsupportedError) as e:
        _sim(b, mode, t_end=1e-5, dt=1e-6)
    assert e.value.feature == "ideal_voltage_loop" and e.value.device.startswith("D")


def _loop_order(msg):
    """The element names of the message's '(loop: a -> b -> ...)'."""
    return msg.split("loop: ", 1)[1].split(")", 1)[0].split(" -> ")


@pytest.mark.parametrize("mode", [None, "lcp"])
def test_a_loop_of_several_ideal_diodes_is_named_element_by_element(mode):
    # V1 -> D1 -> D2 -> D3 -> back: no ideal port is shorted on its own (the
    # LCP poses each open), so the refusal must find the loop through all three
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", 10.0)
    b.add_diode("D1", "a", "b", form="branch", R_on=0.0, g_off=1e-9)
    b.add_diode("D2", "b", "c", form="branch", R_on=0.0, g_off=1e-9)
    b.add_diode("D3", "c", "gnd", form="branch", R_on=0.0, g_off=1e-9)
    b.add_resistor("R0", "b", "gnd", 10.0)
    with pytest.raises(p.UnsupportedError) as e:
        _sim(b, mode, t_end=1e-4, dt=1e-5)
    assert e.value.feature == "ideal_voltage_loop"
    loop = _loop_order(str(e.value))
    ring = ["V1", "D1", "D2", "D3"]
    rots = [ring[k:] + ring[:k] for k in range(4)]
    assert loop in rots + [r[::-1] for r in rots], loop


@pytest.mark.parametrize("R_on", [1e-300, 1e-12])
def test_an_r_on_below_the_port_resolution_is_refused_by_name(R_on):
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match=r'add_diode\("D"\): R_on = .* below 1e-09'):
        b.add_diode("D", "a", "k", form="branch", R_on=R_on, g_off=1e-9)


def test_a_micro_ohm_r_on_runs_as_a_resistor():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_diode("D", "a", "k", form="branch", R_on=1e-6, g_off=1e-9)
    b.add_resistor("R", "k", "gnd", 1.0)
    r = _sim(b, None, t_end=1e-5, dt=1e-6)
    assert np.asarray(r.v("k"))[-1] == pytest.approx(1.0 / (1.0 + 1e-6), rel=1e-12)
