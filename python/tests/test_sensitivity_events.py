"""add-parameter-sensitivities, group 6: sensitivities ACROSS EVENTS on the
time-stepping engines (design D3, DECISIONS 16).

At an event τ(p) the charges are continuous and dq/dp jumps by
(q̇⁻ − q̇⁺)·dτ/dp; dτ/dp = −g_x·S⁻ / (g_x·ẋ⁻) for a diode crossing, T at the
OFF edge of a trailing-edge PWM for its duty, 0 for a fixed gate edge.

Truths, each independent of the simulator:
- closed forms derived here (piecewise exponentials whose event instant
  depends on the parameter), differentiated by the complex step on the
  closed form itself (exact to rounding); each case reports how much of the
  answer is the dτ/dp term;
- the DCM buck table of tasks.md (the exact path's prototype values).
"""

import cmath
import warnings

import numpy as np
import pytest

import pulsim as p

from . import _sens_reference as ref


def _run(b, t_end, engine, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, engine=engine, **kw)


def _cstep(f, x, h=1e-30):
    """d f / d x of an analytic closed form, by the complex step."""
    return f(x + 1j * h * x).imag / (h * x)


PWL_SPLIT = dict(dt=1e-8, enable_substep_state_correction=True, discontinuity_restart=True)
TRBDF2 = dict(rtol=1e-10, atol=1e-13)


# ------------------------------------------- a diode turn-on (state event)
# V -> R -> c ; C: c-gnd ; diode D: c -> k (V_th = 0.7, V_f = 0: the ON law
# g_on*v jumps from g_off*V_th to g_on*V_th at turn-on) ; R2: k-gnd.
# OFF: c sees R and (1/g_off + R2); D turns on at v_c = V_th*(1 + R2*g_off).
# ON: c sees R and (1/g_on + R2). t_end = tau + tau2: the jump is not washed
# out by the fast post-event mode.
V, VTH, GON, GOFF, R2 = 10.0, 0.7, 100.0, 1e-9, 10.0
R0, C0 = 1e3, 1e-6


def _diode_on(R=R0, C=C0):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_resistor("R", "in", "c", R)
    b.add_capacitor("C", "c", "gnd", C)
    b.add_diode("D", "c", "k", GON, GOFF, V_th=VTH)
    b.add_resistor("R2", "k", "gnd", R2)
    return b


def _diode_on_closed(R, C, t_end=None, parts=False):
    Rb = 1 / GOFF + R2
    vinf1, tau1 = V * Rb / (R + Rb), C * R * Rb / (R + Rb)
    von = VTH * Rb * GOFF
    tau = -tau1 * cmath.log(1 - von / vinf1)
    Ra = 1 / GON + R2
    vinf2, tau2 = V * Ra / (R + Ra), C * R * Ra / (R + Ra)
    if parts:
        return tau, vinf2, tau2, von
    return vinf2 + (von - vinf2) * cmath.exp(-(t_end - tau) / tau2)


TAU_ON = _diode_on_closed(R0, C0, parts=True)[0].real
T_END_ON = TAU_ON + _diode_on_closed(R0, C0, parts=True)[2].real


def _truth_on(name, t_end=T_END_ON):
    if name == "R.R":
        return _cstep(lambda r: _diode_on_closed(r, C0, t_end), R0)
    return _cstep(lambda c: _diode_on_closed(R0, c, t_end), C0)


def _dtau_share_on(name):
    """The part of the closed form's derivative carried by dτ/dp."""
    tau, vinf2, tau2, von = (z.real for z in _diode_on_closed(R0, C0, parts=True))
    dv_dtau = (von - vinf2) / tau2 * np.exp(-(T_END_ON - tau) / tau2)
    x0 = R0 if name == "R.R" else C0
    f = ((lambda r: _diode_on_closed(r, C0, parts=True)[0]) if name == "R.R"
         else (lambda c: _diode_on_closed(R0, c, parts=True)[0]))
    return dv_dtau * _cstep(f, x0) / _truth_on(name)


@pytest.mark.parametrize("engine,kw", [("pwl", PWL_SPLIT), ("trbdf2", TRBDF2)])
@pytest.mark.parametrize("name", ["R.R", "C.C"])
def test_diode_turn_on_against_the_closed_form(engine, kw, name):
    """6.1/6.3: a state event whose instant moves with R and C; the dτ/dp
    term is most of the answer (asserted), so dropping it fails."""
    share = _dtau_share_on(name)
    assert share > 0.5, share
    r = _run(_diode_on(), T_END_ON, engine, sensitivities=[name], **kw)
    s = r.sens("c", name)[-1]
    t_last = float(r.sensitivity.times[-1])   # pwl ends on its grid
    assert abs(t_last - T_END_ON) <= 1e-8
    truth = _truth_on(name, t_last)
    assert s == pytest.approx(truth, rel=1e-5), (s, truth, share)


# ------------------------------------------- a duty (parameter time event)
# One pulse: S on for D*T (C charges through R + 1/g_on), then off until
# t_end < T (through R + 1/g_off). d v / d D is ENTIRELY the jump term.
TP, D0, VS = 1e-3, 0.3037, 10.0
GS_ON, GS_OFF = 1e3, 1e-9


def _pulse_rc(R=R0, C=C0):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", VS)
    b.add_switch("S", "in", "a", GS_ON, GS_OFF)
    b.add_resistor("R", "a", "c", R)
    b.add_capacitor("C", "c", "gnd", C)
    return b


def _pulse_closed(R, C, D, t_end):
    t1 = D * TP
    tau1, tau2 = C * (R + 1 / GS_ON), C * (R + 1 / GS_OFF)
    v1 = VS * (1 - cmath.exp(-t1 / tau1))
    return VS - (VS - v1) * cmath.exp(-(t_end - t1) / tau2)


@pytest.mark.parametrize("engine,kw", [("pwl", dict(dt=1e-6,
                                         enable_substep_state_correction=True, discontinuity_restart=True)),
                                       ("trbdf2", TRBDF2)])
def test_duty_of_one_pulse_against_the_closed_form(engine, kw):
    t_end = 0.8 * TP
    b = _pulse_rc()
    sf = p.make_pwm_switch_fn(1 / TP, D0, b.switch_index_of("S"),
                              b.graph.num_switches)
    names = ["S.duty", "R.R", "C.C"]
    r = _run(b, t_end, engine, switch_fn=sf, sensitivities=names, **kw)
    assert r.sensitivity.params == names
    truths = {"S.duty": _cstep(lambda d: _pulse_closed(R0, C0, d, t_end), D0),
              "R.R": _cstep(lambda x: _pulse_closed(x, C0, D0, t_end), R0),
              "C.C": _cstep(lambda x: _pulse_closed(R0, x, D0, t_end), C0)}
    for nm in names:
        s = r.sens("c", nm)[-1]
        assert s == pytest.approx(truths[nm], rel=1e-5), (nm, s, truths[nm])


# ------------------------------------------------ the DCM buck (task 6.3)
def _buck(L=100e-6, R=50.0, C=47e-6, Vin=24.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", Vin)
    b.add_switch("S", "vin", "sw", ref.GON, ref.GOFF)
    b.add_diode("D1", "gnd", "sw", ref.GON, ref.GOFF)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", C)
    b.add_resistor("RL", "vout", "gnd", R)
    return b


@pytest.mark.parametrize("engine,kw", [
    pytest.param("pwl", dict(dt=ref.T / 1000, enable_substep_state_correction=True,
                             discontinuity_restart=True), marks=pytest.mark.xfail(
        strict=True, raises=p.UnsupportedError,
        reason="OPEN (DECISIONS 16, merge onto the PR line): the restart's "
               "consistent start (jump series) moves the state at the turn-off at "
               "29.524 T (cut-set record of L, 1.7e-10 V*s, 1.5e-16 J); dJ/dp is "
               "not carried, refused by name. It was a number (within 1e-5) "
               "before the jump series existed")),
    ("trbdf2", dict(rtol=1e-10, atol=1e-13)),
    pytest.param("trbdf2", dict(rtol=1e-9, atol=1e-12), marks=pytest.mark.xfail(
        strict=True, raises=p.UnsupportedError,
        reason="OPEN (DECISIONS 16): at rtol 1e-9 a turn-off falls within the "
               "conditioning floor of a step start (flipped at t), refused by name"))])
def test_dcm_buck_transient_against_the_table(engine, kw):
    """6.3: ∂v_C/∂L, ∂R, ∂D at 40 T within 1e-5 of the exact path's table."""
    b = _buck()
    sf = p.make_pwm_switch_fn(1 / ref.T, 0.3, b.switch_index_of("S"),
                              b.graph.num_switches)
    r = _run(b, 40 * ref.T, engine, switch_fn=sf,
             sensitivities=["L.L", "RL.R", "S.duty"], **kw)
    for nm, key in (("L.L", "L"), ("RL.R", "R"), ("S.duty", "D")):
        s = r.sens("vout", nm)[-1]
        assert s == pytest.approx(ref.DCM_TRANSIENT[key], rel=1e-5), (nm, s)


# ------------------------------------------------------------- refusals
def test_unlocated_diode_event_on_pwl_is_refused_naming_the_diode():
    """6.3: the fixed engine without event splitting refuses a diode event,
    naming the diode: the event sits on the grid (a staircase in p)."""
    with pytest.raises(p.UnsupportedError) as ei:
        _run(_diode_on(), T_END_ON, "pwl", dt=1e-7, sensitivities=["C.C"])
    assert (ei.value.engine, ei.value.feature) == ("pwl", "sensitivities")
    assert "'D' turns ON" in str(ei.value)
    assert "enable_substep_state_correction" in str(ei.value)


def test_unlocated_duty_edge_on_pwl_is_refused():
    b = _pulse_rc()
    sf = p.make_pwm_switch_fn(1 / TP, D0, b.switch_index_of("S"),
                              b.graph.num_switches)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(b, 0.8 * TP, "pwl", dt=1e-6, switch_fn=sf,
             sensitivities=["S.duty"])
    assert "staircase" in str(ei.value)


def test_fixed_gate_edges_without_splitting_are_carried():
    """A gate edge of a fixed schedule has dτ/dp = 0: carried without event
    splitting (the mask sequence does not move with p), and the gradient is
    the derivative of the same discrete map (central differences)."""
    def run(R=R0, sens=None):
        b = _pulse_rc(R=R)
        sf = p.make_pwm_switch_fn(1 / TP, D0, b.switch_index_of("S"),
                                  b.graph.num_switches)
        kw = dict(sensitivities=sens) if sens else {}
        return _run(b, 0.8 * TP, "pwl", dt=1e-6, switch_fn=sf, **kw)
    s = run(sens=["R.R"]).sens("c", "R.R")[-1]
    h = 1e-3 * R0
    fd = (np.asarray(run(R=R0 + h).v("c"))[-1]
          - np.asarray(run(R=R0 - h).v("c"))[-1]) / (2 * h)
    assert s == pytest.approx(fd, rel=1e-6)


def test_moving_event_with_a_newton_device_is_refused():
    """n·V_T = 0.26 V: the junction leaves v_c free to reach the PWL diode's
    threshold (an earlier version used n = 0.05, which clamped v_c at 36 mV:
    no event happened and the number it returned was right)."""
    b = _diode_on()
    b.add_shockley_diode("Dx", "c", "gnd", 1e-8, 10.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(b, T_END_ON, "pwl", sensitivities=["R.R"], **PWL_SPLIT)
    assert "Newton device" in str(ei.value)
    assert "(D)" in str(ei.value)
