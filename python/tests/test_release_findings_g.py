"""Release findings, group G: the thermal summary on branch-form devices
(F-2-1 = F-6-3), the refusal's ``.device`` under the LCP mode selection
(F-3-2) and the control-graph build refusal under ``control_tick='state'``
(F-3-5).

F-2-1: ``device_thermal_summary`` read a branch-form switch / diode's own
v·i and never applied ``conduction_specs`` (0.0998 W where the closed
form V_f0·I + r_on·I² is 7.991 W for r_on = 0.01) while ``device_loss_summary`` applied it. Both now take
a device's conduction loss from the same loss-summary entry, so they agree
for every device kind and form; the T_j trace follows the same model.
"""
from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

import pulsim as p

COND = {"V_f0": 0.7, "r_on": 0.02}
THERMAL_KINDS = {"resistor", "inductor", "switch", "diode", "shockley_diode",
                 "mosfet_level1", "igbt_level1", "nonlinear_diode"}


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f()


# ---------------------------------------------------------------------------
# F-2-1 / F-6-3: closed form on a diode carrying a known current
# ---------------------------------------------------------------------------

def _dc_device(form, device):
    """10 V into a conducting element with R_on = 1 mOhm (g_on = 1e3) and
    1 Ohm: a constant current, the element's own name 'D'."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vin", "gnd", 10.0)
    sf = None
    if device == "diode":
        b.add_diode("D", "vin", "mid", 1.0e3, 1.0e-9, 0.0, form=form)
    else:
        b.add_switch("D", "vin", "mid", 1.0e3, 1.0e-9, form=form)
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(b.switch_index_of("D"), True)
        sf = lambda t: m  # noqa: E731
    b.add_resistor("Rload", "mid", "gnd", 1.0)
    return b, _quiet(lambda: p.simulate(b, 2e-3, 1e-6, switch_fn=sf,
                                        start_from_dc_op=True)), sf


def _dc_diode(form):
    b, res, _ = _dc_device(form, "diode")
    return b, res


@pytest.mark.parametrize("device", ["diode", "switch"])
@pytest.mark.parametrize("form", ["conductance", "branch"])
def test_thermal_summary_applies_the_conduction_model_closed_form(form, device):
    b, res, sf = _dc_device(form, device)
    i_d = 10.0 / (1.0 + 1e-3)
    P = COND["V_f0"] * i_d + COND["r_on"] * i_d * i_d         # V_f0*I_avg + r_on*I_rms^2
    R_th = 2.0
    th = _quiet(lambda: p.device_thermal_summary(
        b, res, thermal_specs={"D": {"stages": [p.FosterStage(R_th, 1e-9)]}},
        switch_fn=sf, conduction_specs={"D": COND}))[0]
    assert th["P_cond_avg"] == pytest.approx(P, rel=1e-9)
    # tau << dt: the junction follows the power, T_j = T_amb + R_th*P(t)
    assert th["T_j_trace"][-1] == pytest.approx(25.0 + R_th * P, rel=1e-9)
    assert th["T_j_peak"] == pytest.approx(25.0 + R_th * P, rel=1e-9)


# ---------------------------------------------------------------------------
# Every device kind x form: the two summaries agree
# ---------------------------------------------------------------------------

def _buck(form, sw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    if sw == "mosfet":       # the helper with its body diode
        b.add_mosfet("Q1", "vin", "sw", R_on=0.05, form=form)
    elif sw == "igbt":
        b.add_igbt("Q1", "vin", "sw", R_on=0.05, reverse_blocking=False, form=form)
    elif sw == "mbd":
        b.add_mosfet_with_body_diode("Q1", "vin", "sw", R_on=0.05, form=form)
    else:                    # ideal (R_on = 0) branch-form switch and diode
        b.add_ideal_switch("Q1", "vin", "sw")
    if sw == "ideal":
        b.add_diode("D1", "gnd", "sw", form="branch")
    else:
        b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7, form=form)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 2.0)
    pwm = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"),
                               b.graph.num_switches)
    return b, dict(t_end=1e-3, switch_fn=pwm)


def _bridge(form, _sw):      # a composite: four diodes
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "n", 0.0, 230 * np.sqrt(2), 50.0)
    b.add_resistor("Rline", "ac", "x", 0.1)
    p.add_bridge_rectifier(b, "BR", ac_a="x", ac_b="n", dc_pos="dc",
                           dc_neg="gnd", V_th=0.7, form=form)
    b.add_capacitor("C", "dc", "gnd", 470e-6)
    b.add_resistor("RL", "dc", "gnd", 1e3)
    return b, dict(t_end=0.02)


def _shockley(_form, _sw):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0, 1e3)
    b.add_shockley_diode("D", "ac", "m")
    b.add_resistor("R", "m", "gnd", 10.0)
    return b, dict(t_end=2e-3)


CASES = ([("buck", f, s) for s in ("mosfet", "igbt", "mbd")
          for f in ("conductance", "branch")]
         + [("buck", "branch", "ideal")]
         + [("bridge", f, None) for f in ("conductance", "branch")]
         + [("shockley", None, None)])
BUILD = {"buck": _buck, "bridge": _bridge, "shockley": _shockley}


@functools.lru_cache(maxsize=None)
def _run(case, form, sw, engine):
    b, kw = BUILD[case](form, sw)
    if engine == "pwl":
        kw["dt"] = 1e-7 if case == "buck" else 1e-6
    return b, kw, _quiet(lambda: p.simulate(b, engine=engine, **kw))


@pytest.mark.parametrize("engine", ["pwl", "dsed"])
@pytest.mark.parametrize("case,form,sw", CASES,
                         ids=[f"{c}-{f}-{s}" for c, f, s in CASES])
@pytest.mark.parametrize("cond", [False, True], ids=["v*i", "cond_model"])
@pytest.mark.parametrize("window", [False, True], ids=["whole", "window"])
def test_thermal_power_equals_loss_summary(case, form, sw, engine, cond, window):
    if case == "shockley" and engine == "dsed":
        pytest.skip("the exact engine refuses a Shockley diode")
    b, kw, res = _run(case, form, sw, engine)
    comps = [c for c in b.components() if c["kind"] in THERMAL_KINDS]
    R_th = 1.0
    ts = {c["name"]: {"stages": [p.FosterStage(R_th, 1e-12)]} for c in comps}
    cs = ({c["name"]: COND for c in comps if c["kind"] in ("switch", "diode")}
          if cond else None)
    w = {"t_from": 0.5 * kw["t_end"]} if window else {}
    sf = kw.get("switch_fn")
    loss = {e["name"]: e for e in _quiet(lambda: p.device_loss_summary(
        b, res, switch_fn=sf, conduction_specs=cs, **w))}
    th = _quiet(lambda: p.device_thermal_summary(
        b, res, thermal_specs=ts, switch_fn=sf, conduction_specs=cs, **w))
    assert {r["name"] for r in th} == {n for n in ts if n in loss}
    P_max = max(r["P_total_avg"] for r in th)
    for r in th:
        P_loss = p.losses.conduction_power_avg(loss[r["name"]])
        assert r["P_cond_avg"] == P_loss          # the shared reading
        # The T_j trace is the thermal summary's own per-kind reconstruction:
        # with tau << dt it is T_amb + R_th*P(t), so its time average must be
        # R_th times the LOSS SUMMARY's power, per device (a sample mean of a
        # switching waveform: 3 %; before the fix 1.01 W against 2.56 W).
        dT = r["T_j_avg"] - r["T_ambient_C"]
        P_ref = P_loss + r["P_total_avg"] - r["P_cond_avg"]   # + switching / impulse
        assert dT == pytest.approx(R_th * P_ref, rel=0.03,
                                   abs=1e-6 * R_th * P_max), r["name"]


# ---------------------------------------------------------------------------
# thermal_specs on a device the thermal summary has no model for
# ---------------------------------------------------------------------------

def test_thermal_specs_on_an_unsupported_kind_is_refused_by_name():
    b, res = _dc_diode("conductance")
    b2 = p.CircuitBuilder()
    b2.add_voltage_source("V", "a", "gnd", 1.0)
    b2.add_resistor("R", "a", "c", 1.0)
    b2.add_capacitor("C", "c", "gnd", 1e-6)
    r2 = _quiet(lambda: p.simulate(b2, 1e-5, 1e-7))
    for builder, result, name in ((b, res, "Vdc"), (b2, r2, "C")):
        with pytest.raises(ValueError, match=repr(name)):
            p.device_thermal_summary(
                builder, result,
                thermal_specs={name: {"stages": [p.FosterStage(1.0, 1e-3)]}})


# ---------------------------------------------------------------------------
# F-3-2: a refusal's .device is the user's name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [
    "legacy",
    pytest.param("lcp", marks=pytest.mark.xfail(strict=True, reason=(
        "release finding F-3-2: run_transient.hpp builds mode_diode_names with "
        "pwl::branch_label ('D (branch #1)') and passes it as the UnsupportedError's "
        "device; the fix is C++ (pass graph.branch_name, keep the label in the "
        "message) — out of this Python-only slice"))),
])
def test_ideal_voltage_loop_refusal_names_the_device(mode):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_diode("D", "a", "gnd", form="branch")
    b.add_resistor("R", "a", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-6)
    with pytest.raises(p.UnsupportedError) as ei:
        _quiet(lambda: p.simulate(b, 1e-4, dt=1e-6, engine="pwl", mode_selection=mode))
    assert (ei.value.feature, ei.value.device, ei.value.kind) == (
        "ideal_voltage_loop", "D", "Diode")


# ---------------------------------------------------------------------------
# F-3-5: a control-graph build refusal keeps its type on both tick modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tick", ["label", "state"])
def test_rate_below_the_fixed_step_is_a_build_error_on_both_ticks(tick):
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 1e3)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    ch = p.MixedDomainBlockChain(rate=p.Rate(5e-8))
    ch.add("i", p.Integrator(), inputs=dict(x="out", dt="dt"), output="y")
    with pytest.raises(p.ControlGraphBuildError, match="shorter than the fixed"):
        p.simulate(b, 1e-5, engine="pwl", dt=1e-7, control_tick=tick,
                   step_observer=ch.make_step_observer(b))
