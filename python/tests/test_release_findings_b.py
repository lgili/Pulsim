"""Release findings, group B: event samples and result integrals.

Each finding of update-number-changing-release (r2/bucket-N.md) reproduced
on the PR line with the release behaviour turned on EXPLICITLY, against an
independent truth: a closed form, a conservation identity of the circuit
(charge through the diodes = charge into the DC side, from node voltages),
the fixed engine at a small dt, or the exact engine.

The storage convention these tests hold every engine to: a stored sample
at an event instant is the LEFT limit; the right limit, where the next
step starts, is kept beside the samples (``result.right_limit_*``), and
``pulsim.with_event_limits(result)`` is the trace with both.
"""

from __future__ import annotations

import functools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

sys.path.insert(0, str(Path(__file__).resolve().parent))

# the release behaviour, explicitly (the PR line's defaults are the old ones)
PWL_REL = dict(engine="pwl", discontinuity_restart=True,
               enable_substep_state_correction=True)
EXACT_REL = dict(engine="dsed", output_rtol=1e-5)

F_44 = pytest.mark.xfail(strict=True, reason=(
    "release finding F-4-4, rest: the window integrals are the integrator's quadrature "
    "(test_release_findings_b2: step-end windows C·dv to 1e-11), but a window END inside "
    "a restart step (or AT an event, keep='times' reading the right limit) reads the line's "
    "start value 2·w1·X1 + (1 − 2·w1)·X2, the right limit to O(h²) only: buck window 0 "
    "7.4e-4 at dt = 1 us (1.9e-4 at 0.5 us). Needs a true right limit (group A's "
    "consistent start x0+) and a mid-step row keeping the quadrature"))

def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ---------------------------------------------------------------------------
# A 325 V bridge (test_diode_state_reconstruction._bridge): V_th = 0.7 turns
# each diode on with a 1.4 V / 3 mOhm current jump.
# ---------------------------------------------------------------------------

C_DC, R_LOAD = 470e-6, 1000.0


def _bridge(V_th: float) -> p.CircuitBuilder:
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
def _bridge_run(V_th: float, kind: str):
    b = _bridge(V_th)
    kw = dict(PWL_REL, dt=1e-6) if kind == "pwl" else dict(EXACT_REL)
    return b, _quiet(p.simulate, b, 0.06, **kw)


def _charges(view):
    """(through D1 + D2, into the DC side from node voltages), both the
    trapezoid of the SAME trace."""
    t = np.asarray(view.times)
    v_p = np.asarray(view.v("p"))
    q_dc = np.trapezoid(v_p / R_LOAD, t) + C_DC * (v_p[-1] - v_p[0])
    q_in = np.trapezoid(np.asarray(view.i("BR__D1"))
                        + np.asarray(view.i("BR__D2")), t)
    return q_in, q_dc


def test_f54_exact_two_limit_trace_closes_the_bridge_charge():
    """F-5-4 (exact engine): the left-limit samples miss every V_th turn-on
    jump (-1.2e-3); with both limits the trapezoid closes the charge to the
    output rule's interpolation tolerance (measured 1.1e-4)."""
    _, r = _bridge_run(0.7, "exact")
    q_raw, q_dc = _charges(r)
    q_in, q_dc2 = _charges(p.with_event_limits(r))
    assert abs(q_raw / q_dc - 1) > 1e-3          # the left limits alone
    assert abs(q_in / q_dc2 - 1) <= 3e-4, (q_in, q_dc2)


def test_f56_exact_summary_peak_is_a_value_of_the_two_limit_trace():
    """F-5-6: the summary's 466.66 A is the turn-on's RIGHT limit (1.4 V /
    3 mOhm); it is in the two-limit trace, and the source's state carries
    the same peak there."""
    b, r = _bridge_run(0.7, "exact")
    s = {e["name"]: e for e in _quiet(p.device_loss_summary, b, r)}
    v = p.with_event_limits(r)
    for name in ("BR__D1", "BR__D2", "BR__D3", "BR__D4"):
        assert s[name]["i_peak"] == pytest.approx(
            float(np.abs(np.asarray(v.i(name))).max()), rel=1e-12)
    assert max(s[n]["i_peak"] for n in ("BR__D1", "BR__D3")) == pytest.approx(
        float(np.abs(np.asarray(v.i("Vac"))).max()), rel=1e-9)


@pytest.mark.parametrize("V_th", (0.7, 0.0))
def test_f54_f55_pwl_restart_steps_close_the_bridge_charge(V_th):
    """F-5-4 / F-5-5 (fixed engine, restart rule + split): integrated by
    the quadrature of the steps that produced them, the stored samples
    close the DC-side charge to round-off, as the restart-less trapezoidal
    engine did (3e-12); measured on the left-limit samples alone: 4.9e-3
    (V_th = 0.7), 1.3e-5 (V_th = 0)."""
    _, r = _bridge_run(V_th, "pwl")
    q_in, q_dc = _charges(p.with_event_limits(r))
    assert abs(q_in / q_dc - 1) <= 1e-9, (q_in, q_dc)


# ---------------------------------------------------------------------------
# F-6-2: test_loss_window's 230 V bridge on the exact engine
# ---------------------------------------------------------------------------

def test_f62_exact_time_average_of_the_two_limit_trace_is_the_summary():
    """The right limit of each turn-on (13.6 A) is stored; read with it,
    time_average equals the summary's i_avg (which integrates each step
    from its right limit) and steady_state_stats reads it too. The left
    limits alone read -0.21 %."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "n", 0.0, 230 * np.sqrt(2), 50.0)
    b.add_resistor("Rline", "ac", "x", 0.1)
    p.add_bridge_rectifier(b, "BR", ac_a="x", ac_b="n", dc_pos="dc",
                           dc_neg="gnd", V_th=0.7)
    b.add_capacitor("C", "dc", "gnd", 470e-6)
    b.add_resistor("RL", "dc", "gnd", 1e3)
    r = _quiet(p.simulate, b, 0.1, **EXACT_REL)
    s = {e["name"]: e for e in _quiet(p.device_loss_summary, b, r, t_from=0.08)}
    v = p.with_event_limits(r)
    avg = p.time_average(v.times, v.i("BR__D1"), t_from=0.08)
    assert avg == pytest.approx(s["BR__D1"]["i_avg"], rel=1e-9)
    st = p.steady_state_stats(r, "BR__D1", t_from=0.08, kind="i")
    assert st.mean == pytest.approx(avg, rel=1e-12)
    raw = p.time_average(r.times, r.i("BR__D1"), t_from=0.08)
    assert raw < avg * (1 - 1e-3)


# ---------------------------------------------------------------------------
# F-2-4: the thermal summary on the exact engine's output-rule grid
# ---------------------------------------------------------------------------

def _boost(V_th: float, R_load: float):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("Q", "sw", "gnd", g_on=100.0, g_off=1e-9)
    b.add_diode("D1", "sw", "out", 1e3, 1e-9, V_th=V_th)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("RL", "out", "gnd", R_load)
    T = 1e-5
    q = b.switch_index_of("Q")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(q, (t % T) < 0.5 * T)
        return m
    return b, pwm, T


@pytest.mark.parametrize("R_load", (500.0, 20.0))
def test_f24_exact_thermal_ratio_matches_the_fixed_engine(R_load):
    """(T_j_avg - 25)/R_th/P_cond_avg on the output-rule grid read 0.978 /
    1.011 (CCM Q / D1); truth: the fixed engine at T/1000 (0.9946 /
    0.9943), the exact engine on its event grid (0.9947 / 0.9943)."""
    st = [p.FosterStage(R_th_K_per_W=1000.0, tau_s=1e-4)]
    ratio = {}
    for kind, kw in (("pwl", dict(engine="pwl", dt=1e-8)), ("exact", EXACT_REL)):
        b, pwm, T = _boost(0.7, R_load)
        r = _quiet(p.simulate, b, 3e-3, switch_fn=pwm, **kw)
        th = _quiet(p.device_thermal_summary, b, r, switch_fn=pwm,
                    thermal_specs={"D1": {"stages": st}, "Q": {"stages": st}})
        for e in th:
            ratio[(kind, e["name"])] = (e["T_j_avg"] - 25.0) / 1e3 / e["P_cond_avg"]
            assert np.asarray(e["T_j_trace"]).shape == np.asarray(r.times).shape
    for name in ("D1", "Q"):
        assert ratio[("exact", name)] == pytest.approx(ratio[("pwl", name)],
                                                       abs=5e-4), ratio


# ---------------------------------------------------------------------------
# F-5-2: the design module's settle detector under the output rule
# ---------------------------------------------------------------------------

def test_f52_settle_detector_reads_the_landing_not_across_the_edge():
    import _design_reference as ref
    from pulsim.design import Evaluator, SteadySpec
    T = 1e-5

    def opaque_buck(d):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", ref.VIN)
        b.add_switch("S", "vin", "sw", ref.GON, ref.GOFF)
        b.add_diode("D1", "gnd", "sw", ref.GON, ref.GOFF)
        b.add_inductor("L", "sw", "x", d["L"])
        b.add_resistor("RLs", "x", "vout", d["R_L"])
        b.add_capacitor("C", "vout", "gnd", d["C"])
        b.add_resistor("RL", "vout", "gnd", d["R"])
        sf = p.make_pwm_switch_fn(1 / d["T"], d["D"], b.switch_index_of("S"),
                                  b.graph.num_switches)
        return b, sf.__call__        # no next_edge_after: polled edges
    ccm = dict(L=100e-6, C=47e-6, R=1.0, R_L=0.01, T=T, D=0.5)
    ev = Evaluator(opaque_buck, period=T, loads=("RL",), output="vout",
                   engine="dsed", options=dict(output_rtol=1e-5),
                   steady=SteadySpec(method="transient", min_periods=8,
                                     max_periods=256, tol=1e-6))
    rec = _quiet(ev.evaluate, ccm)
    assert rec.status == "ok", rec.reason


# ---------------------------------------------------------------------------
# F-6-4: the loss port's conduction energy on the restart steps
# ---------------------------------------------------------------------------

def _chopper():
    import test_device_losses_variable_step as m
    return m


@pytest.mark.parametrize("dt", (1e-7,))
def test_f64_loss_port_conduction_converges_on_restart_steps(dt):
    """RL chopper, closed-form ∫v·i per device. Restart rule: 1.86e-3 /
    2.0e-3 (S1 / D1), first order; restart-less 1.8e-4 / 2.4e-4."""
    m = _chopper()
    _, cond = m._truth()
    errs = {}
    for h in (dt, dt / 2):
        res = m._run("pwl", dt=h, discontinuity_restart=True,
                     enable_substep_state_correction=True)
        for n in ("S1", "D1"):
            errs[(h, n)] = abs(res.device_energy(n)["E_cond_electrical"] / cond[n] - 1)
    for n in ("S1", "D1"):
        assert errs[(dt, n)] <= 2.5e-4, errs
        # second order: halving dt quarters the error (3x leaves room)
        assert errs[(dt, n)] / errs[(dt / 2, n)] >= 3.0, errs


# ---------------------------------------------------------------------------
# F-6-5: a pulse current source's edge row
# ---------------------------------------------------------------------------

def test_f65_source_current_is_read_on_the_row_side_the_engine_solved():
    """The restart rule solves the row AT a source jump with the source's
    left limit (the row is the left limit); res.i(source) read the
    waveform's right value there: Rsq 1.0 A, Ip 0.0 A at t = 3e-4."""
    for kind in ("pulse", "pwl"):
        b = p.CircuitBuilder()
        if kind == "pulse":
            b.add_pulse_current_source("Ip", "q", "gnd", 0.0, 1.0, 1e-4, 2e-4,
                                       period=5e-4)
        else:   # a ramp into a flat segment: a corner, not a jump
            b.add_pwl_current_source("Ip", "q", "gnd", [0.0, 3e-4, 1e-3],
                                     [0.0, 1.0, 1.0])
        b.add_resistor("Rsq", "q", "q2", 1e-3)
        b.add_resistor("Rq", "q2", "gnd", 5.0)
        b.add_capacitor("C", "q2", "gnd", 1e-6)
        # the exact engine: its event row sits one ulp after the grid sample
        for kw in (dict(engine="pwl", dt=1e-6),
                   dict(engine="pwl", dt=1e-6, discontinuity_restart=True),
                   dict(engine="exact"), dict(engine="exact", output_rtol=1e-4)):
            res = _quiet(p.simulate, b, t_end=1e-3, **kw)
            a, c = np.asarray(res.i("Ip")), np.asarray(res.i("Rsq"))
            assert np.max(np.abs(a - c)) <= 1e-9, (kind, kw, np.asarray(res.times)[np.abs(a - c) > 1e-9])


# ---------------------------------------------------------------------------
# F-2-2 / F-5-7: the boost's diode charge and duty from the result
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R_load", (500.0, 20.0))
def test_f22_pwl_summary_charge_is_the_load_charge(R_load):
    """D1's i_avg x T against the load's KCL charge C·Δv + ∫v/R (same
    trace, same quadrature) over t >= 2 ms; restart rule + split at
    T/1000. Measured: DCM 5.6e-3 off; the restart-less engine 1e-10."""
    b, pwm, T = _boost(0.7, R_load)
    r = _quiet(p.simulate, b, 3e-3, switch_fn=pwm, dt=T / 1000, **PWL_REL)
    v = p.with_event_limits(r)
    t = np.asarray(v.times)
    k = int(np.searchsorted(t, 2e-3))
    tw, vo = t[k:], np.asarray(v.v("out"))[k:]
    q_load = 47e-6 * (vo[-1] - vo[0]) + np.trapezoid(vo / R_load, tw)
    q_d = np.trapezoid(np.asarray(v.i("D1"))[k:], tw)
    assert q_d == pytest.approx(q_load, rel=1e-9)
    s = {e["name"]: e for e in _quiet(p.device_loss_summary, b, r, switch_fn=pwm,
                                      t_from=float(np.asarray(r.times)[
                                          int(np.searchsorted(np.asarray(r.times), 2e-3))]))}
    assert s["D1"]["i_avg"] * (tw[-1] - tw[0]) == pytest.approx(q_load, rel=1e-6)


@pytest.mark.parametrize("V_th", (0.7, 0.0))
def test_f57_pwl_duty_credits_the_part_of_a_step_before_a_commutation(V_th):
    """DCM boost, pwl T/100 with restart + split: D1 duty 0.21713 (V_th = 0)
    / 0.21543 (0.7); truth, the exact engine: 0.22177 / 0.22010."""
    b, pwm, T = _boost(V_th, 500.0)
    r_pwl = _quiet(p.simulate, b, 3e-3, switch_fn=pwm, dt=T / 100, **PWL_REL)
    r_ex = _quiet(p.simulate, b, 3e-3, switch_fn=pwm, engine="dsed")
    d = [{e["name"]: e for e in _quiet(p.device_loss_summary, b, r,
                                       switch_fn=pwm)}["D1"]["duty_conducting"]
         for r in (r_pwl, r_ex)]
    assert abs(d[0] - d[1]) < 2e-3, d


# ---------------------------------------------------------------------------
# F-4-4: the in-loop recorder's window integrals on restart steps
# ---------------------------------------------------------------------------

@F_44
def test_f44_recorder_window_charge_on_restart_steps():
    """∫i_C over a window = C·Δv_C (the integrator's own identity; 2e-4 is
    the trapezoidal engine's residue at the window ends). Restart rule:
    4e-4 off on this buck."""
    import test_recorder_device_readings as m
    b, fn, t_end, _, _ = m._buck()
    wins = [(0.07 * t_end + 1.234e-9, 0.29 * t_end + 4.321e-9),
            (0.31 * t_end + 2.2e-9, 0.97 * t_end)]
    kw = dict(PWL_REL, dt=1e-6, switch_fn=fn)
    res = _quiet(p.simulate, b, t_end, recorders=[
        p.Recorder("r", ["i(C)", "v(out)"], windows=wins)], **kw)
    res2 = _quiet(p.simulate, b, t_end, recorders=[
        p.Recorder("r", ["v(out)"], keep="times", times=[x for w in wins for x in w])], **kw)
    q = res.recorded["r"].windows["i(C)"]["integral"]
    ends = res2.recorded["r"]["v(out)"]
    for k in range(len(wins)):
        dq = 10e-6 * (ends[2 * k + 1] - ends[2 * k])
        assert abs(q[k] - dq) <= 2e-4 * abs(dq) + 1e-12, (k, q[k], dq)


def test_decimated_rule_run_keeps_no_event_limits():
    """store_every > 1 stores some steps only: the result keeps no event
    limits (None) rather than a partial record that reads as "no jumps"."""
    b, pwm, T = _boost(0.7, 20.0)
    r = _quiet(p.simulate, b, 2e-4, switch_fn=pwm, dt=T / 1000, store_every=4, **PWL_REL)
    assert r.right_limit_index is None and r.inner_events is None
    r1 = _quiet(p.simulate, b, 2e-4, switch_fn=pwm, dt=T / 1000, **PWL_REL)
    assert r1.right_limit_index is not None and len(r1.right_limit_index) > 0


# ---------------------------------------------------------------------------
# Integration (batch 14q): the thermal summary hands its event-limit view to
# the loss summary; the inner events must be inserted ONCE
# ---------------------------------------------------------------------------

def test_inner_events_are_inserted_once_through_the_thermal_summary():
    """device_thermal_summary inserts the inner events, then calls
    device_loss_summary on that view, which inserted them again (a view
    reads every attribute it does not set from its base). Truth: the loss
    summary of the result itself, read through the one shared conduction
    function (group G). The bridge's diode turn-ons fall inside steps."""
    b = _bridge(0.7)
    r = _quiet(p.simulate, b, 0.02, dt=1e-5, **PWL_REL)
    assert r.inner_events is not None and len(r.inner_events["times"]) > 0
    once = p._result_views.with_inner_events(r)
    twice = p._result_views.with_inner_events(once)
    assert len(twice.times) == len(once.times)
    loss = {e["name"]: e for e in _quiet(p.device_loss_summary, b, r)}
    st = [p.FosterStage(R_th_K_per_W=1.0, tau_s=1e-4)]
    diodes = [n for n, e in loss.items() if e.get("kind") == "diode"]
    assert diodes
    th = _quiet(p.device_thermal_summary, b, r,
                thermal_specs={n: {"stages": st} for n in diodes})
    assert len(th) == len(diodes)
    for e in th:
        assert e["P_cond_avg"] == pytest.approx(
            p.losses.conduction_power_avg(loss[e["name"]]), rel=1e-9), e["name"]
