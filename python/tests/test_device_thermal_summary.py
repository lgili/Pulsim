"""Tests for the post-hoc `device_thermal_summary` glue helper.

Wires the device-loss summary output (P_cond + P_sw_avg) into a
Foster thermal network and computes T_j(t) per device — closes the
"need example chaining loss → thermal" gap from the v1.6 audit.
"""

from __future__ import annotations

import math

import pytest

import pulsim as p


def _dc_resistor_with_thermal_spec():
    """Two-resistor DC divider. R_top dissipates a known steady-state
    power; we compute T_j on R_top through a 2-stage Foster network."""
    R_top, R_bot, V_dc = 2.0, 3.0, 5.0
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_dc)
    b.add_resistor("R_top", "vin", "mid", R_top)
    b.add_resistor("R_bot", "mid", "gnd", R_bot)
    return b, R_top, R_bot, V_dc


def test_device_thermal_summary_steady_state_matches_p_rth() -> None:
    """A DC resistor running long enough to settle the Foster
    network must converge to T_amb + P_avg · R_th_total."""
    b, R_top, R_bot, V_dc = _dc_resistor_with_thermal_spec()
    i = V_dc / (R_top + R_bot)
    P_top = i * i * R_top

    # Two-stage Foster, slow taus so we know "long enough" is well-
    # defined.
    stages = [p.FosterStage(R_th_K_per_W=0.5, tau_s=1e-3),
              p.FosterStage(R_th_K_per_W=0.3, tau_s=5e-3)]
    R_th_total = sum(s.R_th_K_per_W for s in stages)
    T_amb = 25.0
    T_inf = T_amb + P_top * R_th_total

    # Run 50× the slowest tau ⇒ T_j should be within < 0.1 % of T_inf.
    t_end = 50.0 * stages[-1].tau_s
    res = p.simulate(b, t_end=t_end, dt=stages[0].tau_s / 200)

    out = p.device_thermal_summary(
        b, res,
        thermal_specs={"R_top": {"stages": stages,
                                       "T_ambient_C": T_amb}},
    )
    assert len(out) == 1
    e = out[0]
    assert e["name"] == "R_top"
    assert e["kind"] == "resistor"
    assert math.isclose(e["P_cond_avg"], P_top, rel_tol=2e-2)
    assert math.isclose(e["P_total_avg"], P_top, rel_tol=2e-2)
    assert math.isclose(e["R_th_total"], R_th_total, rel_tol=1e-9)
    # Steady-state junction temperature.
    assert math.isclose(e["T_j_peak"], T_inf, rel_tol=2e-3)
    assert math.isclose(e["T_j_trace"][-1], T_inf, rel_tol=2e-3)
    # T_j must rise monotonically from T_amb (no overshoot for a
    # passive Foster network on a step input).
    assert e["T_j_trace"][0] == pytest.approx(T_amb, rel=1e-3)
    assert e["T_j_trace"][-1] > e["T_j_trace"][0]


def test_device_thermal_summary_switching_loss_folds_in() -> None:
    """PWM switch with non-zero E_on_ref / E_off_ref must boost
    T_j by exactly P_sw_avg · R_th_total in steady state."""
    V_dc, R = 10.0, 1.0
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_dc)
    b.add_resistor("R1", "vin", "mid", R)
    b.add_switch("SW1", "mid", "gnd", g_on=100.0, g_off=1e-9)

    f_sw = 5_000.0
    T_sw = 1.0 / f_sw

    def pwm(t: float):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(0, bool((t % T_sw) < 0.5 * T_sw))
        return m

    # Short, fast Foster ⇒ settles in well under the simulation
    # window.
    stages = [p.FosterStage(R_th_K_per_W=0.5, tau_s=200e-6)]
    R_th = stages[0].R_th_K_per_W

    n_periods = 200
    res = p.simulate(b, t_end=n_periods * T_sw, dt=T_sw / 50,
                       switch_fn=pwm)

    # No switch_specs — expect ZERO switching loss contribution.
    out_no_sw = p.device_thermal_summary(
        b, res,
        thermal_specs={"SW1": {"stages": stages,
                                 "T_ambient_C": 25.0}},
        switch_fn=pwm,
    )
    T_j_peak_no_sw = out_no_sw[0]["T_j_peak"]
    P_cond = out_no_sw[0]["P_cond_avg"]

    # WITH switch_specs the P_sw_avg lifts T_j.
    sw_spec = {"SW1": {"E_on_ref": 100e-6, "E_off_ref": 150e-6,
                            "V_ref": V_dc, "I_ref": V_dc / R}}
    out = p.device_thermal_summary(
        b, res,
        thermal_specs={"SW1": {"stages": stages,
                                 "T_ambient_C": 25.0}},
        switch_fn=pwm,
        switch_specs=sw_spec,
    )
    e = out[0]
    P_sw = e["P_sw_avg"]
    assert P_sw > 0.0
    # ΔT from the switching loss alone ≈ P_sw · R_th.
    expected_delta = P_sw * R_th
    actual_delta = e["T_j_peak"] - T_j_peak_no_sw
    assert math.isclose(actual_delta, expected_delta, rel_tol=5e-2)
    # P_total = P_cond + P_sw
    assert math.isclose(e["P_total_avg"], P_cond + P_sw, rel_tol=1e-2)


def test_device_thermal_summary_diode_includes_q_rr() -> None:
    """Diode in a half-wave rectifier with Q_rr datasheet annotation
    — T_j must include both conduction and reverse-recovery losses."""
    f0 = 1_000.0
    V_pk = 10.0
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "a", "gnd",
                                v_dc=0.0, v_amplitude=V_pk,
                                frequency=f0, phase=0.0)
    b.add_resistor("R1", "a", "b", 1.0)
    b.add_diode("D1", "b", "gnd", g_on=100.0, g_off=1e-6, V_th=0.0)
    res = p.simulate(b, t_end=20.0 / f0, dt=1.0 / (f0 * 200))

    stages = [p.FosterStage(R_th_K_per_W=2.0, tau_s=1e-3)]
    out = p.device_thermal_summary(
        b, res,
        thermal_specs={"D1": {"stages": stages}},
        diode_specs={"D1": {"Q_rr": 100e-9}},
    )
    e = next(x for x in out if x["kind"] == "diode")
    assert e["P_cond_avg"] > 0.0
    assert e["P_sw_avg"] > 0.0
    # T_j should rise above ambient.
    assert e["T_j_peak"] > 25.0
    # No nonsense values.
    assert math.isfinite(e["T_j_avg"])
    assert math.isfinite(e["T_j_peak"])
    assert e["T_j_peak"] >= e["T_j_avg"]


def test_device_thermal_summary_per_device_ambient_override() -> None:
    """The per-device `T_ambient_C` overrides the top-level
    default — useful for staggered hot-spot heatsinks."""
    b, R_top, R_bot, V_dc = _dc_resistor_with_thermal_spec()
    stages = [p.FosterStage(R_th_K_per_W=0.5, tau_s=1e-3)]
    t_end = 20 * stages[-1].tau_s
    res = p.simulate(b, t_end=t_end, dt=stages[0].tau_s / 100)

    out = p.device_thermal_summary(
        b, res,
        thermal_specs={
            "R_top": {"stages": stages, "T_ambient_C": 60.0},
            "R_bot": {"stages": stages},  # falls back to T_ambient_C=25.0
        },
        T_ambient_C=25.0,
    )
    by_name = {e["name"]: e for e in out}
    assert by_name["R_top"]["T_ambient_C"] == pytest.approx(60.0)
    assert by_name["R_bot"]["T_ambient_C"] == pytest.approx(25.0)
    # The 35 °C delta must show up in the steady-state T_j.
    assert (by_name["R_top"]["T_j_trace"][-1]
            - by_name["R_bot"]["T_j_trace"][-1]) >= 30.0


def test_device_thermal_summary_unknown_device_name_errors() -> None:
    b, *_ = _dc_resistor_with_thermal_spec()
    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    with pytest.raises(KeyError, match="no device named"):
        p.device_thermal_summary(
            b, res,
            thermal_specs={"NOT_A_DEVICE":
                                {"stages": [p.FosterStage(0.5, 1e-3)]}},
        )


def test_device_thermal_summary_missing_stages_errors() -> None:
    b, *_ = _dc_resistor_with_thermal_spec()
    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    with pytest.raises(ValueError, match="stages"):
        p.device_thermal_summary(
            b, res,
            thermal_specs={"R_top": {"T_ambient_C": 25.0}},
        )


def test_device_thermal_summary_switch_requires_switch_fn() -> None:
    """A switch listed in thermal_specs without `switch_fn` must
    error — the per-step mask is needed to reconstruct P_cond(t)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_resistor("R1", "vin", "mid", 1.0)
    b.add_switch("SW1", "mid", "gnd", g_on=100.0, g_off=1e-9)

    def pwm(t: float):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(0, bool((t * 1000) % 1 < 0.5))
        return m

    res = p.simulate(b, t_end=1e-3, dt=1e-5, switch_fn=pwm)
    with pytest.raises(ValueError, match="switch_fn"):
        p.device_thermal_summary(
            b, res,
            thermal_specs={"SW1":
                                {"stages": [p.FosterStage(0.5, 1e-3)]}},
            # switch_fn intentionally omitted
        )


def test_device_thermal_summary_empty_specs_yields_empty_list() -> None:
    b, *_ = _dc_resistor_with_thermal_spec()
    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    assert p.device_thermal_summary(b, res, thermal_specs={}) == []


def test_device_thermal_summary_inductor_core_loss_folds_in() -> None:
    """Inductor with Steinmetz core-loss spec: P_core_avg must
    appear on the thermal entry AND drive T_j above ambient.
    Confirms the pipeline `core_loss_specs → P_core_avg → T_j`
    is wired end-to-end (regression for the bug where the field
    was computed but omitted from the output dict)."""
    f0 = 1000.0
    V_pk = 50.0
    R, L = 5.0, 1e-3
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "vin", "gnd",
                                v_dc=0.0, v_amplitude=V_pk,
                                frequency=f0, phase=0.0)
    b.add_inductor("L1", "vin", "mid", L)
    b.add_resistor("R1", "mid", "gnd", R)
    res = p.simulate(b, t_end=20.0 / f0, dt=1.0 / (f0 * 200))

    stages = [p.FosterStage(R_th_K_per_W=10.0, tau_s=1e-3)]
    out = p.device_thermal_summary(
        b, res,
        thermal_specs={"L1": {"stages": stages,
                                 "T_ambient_C": 25.0}},
        core_loss_specs={"L1": {"material": "N87",
                                     "N_turns": 50,
                                     "A_core": 1e-4,
                                     "V_core": 5e-6}},
    )
    e = next(x for x in out if x["kind"] == "inductor")
    assert "P_core_avg" in e
    assert e["P_core_avg"] > 0.0
    # Ideal inductor → P_cond ≈ 0; total drive comes from P_core.
    assert math.isclose(e["P_cond_avg"], 0.0, abs_tol=1e-9)
    assert math.isclose(e["P_total_avg"], e["P_core_avg"],
                          rel_tol=1e-9)
    # T_j must rise above ambient.
    assert e["T_j_peak"] > e["T_ambient_C"]


def test_inductor_core_loss_bad_geometry_raises() -> None:
    """`_inductor_core_loss` previously silently returned 0 for
    non-positive geometry; now it must raise so the user notices
    the bad spec."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "vin", "gnd",
                                v_dc=0.0, v_amplitude=10.0,
                                frequency=1000.0, phase=0.0)
    b.add_inductor("L1", "vin", "mid", 1e-3)
    b.add_resistor("R1", "mid", "gnd", 5.0)
    res = p.simulate(b, t_end=2e-3, dt=5e-6)
    with pytest.raises(ValueError, match="strictly positive"):
        p.device_loss_summary(
            b, res,
            core_loss_specs={"L1": {"material": "N87",
                                         "N_turns": 0,  # invalid
                                         "A_core": 1e-4,
                                         "V_core": 5e-6}},
        )
