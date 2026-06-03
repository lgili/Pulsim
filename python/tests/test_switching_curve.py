"""Tests for nonlinear datasheet E_sw(I) curves (P4).

The default switching-loss annotation scales a single reference energy
linearly with current (``E ∝ I/I_ref``). Real E_on/E_off/E_rr vs current
curves are nonlinear. ``E_on_curve`` / ``E_off_curve`` (switches) and
``E_rr_curve`` (diodes) interpolate the datasheet curve at the actual
switching current per event, with linear extrapolation beyond the
tabulated range.
"""

from __future__ import annotations

import pytest

import pulsim as p
from pulsim.losses import _interp_energy_curve


# ---------------------------------------------------------------------------
# Curve interpolation math
# ---------------------------------------------------------------------------

def test_interp_energy_curve() -> None:
    curve = [(0.0, 0.0), (10.0, 1.0), (20.0, 3.0)]
    assert _interp_energy_curve(5.0, curve)[0] == pytest.approx(0.5)   # interp
    assert _interp_energy_curve(15.0, curve)[0] == pytest.approx(2.0)
    # Linear extrapolation above the curve (last slope = 0.2 J/A).
    assert _interp_energy_curve(30.0, curve)[0] == pytest.approx(5.0)
    # Below-range extrapolation clamps at 0 (energy is non-negative).
    assert _interp_energy_curve(-5.0, curve)[0] == 0.0
    # Single point → constant.
    assert _interp_energy_curve(7.0, [(10.0, 2.0)])[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Switch E_on/E_off curve through the real loss reconstruction
# ---------------------------------------------------------------------------

def _toggled_switch():
    """100 V → ideal switch → 10 Ω, toggled at 1 kHz. When closed the
    switch carries ~10 A; when open it blocks ~100 V."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vin", "gnd", 100.0)
    b.add_switch("S", "vin", "mid", 1.0e3, 1.0e-9)
    b.add_resistor("Rload", "mid", "gnd", 10.0)
    sw = p.make_pwm_switch_fn(frequency=1.0e3, duty=0.5,
                              num_switches=b.graph.num_switches, switch_idx=0)
    res = p.simulate(b, 10.0e-3, 2.0e-6, switch_fn=sw, start_from_dc_op=True)
    return b, res, sw


def _switch_psw(b, res, sw, spec):
    ls = p.device_loss_summary(b, res, switch_fn=sw, switch_specs={"S": spec})
    return next(e for e in ls if e["name"] == "S")["P_sw_avg"]


def test_switch_curve_scales_with_energy() -> None:
    b, res, sw = _toggled_switch()
    p1 = _switch_psw(b, res, sw, {"E_on_curve": [(10.0, 1e-3)], "V_ref": 100.0})
    p2 = _switch_psw(b, res, sw, {"E_on_curve": [(10.0, 2e-3)], "V_ref": 100.0})
    assert p1 > 0.0
    assert p2 == pytest.approx(2.0 * p1, rel=1e-6)


def test_switch_curve_matches_linear_at_the_curve_point() -> None:
    """A single curve point at the operating current equals the linear
    single-point scaling there."""
    b, res, sw = _toggled_switch()
    p_curve = _switch_psw(b, res, sw,
                          {"E_on_curve": [(10.0, 1e-3)], "V_ref": 100.0})
    p_lin = _switch_psw(b, res, sw,
                        {"E_on_ref": 1e-3, "V_ref": 100.0, "I_ref": 10.0})
    assert p_curve == pytest.approx(p_lin, rel=2e-3)


def test_switch_curve_extrapolates_beyond_datasheet() -> None:
    """At ~10 A, a curve tabulated only to 5 A extrapolates to ~2× the
    5 A energy (linear extrapolation), not clamped at the 5 A value."""
    b, res, sw = _toggled_switch()
    p_at10 = _switch_psw(b, res, sw,
                         {"E_on_curve": [(10.0, 1e-3)], "V_ref": 100.0})
    p_extrap = _switch_psw(b, res, sw,
                           {"E_on_curve": [(0.0, 0.0), (5.0, 1e-3)],
                            "V_ref": 100.0})
    assert p_extrap == pytest.approx(2.0 * p_at10, rel=2e-2)


# ---------------------------------------------------------------------------
# Diode E_rr curve
# ---------------------------------------------------------------------------

def test_diode_err_curve_scales_with_energy() -> None:
    """Freewheel diode in a buck: its E_rr(I) curve, interpolated at the
    forward current at each commutation, scales the recovery loss."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vin", "gnd", 100.0)
    b.add_switch("S", "vin", "sw", 1.0e3, 1.0e-9)
    b.add_diode("Dfw", "gnd", "sw", 1.0e3, 1.0e-9, 0.0)
    b.add_inductor("L", "sw", "out", 1.0e-3)
    b.add_capacitor("C", "out", "gnd", 1.0e-4)
    b.add_resistor("R", "out", "gnd", 10.0)
    sw = p.make_pwm_switch_fn(frequency=1.0e3, duty=0.5,
                              num_switches=b.graph.num_switches, switch_idx=0)
    res = p.simulate(b, 10.0e-3, 1.0e-6, switch_fn=sw, start_from_dc_op=True)

    def prr(spec):
        ls = p.device_loss_summary(b, res, switch_fn=sw,
                                   diode_specs={"Dfw": spec})
        return next(e for e in ls if e["name"] == "Dfw").get("P_sw_avg", 0.0)

    p1 = prr({"E_rr_curve": [(0.0, 0.0), (50.0, 1e-6)], "V_R_ref": 100.0})
    p2 = prr({"E_rr_curve": [(0.0, 0.0), (50.0, 2e-6)], "V_R_ref": 100.0})
    assert p1 > 0.0
    assert p2 == pytest.approx(2.0 * p1, rel=1e-6)
