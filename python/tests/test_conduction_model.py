"""Tests for the offset+slope conduction loss model (P3).

The default conduction reconstruction is pure-resistive (``v²·g``). Real
IGBTs and diodes follow ``V = V_f0 + r·I``, whose forward-voltage offset
dominates the conduction loss at low current. ``conduction_specs`` opts a
device into ``P_cond = V_f0·|I| + r_on·I²`` applied to the *actual*
current trace — in both ``device_loss_summary`` (reported power) and
``device_thermal_summary`` (the T_j-driving trace).
"""

from __future__ import annotations

import pytest

import pulsim as p


def _dc_diode_circuit():
    """10 V source → ideal diode → 1 Ω load: a forward diode carrying a
    steady ~10 A. The diode's tiny on-state drop makes the resistive
    reconstruction nearly zero, so the V_f0 offset clearly dominates."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vin", "gnd", 10.0)
    b.add_diode("D", "vin", "mid", 1.0e3, 1.0e-9, 0.0)
    b.add_resistor("Rload", "mid", "gnd", 1.0)
    res = p.simulate(b, 2.0e-3, 1.0e-6, start_from_dc_op=True)
    return b, res


def test_conduction_model_formula() -> None:
    b, res = _dc_diode_circuit()
    ls = p.device_loss_summary(
        b, res, conduction_specs={"D": {"V_f0": 0.7, "r_on": 0.01}})
    d = next(e for e in ls if e["name"] == "D")
    assert "P_cond_model_avg" in d
    # P_cond_model_avg = V_f0·i_avg + r_on·i_rms²
    assert d["P_cond_model_avg"] == pytest.approx(
        0.7 * d["i_avg"] + 0.01 * d["i_rms"] ** 2, rel=1e-6)
    assert d["V_f0"] == 0.7 and d["r_on"] == 0.01
    # The offset term dwarfs the near-zero pure-resistive reconstruction.
    assert d["P_cond_model_avg"] > d["P_avg"]
    assert d["P_cond_model_avg"] > 5.0     # ≈ 0.7·10 + 0.01·100 = 8 W


def test_conduction_model_datasheet_aliases() -> None:
    """IGBT-style V_ce0/r_ce aliases resolve to the same model."""
    b, res = _dc_diode_circuit()
    ls = p.device_loss_summary(
        b, res, conduction_specs={"D": {"V_ce0": 0.7, "r_ce": 0.01}})
    d = next(e for e in ls if e["name"] == "D")
    assert d["V_f0"] == 0.7 and d["r_on"] == 0.01


def test_conduction_specs_unknown_device_raises() -> None:
    b, res = _dc_diode_circuit()
    with pytest.raises(KeyError):
        p.device_loss_summary(
            b, res, conduction_specs={"NOPE": {"V_f0": 0.7}})


def test_thermal_summary_uses_conduction_model() -> None:
    """A device opted into the offset+slope model runs hotter than the
    pure-resistive reconstruction, because its conduction loss is real."""
    b, res = _dc_diode_circuit()
    tspec = {"D": {"stages": [p.FosterStage(R_th_K_per_W=1.0, tau_s=0.005)]}}
    with_model = p.device_thermal_summary(
        b, res, thermal_specs=tspec,
        conduction_specs={"D": {"V_f0": 0.7, "r_on": 0.01}},
        T_ambient_C=25.0)
    resistive = p.device_thermal_summary(
        b, res, thermal_specs=tspec, T_ambient_C=25.0)

    # The conduction power fed to the thermal network is the offset+slope
    # value (≈ 0.7·10 + 0.01·100 = 8 W), not the near-zero resistive one —
    # this is the settling-independent evidence the model is wired in.
    assert with_model[0]["P_cond_avg"] == pytest.approx(7.99, abs=0.1)
    assert with_model[0]["P_cond_avg"] > resistive[0]["P_cond_avg"]
    # Hotter junction follows (exact ΔT depends on Foster settling).
    assert with_model[0]["T_j_avg"] > resistive[0]["T_j_avg"]
