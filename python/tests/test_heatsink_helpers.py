"""Tests for the heatsink + TIM sizing helpers (P5)."""

from __future__ import annotations

import math

import pytest

import pulsim as p


def test_tim_resistance_physics() -> None:
    # R = thickness / (k · area). 0.1 mm grease (k=3) over 10×10 mm.
    R = p.tim_resistance(1e-4, 1e-4, material="thermal_grease")
    assert R == pytest.approx(1e-4 / (3.0 * 1e-4))
    # Explicit conductivity overrides the catalog.
    assert p.tim_resistance(2e-3, 5e-5, k_W_per_mK=5.0) == pytest.approx(
        5e-5 / (5.0 * 2e-3))


def test_tim_catalog_and_errors() -> None:
    assert "thermal_grease" in p.TIM_CATALOG
    assert p.TIM_CATALOG["bare_aluminium"] > 100.0   # ~ no insulator
    with pytest.raises(KeyError):
        p.tim_resistance(1e-3, 1e-4, material="unobtainium")
    with pytest.raises(ValueError):
        p.tim_resistance(1e-3, 1e-4)        # neither material nor k


def test_convection() -> None:
    assert p.convection_coefficient(0.0) == pytest.approx(10.45)
    assert p.convection_coefficient(5.0) == pytest.approx(
        10.45 - 5.0 + 10.0 * math.sqrt(5.0))
    R_still = p.convection_resistance(0.02, airflow_m_per_s=0.0)
    R_forced = p.convection_resistance(0.02, airflow_m_per_s=5.0)
    assert R_still == pytest.approx(1.0 / (10.45 * 0.02))
    assert R_forced < R_still           # forced air lowers R_th


def test_helpers_feed_the_shared_heatsink_api() -> None:
    """The helpers produce the R_th values the shared-heatsink API takes."""
    R_cs = p.tim_resistance(2.5e-4, 1e-4, material="silicone_pad")
    R_sa = p.convection_resistance(0.03, airflow_m_per_s=3.0)
    dev = p.HeatsinkDevice("Q", [p.FosterStage(R_th_K_per_W=0.4, tau_s=0.05)],
                           R_th_case_to_sink_K_per_W=R_cs)
    res = p.shared_heatsink_steady_state(
        [dev], {"Q": 20.0}, R_th_sink_to_amb_K_per_W=R_sa, T_amb_C=40.0)
    assert res["devices"]["Q"]["T_j_C"] > 40.0
