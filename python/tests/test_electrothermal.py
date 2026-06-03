"""Tests for temperature-dependent loss + closed-loop electro-thermal
coupling (P2) on a shared heatsink.

Conduction loss rises with junction temperature (Rds_on(T)), so the
steady state is a self-consistent fixed point: a model with Rds_on fixed
at 25 °C reports an optimistic T_j and cannot see thermal runaway. These
tests pin the feedback analytically (closed form + runaway detection via
the spectral radius of the loss-temperature feedback) and through the
live solver.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulsim as p


def _foster(R, tau):
    return [p.FosterStage(R_th_K_per_W=R, tau_s=tau)]


def test_tempco_loss_power_at() -> None:
    m = p.TempCoLoss(P_cond_ref_W=10.0, a_cond_per_C=0.01, T_ref_C=25.0)
    assert m.power_at(25.0) == pytest.approx(10.0)
    # +0.4…+0.8 %/°C is typical; 1 %/°C doubles Rds_on over 100 °C.
    assert m.power_at(125.0) == pytest.approx(20.0)
    assert m.dP_dT_W_per_C == pytest.approx(0.1)


def test_electrothermal_feedback_raises_tj_above_fixed_loss() -> None:
    """With a positive conduction tempco, the self-consistent T_j is
    higher than the fixed-25 °C-loss prediction — the Rds_on(T) feedback."""
    A = p.HeatsinkDevice("A", _foster(2.0, 0.05), R_th_case_to_sink_K_per_W=0.0)
    m = p.TempCoLoss(P_cond_ref_W=10.0, a_cond_per_C=0.01, T_ref_C=25.0)

    et = p.electrothermal_steady_state(
        [A], {"A": m}, R_th_sink_to_amb_K_per_W=0.0, T_amb_C=25.0)
    fixed = p.shared_heatsink_steady_state(
        [A], {"A": 10.0}, R_th_sink_to_amb_K_per_W=0.0, T_amb_C=25.0)

    assert et["converged"] and not et["runaway"]
    assert et["feedback_gain"] == pytest.approx(0.2)   # ρ(M·K) = 2·0.1
    assert et["devices"]["A"]["T_j_C"] == pytest.approx(50.0)   # vs 45 fixed
    assert fixed["devices"]["A"]["T_j_C"] == pytest.approx(45.0)
    # Converged dissipation grew because the device runs hotter.
    assert et["final_powers_W"]["A"] == pytest.approx(12.5)


def test_electrothermal_reduces_to_p1_when_tempco_zero() -> None:
    A = p.HeatsinkDevice("A", _foster(2.0, 0.05), R_th_case_to_sink_K_per_W=0.5)
    m0 = p.TempCoLoss(P_cond_ref_W=10.0, a_cond_per_C=0.0)
    et = p.electrothermal_steady_state(
        [A], {"A": m0}, R_th_sink_to_amb_K_per_W=1.0, T_amb_C=25.0)
    p1 = p.shared_heatsink_steady_state(
        [A], {"A": 10.0}, R_th_sink_to_amb_K_per_W=1.0, T_amb_C=25.0)
    assert et["feedback_gain"] == 0.0
    assert et["devices"]["A"]["T_j_C"] == pytest.approx(
        p1["devices"]["A"]["T_j_C"])


def test_thermal_runaway_detected() -> None:
    """When the loss-temperature feedback gain reaches 1, there is no
    stable equilibrium — flagged, not silently returned as a number."""
    A = p.HeatsinkDevice("A", _foster(2.0, 0.05))
    m_hot = p.TempCoLoss(P_cond_ref_W=10.0, a_cond_per_C=0.06)   # k=0.6, G=1.2
    r = p.electrothermal_steady_state(
        [A], {"A": m_hot}, R_th_sink_to_amb_K_per_W=0.0, T_amb_C=25.0)
    assert r["runaway"] is True
    assert r["converged"] is False
    assert r["feedback_gain"] == pytest.approx(1.2)
    assert "runaway" in r["message"].lower()


def test_coupled_electrothermal_two_devices() -> None:
    devs = [p.HeatsinkDevice(n, _foster(2.0, 0.05),
                             R_th_case_to_sink_K_per_W=0.5)
            for n in ("A", "B")]
    m = p.TempCoLoss(P_cond_ref_W=10.0, a_cond_per_C=0.005)   # k=0.05
    et = p.electrothermal_steady_state(
        devs, {"A": m, "B": m}, R_th_sink_to_amb_K_per_W=1.0, T_amb_C=25.0)
    fixed = p.shared_heatsink_steady_state(
        devs, {"A": 10.0, "B": 10.0},
        R_th_sink_to_amb_K_per_W=1.0, T_amb_C=25.0)
    # Coupled + feedback runs hotter than the fixed-loss coupled case.
    assert et["devices"]["A"]["T_j_C"] == pytest.approx(83.06, abs=0.05)
    assert fixed["devices"]["A"]["T_j_C"] == pytest.approx(70.0)
    assert et["feedback_gain"] < 1.0


def test_live_electrothermal_transient_matches_steady_state() -> None:
    """The live closed-loop observer (power recomputed from T_j each
    step) settles to the analytic electro-thermal steady state."""
    A = p.HeatsinkDevice("A", _foster(2.0, 0.02), R_th_case_to_sink_K_per_W=0.0)
    m = p.TempCoLoss(P_cond_ref_W=10.0, a_cond_per_C=0.01, T_ref_C=25.0)

    b = p.CircuitBuilder()
    hs = p.add_shared_heatsink(
        b, [A], R_th_sink_to_amb_K_per_W=1.0, C_th_sink_J_per_K=1.0,
        T_amb_C=25.0)
    obs, bex = p.make_electrothermal_heatsink_observer(b, hs, {"A": m})
    res = p.simulate(b, 12.0, 1e-3, step_observer=obs, b_extra_fn=bex,
                     start_from_dc_op=True)

    x_final = np.asarray(res.states)[-1]
    T_j = obs.read_T_j(x_final)["A"]
    ref = p.electrothermal_steady_state(
        [A], {"A": m}, R_th_sink_to_amb_K_per_W=1.0, T_amb_C=25.0)
    # Closed form: ρ=0.3, T_j = 47.5/0.7 ≈ 67.86 °C (vs 55 with fixed loss).
    assert ref["devices"]["A"]["T_j_C"] == pytest.approx(67.857, abs=0.01)
    assert T_j == pytest.approx(ref["devices"]["A"]["T_j_C"], abs=0.6)
