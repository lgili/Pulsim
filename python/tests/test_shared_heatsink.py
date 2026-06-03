"""Tests for the shared-heatsink thermal model (P1).

Several power devices mounted on ONE heatsink are thermally COUPLED:
their dissipations sum at the shared sink, so the sink temperature is
driven by the total power and that rise lifts every junction together.
A per-device (independent) model misses this — these tests pin the
coupling both analytically (``shared_heatsink_steady_state``) and through
the live solver (``add_shared_heatsink`` + ``make_heatsink_observer``).
"""

from __future__ import annotations

import numpy as np
import pytest

import pulsim as p


def _foster(R, tau):
    return [p.FosterStage(R_th_K_per_W=R, tau_s=tau)]


# ---------------------------------------------------------------------------
# Analytic steady state
# ---------------------------------------------------------------------------

def test_steady_state_coupling_zero_power_device_still_heats() -> None:
    """The signature of shared-heatsink coupling: a device dissipating
    *nothing* still runs hot because the shared sink is heated by its
    neighbours."""
    A = p.HeatsinkDevice("A", _foster(2.0, 0.05), R_th_case_to_sink_K_per_W=0.5)
    B = p.HeatsinkDevice("B", _foster(2.0, 0.05), R_th_case_to_sink_K_per_W=0.5)
    r = p.shared_heatsink_steady_state(
        [A, B], {"A": 10.0, "B": 0.0},
        R_th_sink_to_amb_K_per_W=1.0, T_amb_C=25.0)

    # T_sink = T_amb + R_sa * (P_A + P_B) = 25 + 1*10 = 35
    assert r["T_sink_C"] == pytest.approx(35.0)
    assert r["P_total_W"] == pytest.approx(10.0)
    # A: 35 + 0.5*10 + 2*10 = 60
    assert r["devices"]["A"]["T_j_C"] == pytest.approx(60.0)
    # B: 35 + 0 + 0 = 35  (NOT 25, which an independent model would give)
    assert r["devices"]["B"]["T_j_C"] == pytest.approx(35.0)
    # The shared-sink contribution is identical for every device.
    assert (r["devices"]["A"]["delta_T_sink"]
            == r["devices"]["B"]["delta_T_sink"] == pytest.approx(10.0))


def test_steady_state_total_power_drives_shared_sink() -> None:
    """The sink rise tracks TOTAL dissipation, not per-device."""
    devs = [p.HeatsinkDevice(f"Q{i}", _foster(0.3, 0.05),
                             R_th_case_to_sink_K_per_W=0.2)
            for i in range(3)]
    r = p.shared_heatsink_steady_state(
        devs, [18.0, 18.0, 18.0],
        R_th_sink_to_amb_K_per_W=0.5, T_amb_C=40.0)
    assert r["P_total_W"] == pytest.approx(54.0)
    assert r["T_sink_C"] == pytest.approx(40.0 + 0.5 * 54.0)   # 67 °C
    # Each identical device: 67 + 0.2*18 + 0.3*18 = 76 °C
    assert r["devices"]["Q1"]["T_j_C"] == pytest.approx(76.0)


def test_steady_state_mixed_device_types_on_one_sink() -> None:
    """IGBTs and diodes with different impedances share one sink."""
    igbts = [p.HeatsinkDevice(f"Q{i}", _foster(0.30, 0.05),
                              R_th_case_to_sink_K_per_W=0.20)
             for i in range(3)]
    diodes = [p.HeatsinkDevice(f"D{i}", _foster(0.50, 0.03),
                               R_th_case_to_sink_K_per_W=0.20)
              for i in range(3)]
    powers = {**{f"Q{i}": 18.0 for i in range(3)},
              **{f"D{i}": 6.0 for i in range(3)}}
    r = p.shared_heatsink_steady_state(
        igbts + diodes, powers,
        R_th_sink_to_amb_K_per_W=0.5, T_amb_C=40.0)
    # P_total = 3*18 + 3*6 = 72 → T_sink = 40 + 0.5*72 = 76
    assert r["T_sink_C"] == pytest.approx(76.0)
    # IGBT: 76 + 0.2*18 + 0.3*18 = 85; diode: 76 + 0.2*6 + 0.5*6 = 80.2
    assert r["devices"]["Q0"]["T_j_C"] == pytest.approx(85.0)
    assert r["devices"]["D0"]["T_j_C"] == pytest.approx(80.2)
    # Diodes are cooler despite identical sink — coupling + own loss differ.
    assert r["devices"]["D0"]["T_j_C"] < r["devices"]["Q0"]["T_j_C"]


def test_steady_state_input_validation() -> None:
    A = p.HeatsinkDevice("A", _foster(1.0, 0.05))
    with pytest.raises(KeyError):                 # missing power for A
        p.shared_heatsink_steady_state([A], {}, R_th_sink_to_amb_K_per_W=0.5)
    with pytest.raises(ValueError):               # wrong-length sequence
        p.shared_heatsink_steady_state([A], [1.0, 2.0],
                                       R_th_sink_to_amb_K_per_W=0.5)


# ---------------------------------------------------------------------------
# Live coupled transient through the real solver
# ---------------------------------------------------------------------------

def test_live_coupled_transient_matches_analytic_steady_state() -> None:
    """Build the shared-heatsink RC network, inject constant per-device
    power, simulate to steady state, and confirm the coupled junction
    temperatures match the analytic solution — including the coupled
    zero-power device sitting well above ambient."""
    A = p.HeatsinkDevice("A", _foster(2.0, 0.02), R_th_case_to_sink_K_per_W=0.5)
    B = p.HeatsinkDevice("B", _foster(2.0, 0.02), R_th_case_to_sink_K_per_W=0.5)
    powers = {"A": 10.0, "B": 0.0}

    b = p.CircuitBuilder()
    hs = p.add_shared_heatsink(
        b, [A, B],
        R_th_sink_to_amb_K_per_W=1.0, C_th_sink_J_per_K=1.0, T_amb_C=25.0)
    obs, bex = p.make_heatsink_observer(
        b, hs, {"A": lambda t, x: powers["A"],
                "B": lambda t, x: powers["B"]})

    # Slowest constant is C_sink * R_sa = 1 s; run 10 s to settle.
    res = p.simulate(b, 10.0, 1e-3,
                     step_observer=obs, b_extra_fn=bex,
                     start_from_dc_op=True)

    x_final = np.asarray(res.states)[-1]
    T_j = obs.read_T_j(x_final)
    ref = p.shared_heatsink_steady_state(
        [A, B], powers, R_th_sink_to_amb_K_per_W=1.0, T_amb_C=25.0)

    assert T_j["A"] == pytest.approx(ref["devices"]["A"]["T_j_C"], abs=0.5)
    assert T_j["B"] == pytest.approx(ref["devices"]["B"]["T_j_C"], abs=0.5)
    # Coupling end-to-end: B dissipates zero yet is ~35 °C, not 25.
    assert T_j["B"] > 30.0
    assert obs.read_T_sink(x_final) == pytest.approx(ref["T_sink_C"], abs=0.5)


def test_add_shared_heatsink_rejects_duplicate_names() -> None:
    A = p.HeatsinkDevice("dup", _foster(1.0, 0.05))
    A2 = p.HeatsinkDevice("dup", _foster(1.0, 0.05))
    b = p.CircuitBuilder()
    with pytest.raises(ValueError):
        p.add_shared_heatsink(b, [A, A2], R_th_sink_to_amb_K_per_W=0.5)
