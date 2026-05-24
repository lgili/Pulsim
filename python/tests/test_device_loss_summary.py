"""Regression tests for :func:`pulsim.device_loss_summary`.

Verifies that the post-hoc loss summary walks both **inductor** and
**resistor** branches and reconstructs their currents / dissipation
correctly. Added in v1.3 — previously the summary only covered
inductors (see commit history of `python/pulsim/losses.py`).
"""

from __future__ import annotations

import math

import pulsim as p


def _build_dc_resistor_divider(R_top: float = 2.0,
                                 R_bot: float = 3.0,
                                 V_dc: float = 5.0):
    """V_dc → R_top → n_mid → R_bot → gnd. Analytical steady-state:

        v_mid = V_dc · R_bot / (R_top + R_bot)
        i     = V_dc / (R_top + R_bot)
        P_top = i² · R_top
        P_bot = i² · R_bot
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_dc)
    b.add_resistor("R_top", "vin", "mid", R_top)
    b.add_resistor("R_bot", "mid", "gnd", R_bot)
    return b


def test_device_loss_summary_includes_resistor_powers() -> None:
    """Two-resistor divider — both resistors must show up in the
    summary with the analytically expected average power.

    With V_dc=5V, R_top=2Ω, R_bot=3Ω → i=1A → P_top=2W, P_bot=3W.
    """
    R_top, R_bot, V_dc = 2.0, 3.0, 5.0
    b = _build_dc_resistor_divider(R_top, R_bot, V_dc)

    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    summary = p.device_loss_summary(b, res)

    # Both resistors must appear (and so must the inductor count = 0).
    kinds = {entry["kind"] for entry in summary}
    assert "resistor" in kinds, (
        f"resistor walk missing — summary kinds={kinds}"
    )

    by_name = {entry["name"]: entry for entry in summary
                if entry["kind"] == "resistor"}
    assert set(by_name) == {"R_top", "R_bot"}, by_name.keys()

    i_expected = V_dc / (R_top + R_bot)
    P_top_expected = i_expected ** 2 * R_top
    P_bot_expected = i_expected ** 2 * R_bot

    top = by_name["R_top"]
    bot = by_name["R_bot"]

    # Direct kind tagging.
    assert top["R_ohms"] == R_top
    assert bot["R_ohms"] == R_bot

    # Currents — DC, so i_avg ≈ i_rms ≈ i_peak ≈ V/R_total. Loosen
    # the tolerance slightly to absorb the first-sample warm-up
    # before the DC operating point is fully observed.
    for entry in (top, bot):
        assert math.isclose(entry["i_avg"], i_expected, rel_tol=1e-2)
        assert math.isclose(entry["i_rms"], i_expected, rel_tol=1e-2)
        assert math.isclose(entry["i_peak"], i_expected, rel_tol=1e-3)

    # Power dissipated must match i² · R.
    assert math.isclose(top["P_avg"], P_top_expected, rel_tol=1e-2)
    assert math.isclose(bot["P_avg"], P_bot_expected, rel_tol=1e-2)

    # E_total = P_avg · duration.
    duration = float(res.times[-1] - res.times[0])
    assert math.isclose(top["E_total"],
                          P_top_expected * duration, rel_tol=1e-2)
    assert math.isclose(bot["E_total"],
                          P_bot_expected * duration, rel_tol=1e-2)


def test_device_loss_summary_still_reports_inductor_currents() -> None:
    """Series RL settling — the inductor walk must continue to work
    alongside the new resistor walk."""
    V_dc, R, L = 5.0, 1.0, 1e-3
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_dc)
    b.add_inductor("L1", "vin", "mid", L)
    b.add_resistor("R1", "mid", "gnd", R)

    # Long enough to reach steady-state (τ = L/R = 1 ms).
    res = p.simulate(b, t_end=10e-3, dt=1e-5)
    summary = p.device_loss_summary(b, res)

    by_kind: dict[str, list] = {}
    for entry in summary:
        by_kind.setdefault(entry["kind"], []).append(entry)

    # Both kinds present.
    assert "inductor" in by_kind, by_kind.keys()
    assert "resistor" in by_kind, by_kind.keys()
    assert len(by_kind["inductor"]) == 1
    assert len(by_kind["resistor"]) == 1

    ind = by_kind["inductor"][0]
    res_R = by_kind["resistor"][0]

    # i_peak ≤ V/R (asymptote) and i_avg pulled down by the ramp.
    i_inf = V_dc / R  # = 5 A
    assert ind["i_peak"] <= i_inf * 1.02
    assert ind["i_avg"] > 0.0
    assert ind["i_avg"] < i_inf

    # Resistor and inductor share the same branch current ⇒ same i_rms.
    assert math.isclose(ind["i_rms"], res_R["i_rms"], rel_tol=2e-3)
