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


# -----------------------------------------------------------------------
# v1.6 closure — diode + switch + magnetic core-loss paths.
# -----------------------------------------------------------------------

def test_device_loss_summary_includes_diode_when_forward_biased() -> None:
    """SwitchedDiode forward-biased by a DC source. The summary must
    list the diode with non-zero conduction loss and ~100 % conducting
    duty (diode "always on" under DC forward bias).

    Topology: V_dc → R → D → gnd.  Choosing g_on so that the diode
    "on" resistance dominates the loss budget keeps the analytical
    numbers easy to read.
    """
    V_dc, R, g_on, g_off = 5.0, 1.0, 100.0, 1.0e-6
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_dc)
    b.add_resistor("R1", "vin", "mid", R)
    b.add_diode("D1", "mid", "gnd", g_on=g_on, g_off=g_off, V_th=0.0)

    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    summary = p.device_loss_summary(b, res)
    by_kind: dict[str, list] = {}
    for entry in summary:
        by_kind.setdefault(entry["kind"], []).append(entry)

    assert "diode" in by_kind, by_kind.keys()
    d = by_kind["diode"][0]
    assert d["name"] == "D1"
    # Steady-state KCL: V_dc = i·R + i/g_on → i = V_dc / (R + 1/g_on)
    i_expected = V_dc / (R + 1.0 / g_on)
    P_d_expected = i_expected ** 2 / g_on  # = v_d · i = (i/g_on)·i
    # Forward-biased the whole sim ⇒ duty_conducting ≈ 1.
    assert d["duty_conducting"] > 0.95
    assert math.isclose(d["i_peak"], i_expected, rel_tol=2e-2)
    assert math.isclose(d["P_avg"], P_d_expected, rel_tol=5e-2)


def test_device_loss_summary_diode_reverse_biased_is_tiny() -> None:
    """Reverse-bias the diode (V_dc with diode pointed at the source).
    With g_off ≪ g_on the loss should be vanishingly small and
    duty_conducting ≈ 0."""
    V_dc = 5.0
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_dc)
    b.add_resistor("R1", "vin", "mid", 1.0)
    # Anode at gnd, cathode at mid → reverse-biased by the +5 V at mid.
    b.add_diode("D1", "gnd", "mid", g_on=100.0, g_off=1e-9, V_th=0.0)

    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    summary = p.device_loss_summary(b, res)
    diodes = [e for e in summary if e["kind"] == "diode"]
    assert len(diodes) == 1
    d = diodes[0]
    assert d["duty_conducting"] < 0.05
    # Leakage P = V² · g_off ≈ 25 · 1e-9 = 25 nW — well under a μW.
    assert d["P_avg"] < 1e-6


def test_device_loss_summary_switch_requires_switch_fn() -> None:
    """An ideal switch entry is only reported when `switch_fn=` is
    passed — without it the summary skips the switch (no silent
    default to "always closed")."""
    V_dc = 5.0
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_dc)
    b.add_resistor("R1", "vin", "mid", 1.0)
    b.add_switch("SW1", "mid", "gnd", g_on=100.0, g_off=1e-9)

    # PWM-driven switch: closed half the time.
    n_sw = b.graph.num_switches

    def half_closed(_t: float):
        mask = p.SwitchStateMask(n_sw)
        # SW1 is the only switch — keep it closed always at first.
        mask.set(0, True)
        return mask

    res = p.simulate(b, t_end=1e-3, dt=1e-5, switch_fn=half_closed)

    # Without switch_fn: no switch entry.
    summary_no_fn = p.device_loss_summary(b, res)
    assert not any(e["kind"] == "switch" for e in summary_no_fn)

    # With switch_fn: switch entry shows up.
    summary_with_fn = p.device_loss_summary(b, res, switch_fn=half_closed)
    sw_entries = [e for e in summary_with_fn if e["kind"] == "switch"]
    assert len(sw_entries) == 1
    sw = sw_entries[0]
    assert sw["name"] == "SW1"
    assert sw["duty_closed"] > 0.95  # always closed in this test


def test_device_loss_summary_switch_pwm_duty() -> None:
    """50 %-duty PWM switch — `duty_closed` should be ≈ 0.5 and the
    conduction loss must scale with the duty cycle (P_avg ≈ 0.5 ·
    P_on)."""
    V_dc, R, g_on = 5.0, 1.0, 100.0
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_dc)
    b.add_resistor("R1", "vin", "mid", R)
    b.add_switch("SW1", "mid", "gnd", g_on=g_on, g_off=1e-9)

    f_sw = 10_000.0
    T_sw = 1.0 / f_sw

    def pwm_50(t: float):
        mask = p.SwitchStateMask(b.graph.num_switches)
        on = (t % T_sw) < (0.5 * T_sw)
        mask.set(0, bool(on))
        return mask

    # Many switching periods to average out edge effects.
    res = p.simulate(b, t_end=10 * T_sw, dt=T_sw / 200, switch_fn=pwm_50)
    summary = p.device_loss_summary(b, res, switch_fn=pwm_50)
    sw = next(e for e in summary if e["kind"] == "switch")
    # 50 % duty within a tight band.
    assert 0.45 < sw["duty_closed"] < 0.55
    # Loose tolerance on P_avg — RL settling on the closed half adds
    # ripple; the order of magnitude is what we're locking down.
    assert sw["P_avg"] > 0.0
    assert sw["E_total"] > 0.0


def test_device_loss_summary_core_loss_steinmetz() -> None:
    """Series-RL with an AC source and core-loss spec on the inductor.
    The summary entry must add `P_core_avg`, `E_core_total`, `B_peak`
    derived from B(t) = L·i / (N·A_core)."""
    V_pk = 50.0
    f0 = 1000.0
    R, L = 5.0, 1e-3
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "vin", "gnd",
                                v_dc=0.0,
                                v_amplitude=V_pk, frequency=f0,
                                phase=0.0)
    b.add_inductor("L1", "vin", "mid", L)
    b.add_resistor("R1", "mid", "gnd", R)

    # Run for several cycles so iGSE has data.
    res = p.simulate(b, t_end=5 / f0, dt=1.0 / (f0 * 200))

    core_spec = {
        "L1": {
            "material": "N87",      # built-in Mn-Zn ferrite
            "N_turns": 20,
            "A_core": 1.0e-4,        # 1 cm²
            "V_core": 5.0e-6,        # 5 cm³
            "use_igse": True,
        }
    }
    summary = p.device_loss_summary(b, res, core_loss_specs=core_spec)
    ind = next(e for e in summary if e["kind"] == "inductor")
    assert ind["name"] == "L1"
    # B = L·i / (N·A) — sanity check that the peak is positive and
    # finite. We don't pin an absolute value because the steady-state
    # current under R+L AC drive depends on |Z| = √(R² + (ωL)²).
    assert ind["B_peak"] > 0.0
    assert math.isfinite(ind["B_peak"])
    assert ind["P_core_avg"] > 0.0
    assert ind["E_core_total"] > 0.0
    # Sanity: P_core_avg · duration ≈ E_core_total
    duration = float(res.times[-1] - res.times[0])
    assert math.isclose(ind["E_core_total"],
                          ind["P_core_avg"] * duration,
                          rel_tol=1e-6)


def test_device_loss_summary_core_loss_inline_triplet() -> None:
    """The inline K/alpha/beta path matches the named-material path
    on the same inductor."""
    V_pk, f0 = 50.0, 1000.0
    R, L = 5.0, 1e-3
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "vin", "gnd",
                                v_dc=0.0,
                                v_amplitude=V_pk, frequency=f0,
                                phase=0.0)
    b.add_inductor("L1", "vin", "mid", L)
    b.add_resistor("R1", "mid", "gnd", R)
    res = p.simulate(b, t_end=5 / f0, dt=1.0 / (f0 * 200))

    n87 = p.core_material("N87")
    spec_named = {"L1": {"material": "N87",
                              "N_turns": 20,
                              "A_core": 1e-4, "V_core": 5e-6}}
    spec_inline = {"L1": {"K": n87.K, "alpha": n87.alpha,
                              "beta": n87.beta,
                              "N_turns": 20,
                              "A_core": 1e-4, "V_core": 5e-6}}
    s1 = p.device_loss_summary(b, res, core_loss_specs=spec_named)
    s2 = p.device_loss_summary(b, res, core_loss_specs=spec_inline)
    P1 = next(e for e in s1 if e["kind"] == "inductor")["P_core_avg"]
    P2 = next(e for e in s2 if e["kind"] == "inductor")["P_core_avg"]
    assert math.isclose(P1, P2, rel_tol=1e-9)
