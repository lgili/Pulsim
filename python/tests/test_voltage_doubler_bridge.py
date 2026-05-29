"""`add_voltage_doubler_bridge` — universal-input AC-DC front-end.

The helper builds a 4-diode bridge with a series cap stack and a
controlled switch between AC_N and the cap midpoint. When the
switch is open the topology is a standard full-wave bridge; when
the switch is closed it's a voltage doubler. The same DC bus
voltage works on 110 VAC (doubler) and 220 VAC (bridge), which is
the whole point of the universal-input front-end.

These tests pin three things:

1. **Construction** — the helper creates the expected number of
   branches/switches, the dataclass exposes the deterministic
   names, and validation rejects nonsense inputs loudly.
2. **Bridge mode** (SW open) on a high-line input — the DC bus
   settles to ≈ √2·V_RMS minus diode drops.
3. **Doubler mode** (SW closed) on a low-line input — the DC bus
   settles to ≈ 2·√2·V_RMS, matching the bridge-mode output of
   the high-line case.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def _make_uvi_rect(builder, R_load_ohms: float = 1000.0):
    """Build a complete universal-input rectifier plant. Returns
    the :class:`VoltageDoublerBridgeResult` so individual tests can
    drive the switch + probe nodes."""
    builder.add_sine_voltage_source(
        name="Vac", n_pos="ac_l", n_neg="ac_n",
        v_dc=0.0, v_amplitude=311.0,   # 220 VAC peak — high-line nominal
        frequency=60.0)
    rect = p.add_voltage_doubler_bridge(
        builder, name="Rect",
        ac_l="ac_l", ac_n="ac_n",
        dc_pos="vdc", dc_neg="gnd",
        C_top=470e-6, C_bot=470e-6,
        V_th=0.0,         # ignore diode drops in the steady-state check
    )
    builder.add_resistor("R_load", "vdc", "gnd", R_load_ohms)
    return rect


def test_voltage_doubler_bridge_constructs_expected_devices() -> None:
    """The helper must create 4 diodes + 2 caps + 1 switch with the
    documented deterministic names."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source(name="Vac", n_pos="L", n_neg="N",
                                v_dc=0.0, v_amplitude=311.0,
                                frequency=60.0)
    rect = p.add_voltage_doubler_bridge(
        b, name="UVI",
        ac_l="L", ac_n="N",
        dc_pos="vdc", dc_neg="gnd",
    )

    # 1 sine + 4 diodes + 2 caps + 1 switch = 8 branches.
    assert b.graph.num_branches == 8
    # Each diode is also a switch in the internal mask; plus our 1
    # doubler switch. 5 total bits in SwitchStateMask.
    assert b.graph.num_switches == 5

    # Names.
    assert rect.diode_names == ["UVI__D1", "UVI__D2", "UVI__D3", "UVI__D4"]
    assert rect.cap_top_name == "UVI__C_top"
    assert rect.cap_bot_name == "UVI__C_bot"
    assert rect.switch_name == "UVI__SW_dbl"
    assert rect.mid_node == "UVI__mid"

    # The 4 diodes get switch indices 0..3 in registration order;
    # the doubler switch gets the next one. The exact value depends
    # on registration order — we just guard it's the last one.
    assert rect.switch_index == 4
    assert 0 <= rect.switch_branch_id < b.graph.num_branches


def test_voltage_doubler_bridge_custom_mid_node_name() -> None:
    """A caller can override ``mid_node`` so the midpoint shows up
    under their own naming convention (useful for probing /
    instrumentation)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source(name="V", n_pos="a", n_neg="b",
                                v_dc=0.0, v_amplitude=311.0,
                                frequency=60.0)
    rect = p.add_voltage_doubler_bridge(
        b, name="R",
        ac_l="a", ac_n="b",
        dc_pos="vdc", dc_neg="gnd",
        mid_node="bus_mid",
    )
    assert rect.mid_node == "bus_mid"


@pytest.mark.parametrize("kwargs,expected_match", [
    ({"C_top": 0.0}, "C_top"),
    ({"C_bot": -1e-6}, "C_bot"),
    ({"g_on": 0.0}, "g_on"),
    ({"g_off": -1.0}, "g_off"),
    ({"sw_g_on": 0.0}, "sw_g_on"),
    ({"sw_g_off": -1.0}, "sw_g_off"),
])
def test_voltage_doubler_bridge_rejects_bad_params(kwargs, expected_match) -> None:
    b = p.CircuitBuilder()
    b.add_sine_voltage_source(name="V", n_pos="a", n_neg="b",
                                v_dc=0.0, v_amplitude=311.0,
                                frequency=60.0)
    with pytest.raises(ValueError, match=expected_match):
        p.add_voltage_doubler_bridge(
            b, name="R",
            ac_l="a", ac_n="b",
            dc_pos="vdc", dc_neg="gnd",
            **kwargs)


# ---------------------------------------------------------------------------
# Bridge mode (SW open) — high-line input
# ---------------------------------------------------------------------------
def _v_node_final(res, name: str) -> float:
    """Convenience: pull the final-sample voltage at ``name``."""
    return float(np.asarray(res.v(name))[-1])


def test_bridge_mode_high_line_dc_bus_near_sqrt2_vrms() -> None:
    """With the doubler switch OPEN and 311 V peak (220 VAC), the
    DC bus should settle close to the peak (≈ V_peak after a few
    line cycles)."""
    b = p.CircuitBuilder()
    rect = _make_uvi_rect(b, R_load_ohms=1000.0)
    n_sw = b.graph.num_switches

    def sw_fn(_t):
        m = p.SwitchStateMask(n_sw)
        # SW open → bridge mode. (Diode bits default False; the
        # nonlinear refresh resolves diode commutation per step.)
        m.set(rect.switch_index, False)
        return m

    res = p.simulate(b, t_end=0.2, dt=20e-6,
                      switch_fn=sw_fn,
                      enable_nonlinear_refresh=True,
                      max_newton_iterations=20)

    v_dc = _v_node_final(res, "vdc")
    # ≈ V_peak = 311 V. Allow a wide ±15 % envelope because the
    # 1 kΩ load + 470 µF + 470 µF gives a ~25 ms ripple time
    # constant, and we're only running 200 ms.
    assert 0.80 * 311.0 < v_dc < 1.10 * 311.0, v_dc


# ---------------------------------------------------------------------------
# Doubler mode (SW closed) — low-line input
# ---------------------------------------------------------------------------
def test_doubler_mode_low_line_dc_bus_near_2_sqrt2_vrms() -> None:
    """With the doubler switch CLOSED and 156 V peak (110 VAC), the
    DC bus should settle near ``2·V_peak ≈ 311 V`` — matching what
    bridge mode produces on a 220 VAC input."""
    b = p.CircuitBuilder()
    # 110 VAC nominal → 156 V peak.
    b.add_sine_voltage_source(
        name="Vac", n_pos="ac_l", n_neg="ac_n",
        v_dc=0.0, v_amplitude=156.0, frequency=60.0)
    rect = p.add_voltage_doubler_bridge(
        b, name="Rect",
        ac_l="ac_l", ac_n="ac_n",
        dc_pos="vdc", dc_neg="gnd",
        C_top=470e-6, C_bot=470e-6,
        V_th=0.0,
    )
    b.add_resistor("R_load", "vdc", "gnd", 1000.0)
    n_sw = b.graph.num_switches

    def sw_fn(_t):
        m = p.SwitchStateMask(n_sw)
        m.set(rect.switch_index, True)  # doubler mode
        return m

    res = p.simulate(b, t_end=0.2, dt=20e-6,
                      switch_fn=sw_fn,
                      enable_nonlinear_refresh=True,
                      max_newton_iterations=20)

    v_dc = _v_node_final(res, "vdc")
    # ≈ 2·156 = 312 V. Allow ±15 % because of the same RC settle
    # window + diode forward voltage we set to 0.
    assert 0.80 * 312.0 < v_dc < 1.10 * 312.0, v_dc


# ---------------------------------------------------------------------------
# End-to-end: bridge mode and doubler mode produce SIMILAR V_DC at
# their respective canonical input voltages (the whole reason the
# topology exists)
# ---------------------------------------------------------------------------
def test_bridge_220v_matches_doubler_110v_within_15pct() -> None:
    """The universal-input promise: 220 VAC (bridge) and 110 VAC
    (doubler) should land within ±15 % of each other on the DC bus,
    so the downstream PFC / SMPS doesn't need to re-tune."""
    def run(v_amp: float, doubler: bool) -> float:
        b = p.CircuitBuilder()
        b.add_sine_voltage_source(
            name="Vac", n_pos="ac_l", n_neg="ac_n",
            v_dc=0.0, v_amplitude=v_amp, frequency=60.0)
        rect = p.add_voltage_doubler_bridge(
            b, name="Rect",
            ac_l="ac_l", ac_n="ac_n",
            dc_pos="vdc", dc_neg="gnd",
            C_top=470e-6, C_bot=470e-6,
            V_th=0.0,
        )
        b.add_resistor("R_load", "vdc", "gnd", 1000.0)
        n_sw = b.graph.num_switches

        def sw_fn(_t):
            m = p.SwitchStateMask(n_sw)
            m.set(rect.switch_index, doubler)
            return m

        res = p.simulate(b, t_end=0.2, dt=20e-6,
                          switch_fn=sw_fn,
                          enable_nonlinear_refresh=True,
                          max_newton_iterations=20)
        return _v_node_final(res, "vdc")

    v_bridge_220 = run(v_amp=311.0, doubler=False)
    v_doubler_110 = run(v_amp=156.0, doubler=True)

    # The two outputs should match within 15 % of each other.
    assert math.isclose(v_bridge_220, v_doubler_110,
                         rel_tol=0.15), (
        f"Universal-input mismatch: bridge@220V={v_bridge_220:.1f}V, "
        f"doubler@110V={v_doubler_110:.1f}V (rel diff "
        f"{abs(v_bridge_220 - v_doubler_110) / v_bridge_220:.2%}); "
        f"expected ≤ 15 % so the downstream stage doesn't need "
        f"re-tuning across the universal-input range.")


# ---------------------------------------------------------------------------
# Discoverability
# ---------------------------------------------------------------------------
def test_helper_and_result_re_exported_at_top_level() -> None:
    """The helper + result dataclass must be importable directly
    from ``pulsim`` — mirrors :func:`p.add_bridge_rectifier` and
    :func:`p.add_buck` exposure."""
    assert hasattr(p, "add_voltage_doubler_bridge")
    assert hasattr(p, "VoltageDoublerBridgeResult")
    # And they appear in `__all__` (so `dir(p)` shows them).
    assert "add_voltage_doubler_bridge" in p.__all__
    assert "VoltageDoublerBridgeResult" in p.__all__
