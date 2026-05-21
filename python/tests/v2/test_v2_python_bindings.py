"""Layer 7 — Python bindings for v2 (smoke + integration tests)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim.v2 as p


# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------


def test_module_imports() -> None:
    """All public names from `pulsim.v2` must be importable."""
    expected = {
        "CircuitBuilder",
        "Graph",
        "DevicePool",
        "SwitchStateMask",
        "PwlStateSpaceCache",
        "SimulationOptions",
        "SimulationResult",
        "CommutationEvent",
        "run_transient",
        "IdealDiodeParams",
    }
    for name in expected:
        assert hasattr(p, name), f"pulsim.v2 missing public symbol: {name}"


def test_builder_gnd_alias_maps_to_ground() -> None:
    b = p.CircuitBuilder()
    gnd_via_alias = b.node("gnd")
    gnd_via_alias_upper = b.node("GND")
    gnd_via_zero = b.node("0")
    assert gnd_via_alias == gnd_via_alias_upper == gnd_via_zero
    # All alias forms refer to the same ground node.
    # `ground()` is a static method on Graph (returns a
    # compile-time sentinel).
    assert p.Graph.ground() == gnd_via_alias


def test_builder_node_id_of_throws_for_unknown() -> None:
    b = p.CircuitBuilder()
    b.node("known")
    assert b.node_id_of("known") >= 0
    with pytest.raises((IndexError, RuntimeError, ValueError, KeyError)):
        b.node_id_of("never_added")


def test_simulation_options_constructor() -> None:
    opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)
    assert opts.valid()
    assert opts.t_start == 0.0
    assert opts.t_end == 1e-3
    assert opts.dt == 1e-5
    # Default values for the other flags.
    assert opts.enable_newton_line_search is False
    assert opts.enable_newton_lm is False
    assert opts.enable_substep_state_correction is False
    # Settable.
    opts.enable_newton_line_search = True
    assert opts.enable_newton_line_search is True


def test_switch_state_mask_repr() -> None:
    m = p.SwitchStateMask(5)
    s = repr(m)
    assert "SwitchStateMask" in s


# -----------------------------------------------------------------------------
# Integration: V_dc → R → GND
# -----------------------------------------------------------------------------


def test_vdc_resistor_dc_solve() -> None:
    """The simplest sanity check: V_dc=5V → R(1Ω) → GND should give v_n0=5V."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1", "n0", "gnd", 1.0)

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build()

    opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-4)

    mask = p.SwitchStateMask(0)
    res = p.run_transient(
        cache, b.graph, b.pool, opts,
        switch_fn=lambda t: mask,
    )

    assert res.num_steps() >= 2
    # v_n0 = 5V for every recorded sample.
    for state in res.states:
        assert abs(state[0] - 5.0) < 1e-9


# -----------------------------------------------------------------------------
# Integration: half-wave rectifier (the V2 layer5_v2 scenario)
# -----------------------------------------------------------------------------


def test_half_wave_rectifier_from_python() -> None:
    """Half-wave rectifier built and simulated entirely from Python.

    Mirrors the C++ layer5_v2 integration test. Should produce a
    half-wave output that follows V_sine on the positive half and
    stays near zero on the negative half.
    """
    V_amp = 10.0
    f_line = 60.0
    R_load = 10.0
    g_on = 1e3
    g_off = 1e-9

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 0.0)
    b.add_diode("D1", "n0", "n1", g_on, g_off, V_th=0.0)
    b.add_resistor("R_L", "n1", "gnd", R_load)

    dt = 1e-4
    T_line = 1.0 / f_line
    t_end = 2.0 * T_line   # 2 full cycles

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(dt)

    opts = p.SimulationOptions(t_start=0.0, t_end=t_end, dt=dt)

    # b_extra modulates the source's branch-current row with
    # the sinusoidal voltage.
    pool = b.pool
    graph = b.graph

    # The source's branch-current row in the state vector.
    # Convention from the v2 kernel: row =
    #   num_nodes + source_index_among_sources.
    # Here we have 2 nodes (n0 has index 0, n1 has index 1)
    # and 1 source → source's branch var is at row 2.
    # state_size = num_nodes + num_sources + num_inductors
    #            = 2 + 1 + 0 = 3.
    src_var = 2
    state_size = 3

    omega = 2.0 * math.pi * f_line

    def b_extra_fn(t: float) -> np.ndarray:
        bv = np.zeros(state_size, dtype=np.float64)
        V_sine = V_amp * math.sin(omega * t)
        bv[src_var] = -V_sine
        return bv

    mask = p.SwitchStateMask(1)   # diode bit; will be overlaid
    res = p.run_transient(
        cache, graph, pool, opts,
        switch_fn=lambda t: mask,
        b_extra_fn=b_extra_fn,
    )

    assert res.num_steps() > 100

    # Skip the first half-period (transient).
    k_skip = int((T_line / 2.0) / dt)

    n_pos_match = 0
    n_pos_total = 0
    n_neg_match = 0
    n_neg_total = 0
    for k in range(k_skip, res.num_steps()):
        t = res.times[k]
        v_sine = V_amp * math.sin(omega * t)
        v_out = res.states[k][1]   # v_n1

        if v_sine > 0.5:
            n_pos_total += 1
            if abs(v_out - v_sine) < 0.5:
                n_pos_match += 1
        elif v_sine < -0.5:
            n_neg_total += 1
            if abs(v_out) < 0.1:
                n_neg_match += 1

    assert n_pos_total > 0
    assert n_neg_total > 0
    # Loose tolerance — the Python bindings just need to be
    # WIRED correctly, not produce a different answer than
    # the C++ test.
    assert n_pos_match >= n_pos_total * 0.9, (
        f"pos-half match too low: {n_pos_match}/{n_pos_total}"
    )
    assert n_neg_match >= n_neg_total * 0.9, (
        f"neg-half match too low: {n_neg_match}/{n_neg_total}"
    )


# -----------------------------------------------------------------------------
# Integration: nonlinear diode DC load-line via Python
# -----------------------------------------------------------------------------


def test_nonlinear_diode_params_default() -> None:
    """Default IdealDiodeParams should match Layer 4 V3 conventions."""
    params = p.IdealDiodeParams()
    assert params.V_F0 == pytest.approx(0.7)
    assert params.R_d == pytest.approx(0.01)
    assert params.G_off == pytest.approx(1e-9)
    assert params.kappa == pytest.approx(20.0)


def test_nonlinear_diode_params_named() -> None:
    """IdealDiodeParams accepts keyword args."""
    params = p.IdealDiodeParams(
        V_F0=0.5, R_d=0.02, G_off=2e-9, kappa=10.0)
    assert params.V_F0 == pytest.approx(0.5)
    assert params.R_d == pytest.approx(0.02)
    assert params.G_off == pytest.approx(2e-9)
    assert params.kappa == pytest.approx(10.0)


def test_nonlinear_diode_builder_compiles() -> None:
    """add_nonlinear_diode binds correctly and grows num_branches."""
    b = p.CircuitBuilder()
    params = p.IdealDiodeParams(V_F0=0.7, R_d=0.01,
                                  G_off=1e-9, kappa=20.0)
    b.add_voltage_source("Vin", "n0", "gnd", 2.0)
    b.add_nonlinear_diode("D1", "n0", "n1", params)
    b.add_resistor("R_L", "n1", "gnd", 1000.0)
    assert b.num_branches == 3


# -----------------------------------------------------------------------------
# Integration: graph + pool accessors
# -----------------------------------------------------------------------------


def test_graph_accessor() -> None:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1", "n0", "n1", 100.0)
    b.add_resistor("R2", "n1", "gnd", 10.0)

    g = b.graph
    assert g.num_nodes == 2          # n0, n1
    assert g.num_branches == 3       # 1 source + 2 resistors
