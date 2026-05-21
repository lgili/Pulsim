"""Layer 7 — Python bindings for v2 (smoke + integration tests)."""

from __future__ import annotations

import math
import os
from pathlib import Path

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


# -----------------------------------------------------------------------------
# Layer 2 V1 — power-device convenience methods
# -----------------------------------------------------------------------------


def test_add_mosfet_from_python() -> None:
    b = p.CircuitBuilder()
    b.add_mosfet("Q1", "drain", "source")    # all defaults
    assert b.num_branches == 1


def test_add_mosfet_with_body_diode_from_python() -> None:
    b = p.CircuitBuilder()
    b.add_mosfet_with_body_diode("Q1", "drain", "source")
    assert b.num_branches == 2   # switch + body diode


def test_add_mosfet_custom_R_on() -> None:
    b = p.CircuitBuilder()
    b.add_mosfet("Q1", "drain", "source",
                  R_on=2e-3, R_off=1e9)
    # We can't trivially reach pool.switch_g_on from Python
    # in V0 (DevicePool is opaque), but we can verify the
    # branch count + the call succeeded.
    assert b.num_branches == 1


def test_add_igbt_from_python() -> None:
    b = p.CircuitBuilder()
    b.add_igbt("T1", "C", "E")    # all defaults
    assert b.num_branches == 1


# -----------------------------------------------------------------------------
# Layer 2 V2 — transformer
# -----------------------------------------------------------------------------


def test_add_pwm_voltage_source_mean_matches_duty() -> None:
    """Layer 2 V4 — PWM source. Mean output over a full
    cycle should equal v_high · duty (assuming v_low=0)."""
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source(
        "VPWM", "n0", "gnd",
        v_high=24.0, v_low=0.0,
        frequency=100e3, duty=0.5)
    b.add_resistor("R_L", "n0", "gnd", 10.0)

    dt = 1e-8
    T = 1e-5
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(dt)

    opts = p.SimulationOptions(
        t_start=0.0, t_end=5.0 * T, dt=dt)
    mask = p.SwitchStateMask(0)
    res = p.run_transient(
        cache, b.graph, b.pool, opts,
        switch_fn=lambda t: mask)

    # Mean over the last full cycle.
    k_start = res.num_steps() - int(T / dt)
    samples = [res.states[k][0]
               for k in range(k_start, res.num_steps())]
    mean = sum(samples) / len(samples)
    # v_high · duty = 24 · 0.5 = 12 V.
    assert abs(mean - 12.0) < 0.5


def test_add_current_source_from_python() -> None:
    """Layer 2 V3 — CurrentSource device model."""
    b = p.CircuitBuilder()
    b.add_current_source("Ibias", "n0", "gnd", 0.01)
    b.add_resistor("R1", "n0", "gnd", 1000.0)

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build()

    opts = p.SimulationOptions(t_start=0.0,
                                t_end=1e-4, dt=1e-5)
    mask = p.SwitchStateMask(0)
    res = p.run_transient(
        cache, b.graph, b.pool, opts,
        switch_fn=lambda t: mask)
    # I = 10 mA → R = 1 kΩ → v_n0 = +10 V
    # (EE convention: I flows OUT of `from` into circuit)
    assert abs(res.states[-1][0] - 10.0) < 1e-6


def test_add_pulse_voltage_source_step_charges_rc() -> None:
    """Layer 2 V12 — pulse source step-charges an RC circuit
    matching V_C(t) = V · (1 − e^(−t/τ))."""
    import math
    V = 10.0
    R = 1000.0
    C = 1.0e-6
    tau = R * C   # 1 ms

    b = p.CircuitBuilder()
    b.add_pulse_voltage_source(
        "Vstep", "n0", "gnd",
        v_initial=0.0, v_pulsed=V,
        t_start=0.0, pulse_width=1.0)
    b.add_resistor("R", "n0", "vc", R)
    b.add_capacitor("C", "vc", "gnd", C)

    dt = 1e-6
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(dt)
    opts = p.SimulationOptions(
        t_start=0.0, t_end=5.0 * tau, dt=dt)
    mask = p.SwitchStateMask(0)
    res = p.run_transient(
        cache, b.graph, b.pool, opts,
        switch_fn=lambda t: mask)

    vc_idx = b.node_id_of("vc")
    # Sample at 1τ, 2τ, 3τ.
    for k_tau in (1, 2, 3):
        t_check = k_tau * tau
        idx = int(t_check / dt)
        v_sim = res.states[idx][vc_idx]
        v_exp = V * (1.0 - math.exp(-t_check / tau))
        assert abs(v_sim - v_exp) < 0.05, \
            f"At {k_tau}τ: sim={v_sim}, exp={v_exp}"


def test_add_sine_voltage_source_drives_resistor() -> None:
    """Layer 2 V11 — sine source driving a resistor: node
    voltage tracks the analytical sine wave."""
    import math
    V_amp = 10.0
    f_ac  = 60.0
    b = p.CircuitBuilder()
    b.add_sine_voltage_source(
        "Vac", "n0", "gnd",
        v_dc=0.0, v_amplitude=V_amp, frequency=f_ac)
    b.add_resistor("R", "n0", "gnd", 100.0)

    dt = 1e-5
    T_ac = 1.0 / f_ac
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(dt)

    opts = p.SimulationOptions(
        t_start=0.0, t_end=3.0 * T_ac, dt=dt)
    mask = p.SwitchStateMask(0)
    res = p.run_transient(
        cache, b.graph, b.pool, opts,
        switch_fn=lambda t: mask)

    # Check the last cycle tracks the sine analytically.
    err = 0.0
    n = 0
    for k in range(res.num_steps()):
        t = res.times[k]
        v_sim = res.states[k][0]
        v_exp = V_amp * math.sin(2.0 * math.pi * f_ac * t)
        err += (v_sim - v_exp) ** 2
        n += 1
    rms = (err / n) ** 0.5
    assert rms < V_amp * 0.01, \
        f"RMS error {rms} V too large vs amplitude {V_amp} V"


def test_add_transformer_from_python() -> None:
    """add_transformer creates two coupled inductors."""
    b = p.CircuitBuilder()
    b.add_transformer("T1", "p+", "p-", "s+", "s-",
                       L_p=1e-3, L_s=4e-3, k=1.0)
    assert b.num_branches == 2   # 2 inductor branches


def test_make_pwm_switch_fn_from_python() -> None:
    """Layer 2 V5 — make_pwm_switch_fn drives a switched
    resistor end-to-end via run_transient, no lambda needed
    for the switch side."""
    V_src = 10.0
    R     = 100.0
    f_sw  = 50e3
    duty  = 0.4

    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "n1", "gnd", V_src)
    b.add_switch("S1", "n1", "n2", g_on=1e3, g_off=1e-9)
    b.add_resistor("R1", "n2", "gnd", R)

    dt   = 1e-7
    T_sw = 1.0 / f_sw
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(dt)

    opts = p.SimulationOptions(
        t_start=0.0, t_end=5.0 * T_sw, dt=dt)
    n_sw = b.graph.num_switches
    assert n_sw == 1

    sw_fn = p.make_pwm_switch_fn(
        frequency=f_sw, duty=duty,
        switch_idx=0, num_switches=n_sw)

    res = p.run_transient(
        cache, b.graph, b.pool, opts,
        switch_fn=sw_fn)
    assert res.num_steps() > 100

    # Fraction of samples with n2 ≈ V_src (i.e., switch ON).
    n_on = sum(
        1 for k in range(res.num_steps())
        if abs(res.states[k][1] - V_src) < 0.5)
    frac_on = n_on / res.num_steps()
    assert abs(frac_on - duty) < 0.02


def test_make_dead_time_pwm_pair_fn_no_shoot_through() -> None:
    """Layer 2 V6 — complementary HS/LS helper is shoot-
    through-free across a dense sample of duty / dt combos."""
    for duty in (0.1, 0.5, 0.9):
        for dt_dead in (0.0, 1e-2, 5e-2):
            sw = p.make_dead_time_pwm_pair_fn(
                frequency=1.0, duty=duty,
                hs_switch_idx=0, ls_switch_idx=1,
                num_switches=2, dead_time=dt_dead)
            for k in range(200):
                t = (k + 0.317) / 25.0
                m = sw(t)
                assert not (m.get(0) and m.get(1)), \
                    f"shoot-through @ duty={duty} dt={dt_dead}"


def test_make_dead_time_pwm_pair_fn_phase_balance() -> None:
    """Layer 2 V6 — average ON times match duty - dt/T."""
    T = 1.0
    duty = 0.6
    dt_dead = 0.05
    sw = p.make_dead_time_pwm_pair_fn(
        frequency=1.0/T, duty=duty,
        hs_switch_idx=0, ls_switch_idx=1,
        num_switches=2, dead_time=dt_dead)
    N = 20_000
    hs_on = sum(1 for k in range(N)
                if sw((k + 0.5) * T / N).get(0))
    ls_on = sum(1 for k in range(N)
                if sw((k + 0.5) * T / N).get(1))
    hs_frac = hs_on / N
    ls_frac = ls_on / N
    # HS on fraction ≈ duty - dt/T = 0.55.
    assert abs(hs_frac - (duty - dt_dead)) < 5e-3
    # LS on fraction ≈ (1-duty) - dt/T = 0.35.
    assert abs(ls_frac - ((1.0 - duty) - dt_dead)) < 5e-3


def test_make_spwm_pair_fn_no_shoot_through() -> None:
    """Layer 2 V7 — SPWM pair helper is shoot-through-free
    across a modulation cycle."""
    sw = p.make_spwm_pair_fn(
        carrier_frequency=10e3, modulation_frequency=50.0,
        modulation_index=0.9,
        hs_switch_idx=0, ls_switch_idx=1,
        num_switches=2, dead_time=100e-9)
    for k in range(2000):
        t = (k + 0.317) * 0.02 / 2000.0   # 20 ms span
        m = sw(t)
        assert not (m.get(0) and m.get(1)), \
            f"shoot-through @ t={t}"


def test_make_spwm_pair_fn_M_zero_matches_constant_duty() -> None:
    """Layer 2 V7 — with M=0 the SPWM helper is bit-for-bit
    identical to make_dead_time_pwm_pair_fn(duty=0.5)."""
    f_c = 1000.0
    dt_dead = 1e-5
    sw_spwm = p.make_spwm_pair_fn(
        carrier_frequency=f_c, modulation_frequency=50.0,
        modulation_index=0.0,
        hs_switch_idx=0, ls_switch_idx=1,
        num_switches=2, dead_time=dt_dead)
    sw_const = p.make_dead_time_pwm_pair_fn(
        frequency=f_c, duty=0.5,
        hs_switch_idx=0, ls_switch_idx=1,
        num_switches=2, dead_time=dt_dead)
    for k in range(200):
        t = (k + 0.1) * (5.0 / f_c) / 200.0
        assert sw_spwm(t).get(0) == sw_const(t).get(0)
        assert sw_spwm(t).get(1) == sw_const(t).get(1)


def test_make_three_phase_spwm_fn_shoot_through_free() -> None:
    """Layer 2 V8 — 3-phase VSI helper: no shoot-through on
    any of the 3 legs."""
    legs = p.ThreePhaseLegIndices(0, 1, 2, 3, 4, 5)
    sw = p.make_three_phase_spwm_fn(
        carrier_frequency=10e3, modulation_frequency=50.0,
        modulation_index=0.9, legs=legs,
        num_switches=6, dead_time=100e-9)
    for k in range(3000):
        t = (k + 0.317) * 0.04 / 3000.0
        m = sw(t)
        ab = m.get(0) and m.get(1)
        bb = m.get(2) and m.get(3)
        cc = m.get(4) and m.get(5)
        assert not ab and not bb and not cc, \
            f"shoot-through @ t={t} (A={ab} B={bb} C={cc})"


def test_make_three_phase_spwm_fn_120_degree_rotation() -> None:
    """Layer 2 V8 — at ωt = π/2 with M=0.8: leg A peaks (HS
    ON), legs B/C are below 0.5 duty (LS ON). Sampled at
    t_cycle = T_c/2 inside the appropriate carrier period."""
    legs = p.ThreePhaseLegIndices(0, 1, 2, 3, 4, 5)
    f_c = 1000.0
    f_m = 50.0
    sw = p.make_three_phase_spwm_fn(
        carrier_frequency=f_c, modulation_frequency=f_m,
        modulation_index=0.8, legs=legs,
        num_switches=6, dead_time=0.0)
    T_c = 1.0 / f_c
    t = 1.0 / (4.0 * f_m) + T_c * 0.5
    m = sw(t)
    assert m.get(0) is True   # HS_A
    assert m.get(1) is False
    assert m.get(2) is False
    assert m.get(3) is True   # LS_B
    assert m.get(4) is False
    assert m.get(5) is True   # LS_C


def test_make_phase_shift_full_bridge_fn_synchronous_legs() -> None:
    """Layer 2 V9 — at φ=0 the two legs of the full bridge
    are bit-for-bit synchronous; v_AB ≡ 0."""
    import math
    sw = p.make_phase_shift_full_bridge_fn(
        switching_frequency=1000.0, phase_shift=0.0,
        leg_a_hs_idx=0, leg_a_ls_idx=1,
        leg_b_hs_idx=2, leg_b_ls_idx=3,
        num_switches=4, dead_time=0.0)
    for k in range(200):
        t = (k + 0.1) * 5e-3 / 200.0
        m = sw(t)
        assert m.get(0) == m.get(2)   # A_HS ≡ B_HS
        assert m.get(1) == m.get(3)   # A_LS ≡ B_LS


def test_make_phase_shift_full_bridge_fn_anti_phase() -> None:
    """Layer 2 V9 — at φ=π leg B is the mirror of leg A;
    v_AB is a full ±V_bus square wave."""
    import math
    sw = p.make_phase_shift_full_bridge_fn(
        switching_frequency=1000.0, phase_shift=math.pi,
        leg_a_hs_idx=0, leg_a_ls_idx=1,
        leg_b_hs_idx=2, leg_b_ls_idx=3,
        num_switches=4, dead_time=0.0)
    for k in range(200):
        t = (k + 0.1) * 5e-3 / 200.0
        m = sw(t)
        assert m.get(0) == m.get(3)   # A_HS ≡ B_LS
        assert m.get(1) == m.get(2)   # A_LS ≡ B_HS


def test_make_combined_switch_fn_disjoint_bits() -> None:
    """Layer 2 V10 — compose two pair helpers driving
    disjoint bit groups; both must contribute to the
    combined mask."""
    pair_a = p.make_dead_time_pwm_pair_fn(
        frequency=1.0, duty=0.5,
        hs_switch_idx=0, ls_switch_idx=1,
        num_switches=4, dead_time=0.0)
    pair_b = p.make_dead_time_pwm_pair_fn(
        frequency=1.0, duty=0.5,
        hs_switch_idx=2, ls_switch_idx=3,
        num_switches=4, dead_time=0.0,
        phase=0.5)   # half-cycle offset
    combo = p.make_combined_switch_fn(
        num_switches=4, fns=[pair_a, pair_b])

    for k in range(100):
        t = (k + 0.1) * 2.0 / 100.0
        m = combo(t)
        a = pair_a(t)
        b = pair_b(t)
        for i in range(4):
            assert m.get(i) == (a.get(i) or b.get(i))


def test_make_combined_switch_fn_empty_list() -> None:
    """Layer 2 V10 — empty fns list returns an all-OFF mask
    of the requested size."""
    sw = p.make_combined_switch_fn(num_switches=4, fns=[])
    m = sw(0.5)
    assert m.size == 4
    for i in range(4):
        assert m.get(i) is False


def test_make_pwm_switch_fn_phase_and_edge_cases() -> None:
    """Layer 2 V5 — helper handles phase offset, duty=0/1,
    and frequency=0 the same way the C++ helper does."""
    # 50 % duty, no phase: bit 0 ON in first half of cycle.
    sw = p.make_pwm_switch_fn(
        frequency=1.0, duty=0.5,
        switch_idx=0, num_switches=1)
    assert sw(0.0).get(0) is True
    assert sw(0.25).get(0) is True
    assert sw(0.5).get(0) is False    # boundary exclusive
    assert sw(0.75).get(0) is False
    assert sw(1.0).get(0) is True     # cycle wraps

    # Duty=0 → always OFF.
    sw0 = p.make_pwm_switch_fn(1.0, 0.0, 0, 1)
    assert sw0(0.0).get(0) is False
    assert sw0(0.5).get(0) is False

    # Duty=1 → always ON.
    sw1 = p.make_pwm_switch_fn(1.0, 1.0, 0, 1)
    assert sw1(0.0).get(0) is True
    assert sw1(0.99).get(0) is True

    # Frequency=0 → flat OFF (degenerate, no /0).
    sw_dead = p.make_pwm_switch_fn(0.0, 0.5, 0, 1)
    assert sw_dead(0.0).get(0) is False
    assert sw_dead(1e6).get(0) is False

    # Only target switch toggles.
    sw3 = p.make_pwm_switch_fn(1.0, 0.5, 1, 3)
    m_on = sw3(0.1)
    assert m_on.get(0) is False
    assert m_on.get(1) is True
    assert m_on.get(2) is False


# -----------------------------------------------------------------------------
# Layer 8 — YAML loader
# -----------------------------------------------------------------------------


def test_load_yaml_string_basic() -> None:
    """A simple V_dc + R YAML loads + DC-solves."""
    yaml_text = """
circuit:
  devices:
    - type: voltage_source
      name: Vin
      from: n0
      to: gnd
      V: 5.0
    - type: resistor
      name: R1
      from: n0
      to: gnd
      R: 1.0

simulation:
  t_start: 0.0
  t_end: 1e-3
  dt: 1e-4
"""
    loaded = p.load_yaml_string(yaml_text)
    assert loaded.builder.num_branches == 2
    assert loaded.options.dt == pytest.approx(1e-4)


def test_load_yaml_string_missing_field_raises() -> None:
    """Loader propagates validation errors as Python exceptions."""
    yaml_text = """
circuit:
  devices:
    - type: resistor
      name: R_bad
      from: a
      to: b
"""
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        p.load_yaml_string(yaml_text)
    # The error mentions the resistor's name.
    assert "R_bad" in str(exc_info.value)


def test_load_yaml_file_example() -> None:
    """Load and parse the example/v2/half_wave_rectifier.yaml."""
    # Locate the example file relative to the project root.
    # We resolve the repo root by walking up from this test
    # file until we find `examples/v2/`.
    here = Path(__file__).resolve()
    repo_root = here
    for _ in range(8):
        repo_root = repo_root.parent
        if (repo_root / "examples" / "v2").exists():
            break
    else:
        pytest.skip("repo root not located")

    yaml_path = repo_root / "examples" / "v2" / "half_wave_rectifier.yaml"
    assert yaml_path.exists(), f"missing {yaml_path}"

    loaded = p.load_yaml_file(str(yaml_path))
    assert loaded.builder.num_branches == 3
    assert loaded.options.t_end == pytest.approx(0.0333)


def test_load_yaml_buck_example() -> None:
    """Load the buck example and verify topology counts."""
    here = Path(__file__).resolve()
    repo_root = here
    for _ in range(8):
        repo_root = repo_root.parent
        if (repo_root / "examples" / "v2").exists():
            break
    else:
        pytest.skip("repo root not located")

    yaml_path = repo_root / "examples" / "v2" / "buck.yaml"
    loaded = p.load_yaml_file(str(yaml_path))
    # Vin + (MOSFET-w-body-diode = 2) + D_FW + L + Cout +
    # R_L = 7 branches.
    assert loaded.builder.num_branches == 7


def test_transformer_topology_builds_and_factors() -> None:
    """Build a simple primary+secondary topology with a
    transformer + resistors. Should factorize."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 10.0)
    b.add_transformer("T1", "vin", "gnd",
                       "sec", "sec_gnd",
                       L_p=1e-3, L_s=1e-3, k=0.95)
    b.add_resistor("Rsg", "sec_gnd", "gnd", 1e-6)
    b.add_resistor("R_L", "sec", "sec_gnd", 10.0)

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(dt=1e-7)
    # No exception => KLU factorization succeeded.


def test_buck_topology_via_mosfet_helper() -> None:
    """End-to-end builder smoke test: a buck topology
    via the MOSFET helper should produce the expected
    branch count and not crash on cache.build()."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet_with_body_diode("Q1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", 100e-6)
    b.add_capacitor("Cout", "vout", "gnd", 47e-6)
    b.add_resistor("R_L", "vout", "gnd", 10.0)

    # Should be: source + (switch + body diode) + free-
    # wheeling diode + L + C + R = 7 branches.
    assert b.num_branches == 7

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(dt=1e-7)
    # No exception means the topology + parameters are
    # consistent enough for KLU factorization.
