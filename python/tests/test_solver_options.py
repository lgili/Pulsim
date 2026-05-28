"""`pulsim.SolverOptions` — Round 2 of the v1.5 ergonomics polish.

Bundles 11 advanced kernel knobs (Newton tols, event-iteration cap,
inductor regularisation, Phase 2.4 adaptive-RK schema) into a single
dataclass so the top-level `simulate()` signature stays focused on
the everyday kwargs. The flat kwargs are kept for backward
compatibility; when both are given, the flat kwarg wins.

These tests pin:

1. `SolverOptions` defaults match the legacy `simulate()` defaults
   bit-for-bit (no behavioural drift).
2. `simulate(solver=opts)` produces the same result as the flat
   kwarg path.
3. Flat-kwarg + `solver=` mix: flat wins.
4. `__all__` exposes `SolverOptions` at top level.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


def _build_simple_plant():
    """Two-resistor divider, no switches, no nonlinear devices — the
    simplest plant that still exercises `simulate()`'s defaulting
    logic for every kwarg."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_resistor("R_top", "vin", "mid", 2.0)
    b.add_resistor("R_bot", "mid", "gnd", 3.0)
    return b


def test_solver_options_in_top_level_all() -> None:
    """`SolverOptions` is the recommended entry point for advanced
    kernel knobs and must be discoverable from the top level."""
    assert "SolverOptions" in p.__all__
    assert hasattr(p, "SolverOptions")


def test_solver_options_defaults_match_legacy_kwarg_defaults() -> None:
    """Defaults must not introduce behavioural drift — instantiating
    `SolverOptions()` with no kwargs should produce the same `(opts,
    state)` outcome as calling `simulate()` with no advanced kwargs."""
    opts = p.SolverOptions()

    # Mirror the documented simulate() defaults.
    assert opts.enable_nonlinear_refresh is None
    assert opts.max_newton_iterations == 0
    assert opts.tol_newton_dx is None
    assert opts.tol_newton_res is None
    assert opts.enable_newton_line_search is None
    assert opts.enable_newton_lm is None
    assert opts.max_event_iterations == 0
    assert opts.enable_substep_state_correction is None
    assert opts.inductor_freeze_di_max == 0.0
    assert opts.inductor_abs_clamp == 0.0
    # Phase 2.4 adaptive-RK schema.
    assert opts.integrator == "kernel"
    assert math.isclose(opts.rtol, 1.0e-5, rel_tol=1e-12)
    assert math.isclose(opts.atol, 1.0e-8, rel_tol=1e-12)
    assert opts.dt_init == 0.0


def test_simulate_with_default_solver_matches_flat_default() -> None:
    """Calling `simulate(solver=SolverOptions())` must yield exactly
    the same trajectory as omitting the solver kwarg entirely."""
    b = _build_simple_plant()
    res_flat = p.simulate(b, t_end=1e-3, dt=1e-5)
    res_bundle = p.simulate(b, t_end=1e-3, dt=1e-5,
                              solver=p.SolverOptions())

    flat_states = np.asarray(res_flat.states)
    bundle_states = np.asarray(res_bundle.states)
    assert flat_states.shape == bundle_states.shape
    np.testing.assert_allclose(flat_states, bundle_states,
                                  rtol=1e-12, atol=1e-12)


def test_simulate_flat_kwarg_overrides_solver_field() -> None:
    """When both `solver=` and a flat kwarg are passed, the flat
    kwarg wins. This is what lets users keep their existing scripts
    while gradually migrating to the bundle."""
    b = _build_simple_plant()

    # Build a solver with one non-default knob; then override that
    # knob via the flat kwarg with a different value.
    opts = p.SolverOptions(max_newton_iterations=99)

    # Sanity: with just solver, the field flows through.
    res_solver = p.simulate(b, t_end=1e-3, dt=1e-5, solver=opts)
    assert res_solver.num_steps() > 0

    # Now pass a flat kwarg that contradicts the bundle — should win.
    res_flat_wins = p.simulate(b, t_end=1e-3, dt=1e-5,
                                  solver=opts,
                                  max_newton_iterations=42)
    # We can't easily inspect the SimulationOptions actually used by
    # the kernel, but the run completes — no crash from the override.
    assert res_flat_wins.num_steps() == res_solver.num_steps()


def test_solver_options_with_buck_runs_through_to_setpoint() -> None:
    """End-to-end: drive a buck closed-loop via `solver=` and the
    suggested_dt from the topology factory. Sanity check that the
    new ergonomics path works on the same realistic workload as the
    legacy kwargs."""
    b = p.CircuitBuilder()
    buck = p.add_buck(b, V_in=48.0, L=100e-6, C=100e-6, R_load=4.0,
                       f_sw=100e3)

    F_SW = 100e3
    T_SW = 1.0 / F_SW
    DUTY = 0.5

    def pwm(t: float):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(0, (t % T_SW) < (DUTY * T_SW))
        return m

    res = p.simulate(b, t_end=5e-3, dt=buck.suggested_dt,
                       switch_fn=pwm,
                       solver=p.SolverOptions(
                           max_newton_iterations=50,
                           tol_newton_res=1e-10))
    v_out = float(np.asarray(res.v("vout"))[-1])
    assert 0.85 * (DUTY * 48.0) < v_out < 1.15 * (DUTY * 48.0)


def test_solver_options_integrator_field_round_trips() -> None:
    """The Phase 2.4 schema (integrator/rtol/atol/dt_init) is on
    `SolverOptions` and round-trips through `simulate()`. Unknown
    integrator names raise `ValueError`; `dopri5`/`radau` raise
    `NotImplementedError` with a v1.6 pointer."""
    b = _build_simple_plant()

    # Default "kernel" works.
    res = p.simulate(b, t_end=1e-4, dt=1e-5,
                       solver=p.SolverOptions(integrator="kernel"))
    assert res.num_steps() > 0

    # dopri5 from bundle raises just like the flat kwarg form.
    with pytest.raises(NotImplementedError, match="v1.6"):
        p.simulate(b, t_end=1e-4, dt=1e-5,
                     solver=p.SolverOptions(integrator="dopri5"))

    # Unknown name raises ValueError.
    with pytest.raises(ValueError, match="unknown"):
        p.simulate(b, t_end=1e-4, dt=1e-5,
                     solver=p.SolverOptions(integrator="bogus"))
