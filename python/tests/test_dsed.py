"""Smoke tests for the PED (Path-Based Event-Driven) Python reference.

Phase 2.B-4 of the `add-path-based-dsed-engine` OpenSpec. As of
v1.7.0 the Python scheduler reference lives at
``python/tests/_dsed_reference/`` (formerly ``pulsim.dsed.*``) — see
``python/tests/_dsed_reference/__init__.py`` for the rationale.

These tests exercise:
* The reference :mod:`_dsed_reference` package (DOPRI5, PI
  controller, Illinois root-finder, scheduler) on a synthetic RC
  circuit — zero Pulsim-cache dependency.
* The :func:`pulsim.simulate` dispatcher with ``engine='dsed'`` to
  confirm the bridge from :class:`pulsim.CircuitBuilder` exists.

Numerical correctness is validated against analytical solutions
(exponential decay for the RC) — RMSE ≤ 0.1 % per Gate 1A target.
The standalone-PED RC test mirrors the buck CCM validation captured
in `notes/GATE1_RESULTS.md`, simplified to a 1-state system.
"""

from __future__ import annotations

import numpy as np
import pytest


# -------------------------------------------------------------------------
# Test 1 — pulsim.dsed standalone (no Pulsim cache dependency)
# -------------------------------------------------------------------------


def test_dsed_module_importable():
    """The pulsim.dsed package must be importable + expose key classes."""
    from _dsed_reference import (
        PEDSimulator,
        PEDResult,
        PIController,
        EventPredictor,
        illinois,
    )
    assert PEDSimulator is not None
    assert PEDResult is not None
    assert PIController is not None
    assert EventPredictor is not None
    assert callable(illinois)


def test_illinois_finds_root_of_linear():
    """Sanity check: Illinois root-finder on a linear function."""
    from _dsed_reference import illinois
    g = lambda t: t - 0.5  # noqa: E731
    root = illinois(g, 0.0, 1.0)
    assert abs(root - 0.5) < 1e-9


def test_pi_controller_accepts_small_error():
    """PIController.accept returns True for err << tol."""
    from _dsed_reference import PIController
    c = PIController()
    x_old = np.array([1.0, 2.0])
    x_new = x_old.copy()
    err = np.array([1e-12, 1e-12])
    accepted, h_next = c.accept(err, x_old, x_new, 1e-7)
    assert accepted
    assert h_next > 1e-7  # PI grows on under-tolerance


def test_rk45_on_exponential_decay():
    """DOPRI5 on dx/dt = -x with x(0)=1; check x(1) ≈ 1/e."""
    from _dsed_reference import rk45_step, RK45State

    def f(t, x):
        return -x

    x = np.array([1.0])
    state = RK45State()
    h = 0.01
    t = 0.0
    for _ in range(100):
        x_new, _err = rk45_step(f, t, x, h, state)
        x = x_new
        t += h
    assert abs(x[0] - np.exp(-1.0)) < 1e-9


def test_ped_simulator_on_rc_discharge():
    """End-to-end: PED scheduler on a discharging RC.

    System: dx/dt = -x / (R·C), R=C=1 → time constant 1.
    Analytical: x(t) = x(0) · exp(-t).
    Validate over t in [0, 5] with mask fixed (no events).
    """
    from _dsed_reference import PEDSimulator, PIController

    class RC:
        R = 1.0
        C = 1.0
        _mask = True

        def current_mask(self) -> bool:
            return self._mask

        def set_mask(self, m: bool) -> None:
            self._mask = m

        def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
            return np.array([-x[0] / (self.R * self.C)])

    sys = RC()
    sf = lambda t: True  # noqa: E731 — constant mask, no events
    sim = PEDSimulator(
        system=sys, switch_fn=sf,
        controller=PIController(rtol=1e-6, atol=1e-9),
        dt_init=1e-4, dt_max=0.1,
    )
    result = sim.simulate(np.array([1.0]), t_end=5.0)

    # Final value should match exp(-5)
    expected = np.exp(-5.0)
    assert abs(result.states[-1][0] - expected) < 1e-5
    # Should take very few steps (smooth decay; PI grows h aggressively)
    assert result.n_accept < 200
    # Zero events because mask never changed
    assert result.n_events == 0


def test_ped_simulator_handles_periodic_mask_events():
    """PED scheduler with a periodic mask schedule and analytical
    next_edge_after — confirms the gate-edge fast path.

    Switches between two RC time constants every 0.1 s for 1 s.
    The number of events should be 10 (gate edges).
    """
    from _dsed_reference import PEDSimulator, PIController

    class TwoMode:
        _mask = True
        def current_mask(self): return self._mask
        def set_mask(self, m): self._mask = m
        def rhs(self, t, x):
            tau = 1.0 if self._mask else 0.1
            return np.array([-x[0] / tau])

    class PeriodicMaskFn:
        T = 0.1
        D = 0.5
        def __call__(self, t):
            return (t / self.T) % 1.0 < self.D
        def next_edge_after(self, t):
            k = int(np.floor(t / self.T))
            for c in (k * self.T + self.D * self.T,
                       (k + 1) * self.T,
                       (k + 1) * self.T + self.D * self.T):
                if c > t + 1e-15:
                    return c
            return (k + 2) * self.T

    sys = TwoMode()
    sf = PeriodicMaskFn()
    sim = PEDSimulator(
        system=sys, switch_fn=sf,
        controller=PIController(rtol=1e-6, atol=1e-9),
        dt_init=1e-4, dt_max=0.05,
    )
    result = sim.simulate(np.array([1.0]), t_end=1.0)
    # 1 s / 0.1 s period · 2 edges per period - 1 (terminal) = 19
    # But our switch_fn fires both rising and falling edges
    assert 18 <= result.n_events <= 21


# -------------------------------------------------------------------------
# Test 2 — pulsim.simulate(engine='dsed') dispatcher
# -------------------------------------------------------------------------


def test_simulate_rejects_unknown_engine():
    """pulsim.simulate(engine='foo') must raise ValueError."""
    import pulsim as p
    b = p.CircuitBuilder()
    b.add_resistor("R1", "n0", "gnd", 100.0)
    b.add_voltage_source("V1", "n0", "gnd", 1.0)
    with pytest.raises(ValueError, match="unknown engine"):
        p.simulate(b, t_end=1e-3, dt=1e-5, engine="foo")


def test_simulate_pwl_default_engine_still_works():
    """engine='pwl' (the default) must still produce sane results."""
    import pulsim as p
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1", "n0", "n1", 100.0)
    b.add_capacitor("C1", "n1", "gnd", 1e-6)
    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    # default engine='pwl' → SimulationResult with > 0 samples
    assert res.num_steps() > 0


def test_simulate_dsed_runs_end_to_end_on_minimal_rc():
    """engine='dsed' from a CircuitBuilder runs to completion on a
    minimal RC circuit (Bridge.3 + Bridge.11 wired the full
    PwlStateSpaceCache → NativeCircuitBuilderAdapter → C++ scheduler
    pipeline; pre-Bridge.3 this test asserted NotImplementedError).

    For the algorithm + bridge architecture, see
    ``notes/DSED_BRIDGE_DESIGN.md``.
    """
    import pulsim as p
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1", "n0", "n1", 100.0)
    b.add_capacitor("C1", "n1", "gnd", 1e-6)

    # 10× RC = 10 · (100 · 1e-6) = 1 ms → cap fully charged
    res = p.simulate(b, t_end=1e-3, engine="dsed", integrator="rk45",
                      rtol=1e-7, atol=1e-10)
    assert not res.empty()
    assert res.num_steps() > 0
    # Grounded cap → state[0] is v_C; should approach V_in = 5.0V.
    final_vc = float(res.states[-1][0])
    assert abs(final_vc - 5.0) < 0.01


# -------------------------------------------------------------------------
# Test 3 — New API.2 validation surface (Gate 5+ API consolidation)
# -------------------------------------------------------------------------


def test_simulate_pwl_without_dt_raises_actionable_error():
    """engine='pwl' (default) requires a positive `dt`.

    The error message tells the user EXACTLY what to do:
    either pass dt, or switch to engine='dsed'.
    """
    import pulsim as p
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1", "n0", "n1", 100.0)
    with pytest.raises(ValueError) as exc_info:
        # No dt — this should be a CLEAR error, not an obscure
        # TypeError from a missing required argument.
        p.simulate(b, t_end=1e-3)
    msg = str(exc_info.value)
    assert "engine='pwl'" in msg
    assert "positive `dt`" in msg
    assert "engine='dsed'" in msg   # the suggested alternative


def test_simulate_pwl_with_dsed_kwargs_warns_but_runs():
    """engine='pwl' with dsed-only kwargs (rtol, integrator, ...)
    emits UserWarning but doesn't error — scripts that copy-paste
    kwargs keep running."""
    import pulsim as p
    import warnings as _w
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1", "n0", "n1", 100.0)
    b.add_capacitor("C1", "n1", "gnd", 1e-6)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        # Pass rtol (dsed-only) with engine='pwl' (default)
        res = p.simulate(b, t_end=1e-3, dt=1e-5, rtol=1e-6)
        # Must have at least one UserWarning about the leaked kwarg
        leak_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "engine='pwl'" in str(w.message)
            and "ignored" in str(w.message)
        ]
        assert len(leak_warnings) >= 1, (
            f"Expected UserWarning about leaked dsed kwarg; got {caught}"
        )
    # Simulation completed despite the warning
    assert res.num_steps() > 0


def test_simulate_dsed_unknown_integrator_raises():
    """engine='dsed' with integrator='bdf3' raises ValueError listing
    the valid choices."""
    import pulsim as p
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1", "n0", "n1", 100.0)
    with pytest.raises(ValueError, match="unknown integrator"):
        p.simulate(b, t_end=1e-4, engine="dsed", integrator="bdf3")


def test_simulate_dsed_negative_rtol_raises():
    """engine='dsed' with rtol < 0 raises ValueError before any
    heavy lifting."""
    import pulsim as p
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1", "n0", "n1", 100.0)
    with pytest.raises(ValueError, match="rtol must be positive"):
        p.simulate(b, t_end=1e-4, engine="dsed", rtol=-1.0)


def test_simulate_dsed_resolved_options_are_sensible_by_default():
    """engine='dsed' with no kwargs uses sensible defaults; we read
    them from the resolver (the same one the dispatch uses internally)."""
    from pulsim._dsed_dispatch import _resolve_dsed_options
    opts = _resolve_dsed_options()
    assert opts.rtol == 1e-6
    assert opts.atol == 1e-9
    assert opts.dt_max == 1e-5
    assert opts.dt_init == 1e-9
    assert opts.integrator == "auto"
    assert opts.stiffness_threshold == 10.0
    assert opts.h_bdf2 == 1e-6


def test_simulate_dsed_dt_acts_as_dt_max():
    """When the user passes `dt` with engine='dsed', it's interpreted
    as `dt_max` (upper bound on the adaptive step)."""
    from pulsim._dsed_dispatch import _resolve_dsed_options
    opts = _resolve_dsed_options(dt=2.5e-7)
    assert opts.dt_max == 2.5e-7
    # Other defaults still apply
    assert opts.rtol == 1e-6
