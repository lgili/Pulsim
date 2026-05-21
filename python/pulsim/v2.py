"""Pulsim v2 — high-level Python wrapper.

This module re-exports the v2 kernel bindings from
``pulsim._pulsim.v2_kernel`` under a clean ``pulsim.v2``
namespace, plus a high-level :func:`simulate` helper that
encapsulates the cache-build / run_transient dance for the
common case.

Usage:

    import pulsim.v2 as p

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1", "n0", "n1", 100.0)
    b.add_capacitor("C1", "n1", "gnd", 1e-6)

    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    print(res.num_steps(), "samples; last state =", res.states[-1])

For full control, fall back to the explicit pipeline:

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(dt=1e-5)
    opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)
    res = p.run_transient(
        cache, b.graph, b.pool, opts,
        switch_fn=lambda t: p.SwitchStateMask(0),
    )
"""

from __future__ import annotations

from typing import Callable, Optional

from ._pulsim.v2_kernel import (  # type: ignore[import-not-found]
    # Builder API (Layer 6).
    CircuitBuilder,
    # Topology / device handles (opaque — produced by the
    # builder and consumed by the cache).
    Graph,
    DevicePool,
    SwitchStateMask,
    # Cache + simulation.
    PwlStateSpaceCache,
    SimulationOptions,
    SimulationResult,
    CommutationEvent,
    run_transient,
    # Smooth-blend nonlinear diode params (Layer 4 V3).
    IdealDiodeParams,
    # YAML loader (Layer 8).
    LoadedCircuit,
    load_yaml_string,
    load_yaml_file,
    # Source helpers (Layer 2 V5+).
    make_pwm_switch_fn,
    make_dead_time_pwm_pair_fn,
    make_spwm_pair_fn,
    ThreePhaseLegIndices,
    make_three_phase_spwm_fn,
    make_phase_shift_full_bridge_fn,
    make_combined_switch_fn,
)

__all__ = [
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
    "LoadedCircuit",
    "load_yaml_string",
    "load_yaml_file",
    "make_pwm_switch_fn",
    "make_dead_time_pwm_pair_fn",
    "make_spwm_pair_fn",
    "ThreePhaseLegIndices",
    "make_three_phase_spwm_fn",
    "make_phase_shift_full_bridge_fn",
    "make_combined_switch_fn",
    # Proposal #3.3 ergonomics — high-level entry point.
    "simulate",
]


def simulate(
    builder: CircuitBuilder,
    t_end: float,
    dt: float,
    *,
    t_start: float = 0.0,
    switch_fn: Optional[Callable[[float], SwitchStateMask]] = None,
    b_extra_fn: Optional[Callable[[float], "list[float]"]] = None,
    start_from_dc_op: bool = False,
    enable_nonlinear_refresh: Optional[bool] = None,
    max_newton_iterations: int = 0,
    max_event_iterations: int = 0,
    tol_newton_dx: Optional[float] = None,
    tol_newton_res: Optional[float] = None,
    enable_newton_line_search: Optional[bool] = None,
    enable_newton_lm: Optional[bool] = None,
    enable_substep_state_correction: Optional[bool] = None,
) -> SimulationResult:
    """Build the PWL cache and run a fixed-dt transient simulation.

    This is the ergonomic one-call API (proposal #3.3). It collapses
    the cache-build / SimulationOptions / switch_fn dance into a
    single function: pass a populated :class:`CircuitBuilder`, the
    time window, and `dt`, and get a :class:`SimulationResult` back.

    Auto-behaviours (override via keyword args):

    * **`switch_fn`** — defaults to "all switches closed" (i.e. a
      mask with every bit set). For circuits with no switches the
      default is fine. For PWM circuits, pass an explicit
      :func:`make_pwm_switch_fn` or hand-rolled callable.
    * **`enable_nonlinear_refresh`** — auto-detected from the
      builder's pool (proposal #3.4). Set to ``True`` if any of
      smooth-blend `IdealDiode`, SH1 `MosfetLevel1`, Level 1
      `IgbtLevel1`, or `SaturableInductor` is present; ``False``
      otherwise. Pass an explicit bool to override.

    Parameters
    ----------
    builder
        A populated :class:`CircuitBuilder`.
    t_end
        End time, in seconds.
    dt
        Fixed time step, in seconds.
    t_start, default 0.0
        Start time, in seconds.
    switch_fn
        Callable ``t -> SwitchStateMask`` controlling the
        switch state at each sample.  Defaults to all-closed.
    b_extra_fn
        Callable ``t -> list[float]`` adding to the constant
        residual at each step.  Defaults to None (no extras).
    start_from_dc_op
        If ``True``, seed the initial state from
        :func:`compute_dc_op` instead of zero.
    enable_nonlinear_refresh
        Force-enable/-disable the Newton refresh pass.  ``None``
        (default) means auto-detect via
        :meth:`DevicePool.has_nonlinear_devices`.
    max_newton_iterations, max_event_iterations
        Forwarded to :class:`SimulationOptions`.

    Returns
    -------
    SimulationResult
        The full per-sample state-vector history.
    """
    # Build the PWL cache.
    cache = PwlStateSpaceCache(builder.graph, builder.pool)
    cache.build(dt)

    # Construct options.
    opts = SimulationOptions(t_start=t_start, t_end=t_end, dt=dt)
    if max_newton_iterations > 0:
        opts.max_newton_iterations = max_newton_iterations
    if max_event_iterations > 0:
        opts.max_event_iterations = max_event_iterations
    if tol_newton_dx is not None:
        opts.tol_newton_dx = tol_newton_dx
    if tol_newton_res is not None:
        opts.tol_newton_res = tol_newton_res
    if enable_newton_line_search is not None:
        opts.enable_newton_line_search = enable_newton_line_search
    if enable_newton_lm is not None:
        opts.enable_newton_lm = enable_newton_lm
    if enable_substep_state_correction is not None:
        opts.enable_substep_state_correction = enable_substep_state_correction

    # Default switch_fn: all switches closed.
    if switch_fn is None:
        n_sw = builder.graph.num_switches
        default_mask = SwitchStateMask(n_sw)
        for i in range(n_sw):
            default_mask.set(i, True)
        switch_fn = lambda _t: default_mask  # noqa: E731

    # Auto-detect nonlinear devices (proposal #3.4).
    if enable_nonlinear_refresh is None:
        enable_nonlinear_refresh = builder.pool.has_nonlinear_devices()

    # Run the transient — match the keyword convention of the
    # raw `run_transient` binding.
    kwargs: dict = {
        "switch_fn": switch_fn,
        "start_from_dc_op": start_from_dc_op,
        "enable_nonlinear_refresh": enable_nonlinear_refresh,
    }
    if b_extra_fn is not None:
        kwargs["b_extra_fn"] = b_extra_fn

    return run_transient(
        cache, builder.graph, builder.pool, opts, **kwargs,
    )


# Note: SineVoltageSource (Layer 2 V11) is exposed as a
# CircuitBuilder method `add_sine_voltage_source`; there's
# no separate Python-side params class — pass v_dc,
# v_amplitude, frequency, phase as keyword args.

__version__ = "0.1.0"
