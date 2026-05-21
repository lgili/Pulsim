"""Pulsim v2 — high-level Python wrapper.

This module re-exports the v2 kernel bindings from
``pulsim._pulsim.v2_kernel`` under a clean ``pulsim.v2``
namespace.

Usage:

    import pulsim.v2 as p

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1", "n0", "n1", 100.0)
    b.add_capacitor("C1", "n1", "gnd", 1e-6)

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(dt=1e-5)

    opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)

    res = p.run_transient(
        cache, b.graph, b.pool, opts,
        switch_fn=lambda t: p.SwitchStateMask(0),
    )
    print(res.num_steps(), "samples; last state =", res.states[-1])
"""

from __future__ import annotations

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
]

# Note: SineVoltageSource (Layer 2 V11) is exposed as a
# CircuitBuilder method `add_sine_voltage_source`; there's
# no separate Python-side params class — pass v_dc,
# v_amplitude, frequency, phase as keyword args.

__version__ = "0.1.0"
