"""Pulsim — power-electronics circuit simulator.

This top-level module re-exports the C++ kernel bindings from
``pulsim._pulsim`` and the Python helper modules under a flat
``pulsim`` namespace, plus a high-level :func:`simulate` helper
that wraps the cache-build / run_transient dance for the common
case.

Usage:

    import pulsim as p

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

from ._pulsim import (  # type: ignore[import-not-found]
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

from .control import (
    PIController,
    PIDController,
    Comparator,
    RateLimiter,
    SampleHold,
    FirstOrderLowPass,
    LookupTable1D,
    MovingAverageFilter,
    # Math
    Gain,
    Sum,
    Subtract,
    MathBlock,
    # Standalone control
    Integrator,
    Differentiator,
    TransferFunction,
    StateMachine,
    OpAmp,
    # Signal
    Limiter,
    DelayBlock,
    # Modulation
    PwmGenerator,
    SpaceVectorModulator,
    # Transforms
    ClarkeTransform,
    InverseClarkeTransform,
    ParkTransform,
    InverseParkTransform,
    # Sync
    PLL,
    # Routing
    SignalMux,
    SignalDemux,
    # Auto-tuning
    tune_pi_from_bode,
    loop_gain,
    phase_margin_from_loop,
    gain_margin_from_loop,
)
from .ac_analysis import (
    AcSweepResult,
    run_ac_sweep,
    extract_phasor,
    plot_bode,
    plot_nyquist,
    stability_margins,
    save_freq_response,
)
from .discovery import (
    catalog,
    example,
    tour,
    list_topologies,
)
from .plot import (
    scope,
    scope_grid,
    plot_currents,
    scope_fft,
    compare,
)
from .blockchain import (
    MixedDomainBlockChain,
    BlockSpec,
    parse_block_chain,
)
from .dc_strategy import (
    compute_dc_op,
    PseudoTransientConfig,
    SourceStepConfig,
)
from .mna_sweep import (
    MnaSweepResult,
    run_mna_sweep,
)
from .adaptive import (
    AdaptiveResult,
    run_transient_adaptive,
)
from .sweep import (
    SweepResult,
    sweep,
    monte_carlo,
)
from .kpi import (
    KpiGate,
    KpiReport,
    KpiCheckResult,
    load_baseline,
    save_baseline,
)
from .snubber import (
    SnubberRecommendation,
    predict_overshoot,
    recommend_snubber,
)
from .thermal import (
    FosterStage,
    fit_foster_from_zth,
    predict_zth_curve,
    compute_temperature,
    add_foster_network,
    make_thermal_observer,
)
from .grid import (
    add_three_phase_grid,
    add_three_phase_line_impedance,
    sequence_components,
    instantaneous_power_3phase,
    voltage_unbalance_factor,
)
from .magnetic import (
    CoreMaterial,
    core_material,
    list_core_materials,
    steinmetz_loss_density,
    igse_loss_density,
    core_loss,
)
from .switchgear import (
    add_rc_snubber,
    make_thyristor_switch_fn,
    make_fuse_switch_fn,
)
from .motors import (
    Mechanical,
    DcMotor,
    PMSM,
    BLDC,
    add_dc_motor,
    make_dc_motor_observer,
    add_pmsm,
    make_pmsm_observer,
    add_bldc,
    make_bldc_observer,
)
from .spice_import import (
    SpiceElement,
    parse_spice_value,
    parse_spice_netlist,
    spice_to_builder,
)
from .stream import LiveStream

# LiveScope is an optional import — only available when pyqtgraph
# is installed. We tolerate the absence so headless environments
# still load `pulsim` without complaint.
try:
    from .scope import LiveScope  # noqa: F401  (re-exported below via __all__)
    _HAS_SCOPE = True
except ImportError:  # pragma: no cover
    _HAS_SCOPE = False

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
    # Closed-loop control building blocks (Phase 2).
    "PIController",
    "PIDController",
    "Comparator",
    "RateLimiter",
    "SampleHold",
    "FirstOrderLowPass",
    "LookupTable1D",
    "MovingAverageFilter",
    # Math blocks (v1 parity Phase A.1)
    "Gain", "Sum", "Subtract", "MathBlock",
    # Standalone control blocks (v1 parity Phase A.1)
    "Integrator", "Differentiator", "TransferFunction",
    "StateMachine", "OpAmp",
    # Signal-shaping (v1 parity Phase A.1)
    "Limiter", "DelayBlock",
    # Modulation (v1 parity Phase A.1)
    "PwmGenerator", "SpaceVectorModulator",
    # Transforms (v1 parity Phase A.1)
    "ClarkeTransform", "InverseClarkeTransform",
    "ParkTransform", "InverseParkTransform",
    # Sync (v1 parity Phase A.1)
    "PLL",
    # Routing (v1 parity Phase A.1)
    "SignalMux", "SignalDemux",
    # Auto-tuning helpers (loop-shaping from a measured Bode).
    "tune_pi_from_bode",
    "loop_gain",
    "phase_margin_from_loop",
    "gain_margin_from_loop",
    # AC small-signal analysis (swept-sine Bode).
    "AcSweepResult",
    "run_ac_sweep",
    "extract_phasor",
    "plot_bode",
    "plot_nyquist",
    "stability_margins",
    "save_freq_response",
    # Discovery helpers (introspect the available surface).
    "catalog",
    "example",
    "tour",
    "list_topologies",
    # Plot helpers (one-line waveform + multi-panel).
    "scope",
    "scope_grid",
    "plot_currents",
    "scope_fft",
    "compare",
    # Mixed-domain block-chain executor (v1 parity Phase A.1 stage 2).
    "MixedDomainBlockChain",
    "BlockSpec",
    "parse_block_chain",
    # DC operating-point strategies (v1 parity Phase A.2).
    "compute_dc_op",
    "PseudoTransientConfig",
    "SourceStepConfig",
    # Fast frequency sweep via impulse-response (v1 parity Phase A.3).
    "MnaSweepResult",
    "run_mna_sweep",
    # Adaptive (variable-step) transient driver (Phase B.1).
    "AdaptiveResult",
    "run_transient_adaptive",
    # Parameter sweep + Monte Carlo (Phase E.3).
    "SweepResult",
    "sweep",
    "monte_carlo",
    # KPI gates + baselines (Phase E.5).
    "KpiGate",
    "KpiReport",
    "KpiCheckResult",
    "load_baseline",
    "save_baseline",
    # Snubber advisor (Phase E.8).
    "SnubberRecommendation",
    "predict_overshoot",
    "recommend_snubber",
    # Thermal Foster networks + electro-thermal co-sim (Phase C.1).
    "FosterStage",
    "fit_foster_from_zth",
    "predict_zth_curve",
    "compute_temperature",
    "add_foster_network",
    "make_thermal_observer",
    # Three-phase grid helpers (Phase C.2).
    "add_three_phase_grid",
    "add_three_phase_line_impedance",
    "sequence_components",
    "instantaneous_power_3phase",
    "voltage_unbalance_factor",
    # Magnetic core-loss models (Phase C.3).
    "CoreMaterial",
    "core_material",
    "list_core_materials",
    "steinmetz_loss_density",
    "igse_loss_density",
    "core_loss",
    # Switchgear & protection (Phase C.4).
    "add_rc_snubber",
    "make_thyristor_switch_fn",
    "make_fuse_switch_fn",
    # Electromechanical motor models (Phase D).
    "Mechanical",
    "DcMotor",
    "PMSM",
    "BLDC",
    "add_dc_motor",
    "make_dc_motor_observer",
    "add_pmsm",
    "make_pmsm_observer",
    "add_bldc",
    "make_bldc_observer",
    # SPICE netlist import (Phase E.10).
    "SpiceElement",
    "parse_spice_value",
    "parse_spice_netlist",
    "spice_to_builder",
    # Live streaming output + cancellation (foundation for GUI scope).
    "LiveStream",
]

if _HAS_SCOPE:
    __all__.append("LiveScope")


def simulate(
    builder: CircuitBuilder,
    t_end: float,
    dt: float,
    *,
    t_start: float = 0.0,
    switch_fn: Optional[Callable[[float], SwitchStateMask]] = None,
    b_extra_fn: Optional[Callable[[float], "list[float]"]] = None,
    step_observer: Optional[Callable[[float, "object"], None]] = None,
    start_from_dc_op: bool = False,
    enable_nonlinear_refresh: Optional[bool] = None,
    max_newton_iterations: int = 0,
    max_event_iterations: int = 0,
    tol_newton_dx: Optional[float] = None,
    tol_newton_res: Optional[float] = None,
    enable_newton_line_search: Optional[bool] = None,
    enable_newton_lm: Optional[bool] = None,
    enable_substep_state_correction: Optional[bool] = None,
    progress: "bool | int | str" = False,
    initial_state=None,
    should_continue=None,
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

    # Progress bar via step_observer: print a percentage at
    # configurable intervals. The user can pass:
    #   progress=False              — no progress (default)
    #   progress=True               — auto: print every 10%
    #   progress=<int>              — print every N%
    #   progress="bar"              — full ASCII progress bar
    # If the user also supplied a step_observer, we wrap it.
    if progress is not False and progress is not None:
        if progress is True:
            print_every_pct = 10
            bar_mode = False
        elif isinstance(progress, int):
            print_every_pct = max(1, min(100, progress))
            bar_mode = False
        elif isinstance(progress, str) and progress.lower() == "bar":
            print_every_pct = 1
            bar_mode = True
        else:
            print_every_pct = 10
            bar_mode = False

        import sys as _sys
        import time as _time
        _start = _time.perf_counter()
        # Mutable state for closure.
        _last_pct_printed = [-1]
        _t_total = float(t_end - t_start)
        _user_observer = step_observer

        def _progress_observer(t, x):
            if _user_observer is not None:
                _user_observer(t, x)
            if _t_total <= 0:
                return
            pct = int(100.0 * (t - t_start) / _t_total)
            # Only print on multiples of print_every_pct, and only
            # once per multiple (to handle the per-dt observer
            # cadence).
            if pct >= _last_pct_printed[0] + print_every_pct:
                _last_pct_printed[0] = pct - (pct % print_every_pct)
                elapsed = _time.perf_counter() - _start
                if bar_mode:
                    barw = 40
                    filled = int(barw * pct / 100)
                    bar = "█" * filled + "░" * (barw - filled)
                    _sys.stdout.write(
                        f"\r  [{bar}] {pct:3d}% "
                        f"  t={t*1e3:.2f} ms  ({elapsed:5.1f}s)"
                    )
                    _sys.stdout.flush()
                else:
                    _sys.stdout.write(
                        f"  progress: {pct:3d}%  "
                        f"t={t*1e3:.2f} ms  "
                        f"({elapsed:.1f}s)\n"
                    )
                    _sys.stdout.flush()

        step_observer = _progress_observer

    # Fast-path: if step_observer carries an attached `_cxx_chain`
    # attribute (set by `MixedDomainBlockChain.make_step_observer`),
    # use `run_transient_with_chain` which invokes the C++ chain
    # directly each step — skipping the pybind11 std::function
    # wrap. Saves ~10-30 % wall time on chains > 5 blocks.
    cxx_chain = getattr(step_observer, "_cxx_chain", None) \
                  if step_observer is not None else None
    if cxx_chain is not None:
        from . import _pulsim as _k  # type: ignore[import-not-found]
        chain_dt = getattr(step_observer, "_chain_dt", dt)
        kwargs: dict = {
            "switch_fn": switch_fn,
            "start_from_dc_op": start_from_dc_op,
            "enable_nonlinear_refresh": enable_nonlinear_refresh,
        }
        if b_extra_fn is not None:
            kwargs["b_extra_fn"] = b_extra_fn
        if initial_state is not None:
            kwargs["initial_state"] = initial_state
        if should_continue is not None:
            kwargs["should_continue"] = should_continue
        res = _k.run_transient_with_chain(
            cache, builder.graph, builder.pool, opts,
            chain=cxx_chain, chain_dt=chain_dt,
            **kwargs,
        )
    else:
        # Standard path — pybind11-wrapped step_observer (Python or
        # plain C++ via std::function).
        kwargs = {
            "switch_fn": switch_fn,
            "start_from_dc_op": start_from_dc_op,
            "enable_nonlinear_refresh": enable_nonlinear_refresh,
        }
        if b_extra_fn is not None:
            kwargs["b_extra_fn"] = b_extra_fn
        if step_observer is not None:
            kwargs["step_observer"] = step_observer
        if initial_state is not None:
            kwargs["initial_state"] = initial_state
        if should_continue is not None:
            kwargs["should_continue"] = should_continue
        res = run_transient(
            cache, builder.graph, builder.pool, opts, **kwargs,
        )
    # Close the progress bar with a newline if we were in bar mode.
    if progress is True or (isinstance(progress, str)
                              and progress.lower() == "bar"):
        import sys as _sys
        _sys.stdout.write("\n")
        _sys.stdout.flush()
    return res


# Note: SineVoltageSource (Layer 2 V11) is exposed as a
# CircuitBuilder method `add_sine_voltage_source`; there's
# no separate Python-side params class — pass v_dc,
# v_amplitude, frequency, phase as keyword args.

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Graceful failure for legacy v1 symbols
# ---------------------------------------------------------------------------
# v1 used to export ~150 symbols at the top level (Resistor, Circuit,
# Simulator, Preset, AdvancedOptions, codegen, fmu, parse_netlist, …).
# Code that references any of them now hits a NameError at attribute
# resolution. We override ``__getattr__`` (PEP 562) to give an actionable
# migration hint instead of the raw Python error.

_V1_SYMBOL_HINTS = {
    "Circuit":            "p.CircuitBuilder()",
    "RuntimeCircuit":     "p.CircuitBuilder()",
    "Simulator":          "p.simulate(b, t_end=..., dt=...)",
    "SimulationResult":   "p.SimulationResult (still exported)",
    "Resistor":           "b.add_resistor(name, from_node, to_node, R)",
    "Capacitor":          "b.add_capacitor(...)",
    "Inductor":           "b.add_inductor(...)",
    "VoltageSource":      "b.add_voltage_source(...)",
    "CurrentSource":      "b.add_current_source(...)",
    "Diode":              "b.add_diode(name, anode, cathode, g_on, g_off, V_th)",
    "MOSFET":             "b.add_mosfet_with_body_diode(...)",
    "IGBT":               "b.add_igbt(...)",
    "Preset":             "no longer exposed — pass an explicit SimulationOptions",
    "AdvancedOptions":    "no longer exposed — pass an explicit SimulationOptions",
    "PulseParams":        "p.make_pwm_switch_fn(...) for PWM gates",
    "YamlParser":         "p.load_yaml_file(path) returns a LoadedCircuit",
    "YamlParserOptions":  "no equivalent — load_yaml_file / load_yaml_string take no options",
    "RobustnessProfile":  "no equivalent — use enable_nonlinear_refresh + DC-OP strategies",
    "codegen":            "no equivalent in 1.0.0",
    "fmu":                "no equivalent in 1.0.0",
    "templates":          "no equivalent in 1.0.0",
    "schematic":          "no equivalent in 1.0.0",
    "parse_netlist":      "p.parse_spice_netlist (SPICE subset)",
}


def __getattr__(name: str):
    """PEP 562 module-level ``__getattr__`` — fires only when a
    name isn't found in the normal namespace. Translates legacy v1
    attribute accesses into actionable migration hints."""
    if name in _V1_SYMBOL_HINTS:
        hint = _V1_SYMBOL_HINTS[name]
        raise AttributeError(
            f"pulsim.{name} was a v1 symbol and is no longer "
            f"available (pulsim 1.0.0 retired the legacy kernel). "
            f"Migration hint: {hint}. See "
            f"docs/migration-guide.md for the full mapping.")
    raise AttributeError(
        f"module 'pulsim' has no attribute {name!r}. "
        f"pulsim 1.0.0 ships only the modern surface — "
        f"see docs/migration-guide.md if you're porting v1 code.")
