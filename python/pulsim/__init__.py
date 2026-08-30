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

from dataclasses import dataclass
from typing import Callable, Optional

from ._pulsim import (  # type: ignore[import-not-found]
    # v2.0 Phase 2 — raised when a transient cannot continue. Unlike
    # a bare RuntimeError it carries `.partial` (everything computed
    # before the failure) and `.t_failed` (the step it could not
    # take), so a run that dies at 90 % is not a total loss. Still a
    # subclass of RuntimeError: returning a truncated result as if it
    # were whole would be the silent wrong answer this API exists to
    # avoid, so the partial trace has to be asked for.
    SimulationAborted,
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
    PreflightFinding,
    PreflightIssue,
    PreflightOptions,
    PreflightReport,
    # Phase-0 fix #4 helper (private): controlled-vs-diode census.
    _switch_census,
    run_transient,
    run_transient_trbdf2,
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
    # Bridge.12 — native PWM switch_fn classes for the DSED engine.
    # Detected by the C++ scheduler binding and called without GIL
    # roundtrips per scheduler step. ~25× faster per call than the
    # equivalent Python `class PWM` pattern.
    NativePwm2Switch,
    NativeMultiMaskPwm,
)

from .control import (
    PIController,
    PIDController,
    # add-python-closed-loop-helper (v1.5)
    ClosedLoop,
    bind_pi_to_switch,
    bind_pi_to_duty_callable,
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
    SettleConfig,
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
    # v1.4.0 — path-aware variants exploiting refactor_parametric
    sweep_path_aware,
    monte_carlo_path_aware,
)
# Phase-0 fix #8: the migration guide documents
# `p.run_periodic_shooting(...)` but the symbol was never exported —
# users landed on a misleading AttributeError from the PEP 562 hook.
from .periodic import (
    PeriodicShootingResult,
    run_periodic_shooting,
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
    CauerStage,
    fit_foster_from_zth,
    predict_zth_curve,
    compute_temperature,
    add_foster_network,
    add_cauer_thermal_network,
    make_thermal_observer,
    device_thermal_summary,
    ThermalLimitMonitor,
    # Shared heatsink — N devices coupled through one sink (P1).
    HeatsinkDevice,
    SharedHeatsink,
    shared_heatsink_steady_state,
    add_shared_heatsink,
    make_heatsink_observer,
    # Temperature-dependent loss + electro-thermal feedback (P2).
    TempCoLoss,
    electrothermal_steady_state,
    make_electrothermal_heatsink_observer,
    # Heatsink + TIM sizing helpers (P5).
    TIM_CATALOG,
    tim_resistance,
    convection_coefficient,
    convection_resistance,
)
from . import fast_block as _fast_block_module  # noqa: F401  (registers submodule)
from .fast_block import (
    FastBlock,
    fast_block,
)
# Module access: `import pulsim.fast_block` or
# `sys.modules['pulsim.fast_block']`; `pulsim.fast_block` itself
# resolves to the decorator (re-exported above).
from .topology import (
    BuckResult,
    BoostResult,
    FlybackResult,
    add_buck,
    add_boost,
    add_flyback,
    add_bridge_rectifier,
    add_three_phase_vsi,
    add_three_phase_rl_load,
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
    InductionMotor,
    MotorObserverBundle,
    add_dc_motor,
    make_dc_motor_observer,
    add_pmsm,
    make_pmsm_observer,
    add_bldc,
    make_bldc_observer,
    add_induction_motor,
    make_induction_motor_observer,
    im_parameters_from_nameplate,
)
from .observers import (
    SlidingModeObserver,
    FluxMRASObserver,
)
from .hysteresis import (
    JilesAthertonParams,
    JilesAthertonModel,
    reference_material,
    list_reference_materials,
    compute_bh_loop,
    core_loss_jiles_atherton,
    fit_ja_from_bh_curve,
    BHLoopResult,
    HystereticInductor,
    add_hysteretic_inductor,
    make_hysteretic_inductor_observer,
)
# DSED public subpackage — bound as a namespace attribute so users
# can write `pulsim.dsed.run_user_lti(...)`. The PEP 562
# `__getattr__` at the bottom of this module would otherwise reject
# the access with the v1-migration error. v1.6.3 demoted the
# pure-Python schedulers to `python/tests/_dsed_reference/`; the
# public `pulsim.dsed` surface is now just `run_user_lti` + the
# `CircuitBuilderAdapter` for Bridge.10 advanced users.
from . import dsed as dsed  # noqa: F401  (re-export submodule)
from .yaml_chain import wire_chain_from_yaml
from .integrators import (
    AdaptiveSolution,
    DormandPrince5,
    RadauIIA3,
)
from .spice_import import (
    SpiceElement,
    parse_spice_value,
    parse_spice_netlist,
    spice_to_builder,
)
from .stream import LiveStream, NativeLiveStream
from .losses import (
    LossAccumulator,
    EfficiencyCalculator,
    device_loss_summary,
    average_power_at_node,
)
# Custom-code block ("C block"): user code (Python / C / C++) wired to
# the circuit, running at its own sample time.
from .c_block import (
    add_c_block, CBlockHandle, CBLOCK_ABI, wire_c_blocks_from_yaml)
# Records the true per-step switch mask at simulate-time so closed-loop
# loss/thermal summaries don't re-evaluate a stateful switch_fn post-hoc.
from ._result_views import SwitchMaskRecorder
# MMC re-exports — kept as top-level attributes for backward
# compatibility but EXCLUDED from `__all__` so `dir(p)` and
# `from pulsim import *` stay focused on the everyday surface.
# MMC users either name them explicitly (`from pulsim import
# MmcArmAverage`) or qualify via the submodule
# (`from pulsim.mmc import ...`). The `# noqa: F401` quiets the
# linter on these intentional unused imports.
from .mmc import (
    MmcArmAverageParams,  # noqa: F401
    MmcArmAverageResult,  # noqa: F401
    mmc_arm_average_step,  # noqa: F401
    simulate_mmc_arm_average,  # noqa: F401
    # Builder-side wiring (Phase 20.4).
    MmcArmAverage,  # noqa: F401
    add_mmc_arm_average,  # noqa: F401
    make_mmc_arm_observer,  # noqa: F401
    make_mmc_arms_observer,  # noqa: F401
    # Three-phase DC/AC topology helper (Phase 20.4 step 2).
    MmcThreePhaseDcAc,  # noqa: F401
    add_mmc_three_phase_dc_ac,  # noqa: F401
    # L1 multilevel arm — Phase 20.5.
    MmcArmMultilevelParams,  # noqa: F401
    MmcArmMultilevelResult,  # noqa: F401
    ps_pwm_switching_function,  # noqa: F401
    ipd_switching_function,  # noqa: F401
    mmc_arm_multilevel_step,  # noqa: F401
    simulate_mmc_arm_multilevel,  # noqa: F401
    # L2 SM-equivalent arm (dead-time aware) — Phase 20.6.
    MmcArmEquivalentParams,  # noqa: F401
    MmcArmEquivalentState,  # noqa: F401
    MmcArmEquivalentResult,  # noqa: F401
    make_l2_state,  # noqa: F401
    mmc_arm_equivalent_step,  # noqa: F401
    simulate_mmc_arm_equivalent,  # noqa: F401
    # L3 detailed (per-SM balancing) — Phase 20.7.
    MmcArmDetailedParams,  # noqa: F401
    MmcArmDetailedState,  # noqa: F401
    MmcArmDetailedResult,  # noqa: F401
    make_l3_state,  # noqa: F401
    mmc_arm_detailed_step,  # noqa: F401
    simulate_mmc_arm_detailed,  # noqa: F401
    # Builder integration for L1/L2/L3 — Phase 20.8.
    MmcArmMultilevel,  # noqa: F401
    add_mmc_arm_multilevel,  # noqa: F401
    make_mmc_arm_multilevel_observer,  # noqa: F401
    make_mmc_arm_multilevel_observers,  # noqa: F401
    MmcArmEquivalent,  # noqa: F401
    add_mmc_arm_equivalent,  # noqa: F401
    make_mmc_arm_equivalent_observer,  # noqa: F401
    make_mmc_arm_equivalent_observers,  # noqa: F401
    MmcArmDetailed,  # noqa: F401
    add_mmc_arm_detailed,  # noqa: F401
    make_mmc_arm_detailed_observer,  # noqa: F401
    make_mmc_arm_detailed_observers,  # noqa: F401
)

# v2.0 Phase 3 — GGJ Thevenin arm (exact aggregation; supersedes the
# delayed co-simulation L3 path for fixed-step pwl runs).
from .mmc_thevenin import (
    MmcThevArm,  # noqa: F401
    add_mmc_thevenin_arm,  # noqa: F401
)

# Schematic renderer — optional, gated behind `[schematic]` extras
# (schemdraw + networkx + cairosvg + anthropic). Importing the
# submodule does NOT require the heavy deps; render/layout calls
# raise a clear ImportError pointing at the install command.
from . import schematic  # noqa: F401

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
    "PreflightFinding",
    "PreflightIssue",
    "PreflightOptions",
    "PreflightReport",
    "run_transient",
    "IdealDiodeParams",
    "LoadedCircuit",
    "load_yaml_string",
    "load_yaml_file",
    "NativePwm2Switch",
    "NativeMultiMaskPwm",
    "wire_chain_from_yaml",
    "make_pwm_switch_fn",
    "make_dead_time_pwm_pair_fn",
    "make_spwm_pair_fn",
    "ThreePhaseLegIndices",
    "make_three_phase_spwm_fn",
    "make_phase_shift_full_bridge_fn",
    "make_combined_switch_fn",
    # Proposal #3.3 ergonomics — high-level entry point.
    "simulate",
    # add-python-named-lookups (v1.5).
    "NameNotFoundError",
    # add-python-builder-ergonomics (v1.5).
    "Cancelled",
    "SimulationAborted",
    # Closed-loop control building blocks (Phase 2 + add-python-closed-loop-helper v1.5).
    "PIController",
    "ClosedLoop",
    "bind_pi_to_switch",
    "bind_pi_to_duty_callable",
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
    "SettleConfig",
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
    "sweep_path_aware",
    "monte_carlo_path_aware",
    # Periodic steady-state shooting (documented in the migration
    # guide since v1.5; exported as of Phase 0).
    "PeriodicShootingResult",
    "run_periodic_shooting",
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
    "CauerStage",
    "fit_foster_from_zth",
    "predict_zth_curve",
    "compute_temperature",
    "add_foster_network",
    "add_cauer_thermal_network",
    "make_thermal_observer",
    "device_thermal_summary",
    "ThermalLimitMonitor",
    # Switch-mask recorder — exact masks for closed-loop loss/thermal.
    "SwitchMaskRecorder",
    # Shared heatsink — N devices coupled through one sink (P1).
    "HeatsinkDevice",
    "SharedHeatsink",
    "shared_heatsink_steady_state",
    "add_shared_heatsink",
    "make_heatsink_observer",
    # Temperature-dependent loss + electro-thermal feedback (P2).
    "TempCoLoss",
    "electrothermal_steady_state",
    "make_electrothermal_heatsink_observer",
    # Heatsink + TIM sizing helpers (P5).
    "TIM_CATALOG",
    "tim_resistance",
    "convection_coefficient",
    "convection_resistance",
    # PSIM/PLECS-style JIT control block (Numba — optional dep).
    "FastBlock",
    "fast_block",
    # Solver options bundle (v1.5 ergonomics — bundles 11 advanced
    # kernel knobs into one dataclass so `simulate()`'s top-level
    # signature stays focused on the everyday kwargs).
    "SolverOptions",
    # SMPS topology factories (v1.5 ergonomics).
    "BuckResult",
    "BoostResult",
    "FlybackResult",
    "add_buck",
    "add_boost",
    "add_flyback",
    "add_bridge_rectifier",
    "add_three_phase_vsi",
    "add_three_phase_rl_load",
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
    "InductionMotor",
    "add_dc_motor",
    "make_dc_motor_observer",
    "MotorObserverBundle",
    "add_pmsm",
    "make_pmsm_observer",
    "add_bldc",
    "make_bldc_observer",
    "add_induction_motor",
    "make_induction_motor_observer",
    "im_parameters_from_nameplate",
    # Sensorless observers (Phase 2.3).
    "SlidingModeObserver",
    "FluxMRASObserver",
    # Magnetic hysteresis — Jiles-Atherton (Phase 2.2).
    "JilesAthertonParams",
    "JilesAthertonModel",
    "reference_material",
    "list_reference_materials",
    "compute_bh_loop",
    "core_loss_jiles_atherton",
    "fit_ja_from_bh_curve",
    "BHLoopResult",
    "HystereticInductor",
    "add_hysteretic_inductor",
    "make_hysteretic_inductor_observer",
    # Adaptive RK integrators (Phase 2.4).
    "AdaptiveSolution",
    "DormandPrince5",
    "RadauIIA3",
    # SPICE netlist import (Phase E.10).
    "SpiceElement",
    "parse_spice_value",
    "parse_spice_netlist",
    "spice_to_builder",
    # Live streaming output + cancellation (foundation for GUI scope).
    "LiveStream",
    "NativeLiveStream",
    # Post-hoc loss + efficiency helpers (parity with v1 surface).
    "LossAccumulator",
    "EfficiencyCalculator",
    "device_loss_summary",
    "average_power_at_node",
    # Custom-code block (C block).
    "add_c_block",
    "CBlockHandle",
    "CBLOCK_ABI",
    "wire_c_blocks_from_yaml",
    # Modular Multilevel Converter (Phase 20) helpers live as
    # top-level attributes for backward compatibility, but are
    # excluded from `__all__` so `from pulsim import *` and
    # `dir(pulsim)`-driven completion stay focused on the
    # everyday surface. MMC users still import them directly:
    #
    #     from pulsim import MmcArmAverage, add_mmc_arm_average
    #
    # or namespace-qualify via the dedicated submodule:
    #
    #     from pulsim.mmc import MmcArmDetailed, simulate_mmc_arm_average
    #
    # 40+ MMC symbols intentionally NOT listed here.
]

if _HAS_SCOPE:
    __all__.append("LiveScope")


# add-python-builder-ergonomics (v1.5) — patch IC + alias helpers onto
# CircuitBuilder, define Cancelled exception, expose simulate-wrapped
# analyses with `should_continue` cancellation.
from . import _builder_ergonomics as _berg
_berg.install(CircuitBuilder)
Cancelled = _berg.Cancelled



class NameNotFoundError(KeyError):
    """Raised by :meth:`SimulationResult.v` / `.i` / `.power` and the
    :class:`CircuitBuilder` lookup helpers when a requested name
    isn't registered. Carries ``name``, ``kind`` (one of
    ``"node"``, ``"branch"``, ``"switch"``), and up to three
    fuzzy-matched ``suggestions`` so the caller can hint at a
    likely typo.

    Subclasses :class:`KeyError` so existing
    ``try: result.v(name) except KeyError`` patterns keep
    working.
    """

    __slots__ = ("name", "kind", "suggestions")

    def __init__(
        self,
        name: str,
        kind: str,
        suggestions: "list[str] | None" = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.suggestions = list(suggestions or [])
        hint = (
            f" Did you mean {self.suggestions}?"
            if self.suggestions else ""
        )
        super().__init__(
            f"{kind} {name!r} is not registered.{hint}"
        )


def _result_v(self, name: str, t=None):
    """Return the node-voltage trace for ``name``.

    See :class:`NameNotFoundError` for the typed error raised on
    unknown names. The result wrapper stores the source
    :class:`CircuitBuilder` as ``self._builder`` so name lookup
    works without forcing callers to re-pass it.

    Parameters
    ----------
    name : str
        Node name passed to ``CircuitBuilder.add_*`` (or a
        registered alias when alias support is enabled).
    t : int | slice | array-like, optional
        Step selection. ``None`` (default) returns the full
        per-sample trace as a ``numpy.ndarray``. An ``int``
        returns a scalar ``float``.
    """
    import numpy as _np
    builder = getattr(self, "_builder", None)
    if builder is None:
        raise RuntimeError(
            "SimulationResult.v requires the result to carry a "
            "_builder reference. Use pulsim.simulate(...) so the "
            "wrapper is attached automatically, or set "
            "result._builder = builder by hand."
        )
    # add-python-builder-ergonomics: consult the alias map first.
    # If `name` is a registered alias for a node, route through to
    # the canonical name's node_id_of.
    alias = builder._resolve_alias(name) if hasattr(
        builder, "_resolve_alias") else None
    if alias is not None and alias[0] == "node":
        name = alias[1]
    try:
        idx = builder.node_id_of(name)
    except IndexError:
        # Pulsim's Graph exposes nodes as a list of dicts
        # ``[{"id": int, "name": str}, ...]``. We walk it to suggest
        # close matches when a typo lands here.
        candidates: list[str] = []
        try:
            for node in builder.graph.nodes:
                n_name = node.get("name") if isinstance(node, dict) \
                          else getattr(node, "name", "")
                if n_name:
                    candidates.append(str(n_name))
        except Exception:  # noqa: BLE001 — defensive only
            pass
        import difflib as _difflib
        sugg = _difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
        raise NameNotFoundError(name, "node", sugg) from None
    states = _np.asarray(self.states)
    # `states` is a READ-ONLY zero-copy view over the kernel buffer
    # (v2.0). A per-signal trace is small (one column), and v1.x
    # handed back a writable owned array here — so copy the column
    # out rather than leaking read-onlyness (and a reference pinning
    # the whole run) into `res.v(...)`.
    col = _np.array(states[:, idx], dtype=float)
    if t is None:
        return col
    return col[t]


def _result_i(self, name: str, t=None):
    """Return the branch-current trace for ``name``.

    ``result.i()`` aims at the PLECS-style output-equation convention:
    the simulator's state vector ``x`` is the small set of MNA
    natives (node voltages + augmented inductor / voltage-source
    currents), and any branch current is a deterministic function of
    that state plus the device's stored params + (for switched
    devices) the active switch mask. Reading any current is therefore
    a per-step evaluation of that function — no need for sentinel
    probes or sense resistors in the topology.

    Supported branch kinds and their reconstruction:

    +----------------------------+-----------------------------------------+
    | kind                        | ``i(t)``                                |
    +============================+=========================================+
    | ``inductor``               | ``states[:, branch_var_id_for_inductor]`` |
    | ``voltage_source``         | ``states[:, branch_var_id_for_source]`` |
    | ``pwm_voltage_source``     | ditto                                   |
    | ``sine_voltage_source``    | ditto                                   |
    | ``pulse_voltage_source``   | ditto                                   |
    | ``resistor``               | ``(V_from − V_to) / R_ohms``            |
    | ``capacitor``              | ``C · d(V_from − V_to)/dt``             |
    | ``current_source``         | constant ``I`` from params              |
    | ``diode`` (PWL switched)   | ``(V_from − V_to − V_th)·G`` where      |
    |                            | ``G = g_on`` when forward-biased,       |
    |                            | ``g_off`` otherwise                     |
    | ``switch``                 | ``(V_from − V_to)·G`` where             |
    |                            | ``G = g_on`` when ``switch_fn`` mask    |
    |                            | bit is set, ``g_off`` otherwise         |
    +----------------------------+-----------------------------------------+

    Sentinel handling. Pulsim uses ``node_id = −1`` for the
    ground / reference node (no column in the state vector;
    implicit zero in the MNA solve). The reconstruction treats
    ``node_id < 0`` as ``V = 0`` instead of Python's
    negative-indexing into ``states``.

    Switch-fn requirement. Switch-branch reconstruction needs the
    per-step mask schedule. ``pulsim.simulate(...)`` auto-stashes the
    composed ``switch_fn`` on ``result._switch_fn``; if you're
    operating on a result built another way, set ``result._switch_fn
    = your_switch_fn`` before calling ``.i()``.

    Not yet supported (deferred to future PRs because their device
    params aren't exposed by ``builder.components()`` today):
    ``mosfet_level1``, ``igbt_level1``, ``nonlinear_diode``,
    ``vcvs``, ``saturable_inductor``. Calls on those kinds raise
    :class:`NotImplementedError` pointing at
    :func:`pulsim.losses.device_loss_summary`, which already
    implements per-step nonlinear stamp evaluation for the loss
    summary.

    Sign convention. Current is positive flowing from the ``from``
    terminal to the ``to`` terminal of the ``add_*`` call (matches
    the inductor / voltage-source convention).

    Parameters
    ----------
    name
        Branch name (any supported kind from the table above).
    t
        Optional step selection (``int`` / ``slice`` / array-like).
        ``None`` returns the full per-sample trace.
    """
    import numpy as _np
    from ._result_views import (
        node_voltage_trace as _node_v,
        resolve_switch_closed_trace as _resolve_switch_closed,
        states_as_array as _sa,
        times_as_array as _ta,
    )
    builder = getattr(self, "_builder", None)
    if builder is None:
        raise RuntimeError(
            "SimulationResult.i requires the result to carry a "
            "_builder reference. Use pulsim.simulate(...) so the "
            "wrapper is attached automatically."
        )
    # Same alias resolution as `.v`.
    alias = builder._resolve_alias(name) if hasattr(
        builder, "_resolve_alias") else None
    if alias is not None and alias[0] == "branch":
        name = alias[1]
    try:
        b_id = builder.branch_index_of(name)
    except IndexError:
        candidates = [d.name for d in builder.devices()]
        import difflib as _difflib
        sugg = _difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
        raise NameNotFoundError(name, "branch", sugg) from None

    # ---- Fast path: state-vector-native kinds (inductor, VS family) ----
    # Try inductor first (most common case), then the source family
    # (which also covers PWM / sine / pulse voltage sources — they
    # all share ``branch_var_id_for_source``).
    state_idx: "int | None" = None
    for accessor in (
        "branch_var_id_for_inductor",
        "branch_var_id_for_source",
    ):
        fn = getattr(builder.pool, accessor, None)
        if fn is None:
            continue
        try:
            state_idx = int(fn(b_id, builder.graph))
            break
        except Exception:  # noqa: BLE001 — wrong device kind
            continue
    states = _sa(self)
    if state_idx is not None:
        # Owned, writable copy of the single column — see the note
        # in `_result_v`. Keeps `res.i(...)` writable regardless of
        # whether the device is state-native (inductor branch var)
        # or computed (resistor), so writability never depends on
        # the device kind.
        col = _np.array(states[:, state_idx], dtype=float)
        if t is None:
            return col
        return col[t]

    # ---- Slow path: dispatch on `kind` from components() ----
    desc = None
    try:
        for d in builder.components():
            if int(d.get("branch_id", -1)) == int(b_id):
                desc = d
                break
    except Exception:  # noqa: BLE001 — defensive
        desc = None
    if desc is None:
        raise RuntimeError(
            f"result.i({name!r}): branch_id {b_id} not found in "
            f"`builder.components()`. This usually means the binding "
            f"signature drifted — check `python/bindings.cpp::components`.")

    kind = desc.get("kind", "unknown")
    params = desc.get("params", {}) or {}
    from_id, to_id = desc.get("nodes", (None, None))

    def _need_nodes() -> "tuple[int, int]":
        if from_id is None or to_id is None:
            raise RuntimeError(
                f"result.i({name!r}): descriptor missing terminal "
                f"node IDs.")
        return int(from_id), int(to_id)

    def _v_branch() -> "_np.ndarray":
        f, to = _need_nodes()
        return _node_v(states, f) - _node_v(states, to)

    def _finalize(col):
        return col if t is None else col[t]

    # ----------------------- resistor --------------------------------
    if kind == "resistor":
        try:
            R_ohms = float(params["R_ohms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"result.i({name!r}): resistor params missing "
                f"`R_ohms` ({exc!r})") from None
        if R_ohms <= 0.0:
            raise ValueError(
                f"result.i({name!r}): non-positive R_ohms={R_ohms}")
        return _finalize(_v_branch() / R_ohms)

    # ----------------------- capacitor -------------------------------
    if kind == "capacitor":
        try:
            C_F = float(params["C_farads"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"result.i({name!r}): capacitor params missing "
                f"`C_farads` ({exc!r})") from None
        if C_F <= 0.0:
            raise ValueError(
                f"result.i({name!r}): non-positive C_farads={C_F}")
        v_C = _v_branch()
        times_arr = _ta(self)
        # Numerical derivative — uses second-order central differences
        # at interior samples, one-sided at the endpoints. The kernel's
        # trapezoidal scheme already produces a smooth v_C, so np.gradient
        # gives results consistent with i_C = C·dv/dt to within solver
        # tolerance (typically ≲ 1 % of peak |i_C|). For exact
        # representations, the future kernel-side C matrix expansion
        # would emit i_C directly during the solve.
        if times_arr.size < 2 or v_C.size < 2:
            return _finalize(_np.zeros_like(v_C))
        col = C_F * _np.gradient(v_C, times_arr)
        return _finalize(col)

    # ----------------------- current source --------------------------
    if kind == "current_source":
        try:
            I_A = float(params["I"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"result.i({name!r}): current_source params missing "
                f"`I` ({exc!r})") from None
        return _finalize(_np.full(states.shape[0], I_A))

    # ----------------------- diode -----------------------------------
    if kind == "diode":
        # PWL switched diode: closed when v_D > V_th. Use g_on while
        # forward-biased, g_off while reverse-biased. Mirrors the
        # convention from `pulsim.losses._conduction_stats(diode)`.
        try:
            g_on = float(params["g_on"])
            g_off = float(params["g_off"])
            V_th = float(params.get("V_th", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"result.i({name!r}): diode params missing "
                f"`g_on`/`g_off`/`V_th` ({exc!r})") from None
        v_D = _v_branch()
        closed = v_D > V_th
        G_arr = _np.where(closed, g_on, g_off)
        # The diode stamp models the on-state as a conductance to a
        # virtual V_th supply: i_D = G·(v_D − V_th) when closed,
        # i_D = G·v_D when open (g_off is leakage with no threshold).
        col = _np.where(closed,
                          G_arr * (v_D - V_th),
                          G_arr * v_D)
        return _finalize(col)

    # ----------------------- switch ----------------------------------
    if kind == "switch":
        try:
            g_on = float(params["g_on"])
            g_off = float(params["g_off"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"result.i({name!r}): switch params missing "
                f"`g_on`/`g_off` ({exc!r})") from None
        sf = getattr(self, "_switch_fn", None)
        if sf is None:
            raise NotImplementedError(
                f"result.i({name!r}): switch current reconstruction "
                f"requires the original `switch_fn` to be available on "
                f"the result. `pulsim.simulate(...)` auto-stashes it "
                f"on `result._switch_fn`; if you built this result "
                f"another way, set "
                f"`result._switch_fn = your_switch_fn` before calling "
                f"`.i()`.")
        try:
            seq_idx = int(builder.switch_index_of(name))
        except Exception as exc:  # noqa: BLE001 — usage error
            raise RuntimeError(
                f"result.i({name!r}): switch_index_of failed "
                f"({exc!r}); make sure {name!r} is a switch added "
                f"via `builder.add_switch`.") from None
        times_arr = _ta(self)
        v_sw = _v_branch()
        # Prefer the recorded mask; else re-evaluate switch_fn with a
        # voltage-consistency guard (robust to the closed-loop post-hoc
        # mask desync — see _result_views.resolve_switch_closed_trace).
        closed = _resolve_switch_closed(
            self, sf, times_arr, seq_idx, v_sw, name=name)
        G_arr = _np.where(closed, g_on, g_off)
        return _finalize(v_sw * G_arr)

    # ----------------------- not yet supported -----------------------
    raise NotImplementedError(
        f"branch {name!r} has kind={kind!r}; result.i() does not yet "
        f"reconstruct currents for this device family. Currently "
        f"supported: inductor, voltage_source (incl. pwm / sine / "
        f"pulse variants), resistor, capacitor, current_source, "
        f"diode, switch. For mosfet_level1 / igbt_level1 / "
        f"nonlinear_diode / vcvs / saturable_inductor, the per-step "
        f"nonlinear stamp evaluation lives in "
        f"`pulsim.losses.device_loss_summary` — use that for now. "
        f"Lifting these into `result.i()` is tracked as a follow-up.")


def _result_power(self, device_name: str) -> float:
    """Return the average dissipated power for ``device_name``,
    sourced from :func:`device_loss_summary`.

    Only resistors carry a meaningful ``P_avg`` today (inductors
    are ideal — no loss; switches/diodes aren't in the summary
    yet). For other device kinds the method raises
    :class:`NotImplementedError` with the same migration hint as
    :meth:`i`.
    """
    builder = getattr(self, "_builder", None)
    if builder is None:
        raise RuntimeError(
            "SimulationResult.power requires the result to carry a "
            "_builder reference. Use pulsim.simulate(...) so the "
            "wrapper is attached automatically."
        )
    # ``device_loss_summary`` returns a list[dict[str, Any]] keyed
    # by ``name``. Walk it once to find the match.
    summary = device_loss_summary(builder, self)
    by_name = {row.get("name"): row for row in summary if "name" in row}
    if device_name not in by_name:
        candidates = list(by_name.keys())
        import difflib as _difflib
        sugg = _difflib.get_close_matches(device_name, candidates, n=3, cutoff=0.6)
        raise NameNotFoundError(device_name, "branch", sugg)
    row = by_name[device_name]
    if "P_avg" in row:
        return float(row["P_avg"])
    raise NotImplementedError(
        f"device {device_name!r} ({row.get('kind', '?')}) doesn't "
        f"expose P_avg in device_loss_summary today — only "
        f"resistors do. Switches / diodes / MOSFETs need the "
        f"v_ds × i_d reconstruction via the device pool, which "
        f"isn't bound to Python yet."
    )


def _result_signal(self, name: str):
    """Return a user-recorded trace by name (e.g.
    ``result.signal("M1.omega")``).

    Sources, in resolution order:

    1. ``self._motor_traces`` — populated by motor observer bundles
       (PMSM / BLDC / DC motor / IM) when their ``attach_to_result``
       was called. :func:`pulsim.simulate` does this automatically
       when it detects a :class:`MotorObserverBundle` in
       ``step_observer`` or inside ``closed_loops``.
    2. Future-extensible: any other observer that registers traces
       under the same dict can be reached the same way.

    Returns
    -------
    numpy.ndarray
        The recorded samples. Use ``self.signal(f"{name}.t")`` if you
        also need the timestamp array (motor bundles publish their
        own ``<motor>.t`` alongside each motor for convenience —
        these typically equal ``self.times`` but the motor observer
        captures the value at observer-call time, which can differ
        from kernel sample time in an adaptive integrator).

    Raises
    ------
    NameNotFoundError
        When `name` isn't registered. The error carries fuzzy
        suggestions over the registered trace names.
    """
    traces = getattr(self, "_motor_traces", None) or {}
    if name in traces:
        return traces[name]
    candidates = list(traces.keys())
    import difflib as _difflib
    sugg = _difflib.get_close_matches(name, candidates, n=3, cutoff=0.5)
    raise NameNotFoundError(name, "signal", sugg)


def _result_signals(self) -> "list[str]":
    """Return the sorted list of registered signal names (e.g.
    ``['M1.T_em', 'M1.i_a', 'M1.i_b', 'M1.i_c', 'M1.i_d',
       'M1.i_q', 'M1.omega', 'M1.t', 'M1.theta']``).
    """
    return sorted((getattr(self, "_motor_traces", None) or {}).keys())


def _result_currents(self, *, skip_unsupported: bool = True
                        ) -> "dict[str, object]":
    """Return ``{branch_name: ndarray}`` with every supported branch's
    current trace — pulsim's nearest equivalent to PLECS' "all currents
    from the output equation" UX.

    Walks ``builder.components()`` and calls :meth:`i` on every branch
    whose kind has a reconstruction implementation (inductor,
    voltage_source family, resistor, capacitor, current_source, diode,
    switch). When ``skip_unsupported`` is True (default), kinds without
    a Python-side reconstruction
    (``mosfet_level1``/``igbt_level1``/``nonlinear_diode``/``vcvs``/
    ``saturable_inductor``) are silently omitted from the result —
    pass ``False`` to surface their ``NotImplementedError`` instead.

    Returns
    -------
    dict[str, numpy.ndarray]
        Branch name → per-sample current trace. Empty dict if the
        result wasn't produced via ``pulsim.simulate(...)`` (no builder
        reference attached).
    """
    builder = getattr(self, "_builder", None)
    out: "dict[str, object]" = {}
    if builder is None:
        return out
    try:
        comps = list(builder.components())
    except Exception:  # noqa: BLE001 — defensive
        return out
    for d in comps:
        n = d.get("name", "")
        if not n:
            continue
        try:
            out[n] = self.i(n)
        except (NotImplementedError, RuntimeError, ValueError):
            if not skip_unsupported:
                raise
            continue
    return out


# Monkey-patch the methods onto the C++-bound SimulationResult class.
# We can't subclass it cleanly because run_transient returns the
# concrete C++ type, but the type itself accepts attribute injection
# (py::dynamic_attr() on the binding).
SimulationResult.v = _result_v          # type: ignore[attr-defined]
# v2.0 Phase 3 item 5 — one result surface for both engines. The
# dsed wrapper's `.states` is the reconstructed full-MNA trajectory
# (same row layout), so the same accessors bind verbatim. On a
# result without full states (a non-builder path) they fail with
# the shape mismatch naming the limitation rather than silently
# indexing reduced states as if they were node voltages.
from ._dsed_dispatch import (  # noqa: E402 — must follow the
    _PEDSimulationResult as _PEDRes,  # accessor definitions above
)
_PEDRes.v = _result_v                   # type: ignore[attr-defined]
SimulationResult.i = _result_i
_PEDRes.i = _result_i       # type: ignore[attr-defined]          # type: ignore[attr-defined]
SimulationResult.power = _result_power  # type: ignore[attr-defined]
SimulationResult.signal = _result_signal      # type: ignore[attr-defined]
SimulationResult.signals = _result_signals    # type: ignore[attr-defined]
SimulationResult.currents = _result_currents  # type: ignore[attr-defined]


def _result_plot(self, *signals, save=None, show=None, **kwargs):
    """Plot one or more signals from this simulation result.

    Each ``signal`` can be:

    * ``"vout"``, ``"node_name"`` — a node voltage. Resolved via
      :meth:`~SimulationResult.v`.
    * ``"L1"``, ``"R_load"``, ``"M1"``, ... — a branch current.
      Resolved via :meth:`~SimulationResult.i`.

    The result must carry a back-reference to its builder (which
    :func:`simulate` sets automatically). Standalone
    ``run_transient`` consumers without the builder attached must
    use :func:`pulsim.scope` directly.

    Parameters
    ----------
    signals
        Variable-length list of signal names. Each must resolve as
        either a node voltage or a branch current — unknown names
        raise :class:`NameNotFoundError` with the same fuzzy
        "Did you mean ...?" suggestion path used by
        :meth:`SimulationResult.v` / :meth:`.i`.
    save
        Optional path; passed through to :func:`pulsim.scope`.
    show
        If ``True`` (default when called interactively), display
        the figure. If ``False``, just return it.
    **kwargs
        Forwarded into :func:`pulsim.scope` (titles, axes options,
        etc.).

    Returns
    -------
    matplotlib.figure.Figure
        The figure produced by :func:`pulsim.scope`.

    Raises
    ------
    AttributeError
        If the result has no `_builder` back-reference. Use
        :func:`pulsim.scope(builder, result, signals=[...])` for
        that case.
    NameNotFoundError
        If any signal can't be resolved as a node OR a branch.
    """
    builder = getattr(self, "_builder", None)
    if builder is None:
        raise AttributeError(
            "SimulationResult.plot() requires the result to carry a "
            "back-reference to its builder. Use "
            "`pulsim.scope(builder, result, signals=[...])` directly "
            "when the result came from `run_transient` without going "
            "through `simulate()`.")
    from .plot import scope as _scope
    return _scope(builder, self, signals=list(signals),
                    save=save, show=show, **kwargs)


SimulationResult.plot = _result_plot  # type: ignore[attr-defined]


# =============================================================================
# SolverOptions — bundle advanced kernel knobs (v1.5 ergonomics)
# =============================================================================

@dataclass
class SolverOptions:
    """Advanced kernel options for :func:`simulate`.

    Most users never need to touch any of these — the defaults are
    tuned for typical SMPS workloads. They cover three families:

    * **Newton / nonlinear refresh** — when the circuit has smooth
      diodes, MOSFET / IGBT level-1 models, or saturable inductors.
    * **Event detection** — sub-step commutation timing accuracy.
    * **Adaptive RK schema** (Phase 2.4) — `integrator`, `rtol`,
      `atol`, `dt_init`. Today only `integrator="kernel"` is wired
      to actually run; `"dopri5"` / `"radau"` raise
      ``NotImplementedError`` pointing at the v1.6 cache refactor.

    Two equivalent ways to use:

    1. **Bundle**:
       ::

           opts = p.SolverOptions(max_newton_iterations=200,
                                     tol_newton_res=1e-11)
           res = p.simulate(b, t_end=..., dt=..., solver=opts)

    2. **Flat kwargs** (backward compatible):
       ::

           res = p.simulate(b, t_end=..., dt=...,
                              max_newton_iterations=200,
                              tol_newton_res=1e-11)

    When both are given, the flat kwargs override the corresponding
    fields on `solver`.
    """
    # Newton / nonlinear refresh ------------------------------------
    enable_nonlinear_refresh: Optional[bool] = None
    max_newton_iterations: int = 0
    tol_newton_dx: Optional[float] = None
    tol_newton_res: Optional[float] = None
    enable_newton_line_search: Optional[bool] = None
    enable_newton_lm: Optional[bool] = None

    # Event detection -----------------------------------------------
    # None = engine default (16). 0 is a MEANINGFUL value — it
    # disables event iteration entirely (the pre-V2.1 behaviour) —
    # so it cannot double as the "not set" sentinel.
    max_event_iterations: Optional[int] = None
    enable_substep_state_correction: Optional[bool] = None

    # Topology preflight ---------------------------------------------
    # None = engine default (True). False must mean "the user
    # explicitly opted out", which a plain `bool = True` could not
    # express — see the sentinel rules above.
    auto_regularize: Optional[bool] = None

    # Output ---------------------------------------------------------
    # Record every m-th step (1 = every step). The solver still
    # integrates at `dt`; only what is stored changes, and the
    # recorded grid stays uniform at m*dt so FFT / harmonic analysis
    # remains valid.
    store_every: int = 1

    # Inductor regularisation (rare — kept for completeness) --------
    # Optional because 0.0 is the documented OFF switch for both
    # guards, not merely "engine default" — using it as the "not
    # set" sentinel silently overrode an explicit flat request to
    # DISABLE them (same contract violation as the store_every bug).
    inductor_freeze_di_max: Optional[float] = None
    inductor_abs_clamp: Optional[float] = None

    # Adaptive RK schema (Phase 2.4 — execution deferred to v1.6) ---
    #
    # All Optional with a None default: a dataclass cannot otherwise
    # tell "the user set this" from "this is the class default", and
    # these four carry PWL-flavoured values. Merging them
    # unconditionally pushed dt_init=0.0 and integrator="kernel"
    # into the DSED path, so ANY `simulate(engine='dsed',
    # solver=...)` died on "dt_init must be positive, got 0.0" for a
    # parameter the caller never touched — and every PWL run with a
    # bundle emitted a bogus "these kwargs are ignored" warning.
    integrator: Optional[str] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None
    dt_init: Optional[float] = None


def simulate(
    builder: CircuitBuilder,
    t_end: float,
    dt: Optional[float] = None,
    *,
    # --- engine selector ---
    engine: str = "auto",
    # --- DSED variable-step kwargs (only used when engine='dsed') ---
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    dt_init: Optional[float] = None,
    integrator: Optional[str] = None,
    stiffness_threshold: Optional[float] = None,
    h_bdf2: Optional[float] = None,
    # --- common kwargs ---
    t_start: float = 0.0,
    switch_fn: Optional[Callable[[float], SwitchStateMask]] = None,
    b_extra_fn: Optional[Callable[[float], "list[float]"]] = None,
    step_observer: Optional[Callable[[float, "object"], None]] = None,
    start_from_dc_op: bool = False,
    enable_nonlinear_refresh: Optional[bool] = None,
    max_newton_iterations: int = 0,
    max_event_iterations: Optional[int] = None,
    strict_event_iterations: bool = False,
    max_dt_halvings: Optional[int] = None,
    voltage_sanity_factor: Optional[float] = None,
    store_every: Optional[int] = None,
    auto_regularize: Optional[bool] = None,
    tol_newton_dx: Optional[float] = None,
    tol_newton_res: Optional[float] = None,
    enable_newton_line_search: Optional[bool] = None,
    enable_newton_lm: Optional[bool] = None,
    enable_substep_state_correction: Optional[bool] = None,
    inductor_freeze_di_max: Optional[float] = None,
    inductor_abs_clamp: Optional[float] = None,
    progress: "bool | int | str" = False,
    initial_state=None,
    should_continue=None,
    closed_loops=None,
    live_stream=None,
    mmc_arms=None,
    controller_period: Optional[float] = None,
    # --- SolverOptions bundle (v1.5 Round 2 ergonomics polish) ---
    solver: Optional["SolverOptions"] = None,
) -> SimulationResult:
    """Build the cache and run a transient simulation.

    **The one rule** (``engine='auto'``, the v2.0 default)::

        dt given  ->  fixed step, exactly as before
        no dt     ->  the engine picks, and takes the
                      variable-step path whenever it can

    An explicit ``dt`` is a REQUEST for a fixed step, not a hint —
    reading it as the variable engine's step ceiling would silently
    change the answer of every script that already passes one. So
    the modern engine is opt-in by OMISSION, and
    ``result.engine_used`` always says which one ran (with
    ``result.engine_route_reason`` when ``auto`` did not pick the
    variable-step one).

    The engines:

    * ``engine='auto'`` (default) — routes. Variable-step TR-BDF2
      when the circuit and kwargs qualify, fixed-step otherwise.
      Raises, naming the blocker and asking for a ``dt``, when
      neither can serve.

    * ``engine='trbdf2'`` — the variable-step engine by name:
      L-stable second-order composite (TR + BDF2), LTE-controlled
      step, gate edges and controller ticks landed exactly, diode
      crossings localized between steps. ``dt`` (if given) is the
      step CEILING. REFUSES rather than routing, which is what a
      caller who asked for it by name wants to know. Scope: linear
      PWL circuits (nonlinear devices need a Newton loop per stage,
      not wired yet).

    * ``engine='pwl'`` — fixed-step trapezoidal companion + PWL
      state-space cache. **Requires a positive `dt`.** Bit-exact
      reproducibility; matches v1.4.0 output to machine precision.
      Everything the other engines refuse runs here.

    * ``engine='dsed'`` — Path-Based Event-Driven scheduler with
      automatic RK45 / BDF2 dispatch and exact LTI stepping.
      ``dt`` (if supplied) acts as the maximum-step cap ``dt_max``.

    Decision tree, short version:

    * **"Não sei qual escolher"** → pass nothing. That is what
      ``auto`` is for.
    * **"Preciso do grid fixo / reprodutibilidade bit-exata"** →
      pass ``dt``.

    Examples
    --------

    Fixed-step (v1.4.0 behaviour)::

        result = pulsim.simulate(b, t_end=5e-3, dt=1e-7)

    Variable-step with default tolerances::

        result = pulsim.simulate(b, t_end=5e-3, engine='dsed')

    Variable-step with tighter accuracy::

        result = pulsim.simulate(
            b, t_end=5e-3, engine='dsed',
            rtol=1e-8, atol=1e-11,
        )

    Variable-step forcing RK45-only (e.g. debug / repeatability)::

        result = pulsim.simulate(
            b, t_end=5e-3, engine='dsed',
            integrator='rk45',
        )

    Parameters
    ----------
    builder
        A populated :class:`CircuitBuilder`.
    t_end
        End time, in seconds.
    dt
        Fixed time step (engine='pwl', required) OR upper-bound
        ``dt_max`` (engine='dsed', optional — defaults to 10 µs).
    engine
        ``'pwl'`` (default, fixed-step) or ``'dsed'`` (variable-step).
    rtol, atol
        Relative / absolute tolerance for the DSED PI controller.
        Ignored if ``engine='pwl'`` (warning emitted). Defaults:
        ``rtol=1e-6, atol=1e-9``.
    dt_init
        Initial step for the DSED PI controller (default 1e-9).
    integrator
        DSED override: ``'auto'`` (default), ``'rk45'``, or ``'bdf2'``.
        ``'auto'`` picks per mode via stiffness detector. Force one
        for debug / repeatability.
    stiffness_threshold
        DSED override: ``|λ_max|·h`` ratio above which auto-dispatch
        picks BDF2 (default 10.0).
    h_bdf2
        Fixed step for BDF2 segments under ``integrator='bdf2'`` or
        ``'auto'`` (default 1e-6).
    t_start, default 0.0
        Start time, in seconds.
    switch_fn
        Callable ``t -> SwitchStateMask`` controlling the
        switch state at each sample.  Defaults to all-closed.
    b_extra_fn
        PWL-only: Callable ``t -> list[float]`` adding to the constant
        residual at each step.  Defaults to None.
    start_from_dc_op
        PWL-only: If ``True``, seed the initial state from
        :func:`compute_dc_op` instead of zero.
    enable_nonlinear_refresh
        PWL-only: Force-enable/-disable the Newton refresh pass.
        ``None`` (default) means auto-detect.
    max_newton_iterations, max_event_iterations
        PWL-only: forwarded to :class:`SimulationOptions`.

    Returns
    -------
    SimulationResult
        The full per-sample state-vector history.

    Raises
    ------
    ValueError
        On unknown ``engine``; on ``engine='pwl'`` without ``dt``;
        on invalid DSED options (negative tolerance, unknown
        ``integrator``).
    UserWarning
        When DSED-only kwargs are passed with ``engine='pwl'`` (they
        are silently ignored — the warning is informational so
        scripts that copy-paste kwargs don't break suddenly).
    NotImplementedError
        On ``engine='dsed'`` from a :class:`CircuitBuilder` when
        the per-mask (A, b) extractor rejects the circuit (e.g.
        nonlinear device the extractor doesn't yet support). For
        PED today with a user-supplied LTI system, use
        :func:`pulsim.dsed.run_user_lti` directly.
    """
    # ---- Merge `solver` bundle (v1.5 Round 2 ergonomics). ----
    # For every advanced kernel knob, the flat kwarg wins when the
    # user explicitly passed something non-default; otherwise we pull
    # from `solver` if supplied. Purely additive — every existing
    # call site that uses the flat kwargs continues to behave
    # identically. This runs BEFORE the engine dispatch so the merged
    # rtol/atol/dt_init/integrator values also flow into the DSED
    # path when the user authors via `solver=`.
    if solver is not None:
        if enable_nonlinear_refresh is None:
            enable_nonlinear_refresh = solver.enable_nonlinear_refresh
        if max_newton_iterations == 0:
            max_newton_iterations = solver.max_newton_iterations
        if max_event_iterations is None:
            max_event_iterations = solver.max_event_iterations
        # `1` is a MEANINGFUL value here ("record every step"), not a
        # "not passed" sentinel like the 0s above — so the flat kwarg
        # only defers to the bundle when it was genuinely omitted.
        # Using 1 as the sentinel silently discarded an explicit
        # store_every=1 in favour of a bundle's decimation.
        if store_every is None:
            store_every = solver.store_every
        if auto_regularize is None:
            auto_regularize = solver.auto_regularize
        if tol_newton_dx is None:
            tol_newton_dx = solver.tol_newton_dx
        if tol_newton_res is None:
            tol_newton_res = solver.tol_newton_res
        if enable_newton_line_search is None:
            enable_newton_line_search = solver.enable_newton_line_search
        if enable_newton_lm is None:
            enable_newton_lm = solver.enable_newton_lm
        if enable_substep_state_correction is None:
            enable_substep_state_correction = (
                solver.enable_substep_state_correction)
        if inductor_freeze_di_max is None:
            inductor_freeze_di_max = solver.inductor_freeze_di_max
        if inductor_abs_clamp is None:
            inductor_abs_clamp = solver.inductor_abs_clamp
        if integrator is None:
            integrator = solver.integrator
        if rtol is None:
            rtol = solver.rtol
        if atol is None:
            atol = solver.atol
        if dt_init is None:
            dt_init = solver.dt_init

    # ---- Validate first — fail fast on user mistakes (DSED side). ----
    # `_validate_engine_kwargs` validates engine + DSED-specific kwargs
    # (rtol/atol/dt_init/integrator/stiffness_threshold/h_bdf2). For
    # engine='pwl' it just warns on DSED-only kwarg leakage.
    from . import _dsed_dispatch as _dsed
    _dsed._validate_engine_kwargs(
        engine=engine,
        dt=dt,
        rtol=rtol,
        atol=atol,
        dt_init=dt_init,
        integrator=integrator,
        stiffness_threshold=stiffness_threshold,
        h_bdf2=h_bdf2,
    )

    # ---- Topology preflight (v2.0 Phase 2) --------------------------
    #
    # Placed AFTER kwarg validation and BEFORE the engine dispatch,
    # both deliberately. After validation, because this MUTATES the
    # caller's builder (it appends reference ties) and a call that
    # dies on a typo'd kwarg must not leave the circuit changed.
    # Before the dispatch, because mutating the builder is what lets
    # BOTH engines inherit the fix — placing it later would have made
    # auto_regularize a PWL-only feature the DSED path ignored.
    #
    # `None` means "not set"; resolve it here rather than relying on
    # an `is not False` identity test, which would silently treat
    # numpy.False_ or 0 as opt-IN.
    if auto_regularize is None:
        auto_regularize = True
    _preflight_report = None
    if auto_regularize:
        _preflight_report = builder.run_preflight(
            PreflightOptions(auto_regularize=True))
        if not _preflight_report.empty():
            import warnings
            warnings.warn(
                "simulate(): " + _preflight_report.summary() +
                "\n  These ties give the MNA a voltage reference "
                "without loading the circuit (1 GΩ draws nanoamps). "
                "Inspect them via result._preflight, or pass "
                "auto_regularize=False to get the original singular"
                "-matrix error instead. Note the ties persist on the "
                "builder, so re-running the SAME builder with "
                "auto_regularize=False will not restore the error — "
                "rebuild the circuit for that.",
                stacklevel=2)

    # ---- Engine dispatch: DSED takes the early-return path. ----
    # For engine='dsed', `integrator` ∈ {'auto', 'rk45', 'bdf2', None}.
    # The DSED dispatcher owns its own validation; main's Phase 2.4
    # adaptive-RK selector below is PWL-specific and must NOT apply
    # to DSED integrator names.
    if engine == "dsed":
        if getattr(builder, "_c_blocks", None):
            import warnings
            warnings.warn(
                "simulate(engine='dsed'): C blocks added via add_c_block() "
                "are not yet wired into the DSED engine (their outputs use "
                "b_extra, which DSED does not consume). Use engine='pwl' to "
                "run C blocks.")
        # Hard-fail on kwargs the DSED path cannot honor. Silently
        # dropping these is the worst failure mode a simulator can
        # have: a closed-loop circuit would run OPEN-loop and return
        # plausible-looking (wrong) waveforms. Mirror of the C-block
        # warning above, but these change the *physics* of the run,
        # so they raise instead of warn.
        _dsed_unsupported = {
            "step_observer": step_observer is not None,
            "closed_loops": bool(closed_loops),
            "should_continue": should_continue is not None,
            "live_stream": live_stream is not None,
            "start_from_dc_op": bool(start_from_dc_op),
            # review finding PY-4: DSED has no diode event iteration,
            # so silently accepting this flag would be the exact
            # pattern this block exists to prevent.
            "strict_event_iterations": bool(strict_event_iterations),
            # A fixed-step concept: the DSED engine already adapts its
            # own step, so accepting this would promise a mechanism
            # that does not exist there.
            "max_dt_halvings": max_dt_halvings is not None,
            # v2.0 Phase 1: output decimation is a fixed-grid
            # concept; the DSED engine emits variable-step samples,
            # so honouring `store_every` there would mean something
            # different (and unstated). Fail loudly rather than
            # returning a full-rate trace the caller did not ask for.
            "store_every": store_every is not None
                            and int(store_every) != 1,
            # v2.0 Phase 3 "obra №2": the Thevenin arm's exact
            # coupling is driven by the pwl engine's per-step
            # observer/b_extra pair and the parametric refactor of
            # its cache — none of which exist in the DSED path.
            "mmc_arms": bool(mmc_arms),
        }
        _offending = [k for k, hit in _dsed_unsupported.items() if hit]
        if _offending:
            raise ValueError(
                f"simulate(engine='dsed'): {', '.join(_offending)} "
                "is/are not supported by the DSED engine yet — the "
                "run would silently ignore them (e.g. a closed-loop "
                "converter would simulate OPEN-loop). Use "
                "engine='pwl' for these features, or drop the "
                "kwarg(s) if the open-loop behaviour is intended. "
                "Observer/closed-loop support inside DSED is tracked "
                "for v2.0 (event-synchronised controller cadence)."
            )
        # The DSED branch returns here, ~400 lines before the PWL
        # tail that attaches `_builder` / `_preflight`. Attach the
        # preflight report on the way out, so `result._preflight` —
        # which the warning tells every user to read — is not a
        # PWL-only attribute.
        _dsed_res = _dsed.run_dsed_from_builder(
            builder=builder,
            t_end=t_end,
            dt=dt,
            rtol=rtol,
            atol=atol,
            dt_init=dt_init,
            integrator=integrator,
            stiffness_threshold=stiffness_threshold,
            h_bdf2=h_bdf2,
            t_start=t_start,
            switch_fn=switch_fn,
            b_extra_fn=b_extra_fn,
            initial_state=initial_state,
            progress=progress,
        )
        try:
            _dsed_res._preflight = _preflight_report
            _dsed_res.engine_used = "dsed"
            _dsed_res.engine_route_reason = None
            # v2.0 Phase 3 item 5 — the unified result: with the
            # full-MNA trajectory reconstructed, the same name-based
            # accessors the pwl engine gets now mean the same thing
            # here. `_builder` is what they resolve names against.
            _dsed_res._builder = builder
        except AttributeError:  # pragma: no cover
            pass
        return _dsed_res

    _engine_asked = engine
    _route_reasons: "list[str]" = []
    if engine in ("auto", "trbdf2"):
        _route_reasons = _trbdf2_blockers(
            builder, dt=dt, step_observer=step_observer,
            closed_loops=closed_loops,
            controller_period=controller_period,
            live_stream=live_stream, progress=progress,
            start_from_dc_op=start_from_dc_op,
            strict_event_iterations=strict_event_iterations,
            max_dt_halvings=max_dt_halvings,
            store_every=store_every, mmc_arms=mmc_arms,
            enable_substep_state_correction=(
                enable_substep_state_correction),
            inductor_freeze_di_max=inductor_freeze_di_max,
            inductor_abs_clamp=inductor_abs_clamp,
            switch_fn=switch_fn)
        if engine == "auto":
            # ROUTE, don't refuse: 'auto' means "pick the engine",
            # and a script written against the fixed-step API must
            # keep working when it becomes the default.
            #
            # AN EXPLICIT dt IS A REQUEST FOR A FIXED STEP. Reading
            # it as the variable engine's step CEILING instead
            # would silently change the answer of every script that
            # already passes one (measured: a closed-loop buck
            # moved 4.985 -> 4.968 V). So the rule is simply:
            #   dt given      -> fixed step, as before
            #   no dt         -> the engine picks, and uses the
            #                    variable-step path when it can
            # which makes the new engine opt-in by OMISSION.
            if dt is not None and dt > 0:
                _route_reasons = _route_reasons or [
                    "an explicit dt was given, which requests a "
                    "fixed step"]
            if _route_reasons:
                if dt is None or dt <= 0:
                    raise ValueError(
                        "simulate(): this circuit cannot use the "
                        "variable-step engine — "
                        + "; ".join(_route_reasons)
                        + ". The fixed-step engine can run it, but "
                        "needs a step: pass dt=<seconds> (or "
                        "engine='dsed' for the event-driven "
                        "path).")
                engine = "pwl"
                # Validation ran against the engine the caller
                # ASKED for, so kwargs the fixed engine ignores
                # (rtol/atol/dt_init/integrator) passed silently.
                # Re-check now that we know where the run lands —
                # a dropped kwarg the user believed in is exactly
                # the failure mode the leak warning exists for.
                _dsed._validate_engine_kwargs(
                    engine="pwl", dt=dt, rtol=rtol, atol=atol,
                    dt_init=dt_init, integrator=integrator,
                    stiffness_threshold=stiffness_threshold,
                    h_bdf2=h_bdf2)
            else:
                engine = "trbdf2"
        elif _route_reasons:
            raise ValueError(
                "simulate(engine='trbdf2'): "
                + "; ".join(_route_reasons)
                + ". Use engine='auto' to route to the fixed-step "
                "engine automatically, or engine='pwl' explicitly.")

    if engine == "trbdf2":
        # ---- v2.0 Phase 3: variable-step TR-BDF2 on the sparse
        #      MNA kernel. L-stable, LTE-controlled — no dt to
        #      guess; `dt` (optional) is the step CEILING h_max,
        #      which is also the gate-edge sampling resolution
        #      (a gate pulse narrower than h_max can be missed,
        #      exactly like dsed's dt_max). ----
        # Every unsupported combination was already caught by
        # _trbdf2_blockers above (which is also what routes
        # engine='auto' away from here).
        if integrator is not None:
            import warnings
            warnings.warn(
                f"simulate(engine={_engine_asked!r}): integrator= "
                "selects a dsed integrator and is ignored by the "
                "variable-step engine, which is TR-BDF2 by "
                "construction.",
                stacklevel=2)
        # ---- controllers on an EXACT cadence ----
        # A digital controller samples at k·T_ctrl. The fixed
        # engine only THROTTLES an every-step observer on that
        # period, so the tick lands on whichever step first crosses
        # the boundary and the error accumulates (measured: 198
        # ticks instead of 200 over 20 ms at 10 kHz). Here the tick
        # instants are SCHEDULED like gate edges — landed exactly,
        # observer fired there, and the duty it sets is what the
        # coming step's mask is built from.
        _cl_list = list(closed_loops) if closed_loops else []
        _auto_obs_list = []
        _auto_periods = []
        for _cl in _cl_list:
            # `.tick` is the loop's UNTHROTTLED update. Calling the
            # throttled `step_observer` on an exactly-scheduled
            # cadence loses ticks to floating point (measured 129
            # of 200), which quietly changes the loop gain.
            _auto_obs_list.append(
                getattr(_cl.step_observer, "tick", _cl.step_observer))
            _p = float(getattr(_cl, "period", 0.0) or 0.0)
            if _p > 0.0:
                _auto_periods.append(_p)
        if step_observer is not None:
            _auto_obs_list.append(
                getattr(step_observer, "tick", step_observer))
            _p_obs = controller_period
            if _p_obs is None:
                _p_obs = getattr(step_observer, "period", None)
            if _p_obs is not None:
                _auto_periods.append(float(_p_obs))
        if _auto_obs_list and not _auto_periods:
            raise ValueError(
                "simulate(engine='auto'): a step_observer needs a "
                "cadence. This engine has no fixed grid to ride, "
                "so pass controller_period=<T_ctrl seconds> (the "
                "rate your controller samples at) and the tick "
                "instants are scheduled exactly. A ClosedLoop from "
                "bind_pi_to_switch carries its own period.")
        _auto_period = min(_auto_periods) if _auto_periods else 0.0
        if len(set(_auto_periods)) > 1:
            # Different rates: tick every observer on the fastest
            # schedule and let each one's own throttle decide (the
            # binder's observers already throttle on their period).
            pass
        if len(_auto_obs_list) == 1:
            _auto_observer = _auto_obs_list[0]
        elif _auto_obs_list:
            def _auto_observer(t, x, _obs=tuple(_auto_obs_list)):
                for _o in _obs:
                    _o(t, x)
        else:
            _auto_observer = None
        if _cl_list:
            _cl_switch_fns = [c.switch_fn for c in _cl_list]
            if switch_fn is None and len(_cl_switch_fns) == 1:
                switch_fn = _cl_switch_fns[0]
            elif _cl_switch_fns:
                _n_sw_cl = builder.graph.num_switches
                _user_sf = switch_fn

                def switch_fn(t, _fns=tuple(_cl_switch_fns),
                               _u=_user_sf, _n=_n_sw_cl):
                    m = SwitchStateMask(_n)
                    if _u is not None:
                        m = _u(t)
                    for _f in _fns:
                        m = m | _f(t)
                    return m

        _auto_nl = (enable_nonlinear_refresh
                     if enable_nonlinear_refresh is not None
                     else builder.pool.has_nonlinear_devices())
        _span = float(t_end) - float(t_start)
        # 0 = let the kernel pick: it knows the circuit's fastest
        # periodic source and caps the ceiling at 20 steps per
        # period / a third of the narrowest pulse. Computing
        # span/1000 here instead would step straight over a narrow
        # pulse train on a long run (measured: a peak detector read
        # 0 V instead of 9.87).
        _h_max = float(dt) if dt is not None else 0.0
        _h_init = float(dt_init) if dt_init else 0.0
        _rtol = float(rtol) if rtol is not None else 1e-5
        _atol = float(atol) if atol is not None else 1e-8
        _n_sw = builder.graph.num_switches
        if switch_fn is None:
            # No driver: all-OPEN here (the v2.0 semantics; the
            # fixed engine's all-CLOSED legacy default is what the
            # router sends this case to instead — flipping THAT is
            # its own breaking change, not this one).
            switch_fn = (lambda _n=_n_sw:
                          (lambda t: SwitchStateMask(_n)))()
        cache = PwlStateSpaceCache(builder.graph, builder.pool)
        # Lazy build only needs a positive dt to mark the cache
        # dynamic; the stepper solves at its own per-step dt via
        # solve_at, never at this value.
        cache.build_lazy(_h_max if _h_max > 0.0 else _span / 1000.0)
        try:
            res, _stats = run_transient_trbdf2(
                cache, builder.graph, builder.pool,
                t_start=float(t_start), t_end=float(t_end),
                rtol=_rtol, atol=_atol,
                h_init=_h_init, h_max=_h_max,
                switch_fn=switch_fn,
                b_extra_fn=b_extra_fn,
                initial_state=initial_state,
                max_event_iterations=(
                    0 if max_event_iterations is None
                    else int(max_event_iterations)),
                should_continue=should_continue,
                observer_period=_auto_period,
                step_observer=_auto_observer,
                enable_nonlinear_refresh=_auto_nl,
                max_newton_iterations=int(max_newton_iterations or 0),
                tol_newton_dx=float(tol_newton_dx or 0.0),
                tol_newton_res=float(tol_newton_res or 0.0),
                enable_newton_line_search=bool(
                    enable_newton_line_search),
                enable_newton_lm=bool(enable_newton_lm),
            )
        except SimulationAborted as _aborted:
            try:
                _partial = _aborted.partial
                _partial._builder = builder
                _partial._preflight = _preflight_report
                _partial._switch_fn = switch_fn
                # The wreckage gets the same handles the survivors
                # get — including an empty-but-present stats dict,
                # so post-mortem code can read it unconditionally.
                _partial._trbdf2_stats = {}
            except AttributeError:  # pragma: no cover
                pass
            raise
        res._builder = builder
        res._preflight = _preflight_report
        res._switch_fn = switch_fn
        res._trbdf2_stats = _stats
        res.engine_used = "trbdf2"
        res.engine_route_reason = None
        # The implausible-voltage detector is engine-independent
        # (it reads the trace, not the stepper) and this engine's
        # new all-OPEN default mask makes the inductor-open case it
        # exists for MORE likely, not less. Run it here too — the
        # pwl tail that normally does it is 500 lines below the
        # auto return.
        _auto_check_voltage_sanity(res, builder,
                                    voltage_sanity_factor)
        if _stats.get("n_forced_accepts", 0) > 0:
            import warnings
            warnings.warn(
                f"simulate(engine='auto'): "
                f"{_stats['n_forced_accepts']} step(s) were "
                "accepted with the error estimate above tolerance "
                "because h had already reached its floor (usually "
                "the sliver between two events). The trace is "
                "still finite, but those steps carry more error "
                "than rtol promises — inspect "
                "result._trbdf2_stats and the event times if the "
                "waveform matters there.",
                stacklevel=2)
        # Threshold 3: a single boundary graze at startup is normal
        # (measured 1 on the hysteresis-banded flyback); a sliding
        # model produces them every few cycles (5 per 5 ms on the
        # V_th=0 buck, hundreds on tightened tolerances).
        if _stats.get("n_chatter_breaks", 0) > 3:
            import warnings
            warnings.warn(
                f"simulate(engine='auto'): a diode rode its "
                f"conduction boundary on "
                f"{_stats['n_chatter_breaks']} step(s) — the "
                "signature of an un-hysteresed ideal diode "
                "(V_th=0) in DCM, which is a SLIDING-MODE model: "
                "the reference engine chatters ~600 flips per "
                "period on it. This engine AVERAGES the slide; at "
                "the default rtol the average is correct (checked "
                "at 0.1 mV on the flyback), but tightening rtol "
                "partially resolves the slide and can BIAS the "
                "mean by ~0.5%. Give the diode its physical V_th "
                "(that IS the hysteresis band) for "
                "tolerance-proportional convergence, or use "
                "engine='pwl'.",
                stacklevel=2)
        return res

    # ---- engine='pwl' from here on ----
    # mypy/pyright: narrow Optional[float] → float.
    assert dt is not None and dt > 0

    # T2.2: snapshot the user's original observer/closed_loops args
    # so the post-run motor-trace auto-attach can walk them. The
    # local names get rebound below by the progress / closed_loops
    # / compose-observer wrappers.
    _step_observer_user = step_observer
    _b_extra_fn_user = b_extra_fn
    _closed_loops_user = closed_loops
    _switch_fn_user = switch_fn

    # Fall back to documented PWL defaults for any remaining None values
    # introduced by the `solver=` plumbing (these used to be flat
    # defaults at the signature; the DSED engine has its own defaults
    # owned by the dispatcher above).
    if integrator is None:
        integrator = "kernel"
    if rtol is None:
        rtol = 1.0e-5
    if atol is None:
        atol = 1.0e-8
    if dt_init is None:
        dt_init = 0.0

    # Phase 2.4 — adaptive RK selector (schema v1.5, wiring v1.6).
    # Today only the kernel trap path is wired into run_transient.
    # Reject other choices with a clear pointer at the deferred
    # cache refactor so users get actionable feedback instead of a
    # silent default-back to "kernel".
    if integrator not in ("kernel", "default", None):
        if integrator not in ("dopri5", "radau"):
            raise ValueError(
                f"simulate(integrator={integrator!r}): unknown "
                "integrator for engine='pwl'. Supported names: "
                "'kernel' (default), 'dopri5', 'radau'. For DSED's "
                "RK45/BDF2/auto integrators, use engine='dsed'."
            )
        raise NotImplementedError(
            f"simulate(integrator={integrator!r}) is reserved for "
            "the v1.6 cache refactor — `PwlStateSpaceCache` would "
            "need to expose continuous-time (G, M, b) and the "
            "Python adaptive RK integrators would need DAE Index-1 "
            "support (M·dx/dt = g, with M structurally singular "
            "for augmented MNA). Track this under the "
            "'add-adaptive-runge-kutta-solvers' OpenSpec proposal. "
            "For Phase 2.4, the integrator name and its tolerances "
            "round-trip through `SimulationOptions` and YAML so "
            "your config stays forward-compatible — drop "
            "`integrator='kernel'` (or omit) to run today. "
            "Note: engine='dsed' offers RK45/BDF2/auto today via the "
            "Path-Based Event-Driven scheduler."
        )
    _ = rtol, atol, dt_init  # Reserved for the v1.6 RK path; recorded only.

    if closed_loops:
        # Compose closed_loops with any user-supplied switch_fn /
        # step_observer. Pre-v1.6.5 this raised ValueError because
        # the loop owns its own switch indices and observer state;
        # but real drives need this composition: e.g. a closed-loop
        # PFC boost stage (owns the boost MOSFET via ClosedLoop) +
        # an openly-switched 3φ VSI (owns its own switches via a
        # separate switch_fn) + a PMSM observer (b_extra_fn /
        # step_observer). Each owner addresses a disjoint set of
        # switch indices, so the merged switch_fn is the bitwise OR
        # of every contributor's mask — exactly what
        # `make_combined_switch_fn` already does. Step observers
        # compose by running each callback in registration order:
        # closed-loop observers first (they update PI state /
        # closed-loop bookkeeping), then the user's step_observer
        # (e.g. PMSM mechanical state, custom probes).
        loops = list(closed_loops)
        n_sw_compose = builder.graph.num_switches
        per_switch_fns = [loop.switch_fn for loop in loops]
        if switch_fn is not None:
            per_switch_fns.append(switch_fn)
        switch_fn = make_combined_switch_fn(n_sw_compose, per_switch_fns)

        per_observers: list[Callable[[float, object], None]] = [
            loop.step_observer for loop in loops]
        if step_observer is not None:
            per_observers.append(step_observer)

        def _composed_observer(t: float, x) -> None:
            for obs in per_observers:
                obs(t, x)

        # T2.2: stash the inner observers so the post-run
        # _auto_attach_motor_traces can walk into any MotorObserverBundle
        # that the user passed via step_observer= (now wrapped here).
        _composed_observer._inner_observers = per_observers  # type: ignore[attr-defined]

        step_observer = _composed_observer

    # Custom-code blocks (C blocks) registered via add_c_block(): compose
    # each block's throttled observer (reads inputs + runs user code) and
    # its b_extra injector (drives the controlled output sources) on top
    # of any user/closed-loop callbacks. PWL path only — outputs use
    # b_extra, which the DSED engine does not consume.
    _c_blocks = list(getattr(builder, "_c_blocks", []) or [])
    if _c_blocks:
        _cb_obs = [cb.step_observer for cb in _c_blocks]
        _cb_bex = [cb.b_extra_fn for cb in _c_blocks]
        _prior_obs = step_observer
        _prior_bex = b_extra_fn

        def _cblock_observer(t: float, x) -> None:
            if _prior_obs is not None:
                _prior_obs(t, x)
            for _obs in _cb_obs:
                _obs(t, x)

        def _cblock_b_extra(t: float) -> list:
            total: Optional[list] = (
                list(_prior_bex(t)) if _prior_bex is not None else None)
            for _fn in _cb_bex:
                contrib = _fn(t)
                if total is None:
                    total = list(contrib)
                else:
                    for i, v in enumerate(contrib):
                        total[i] += v
            return total if total is not None else []

        step_observer = _cblock_observer
        b_extra_fn = _cblock_b_extra

    # add-python-builder-ergonomics: if the caller didn't pass an
    # explicit initial_state, ask the builder for its recorded ICs.
    # The C++ `initial_state()` synthesises from `c0=` / `i0=` /
    # `set_initial` calls. Returns an all-zero vector when no ICs
    # were set; we only override the simulate path when there's at
    # least one non-zero entry to avoid forcing a needless copy.
    if initial_state is None:
        try:
            import numpy as _np_check
            candidate = builder.initial_state()
            if _np_check.any(_np_check.asarray(candidate) != 0.0):
                initial_state = candidate
        except Exception:  # noqa: BLE001 — test mocks etc.
            pass

    # Build the PWL cache — LAZY by default (Phase-0 fix #7).
    # A PWM converter visits only a handful of the 2^N switch
    # states, and the eager enumeration made many-switch circuits
    # (3φ NPC: 2^12+ factorizations; MMC: unbuildable) hang before
    # the first step. Lazy mode factorises each mask on first
    # visit and produces bit-identical results for every visited
    # mask. This also makes the docs (gotchas.md "lazy-build is
    # the default") true — they previously described intent, not
    # behaviour.
    cache = PwlStateSpaceCache(builder.graph, builder.pool)
    cache.build_lazy(dt)

    # v2.0 Phase 3 "obra №2" — GGJ Thevenin MMC arms. The driver
    # needs THIS cache (its observer refactors these factors in
    # place on gating events) and the run's dt, which is why the
    # wiring happens here and not in add_mmc_thevenin_arm. Compose
    # AROUND whatever hooks are already in place: the MMC observer
    # runs first (it mutates pool/cache before the solve), and the
    # b_extra vectors sum.
    _mmc_finalize = None
    if mmc_arms:
        # Combinations the arm bookkeeping cannot honour yet — each
        # would return plausible-looking wrong capacitor voltages
        # (all four demonstrated by the adversarial review), so they
        # refuse with the mechanism named instead.
        if store_every is not None and int(store_every) != 1:
            raise ValueError(
                "simulate(mmc_arms=..., store_every>1): decimation "
                "can drop the run's FINAL state, and the end-of-run "
                "capacitor back-solve would then fold the last "
                "step's stamp with a state up to store_every-1 "
                "steps old — volts-scale v_c error in a switching "
                "arm. Record at full rate and decimate afterwards.")
        if start_from_dc_op:
            raise ValueError(
                "simulate(mmc_arms=..., start_from_dc_op=True): the "
                "DC operating point is computed BEFORE the arm's "
                "first stamp and without its V_eq back-EMF, so a "
                "pre-charged arm appears as a milliohm short and "
                "the DC solve seeds kA-scale phantom currents. "
                "Start from t=0 state (the default) or pass "
                "initial_state= explicitly.")
        if max_dt_halvings is not None and int(max_dt_halvings) > 0:
            raise ValueError(
                "simulate(mmc_arms=..., max_dt_halvings>0): the "
                "dt-halving retry re-solves a failed step at sub-dt "
                "with the arm's FULL-dt companion (R_c = dt/2C and "
                "V_eq are baked per step, and the observer does not "
                "re-fire), silently corrupting the capacitor "
                "bookkeeping. The ladder is disabled automatically "
                "when mmc_arms is set; do not request it.")
        if enable_substep_state_correction:
            raise ValueError(
                "simulate(mmc_arms=..., "
                "enable_substep_state_correction=True): the "
                "commutation sub-split re-solves at dt1/dt2 with "
                "the arm's full-dt Thevenin stamp — same corruption "
                "mechanism as the dt-halving retry.")
        if initial_state is not None:
            _unset = [a.name for a in mmc_arms
                       if a.v_c_preset is None]
            if _unset:
                raise ValueError(
                    "simulate(mmc_arms=..., initial_state=...): "
                    f"arm(s) {_unset} have no v_c_preset. A "
                    "continuation restarts every arm from "
                    "v_c_init, which silently resets the submodule "
                    "capacitors while the network continues — an "
                    "unphysical energy jump. Recipe: re-register "
                    "the arm with v_c_preset=old_arm.v_c alongside "
                    "initial_state=res.states[-1].")
        # A failed step must FAIL (SimulationAborted carries the
        # partial trace), not silently retry with a wrong-dt arm.
        max_dt_halvings = 0
        from .mmc_thevenin import make_mmc_thevenin_driver
        _mmc_obs, _mmc_b, _mmc_finalize = make_mmc_thevenin_driver(
            mmc_arms, builder, cache, dt,
            builder.pool.state_size(builder.graph))
        if step_observer is None:
            step_observer = _mmc_obs
        else:
            _prev_obs = step_observer

            def _obs_with_mmc(t, x, _mmc=_mmc_obs, _u=_prev_obs):
                _mmc(t, x)
                _u(t, x)
            step_observer = _obs_with_mmc
        if b_extra_fn is None:
            b_extra_fn = _mmc_b
        else:
            _prev_b = b_extra_fn
            _mmc_state_size = builder.pool.state_size(builder.graph)

            def _b_with_mmc(t, _mmc=_mmc_b, _u=_prev_b,
                             _n=_mmc_state_size):
                import numpy as _np
                u = _np.asarray(_u(t), dtype=float)
                if u.shape != (_n,):
                    # A scalar or short vector would BROADCAST over
                    # the sum and inject current into every row.
                    raise ValueError(
                        f"b_extra_fn returned shape {u.shape}, "
                        f"expected ({_n},) when composed with "
                        "mmc_arms")
                return _np.asarray(_mmc(t)) + u
            b_extra_fn = _b_with_mmc

    # Construct options.
    opts = SimulationOptions(t_start=t_start, t_end=t_end, dt=dt)
    if store_every is not None and int(store_every) != 1:
        if int(store_every) < 1:
            raise ValueError(
                f"simulate(store_every={store_every}): must be >= 1 "
                "(1 records every step)")
        opts.store_every = int(store_every)
    if max_newton_iterations > 0:
        opts.max_newton_iterations = max_newton_iterations
    if max_event_iterations is not None:
        # Including 0 — "disable event iteration" is a documented
        # request, and gating on `> 0` silently discarded it (the
        # kernel default of 16 applied instead).
        if int(max_event_iterations) < 0:
            raise ValueError(
                f"simulate(max_event_iterations={max_event_iterations}"
                "): must be >= 0 (0 disables event iteration)")
        opts.max_event_iterations = int(max_event_iterations)
    if strict_event_iterations:
        opts.strict_event_iterations = True
    if max_dt_halvings is not None:
        # 0 is a MEANINGFUL value here (disable the retry and restore
        # the pre-v2.0 hard failure), which is why the default is
        # None rather than 0.
        if int(max_dt_halvings) < 0:
            raise ValueError(
                f"simulate(max_dt_halvings={max_dt_halvings}): must "
                "be >= 0 (0 disables the retry)")
        opts.max_dt_halvings = int(max_dt_halvings)
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
    if inductor_freeze_di_max is not None and inductor_freeze_di_max > 0:
        opts.inductor_freeze_di_max = float(inductor_freeze_di_max)
    if inductor_abs_clamp is not None and inductor_abs_clamp > 0:
        opts.inductor_abs_clamp = float(inductor_abs_clamp)

    # Default switch_fn: all switches closed (v1.x behaviour).
    #
    # Phase-0 fix #4: this default is DANGEROUS for circuits with
    # controlled switches — a half-bridge with a forgotten gate
    # assignment becomes a dead short across the DC link (shoot-
    # through), and either dies with an unexplained singular-matrix
    # error or converges to absurd currents. Diode bits are solver-
    # owned (combine_masks overlays them), so for diode-only
    # rectifiers the default is harmless and stays silent. When
    # CONTROLLED switches exist, warn loudly; v2.0 flips the
    # default to all-OPEN and makes this an error.
    if switch_fn is None:
        n_sw = builder.graph.num_switches
        try:
            _, _n_diode, _controlled = _switch_census(
                builder.graph, builder.pool)
        except Exception:  # noqa: BLE001 — test doubles w/o pool
            _controlled = []
        if _controlled:
            import warnings
            warnings.warn(
                f"simulate(): no switch_fn was given, but the "
                f"circuit has {len(_controlled)} controlled "
                f"switch(es) (indices {list(_controlled)}). They "
                "default to ALL CLOSED, which short-circuits "
                "bridge legs (shoot-through). Pass an explicit "
                "switch_fn / closed_loops that drives these bits. "
                "In Pulsim v2.0 this becomes an error and the "
                "default flips to all-OPEN.",
                stacklevel=2)
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

    # Live streaming hook: when the caller passes a NativeLiveStream
    # (or any object with ``.attach(state_size)`` + ``.native_ring``),
    # attach it now so the kernel can push (t, x) samples straight
    # into its ring buffer during the run. The kernel-side ``live_ring``
    # path uses atomics only — zero GIL contention with the GUI thread
    # that polls ``stream.get_new_samples()``.
    live_ring = None
    if live_stream is not None:
        state_size = builder.pool.state_size(builder.graph)
        # ``builder.state_var_names()`` returns a labelled list of
        # the same length as the kernel state vector — passes it
        # along so live-scope GUIs (``pulsim.LiveScope`` /
        # PulsimGUI) can resolve names → state indices without
        # re-computing the layout themselves.
        names = None
        try:
            names = list(builder.state_var_names())
        except Exception:  # noqa: BLE001 — older binaries lack it
            names = None
        live_stream.attach(state_size, names=names)
        live_ring = live_stream.native_ring
        # Auto-wire pause/stop into ``should_continue`` so the GUI's
        # pause button actually halts the kernel (instead of letting
        # samples accumulate during a "display-only" pause). Only when
        # the caller didn't pass their own should_continue — we don't
        # want to override a user-supplied cancellation hook.
        if should_continue is None \
                and hasattr(live_stream, "should_continue"):
            should_continue = live_stream.should_continue

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
        if live_ring is not None:
            kwargs["live_ring"] = live_ring
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
        if live_ring is not None:
            kwargs["live_ring"] = live_ring
        try:
            res = run_transient(
                cache, builder.graph, builder.pool, opts, **kwargs,
            )
        except SimulationAborted as _aborted:
            # The partial trace is a fresh pybind-owned object with
            # none of the side-table attributes simulate() attaches
            # on success, so without this every name-based accessor
            # on it — .v(), .i(), .power(), .signal(), .plot() —
            # fails with a message telling the user to do the thing
            # they already did. Give the wreckage the same handles
            # the survivors get.
            try:
                _partial = _aborted.partial
                _partial._builder = builder
                _partial._preflight = _preflight_report
                _partial._switch_fn = kwargs.get("switch_fn")
            except AttributeError:  # pragma: no cover — defensive
                pass
            raise
    # Close the progress bar with a newline if we were in bar mode.
    if progress is True or (isinstance(progress, str)
                              and progress.lower() == "bar"):
        import sys as _sys
        _sys.stdout.write("\n")
        _sys.stdout.flush()
    # Phase-0 fix #9: the kernel no longer throws away a whole run
    # on diode event-iteration cycles/budget exhaustion — it records
    # breaches and continues. Surface them LOUDLY so the accepted
    # per-step diode-state error is never silent.
    _breaches = getattr(res, "event_iteration_breaches", None)
    if _breaches:
        import warnings
        _b0 = _breaches[0]
        warnings.warn(
            f"simulate(): diode event iteration failed to settle on "
            f"{len(_breaches)} of {len(res.times)} steps (first at "
            f"t={_b0.t:.6g}, "
            f"{'mask cycle' if _b0.cycle_detected else 'budget hit'}). "
            "The solver kept the last consistent solve each time and "
            "continued; waveforms near those instants carry a one-"
            "flip diode-state error. Inspect "
            "result.event_iteration_breaches, reduce dt, or pass "
            "strict_event_iterations=True to restore the old hard "
            "failure.",
            stacklevel=2)

    # Attach the builder so `result.v(name)` / `.i(name)` / `.power(name)`
    # can resolve names without forcing the caller to re-pass the
    # builder. The binding enables `py::dynamic_attr()` precisely so
    # this assignment works.
    try:
        res._builder = builder
    except AttributeError:  # pragma: no cover — pre-dynamic_attr builds
        pass

    # Stash the (composed) switch_fn so `result.i(switch_name)` can
    # reconstruct switch-branch currents post-hoc by replaying the
    # mask schedule at each result time. ``switch_fn`` here is the
    # post-compose version (closed_loops + user) so it sees every
    # switch the result actually exercised.
    try:
        # The preflight facts are properties of the CIRCUIT, but a
        # user reads them off the result, so attach both.
        res._preflight = _preflight_report
    except AttributeError:  # pragma: no cover
        pass
    try:
        # Which engine actually ran, and — when the caller said
        # 'auto' and did not get the variable-step one — why. A
        # router that cannot be interrogated is a router that
        # surprises people.
        res.engine_used = "pwl"
        res.engine_route_reason = (
            "; ".join(_route_reasons) if _route_reasons else None)
    except AttributeError:  # pragma: no cover
        pass
    try:
        res._switch_fn = switch_fn
    except AttributeError:  # pragma: no cover
        pass

    # A post-solve inductor guard that FIRED replaced the solver's
    # answer with a number the caller configured. Reading that limit
    # off a plot as if it were current is the failure this warning
    # exists to prevent: on the drive these guards were written for,
    # the reported line current peaks at exactly 100.000 A because
    # the clamp is 100 A.
    # An inductor whose conduction path opens produces an unbounded
    # voltage in an idealized model, and Pulsim reports it finitely
    # and without comment — 2.9 MV on a 48 V circuit, `isfinite` true
    # throughout. The current guards cannot see it: they watch i_L,
    # which stays believable. Check the voltage against what the
    # circuit's own sources can account for.
    # The try covers ONLY the kernel call. Wrapping the warn() too
    # meant that under `-W error` — where a warning IS an exception —
    # the detector produced no signal at all: not the warning, and
    # not the attribute either. A user who asked for warnings to be
    # fatal is the last one who should be silently spared this.
    _volt = None
    try:
        from . import _pulsim as _k  # type: ignore[import-not-found]
        _volt = _k.find_implausible_voltage(
            builder.graph, builder.pool, res,
            100.0 if voltage_sanity_factor is None
            else float(voltage_sanity_factor))
    except Exception:  # pragma: no cover — never break a good run
        _volt = None
    if _volt is not None and _volt.node >= 0:
        import warnings
        res._implausible_voltage = _volt
        warnings.warn(
            "simulate(): " + _k.describe_implausible_voltage(
                builder.graph, builder.pool, _volt),
            stacklevel=2)

    # A retried step integrated its interval more finely than the
    # caller asked for. That is a change in ACCURACY, and the record
    # exists because they are entitled to know — which is worth
    # nothing if nobody says it out loud.
    _retries = list(getattr(res, "dt_retries", []) or [])
    if _retries:
        import warnings
        worst = max(d.halvings for d in _retries)
        first = min(_retries, key=lambda d: d.t)
        warnings.warn(
            f"simulate(): {len(_retries)} step(s) would not converge "
            f"at dt = {dt:g} and were re-taken at a smaller step "
            f"(down to dt/{1 << worst}), first at t = "
            f"{first.t:.6g} s. The sampling grid is unchanged — what "
            f"changed is that those intervals were integrated more "
            f"finely than you asked for. If it happens often, the "
            f"run is telling you dt is too coarse for this circuit. "
            f"Inspect result.dt_retries; pass max_dt_halvings=0 to "
            f"get the failure back instead.\n  First reason: "
            f"{first.reason[:200]}",
            stacklevel=2)

    _guards = list(getattr(res, "inductor_guard_actions", []) or [])
    if _guards:
        import warnings
        lines = []
        for g in _guards:
            name = builder.graph.branch_name(g.branch_id) or \
                f"branch #{g.branch_id}"
            lines.append(
                f"  * {name}: the guard replaced the solver's "
                f"current on {g.total()} step(s), first at "
                f"t = {g.t_first:.6g} s. Peak raw solve "
                f"{g.worst_solved:.4g} A; limit reported "
                f"{g.reported_limit:.4g} A. A guard that stays "
                f"engaged over many steps is HOLDING the trajectory "
                f"there, not trimming an outlier.")
        warnings.warn(
            "simulate(): a post-solve inductor guard replaced the "
            "solver's answer.\n" + "\n".join(lines) +
            "\n  These guards do not solve anything — they substitute "
            "a limit you configured for a current the solver "
            "computed, so what you plot for these inductors is the "
            "limit. Before trusting the trace, find out WHY the "
            "underlying current went there: re-run with the guard "
            "off and with a smaller dt. If the value is unchanged by "
            "dt, it is your model's own trajectory (an unbounded "
            "open-loop stage, a missing snubber, a missing inrush "
            "limiter), not solver noise — and clipping it is hiding "
            "a modelling result, not a numerical one. Inspect "
            "result.inductor_guard_actions.",
            stacklevel=2)

    # T2.2: auto-attach motor observer traces so
    # `res.signal('M1.omega')` works without manual wiring. We walk
    # the step_observer / b_extra_fn / closed_loops the user passed
    # in and any MotorObserverBundle gets `.attach_to_result(res)`.
    _auto_attach_motor_traces(
        res,
        step_observer=_step_observer_user,
        b_extra_fn=_b_extra_fn_user,
        closed_loops=_closed_loops_user,
    )
    # v2.0 Phase 3: fold the last step's charge transfer into the
    # Thevenin arms — the per-step observer only fires BEFORE
    # solves, so without this `arm.v_c` would be one step behind
    # the trace it was solved against.
    if _mmc_finalize is not None and len(res.states) > 0:
        _mmc_finalize(res.times[-1], res.states[-1])
    return res


def _trbdf2_blockers(builder, *, dt, step_observer, closed_loops,
                      controller_period, live_stream, progress,
                      start_from_dc_op, strict_event_iterations,
                      max_dt_halvings, store_every, mmc_arms,
                      enable_substep_state_correction,
                      inductor_freeze_di_max, inductor_abs_clamp,
                      switch_fn):
    """Why the variable-step engine cannot serve this run.

    Returns a list of human-readable reasons — empty means it can.
    `engine='auto'` routes on this: no blockers → TR-BDF2, else the
    fixed-step engine, so a script written for `simulate(b, t_end,
    dt=...)` keeps working while a dt-less one gets the variable
    engine whenever the circuit qualifies. `engine='trbdf2'` uses
    the same list to REFUSE, which is what a caller who explicitly
    asked for it wants to know.
    """
    why = []
    try:
        from ._pulsim import BranchKind as _BK  # type: ignore
        n_sat = 0
        for br in builder.graph.branches:
            if br.get("kind") != _BK.Nonlinear:
                continue
            # Nonlinear diodes / MOSFETs / IGBTs are fine — each
            # stage becomes a Newton solve on the same companion,
            # and their re-stamp has no dt in it. A SATURABLE
            # inductor's does, and its flux cannot be rolled back
            # when a step is rejected.
            if str(builder.pool.kind_of(br["id"])).endswith(
                    "SaturableInductor"):
                n_sat += 1
        if n_sat:
            why.append(
                f"{n_sat} saturable inductor(s) — their Newton "
                "stamp divides by the step size and their flux "
                "history cannot be rolled back when a step is "
                "rejected")
    except Exception:  # pragma: no cover — never block on a probe
        pass
    if getattr(builder, "_c_blocks", None):
        why.append("C blocks sample on a fixed dt grid")
    if live_stream is not None:
        why.append("live_stream needs the fixed-step kernel's "
                    "per-step ring push")
    if progress:
        why.append("progress has no per-step hook here")
    if start_from_dc_op:
        why.append("start_from_dc_op is a fixed-step entry point")
    if strict_event_iterations:
        why.append("strict_event_iterations belongs to the "
                    "fixed-step diode iteration")
    if max_dt_halvings is not None and int(max_dt_halvings) > 0:
        why.append("max_dt_halvings re-solves at sub-dt against a "
                    "full-dt companion")
    if enable_substep_state_correction:
        why.append("enable_substep_state_correction is the "
                    "fixed-step commutation split")
    if store_every is not None and int(store_every) != 1:
        why.append("store_every decimates a uniform grid")
    if mmc_arms:
        why.append("mmc_arms drives the fixed-step observer/"
                    "b_extra pair")
    if inductor_freeze_di_max is not None:
        why.append("inductor_freeze_di_max is a fixed-step "
                    "post-solve guard")
    if inductor_abs_clamp is not None:
        why.append("inductor_abs_clamp is a fixed-step post-solve "
                    "guard")
    # An observer needs a CADENCE here — there is no grid to ride.
    _has_cadence = controller_period is not None
    if closed_loops:
        for _cl in closed_loops:
            if float(getattr(_cl, "period", 0.0) or 0.0) > 0.0:
                _has_cadence = True
    if step_observer is not None and not _has_cadence:
        if getattr(step_observer, "period", None) is None:
            why.append(
                "a step_observer with no cadence (pass "
                "controller_period=<T_ctrl>, or use a ClosedLoop, "
                "which carries its own)")
    if closed_loops and not _has_cadence:
        why.append("closed_loops whose handles carry no period")
    if switch_fn is None and not closed_loops:
        # (A ClosedLoop brings its own switch_fn, so the circuit is
        # driven even when the caller passed none.)
        # The two engines disagree on what an undriven controlled
        # switch means (fixed: all-CLOSED + warning, the v1 legacy;
        # variable: all-OPEN). Route to the one whose answer the
        # caller's script was written against; flipping the legacy
        # default is its own breaking change.
        try:
            _, _nd, _ctl = _switch_census(builder.graph,
                                           builder.pool)
        except Exception:  # noqa: BLE001 — test doubles
            _ctl = []
        if _ctl:
            why.append(
                f"{len(_ctl)} controlled switch(es) with no "
                "switch_fn (the two engines' undriven-switch "
                "defaults differ)")
    del dt
    return why


def _auto_check_voltage_sanity(res, builder, factor):
    """Run the implausible-voltage detector on any result.

    Extracted so `engine='auto'` gets the same guard the pwl tail
    applies: the detector reads the recorded trace, not the
    stepper, so it is engine-independent — and the variable-step
    engine's all-OPEN default gate mask makes the inductor-open
    case it exists for (2.9 MV on a 48 V circuit, isfinite
    throughout) MORE likely, not less.
    """
    try:
        from . import _pulsim as _k  # type: ignore[import-not-found]
        volt = _k.find_implausible_voltage(
            builder.graph, builder.pool, res,
            100.0 if factor is None else float(factor))
    except Exception:  # pragma: no cover — never break a good run
        return
    if volt is not None and volt.node >= 0:
        import warnings
        res._implausible_voltage = volt
        warnings.warn(
            "simulate(): " + _k.describe_implausible_voltage(
                builder.graph, builder.pool, volt),
            stacklevel=3)


def _auto_attach_motor_traces(
    result,
    *,
    step_observer=None,
    b_extra_fn=None,
    closed_loops=None,
) -> None:
    """Find every :class:`MotorObserverBundle` reachable from the
    caller's `simulate(...)` arguments and stash its traces on
    ``result._motor_traces``. Idempotent and exception-safe — a
    misbehaving custom step_observer must not break the post-sim
    result.
    """
    from .motors import MotorObserverBundle as _Bundle
    seen: set = set()

    def _visit(obj) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, _Bundle):
            try:
                obj.attach_to_result(result)
            except Exception:  # noqa: BLE001 — defensive
                pass
            return
        # Composed observer (closed_loops path) — walks the inner
        # list we stash on the wrapper at compose time.
        for attr in ("_inner_observers", "_inner_bundles"):
            inner = getattr(obj, attr, None)
            if inner is None:
                continue
            try:
                for child in inner:
                    _visit(child)
            except TypeError:  # not iterable — skip
                pass

    _visit(step_observer)
    _visit(b_extra_fn)
    if closed_loops:
        try:
            for loop in closed_loops:
                _visit(loop)
                # The closed-loop wrapper itself bundles a
                # step_observer / switch_fn; visit those too.
                _visit(getattr(loop, "step_observer", None))
                _visit(getattr(loop, "switch_fn", None))
        except TypeError:  # not iterable
            pass


# Note: SineVoltageSource (Layer 2 V11) is exposed as a
# CircuitBuilder method `add_sine_voltage_source`; there's
# no separate Python-side params class — pass v_dc,
# v_amplitude, frequency, phase as keyword args.

__version__ = "1.8.0"


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
    "templates":          "moved to pulsim.schematic.templates",
    # schematic intentionally NOT listed here — `pulsim.schematic` is
    # the restored subpackage (Phase 1-17 of the schematic-renderer-v2
    # effort), so attribute access resolves normally and never falls
    # through to this hint dict.
    "parse_netlist":      "p.parse_spice_netlist (SPICE subset)",
    # ----- T3.1: 1.5 → 1.6 churn hints ---------------------------------
    # The retired pulsim.sweep helper *classes* (1.5 → 1.6).
    # `pulsim.sweep` ITSELF is now a function and resolves normally —
    # only these former subpackage attributes hit the hint path.
    "Distribution":       "lambda rng: rng.normal(mu, sigma) — pass the "
                          "distribution callable directly to p.monte_carlo("
                          "distributions={...}). See docs/migration-guide.md "
                          "§ '1.5 → 1.6 — API stability notes'.",
    "Cartesian":          "params={...} kwarg on p.sweep(builder_factory, "
                          "params={'R': [10, 20, 50]}, kpi_fn=..., ...). "
                          "See docs/migration-guide.md.",
    "metrics":            "build your own KPI lambda — `kpi_fn(res, "
                          "params) -> dict[str, float]` is the 1.6 contract. "
                          "See docs/migration-guide.md for examples "
                          "(rms_voltage, peak_current, settling_time).",
    "PmsmParams":         "no params struct — p.add_pmsm takes the parameters "
                          "as direct kwargs (R_s=, Ld=, Lq=, psi_pm=, "
                          "pole_pairs=, J=, ...). See docs/migration-guide.md "
                          "§ '1.5 → 1.6 — API stability notes'.",
    "ThreePhaseVsiParams": "no params struct — p.add_three_phase_vsi takes the "
                          "parameters as direct kwargs.",
    "BldcParams":         "no params struct — p.add_bldc takes the parameters "
                          "as direct kwargs.",
}


# Note: the original `_simulate_dsed()` stub (Gate 2 Phase 2.B-3) was
# replaced by `pulsim._dsed_dispatch.run_dsed_from_builder()` in the
# Gate 5+ API consolidation. The new dispatch lives in its own module
# (`_dsed_dispatch.py`) and provides validation + option resolution +
# the honest NotImplementedError-with-resolved-options message. See
# the `simulate(...)` body above for how it is wired in.


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
