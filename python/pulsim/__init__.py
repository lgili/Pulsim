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
    # Phase-0 fix #4 helper (private): controlled-vs-diode census.
    _switch_census,
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
SimulationResult.i = _result_i          # type: ignore[attr-defined]
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
    max_event_iterations: int = 0
    enable_substep_state_correction: Optional[bool] = None

    # Output ---------------------------------------------------------
    # Record every m-th step (1 = every step). The solver still
    # integrates at `dt`; only what is stored changes, and the
    # recorded grid stays uniform at m*dt so FFT / harmonic analysis
    # remains valid.
    store_every: int = 1

    # Inductor regularisation (rare — kept for completeness) --------
    inductor_freeze_di_max: float = 0.0
    inductor_abs_clamp: float = 0.0

    # Adaptive RK schema (Phase 2.4 — execution deferred to v1.6) ---
    integrator: str = "kernel"
    rtol: float = 1.0e-5
    atol: float = 1.0e-8
    dt_init: float = 0.0


def simulate(
    builder: CircuitBuilder,
    t_end: float,
    dt: Optional[float] = None,
    *,
    # --- engine selector ---
    engine: str = "pwl",
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
    max_event_iterations: int = 0,
    strict_event_iterations: bool = False,
    store_every: int = 1,
    tol_newton_dx: Optional[float] = None,
    tol_newton_res: Optional[float] = None,
    enable_newton_line_search: Optional[bool] = None,
    enable_newton_lm: Optional[bool] = None,
    enable_substep_state_correction: Optional[bool] = None,
    inductor_freeze_di_max: float = 0.0,
    inductor_abs_clamp: float = 0.0,
    progress: "bool | int | str" = False,
    initial_state=None,
    should_continue=None,
    closed_loops=None,
    live_stream=None,
    # --- SolverOptions bundle (v1.5 Round 2 ergonomics polish) ---
    solver: Optional["SolverOptions"] = None,
) -> SimulationResult:
    """Build the cache and run a transient simulation.

    Two simulation paradigms, selected by the ``engine`` keyword:

    * ``engine='pwl'`` (default, v1.0–v1.4 compatibility) — fixed-step
      trapezoidal companion + PWL state-space cache. **Requires a
      positive `dt`.** Bit-exact reproducibility; matches v1.4.0
      output to machine precision.

    * ``engine='dsed'`` (variable-step, opt-in) — Path-Based
      Event-Driven scheduler with automatic RK45 / BDF2 dispatch.
      Tolerance ``rtol`` controls accuracy; the kernel handles
      integrator selection, step sizing, event prediction, and
      LU caching transparently. ``dt`` (if supplied) acts as
      the maximum-step cap ``dt_max``.

    Two simple decision trees for users:

    * **"Não sei qual escolher"** → ``engine='pwl'`` with the
      smallest ``dt`` you can afford. Always works; matches every
      legacy script.
    * **"Quero performance e meu circuito tem PWM/DCM/eventos"** →
      ``engine='dsed', rtol=1e-6``. Kernel does the rest.

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
        if max_event_iterations == 0:
            max_event_iterations = solver.max_event_iterations
        if store_every == 1:
            store_every = solver.store_every
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
        if inductor_freeze_di_max == 0.0:
            inductor_freeze_di_max = solver.inductor_freeze_di_max
        if inductor_abs_clamp == 0.0:
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
            # v2.0 Phase 1: output decimation is a fixed-grid
            # concept; the DSED engine emits variable-step samples,
            # so honouring `store_every` there would mean something
            # different (and unstated). Fail loudly rather than
            # returning a full-rate trace the caller did not ask for.
            "store_every": int(store_every) != 1,
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
        return _dsed.run_dsed_from_builder(
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

    # Construct options.
    opts = SimulationOptions(t_start=t_start, t_end=t_end, dt=dt)
    if store_every != 1:
        if int(store_every) < 1:
            raise ValueError(
                f"simulate(store_every={store_every}): must be >= 1 "
                "(1 records every step)")
        opts.store_every = int(store_every)
    if max_newton_iterations > 0:
        opts.max_newton_iterations = max_newton_iterations
    if max_event_iterations > 0:
        opts.max_event_iterations = max_event_iterations
    if strict_event_iterations:
        opts.strict_event_iterations = True
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
    if inductor_freeze_di_max > 0:
        opts.inductor_freeze_di_max = float(inductor_freeze_di_max)
    if inductor_abs_clamp > 0:
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
        res = run_transient(
            cache, builder.graph, builder.pool, opts, **kwargs,
        )
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
        res._switch_fn = switch_fn
    except AttributeError:  # pragma: no cover
        pass

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
    return res


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
