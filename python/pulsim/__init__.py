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
from .stream import LiveStream, NativeLiveStream
from .losses import (
    LossAccumulator,
    EfficiencyCalculator,
    device_loss_summary,
    average_power_at_node,
)
from .mmc import (
    MmcArmAverageParams,
    MmcArmAverageResult,
    mmc_arm_average_step,
    simulate_mmc_arm_average,
    # Builder-side wiring (Phase 20.4).
    MmcArmAverage,
    add_mmc_arm_average,
    make_mmc_arm_observer,
    make_mmc_arms_observer,
    # Three-phase DC/AC topology helper (Phase 20.4 step 2).
    MmcThreePhaseDcAc,
    add_mmc_three_phase_dc_ac,
    # L1 multilevel arm — Phase 20.5.
    MmcArmMultilevelParams,
    MmcArmMultilevelResult,
    ps_pwm_switching_function,
    ipd_switching_function,
    mmc_arm_multilevel_step,
    simulate_mmc_arm_multilevel,
    # L2 SM-equivalent arm (dead-time aware) — Phase 20.6.
    MmcArmEquivalentParams,
    MmcArmEquivalentState,
    MmcArmEquivalentResult,
    make_l2_state,
    mmc_arm_equivalent_step,
    simulate_mmc_arm_equivalent,
    # L3 detailed (per-SM balancing) — Phase 20.7.
    MmcArmDetailedParams,
    MmcArmDetailedState,
    MmcArmDetailedResult,
    make_l3_state,
    mmc_arm_detailed_step,
    simulate_mmc_arm_detailed,
    # Builder integration for L1/L2/L3 — Phase 20.8.
    MmcArmMultilevel,
    add_mmc_arm_multilevel,
    make_mmc_arm_multilevel_observer,
    make_mmc_arm_multilevel_observers,
    MmcArmEquivalent,
    add_mmc_arm_equivalent,
    make_mmc_arm_equivalent_observer,
    make_mmc_arm_equivalent_observers,
    MmcArmDetailed,
    add_mmc_arm_detailed,
    make_mmc_arm_detailed_observer,
    make_mmc_arm_detailed_observers,
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
    "NativeLiveStream",
    # Post-hoc loss + efficiency helpers (parity with v1 surface).
    "LossAccumulator",
    "EfficiencyCalculator",
    "device_loss_summary",
    "average_power_at_node",
    # Modular Multilevel Converter — L0 average-value arm model
    # (Phase 20, Sousa 2022 eqs 2.13/2.14).
    "MmcArmAverageParams",
    "MmcArmAverageResult",
    "mmc_arm_average_step",
    "simulate_mmc_arm_average",
    # MMC L0 CircuitBuilder integration (Phase 20.4).
    "MmcArmAverage",
    "add_mmc_arm_average",
    "make_mmc_arm_observer",
    "make_mmc_arms_observer",
    # MMC three-phase DC/AC topology helper.
    "MmcThreePhaseDcAc",
    "add_mmc_three_phase_dc_ac",
    # MMC L1 — discrete multilevel arm (PS-PWM).
    "MmcArmMultilevelParams",
    "MmcArmMultilevelResult",
    "ps_pwm_switching_function",
    "ipd_switching_function",
    "mmc_arm_multilevel_step",
    "simulate_mmc_arm_multilevel",
    # MMC L2 — SM-equivalent (dead-time + min-pulse-width).
    "MmcArmEquivalentParams",
    "MmcArmEquivalentState",
    "MmcArmEquivalentResult",
    "make_l2_state",
    "mmc_arm_equivalent_step",
    "simulate_mmc_arm_equivalent",
    # MMC L3 — detailed per-SM with balancing.
    "MmcArmDetailedParams",
    "MmcArmDetailedState",
    "MmcArmDetailedResult",
    "make_l3_state",
    "mmc_arm_detailed_step",
    "simulate_mmc_arm_detailed",
    # MMC L1/L2/L3 builder integration.
    "MmcArmMultilevel",
    "add_mmc_arm_multilevel",
    "make_mmc_arm_multilevel_observer",
    "make_mmc_arm_multilevel_observers",
    "MmcArmEquivalent",
    "add_mmc_arm_equivalent",
    "make_mmc_arm_equivalent_observer",
    "make_mmc_arm_equivalent_observers",
    "MmcArmDetailed",
    "add_mmc_arm_detailed",
    "make_mmc_arm_detailed_observer",
    "make_mmc_arm_detailed_observers",
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
    col = states[:, idx]
    if t is None:
        return col
    return col[t]


def _result_i(self, name: str, t=None):
    """Return the branch-current trace for ``name``.

    The state vector only carries currents for branches that
    contribute MNA state variables — inductors and independent
    voltage sources. For resistors / capacitors / diodes /
    MOSFETs, the current must be reconstructed from node
    voltages and the device's parameters; that path lives in
    :mod:`pulsim.losses` (today via
    :func:`average_power_at_node` and
    :func:`device_loss_summary`). When called on a non-state
    branch this method raises :class:`NotImplementedError`
    pointing the caller at those helpers.

    Sign convention follows the ``add_*`` call: current is
    positive flowing from the ``from`` terminal to the ``to``
    terminal.
    """
    import numpy as _np
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
    # Map branch_id to its state-vector column. Only inductors and
    # voltage sources have a dedicated current state variable; the
    # pool exposes the lookup but each device family has its own
    # accessor in v1.4. Try inductor first (most common case), fall
    # back to source.
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
    if state_idx is None:
        raise NotImplementedError(
            f"branch {name!r} has no MNA current state variable "
            f"(it's likely a resistor, capacitor, diode, or MOSFET — "
            f"reconstruct its current via pulsim.losses helpers, e.g. "
            f"device_loss_summary(result, builder)). result.i() is "
            f"defined for inductors and voltage sources only."
        )
    states = _np.asarray(self.states)
    col = states[:, state_idx]
    if t is None:
        return col
    return col[t]


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


# Monkey-patch the methods onto the C++-bound SimulationResult class.
# We can't subclass it cleanly because run_transient returns the
# concrete C++ type, but the type itself accepts attribute injection
# (py::dynamic_attr() on the binding).
SimulationResult.v = _result_v          # type: ignore[attr-defined]
SimulationResult.i = _result_i          # type: ignore[attr-defined]
SimulationResult.power = _result_power  # type: ignore[attr-defined]


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
    closed_loops=None,
    live_stream=None,
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
    # add-python-closed-loop-helper composition: when caller passes
    # one or more `ClosedLoop` instances, derive `switch_fn` and
    # `step_observer` from them. Mutually exclusive with explicit
    # `switch_fn=` / `step_observer=` to avoid silent override.
    if closed_loops:
        if switch_fn is not None or step_observer is not None:
            raise ValueError(
                "pass closed_loops OR switch_fn/step_observer, not "
                "both — the helper composes both callbacks "
                "internally."
            )
        loops = list(closed_loops)
        n_sw_compose = builder.graph.num_switches
        per_switch_fns = [loop.switch_fn for loop in loops]
        switch_fn = make_combined_switch_fn(n_sw_compose, per_switch_fns)
        per_observers = [loop.step_observer for loop in loops]

        def _composed_observer(t: float, x) -> None:
            for obs in per_observers:
                obs(t, x)

        step_observer = _composed_observer

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
    # Attach the builder so `result.v(name)` / `.i(name)` / `.power(name)`
    # can resolve names without forcing the caller to re-pass the
    # builder. The binding enables `py::dynamic_attr()` precisely so
    # this assignment works.
    try:
        res._builder = builder
    except AttributeError:  # pragma: no cover — pre-dynamic_attr builds
        pass
    return res


# Note: SineVoltageSource (Layer 2 V11) is exposed as a
# CircuitBuilder method `add_sine_voltage_source`; there's
# no separate Python-side params class — pass v_dc,
# v_amplitude, frequency, phase as keyword args.

__version__ = "1.4.2"


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
