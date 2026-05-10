"""PulsimCore - High-performance circuit simulator for power electronics.

This is the API with C++23 features and SIMD optimization.
"""

__version__ = "0.5.1"

import weakref

import numpy as np

from ._pulsim import (
    # Enums
    DeviceType,
    SolverStatus,
    DCStrategy,
    RLCDamping,
    DeviceHint,
    SIMDLevel,
    Integrator,
    TimestepMethod,
    StepMode,
    FormulationMode,
    SwitchingMode,

    # Device Classes - Linear
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,

    # Device Classes - Nonlinear
    IdealDiode,
    IdealSwitch,
    MOSFETParams,
    MOSFET,
    IGBTParams,
    IGBT,

    # Time-Varying Sources
    PWMParams,
    PWMVoltageSource,
    SineParams,
    SineVoltageSource,
    RampParams,
    RampGenerator,
    PulseParams,
    PulseVoltageSource,

    # Control Blocks
    PIController,
    PIDController,
    Comparator,
    SampleHold,
    RateLimiter,
    MovingAverageFilter,
    HysteresisController,
    LookupTable1D,

    # Circuit Builder
    Circuit,
    VirtualComponent,
    MixedDomainStepResult,
    VirtualChannelMetadata,

    # DC Solver
    solve_dc,
    dc_operating_point,

    # Transient Simulation
    run_transient as _run_transient_native,
    run_transient_streaming as _run_transient_streaming_native,
    SimulationOptions,
    SimulationResult,
    Simulator,
    PeriodicSteadyStateOptions,
    PeriodicSteadyStateResult,
    HarmonicBalanceOptions,
    HarmonicBalanceResult,
    StiffnessConfig,
    SwitchingEnergy,
    FallbackReasonCode,
    FallbackPolicyOptions,
    FallbackTraceEntry,
    BackendTelemetry,
    ThermalCouplingPolicy,
    ThermalCouplingOptions,
    ThermalDeviceConfig,
    DeviceThermalTelemetry,
    ThermalSummary,
    ComponentElectrothermalTelemetry,
    SimulationEventType,
    SimulationEvent,
    LinearSolverTelemetry,
    YamlParserOptions,
    YamlParser,

    # Solver Configuration
    Tolerances,
    NewtonOptions,
    NewtonResult,

    # Convergence Monitoring
    IterationRecord,
    ConvergenceHistory,
    VariableConvergence,
    PerVariableConvergence,

    # Convergence Aids
    GminConfig,
    SourceSteppingConfig,
    PseudoTransientConfig,
    InitializationConfig,
    DCConvergenceConfig,
    DCAnalysisResult,

    # Analytical Solutions (Validation)
    RCAnalytical,
    RLAnalytical,
    RLCAnalytical,

    # Validation Framework
    ValidationResult_v2 as ValidationResult,
    compare_waveforms,
    export_validation_csv,
    export_validation_json,

    # Benchmark Framework
    BenchmarkTiming,
    BenchmarkResult,
    export_benchmark_csv,
    export_benchmark_json,

    # Integration Methods
    BDFOrderConfig,
    TimestepConfig,
    AdvancedTimestepConfig,
    RichardsonLTEConfig,

    # High-Performance Features
    LinearSolverConfig,
    LinearSolverKind,
    PreconditionerKind,
    IterativeSolverConfig,
    LinearSolverStackConfig,
    detect_simd_level,
    simd_vector_width,
    backend_capabilities,
    solver_status_to_string,

    # Thermal Simulation
    FosterStage,
    FosterNetwork,
    CauerStage,
    CauerNetwork,
    ThermalSimulator,
    ThermalLimitMonitor,
    ThermalResult,
    create_mosfet_thermal_model,
    create_from_datasheet_4param,
    create_simple_thermal_model,

    # Power Loss Calculation
    MOSFETLossParams,
    IGBTLossParams,
    DiodeLossParams,
    ConductionLoss,
    SwitchingLoss,
    LossBreakdown,
    LossAccumulator,
    EfficiencyCalculator,
    LossResult,
    SystemLossSummary,

    # Frequency-domain analysis (add-frequency-domain-analysis)
    LinearSystem,
    AcSweepScale,
    AcSweepOptions,
    AcMeasurement,
    AcSweepResult,
    FraOptions,
    FraMeasurement,
    FraResult,
)

# Netlist Parser (Pure Python)
from .netlist import (
    parse_netlist,
    parse_netlist_verbose,
    parse_value,
    NetlistParseError,
    NetlistWarning,
    ParsedNetlist,
)

# Signal-Flow Evaluator (Pure Python, no GUI dependency)
from .signal_evaluator import (
    SignalEvaluator,
    AlgebraicLoopError,
    SIGNAL_TYPES,
)

# Frequency-domain analysis plotting helpers
# (matplotlib imported lazily inside the helpers — no required dependency)
from .frequency_analysis import (
    bode_plot,
    nyquist_plot,
    fra_overlay,
    export_ac_csv,
    export_fra_csv,
    export_ac_json,
    export_fra_json,
    load_ac_result_csv,
)

# Converter templates (add-converter-templates)
from . import templates as templates  # noqa: E402  (re-export submodule)

# Real-time code generation (add-realtime-code-generation)
from . import codegen as codegen  # noqa: E402

# Parameter sweep (add-monte-carlo-parameter-sweep)
from . import sweep as sweep  # noqa: E402

# FMI 2.0 Co-Simulation export (add-fmi-export)
from . import fmu as fmu  # noqa: E402


# =============================================================================
# Ergonomic API extension: string node names in Circuit.add_*
# =============================================================================
#
# Pulsim's C++ Circuit API takes integer node IDs (returned by `add_node`).
# That's awkward for higher-level code that wants to reference nodes by
# name (`"in"`, `"out"`, `"sw"`, ...). This wrapper monkey-patches the
# Pulsim Circuit class so every `add_*` method auto-resolves a string
# argument by:
#   1. Looking it up via `circuit.get_node(name)`. If found, use that
#      integer ID.
#   2. Otherwise, calling `circuit.add_node(name)` to create the node and
#      use the returned ID.
#
# Special string `"0"` is treated as ground (no `add_node`, returns
# `circuit.ground()` = -1).
#
# This is purely additive — passing integer IDs continues to work
# untouched, so existing user code is not affected.

def _resolve_node(circuit, node):
    """Resolve a node argument: integer → unchanged; string → looked up
    in the circuit (or auto-added). The string `"0"` always means ground."""
    if isinstance(node, (int, np.integer)):
        return int(node)
    if isinstance(node, str):
        if node in ("0", "gnd", "GND", "ground"):
            return circuit.ground()
        try:
            existing = circuit.get_node(node)
            return existing
        except RuntimeError:
            return circuit.add_node(node)
    raise TypeError(
        f"node argument must be int or str, got {type(node).__name__}"
    )


# Map of (method_name, number_of_leading_node_args).
# Two-terminal: voltage_source, resistor, inductor, capacitor, diode,
#   current_source, snubber_rc, switch, pulse/pwm/sine_voltage_source.
# Three-terminal: vcswitch ([ctrl, A, B]), mosfet/igbt ([gate, drain/coll,
#   source/em]).
# Four-terminal: transformer ([p1, p2, s1, s2]).
_ADD_METHOD_NODE_COUNTS = {
    "add_voltage_source": 2,
    "add_pulse_voltage_source": 2,
    "add_pwm_voltage_source": 2,
    "add_sine_voltage_source": 2,
    "add_current_source": 2,
    "add_resistor": 2,
    "add_inductor": 2,
    "add_capacitor": 2,
    "add_diode": 2,
    "add_snubber_rc": 2,
    "add_switch": 2,
    "add_vcswitch": 3,
    "add_mosfet": 3,
    "add_igbt": 3,
    "add_transformer": 4,
}


class _CircuitWrapper(Circuit):
    """Subclass of the pybind11-bound Circuit that auto-resolves
    string node names in every `add_*` method. pybind11 doesn't permit
    monkey-patching the bound class methods directly, so subclassing
    is the cleanest path. The subclass overrides each known `add_*`
    method by name and forwards to `super().<method>()` after
    converting string args to integer IDs."""
    pass


def _make_string_resolving_method(method_name, n_node_args):
    """Generate an override for `_CircuitWrapper.<method_name>`."""
    raw = getattr(Circuit, method_name)

    def override(self, name, *args, **kwargs):
        if len(args) < n_node_args:
            return raw(self, name, *args, **kwargs)
        nodes = [_resolve_node(self, args[i]) for i in range(n_node_args)]
        rest = args[n_node_args:]
        return raw(self, name, *nodes, *rest, **kwargs)

    override.__name__ = method_name
    override.__qualname__ = f"_CircuitWrapper.{method_name}"
    override.__doc__ = (raw.__doc__ or "") + (
        "\n\nAccepts string node names — they're auto-resolved to integer "
        "IDs (looked up via `get_node` or created via `add_node`). The "
        "string `'0'` (or `'gnd'` / `'GND'` / `'ground'`) always resolves "
        "to ground (-1)."
    )
    return override


for _name, _n_nodes in _ADD_METHOD_NODE_COUNTS.items():
    if hasattr(Circuit, _name):
        setattr(_CircuitWrapper, _name,
                _make_string_resolving_method(_name, _n_nodes))


# Re-export the wrapped class as the public `Circuit`. Existing user
# code that does `from pulsim import Circuit` and `isinstance(c, Circuit)`
# checks continues to work — `_CircuitWrapper` IS-A `Circuit` (subclass).
# Callers that constructed via the C-side `_pulsim.Circuit` directly are
# unaffected.
Circuit = _CircuitWrapper


# =============================================================================
# Legacy API aliases on SimulationOptions
# =============================================================================
#
# Older test/example code uses different attribute names than the
# current C++ binding. Provide aliases so legacy code still works.
#
# Mappings:
#   `opts.use_ic`              → `opts.uic`
#   `opts.dtmin`               → `opts.dt_min`
#   `opts.integration_method`  → `opts.integrator`
#
# Subclass approach (same as Circuit) since pybind11 doesn't permit
# monkey-patching properties on a bound class.

class _SimulationOptionsWrapper(SimulationOptions):
    """Subclass of `SimulationOptions` with legacy attribute aliases."""

    @property
    def use_ic(self):
        return self.uic

    @use_ic.setter
    def use_ic(self, value):
        self.uic = bool(value)

    @property
    def dtmin(self):
        return self.dt_min

    @dtmin.setter
    def dtmin(self, value):
        self.dt_min = float(value)

    @property
    def integration_method(self):
        return self.integrator

    @integration_method.setter
    def integration_method(self, value):
        self.integrator = value


SimulationOptions = _SimulationOptionsWrapper


# Module-level alias: `pulsim.IntegrationMethod` → `pulsim.Integrator`.
# Plus map legacy enum names like `GEAR2` to their current equivalents.
class _IntegrationMethodAlias:
    """Alias for the older `IntegrationMethod` enum name. Legacy names
    map to current `Integrator` values:
      GEAR2 → BDF2  (second-order GEAR is the BDF2 multistep family)
      BACKWARD_EULER → BDF1
      TRAPEZOIDAL → Trapezoidal
    """

    def __getattr__(self, name):
        # Direct passthroughs.
        if hasattr(Integrator, name):
            return getattr(Integrator, name)
        # Legacy aliases.
        legacy_map = {
            "GEAR2": Integrator.BDF2,
            "BACKWARD_EULER": Integrator.BDF1,
            "TRAPEZOIDAL": Integrator.Trapezoidal,
            "TRBDF2": Integrator.TRBDF2,
        }
        if name in legacy_map:
            return legacy_map[name]
        raise AttributeError(
            f"IntegrationMethod has no member {name!r}. "
            f"Use one of {[v for v in dir(Integrator) if not v.startswith('_')]}"
        )


IntegrationMethod = _IntegrationMethodAlias()


_AUTO_BLEEDER_CIRCUITS = weakref.WeakSet()


def _copy_newton_options(source):
    opts = NewtonOptions()
    if source is None:
        return opts

    for name in dir(source):
        if name.startswith("_") or name == "tolerances":
            continue
        value = getattr(source, name)
        if callable(value):
            continue
        try:
            setattr(opts, name, value)
        except Exception:
            # Keep compatibility if bindings change fields across versions
            pass

    if hasattr(source, "tolerances"):
        dst_tol = Tolerances()
        src_tol = source.tolerances
        for name in dir(src_tol):
            if name.startswith("_"):
                continue
            value = getattr(src_tol, name)
            if callable(value):
                continue
            try:
                setattr(dst_tol, name, value)
            except Exception:
                pass
        opts.tolerances = dst_tol

    return opts


def _clone_state_vector(x0):
    if x0 is None:
        return None
    if isinstance(x0, np.ndarray):
        return x0.copy()
    if isinstance(x0, (list, tuple)):
        return list(x0)
    try:
        return np.array(x0, dtype=float, copy=True)
    except Exception:
        return x0


def _is_state_vector(value):
    if value is None:
        return False
    if isinstance(value, (NewtonOptions, LinearSolverStackConfig)):
        return False
    return isinstance(value, (list, tuple, np.ndarray))


def _parse_run_transient_args(args):
    if len(args) > 3:
        raise TypeError(
            "run_transient accepts at most 3 positional extras: "
            "[x0], [newton_options], [linear_solver]"
        )

    x0 = None
    newton_options = None
    linear_solver = None

    if len(args) == 0:
        return x0, newton_options, linear_solver

    first = args[0]
    if isinstance(first, NewtonOptions):
        newton_options = first
        if len(args) >= 2:
            linear_solver = args[1]
    elif isinstance(first, LinearSolverStackConfig):
        linear_solver = first
    elif _is_state_vector(first):
        x0 = first
        if len(args) >= 2:
            newton_options = args[1]
        if len(args) >= 3:
            linear_solver = args[2]
    else:
        raise TypeError(
            "Invalid run_transient positional arguments. Expected x0, "
            "NewtonOptions and/or LinearSolverStackConfig."
        )

    return x0, newton_options, linear_solver


def _run_transient_once(circuit, t_start, t_stop, dt, x0, newton_options, linear_solver):
    if x0 is None:
        return _run_transient_native(
            circuit, t_start, t_stop, dt, newton_options, linear_solver
        )
    return _run_transient_native(
        circuit, t_start, t_stop, dt, x0, newton_options, linear_solver
    )


def _is_retryable_failure(message: str) -> bool:
    text = (message or "").lower()
    tokens = (
        "max iterations",
        "diverg",
        "singular",
        "numerical",
        "transient failed",
        "not finite",
        "nan",
    )
    return any(token in text for token in tokens)


def _tune_linear_solver_for_robust(linear_solver):
    try:
        has_default_order = (
            len(linear_solver.order) == 0
            or (
                len(linear_solver.order) == 1
                and linear_solver.order[0] == LinearSolverKind.SparseLU
            )
        )
        if has_default_order:
            linear_solver.order = [
                LinearSolverKind.KLU,
                LinearSolverKind.EnhancedSparseLU,
                LinearSolverKind.GMRES,
                LinearSolverKind.BiCGSTAB,
            ]

        if len(linear_solver.fallback_order) == 0:
            linear_solver.fallback_order = [
                LinearSolverKind.EnhancedSparseLU,
                LinearSolverKind.SparseLU,
                LinearSolverKind.GMRES,
                LinearSolverKind.BiCGSTAB,
            ]

        linear_solver.allow_fallback = True
        linear_solver.auto_select = True
        linear_solver.size_threshold = min(linear_solver.size_threshold, 1200)
        linear_solver.nnz_threshold = min(linear_solver.nnz_threshold, 120000)
        linear_solver.diag_min_threshold = max(linear_solver.diag_min_threshold, 1e-12)

        it = linear_solver.iterative_config
        it.max_iterations = max(it.max_iterations, 300)
        it.tolerance = min(it.tolerance, 1e-8)
        it.restart = max(it.restart, 40)
        it.enable_scaling = True
        it.scaling_floor = min(it.scaling_floor, 1e-12)
        if it.preconditioner in (PreconditionerKind.None_, PreconditionerKind.Jacobi):
            it.preconditioner = PreconditionerKind.ILUT
        it.ilut_drop_tolerance = min(it.ilut_drop_tolerance, 1e-3)
        it.ilut_fill_factor = max(it.ilut_fill_factor, 10.0)
    except Exception:
        pass


def _tune_newton_for_robust(opts):
    try:
        opts.max_iterations = max(int(opts.max_iterations), 120)
        opts.auto_damping = True
        opts.min_damping = min(float(opts.min_damping), 1e-4)
        opts.enable_limiting = True
        opts.max_voltage_step = max(float(opts.max_voltage_step), 10.0)
        opts.max_current_step = max(float(opts.max_current_step), 20.0)
        opts.enable_trust_region = True
        opts.trust_radius = max(float(opts.trust_radius), 8.0)
        opts.trust_shrink = min(float(opts.trust_shrink), 0.5)
        opts.trust_expand = max(float(opts.trust_expand), 1.5)
        opts.detect_stall = False
    except Exception:
        pass


def _apply_auto_bleeders(circuit, resistance=1e7):
    if circuit in _AUTO_BLEEDER_CIRCUITS:
        return False
    for node_idx in range(circuit.num_nodes()):
        circuit.add_resistor(f"__auto_bleed_n{node_idx}", node_idx, -1, resistance)
    _AUTO_BLEEDER_CIRCUITS.add(circuit)
    return True


def _apply_nonlinear_regularization(circuit):
    if not hasattr(circuit, "apply_numerical_regularization"):
        return 0
    try:
        return int(circuit.apply_numerical_regularization())
    except Exception:
        return 0


def run_transient(
    circuit,
    t_start,
    t_stop,
    dt,
    *args,
    robust=True,
    auto_regularize=True,
):
    """Run transient simulation with automatic retry and stabilization fallback.

    By default, this wrapper keeps the original API behavior and adds automatic
    retry profiles for difficult switching steps. If convergence still fails,
    it can inject tiny high-value bleeder resistors (once per circuit) to
    regularize floating-node situations common with idealized converter models.
    """
    x0, user_newton, user_linear = _parse_run_transient_args(args)

    base_newton = _copy_newton_options(user_newton)
    linear_solver = user_linear if user_linear is not None else LinearSolverStackConfig.defaults()

    if robust:
        _tune_newton_for_robust(base_newton)
        _tune_linear_solver_for_robust(linear_solver)

    attempts = [
        (1.00, max(80, int(base_newton.max_iterations)), False),
        (0.75, max(160, int(base_newton.max_iterations)), auto_regularize),
        (0.50, max(260, int(base_newton.max_iterations)), auto_regularize),
    ]

    last_result = None
    regularized = False

    for idx, (dt_scale, max_iter, apply_regularization) in enumerate(attempts):
        if idx > 0 and not robust:
            break

        if apply_regularization:
            nonlinear_updates = _apply_nonlinear_regularization(circuit)
            bleeders_added = _apply_auto_bleeders(circuit)
            regularized = regularized or bleeders_added or (nonlinear_updates > 0)

        trial_newton = _copy_newton_options(base_newton)
        trial_newton.max_iterations = max_iter

        trial_x0 = _clone_state_vector(x0)
        trial_dt = float(dt) * dt_scale
        last_result = _run_transient_once(
            circuit,
            t_start,
            t_stop,
            trial_dt,
            trial_x0,
            trial_newton,
            linear_solver,
        )

        if last_result[2]:
            return last_result

        if not _is_retryable_failure(last_result[3]):
            return last_result

    if last_result is None:
        raise RuntimeError("run_transient failed before attempting simulation")

    if regularized:
        return (
            last_result[0],
            last_result[1],
            last_result[2],
            f"{last_result[3]} (automatic regularization attempted)",
        )
    return last_result


run_transient_streaming = _run_transient_streaming_native

__all__ = [
    # Version
    "__version__",

    # Enums
    "DeviceType",
    "SolverStatus",
    "DCStrategy",
    "RLCDamping",
    "DeviceHint",
    "SIMDLevel",
    "Integrator",
    "TimestepMethod",
    "StepMode",
    "FormulationMode",
    "SwitchingMode",

    # Device Classes - Linear
    "Resistor",
    "Capacitor",
    "Inductor",
    "VoltageSource",
    "CurrentSource",

    # Device Classes - Nonlinear
    "IdealDiode",
    "IdealSwitch",
    "MOSFETParams",
    "MOSFET",
    "IGBTParams",
    "IGBT",

    # Time-Varying Sources
    "PWMParams",
    "PWMVoltageSource",
    "SineParams",
    "SineVoltageSource",
    "RampParams",
    "RampGenerator",
    "PulseParams",
    "PulseVoltageSource",

    # Control Blocks
    "PIController",
    "PIDController",
    "Comparator",
    "SampleHold",
    "RateLimiter",
    "MovingAverageFilter",
    "HysteresisController",
    "LookupTable1D",

    # Circuit Builder
    "Circuit",
    "VirtualComponent",
    "MixedDomainStepResult",
    "VirtualChannelMetadata",

    # DC Solver
    "solve_dc",
    "dc_operating_point",

    # Transient Simulation
    "run_transient",
    "run_transient_streaming",
    "SimulationOptions",
    "SimulationResult",
    "Simulator",
    "PeriodicSteadyStateOptions",
    "PeriodicSteadyStateResult",
    "HarmonicBalanceOptions",
    "HarmonicBalanceResult",
    "StiffnessConfig",
    "SwitchingEnergy",
    "FallbackReasonCode",
    "FallbackPolicyOptions",
    "FallbackTraceEntry",
    "BackendTelemetry",
    "ThermalCouplingPolicy",
    "ThermalCouplingOptions",
    "ThermalDeviceConfig",
    "DeviceThermalTelemetry",
    "ThermalSummary",
    "ComponentElectrothermalTelemetry",
    "SimulationEventType",
    "SimulationEvent",
    "LinearSolverTelemetry",
    "YamlParserOptions",
    "YamlParser",

    # Solver Configuration
    "Tolerances",
    "NewtonOptions",
    "NewtonResult",

    # Convergence Monitoring
    "IterationRecord",
    "ConvergenceHistory",
    "VariableConvergence",
    "PerVariableConvergence",

    # Convergence Aids
    "GminConfig",
    "SourceSteppingConfig",
    "PseudoTransientConfig",
    "InitializationConfig",
    "DCConvergenceConfig",
    "DCAnalysisResult",

    # Analytical Solutions (Validation)
    "RCAnalytical",
    "RLAnalytical",
    "RLCAnalytical",

    # Validation Framework
    "ValidationResult",
    "compare_waveforms",
    "export_validation_csv",
    "export_validation_json",

    # Benchmark Framework
    "BenchmarkTiming",
    "BenchmarkResult",
    "export_benchmark_csv",
    "export_benchmark_json",

    # Integration Methods
    "BDFOrderConfig",
    "TimestepConfig",
    "AdvancedTimestepConfig",
    "RichardsonLTEConfig",

    # High-Performance Features
    "LinearSolverConfig",
    "LinearSolverKind",
    "PreconditionerKind",
    "IterativeSolverConfig",
    "LinearSolverStackConfig",
    "detect_simd_level",
    "simd_vector_width",
    "backend_capabilities",
    "solver_status_to_string",

    # Thermal Simulation
    "FosterStage",
    "FosterNetwork",
    "CauerStage",
    "CauerNetwork",
    "ThermalSimulator",
    "ThermalLimitMonitor",
    "ThermalResult",
    "create_mosfet_thermal_model",
    "create_from_datasheet_4param",
    "create_simple_thermal_model",

    # Power Loss Calculation
    "MOSFETLossParams",
    "IGBTLossParams",
    "DiodeLossParams",
    "ConductionLoss",
    "SwitchingLoss",
    "LossBreakdown",
    "LossAccumulator",
    "EfficiencyCalculator",
    "LossResult",
    "SystemLossSummary",

    # Netlist Parser
    "parse_netlist",
    "parse_netlist_verbose",
    "parse_value",
    "NetlistParseError",
    "NetlistWarning",
    "ParsedNetlist",

    # Signal-Flow Evaluator
    "SignalEvaluator",
    "AlgebraicLoopError",
    "SIGNAL_TYPES",
]
