"""PulsimCore - High-performance circuit simulator for power electronics.

This is the API with C++23 features and SIMD optimization.
"""

__version__ = "0.7.3"

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
    ControlUpdateMode,
    FormulationMode,
    FrequencyAnalysisMode,
    FrequencyAnchorMode,
    FrequencySweepScale,
    FrequencyMetricUndefinedReason,
    AveragedConverterTopology,
    AveragedOperatingMode,
    AveragedEnvelopePolicy,
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
    run_frequency_analysis as _run_frequency_analysis_native,
    run_transient_streaming as _run_transient_streaming_native,
    SimulationOptions,
    SimulationResult,
    Simulator,
    PeriodicSteadyStateOptions,
    PeriodicSteadyStateResult,
    HarmonicBalanceOptions,
    HarmonicBalanceResult,
    FrequencyAnalysisPort,
    FrequencyAnalysisOptions,
    FrequencyAnalysisResult,
    AveragedConverterOptions,
    StiffnessConfig,
    SwitchingEnergy,
    SwitchingEnergySurface3D,
    FallbackReasonCode,
    FallbackPolicyOptions,
    FallbackTraceEntry,
    BackendTelemetry,
    ThermalCouplingPolicy,
    ThermalNetworkKind,
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

# Custom C / Python Computation Blocks
from .cblock import (
    CBlockCompileError,
    CBlockABIError,
    CBlockRuntimeError,
    detect_compiler,
    compile_cblock,
    CBlockLibrary,
    PythonCBlock,
)

# Waveform Post-Processing (Pure Python, backend-owned metric pipeline)
from .post_processing import (
    PostProcessingWindowMode,
    PostProcessingJobKind,
    PostProcessingDiagnosticCode,
    WindowFunction,
    PostProcessingWindowSpec,
    PostProcessingJob,
    PostProcessingOptions,
    ScalarMetric,
    SpectralBin,
    HarmonicEntry,
    UndefinedMetricEntry,
    PostProcessingJobResult,
    PostProcessingResult,
    run_post_processing,
    parse_post_processing_yaml,
)


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


def _run_transient_once(
    circuit, t_start, t_stop, dt, x0, newton_options, linear_solver
):
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
        has_default_order = len(linear_solver.order) == 0 or (
            len(linear_solver.order) == 1
            and linear_solver.order[0] == LinearSolverKind.SparseLU
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
    linear_solver = (
        user_linear if user_linear is not None else LinearSolverStackConfig.defaults()
    )

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


_FREQUENCY_DIAGNOSTIC_REASON_CODES = {
    "FrequencyInvalidConfiguration": "frequency_invalid_configuration",
    "FrequencyUnsupportedConfiguration": "frequency_unsupported_configuration",
    "FrequencySolverFailure": "frequency_solver_failure",
}


def _enum_name(value):
    if value is None:
        return "Unknown"
    return getattr(value, "name", str(value))


class FrequencyAnalysisError(RuntimeError):
    """Structured exception for failed frequency-domain analysis runs."""

    def __init__(
        self,
        *,
        diagnostic,
        reason_code,
        message,
        failed_point_index,
        failed_frequency_hz,
        mode,
        anchor_mode_selected,
    ):
        summary = f"{reason_code}: {message}" if reason_code else message
        super().__init__(summary)
        self.diagnostic = diagnostic
        self.reason_code = reason_code
        self.message = message
        self.failed_point_index = int(failed_point_index)
        self.failed_frequency_hz = float(failed_frequency_hz)
        self.mode = mode
        self.anchor_mode_selected = anchor_mode_selected

    @classmethod
    def from_result(cls, result):
        diagnostic = getattr(result, "diagnostic", None)
        diagnostic_name = _enum_name(diagnostic)
        reason_code = _FREQUENCY_DIAGNOSTIC_REASON_CODES.get(
            diagnostic_name, diagnostic_name.lower()
        )
        return cls(
            diagnostic=diagnostic,
            reason_code=reason_code,
            message=str(getattr(result, "message", "")),
            failed_point_index=int(getattr(result, "failed_point_index", -1)),
            failed_frequency_hz=float(
                getattr(result, "failed_frequency_hz", float("nan"))
            ),
            mode=getattr(result, "mode", None),
            anchor_mode_selected=getattr(result, "anchor_mode_selected", None),
        )


def run_frequency_analysis(circuit, options, *, raise_on_failure=False):
    """Run frequency-domain AC sweep using the procedural API.

    Args:
        circuit: Pulsim circuit instance.
        options: Frequency analysis configuration.
        raise_on_failure: When true, raise ``FrequencyAnalysisError`` on
            deterministic kernel failures instead of returning ``success=False``.
    """
    result = _run_frequency_analysis_native(circuit, options)
    if raise_on_failure and not bool(getattr(result, "success", False)):
        raise FrequencyAnalysisError.from_result(result)
    return result


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
    "ControlUpdateMode",
    "FormulationMode",
    "FrequencyAnalysisMode",
    "FrequencyAnchorMode",
    "FrequencySweepScale",
    "FrequencyMetricUndefinedReason",
    "AveragedConverterTopology",
    "AveragedOperatingMode",
    "AveragedEnvelopePolicy",
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
    "run_frequency_analysis",
    "run_transient_streaming",
    "FrequencyAnalysisError",
    "SimulationOptions",
    "SimulationResult",
    "Simulator",
    "PeriodicSteadyStateOptions",
    "PeriodicSteadyStateResult",
    "HarmonicBalanceOptions",
    "HarmonicBalanceResult",
    "FrequencyAnalysisPort",
    "FrequencyAnalysisOptions",
    "FrequencyAnalysisResult",
    "AveragedConverterOptions",
    "StiffnessConfig",
    "SwitchingEnergy",
    "SwitchingEnergySurface3D",
    "FallbackReasonCode",
    "FallbackPolicyOptions",
    "FallbackTraceEntry",
    "BackendTelemetry",
    "ThermalCouplingPolicy",
    "ThermalNetworkKind",
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
    # Custom C / Python Computation Blocks
    "CBlockCompileError",
    "CBlockABIError",
    "CBlockRuntimeError",
    "detect_compiler",
    "compile_cblock",
    "CBlockLibrary",
    "PythonCBlock",
    # Waveform Post-Processing
    "PostProcessingWindowMode",
    "PostProcessingJobKind",
    "PostProcessingDiagnosticCode",
    "WindowFunction",
    "PostProcessingWindowSpec",
    "PostProcessingJob",
    "PostProcessingOptions",
    "ScalarMetric",
    "SpectralBin",
    "HarmonicEntry",
    "UndefinedMetricEntry",
    "PostProcessingJobResult",
    "PostProcessingResult",
    "run_post_processing",
    "parse_post_processing_yaml",
]

# ---------------------------------------------------------------------------
# Module-level capabilities registry
# ---------------------------------------------------------------------------

capabilities: dict[str, bool] = {
    "c_block": True,
}
