"""PulsimCore - High-performance circuit simulator for power electronics.

This is the API with C++23 features and SIMD optimization.
"""

__version__ = "0.2.0"

import weakref
from typing import Tuple

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
    ThermalCouplingPolicy,
    ThermalCouplingOptions,
    ThermalDeviceConfig,
    DeviceThermalTelemetry,
    ThermalSummary,
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
    "ThermalCouplingPolicy",
    "ThermalCouplingOptions",
    "ThermalDeviceConfig",
    "DeviceThermalTelemetry",
    "ThermalSummary",
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
]
