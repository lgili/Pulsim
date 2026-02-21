"""PulsimCore - High-performance circuit simulator for power electronics.

This is the API with C++23 features and SIMD optimization.
"""

__version__ = "0.2.0"

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
    run_transient,
    run_transient_streaming,
    SimulationOptions,
    SimulationResult,
    Simulator,
    PeriodicSteadyStateOptions,
    PeriodicSteadyStateResult,
    HarmonicBalanceOptions,
    HarmonicBalanceResult,
    StiffnessConfig,
    SwitchingEnergy,
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
