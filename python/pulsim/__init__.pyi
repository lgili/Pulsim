"""Type stubs for PulsimCore High-Performance API."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from enum import Enum

__version__: str

# =============================================================================
# Enums
# =============================================================================

class DeviceType(Enum):
    Resistor = ...
    Capacitor = ...
    Inductor = ...
    VoltageSource = ...
    CurrentSource = ...
    Diode = ...
    Switch = ...
    MOSFET = ...
    IGBT = ...
    Transformer = ...

class SolverStatus(Enum):
    Success = ...
    MaxIterationsReached = ...
    SingularMatrix = ...
    NumericalError = ...
    ConvergenceStall = ...
    Diverging = ...

class DCStrategy(Enum):
    Direct = ...
    GminStepping = ...
    SourceStepping = ...
    PseudoTransient = ...
    Auto = ...

class RLCDamping(Enum):
    Underdamped = ...
    Critical = ...
    Overdamped = ...

class DeviceHint(Enum):
    None_ = ...
    DiodeAnode = ...
    DiodeCathode = ...
    MOSFETGate = ...
    MOSFETDrain = ...
    MOSFETSource = ...
    BJTBase = ...
    BJTCollector = ...
    BJTEmitter = ...
    SupplyPositive = ...
    SupplyNegative = ...
    Ground = ...

class SIMDLevel(Enum):
    None_ = ...
    SSE2 = ...
    SSE4 = ...
    AVX = ...
    AVX2 = ...
    AVX512 = ...
    NEON = ...

class Integrator(Enum):
    Trapezoidal = ...
    BDF1 = ...
    BDF2 = ...
    BDF3 = ...
    BDF4 = ...
    BDF5 = ...
    Gear = ...
    TRBDF2 = ...
    RosenbrockW = ...
    SDIRK2 = ...

class TimestepMethod(Enum):
    StepDoubling = ...
    Richardson = ...

class StepMode(Enum):
    Fixed = ...
    Variable = ...

class ControlUpdateMode(Enum):
    Auto = ...
    Continuous = ...
    Discrete = ...

class FormulationMode(Enum):
    ProjectedWrapper = ...
    Direct = ...

class FrequencyAnalysisMode(Enum):
    OpenLoopTransfer = ...
    ClosedLoopTransfer = ...
    InputImpedance = ...
    OutputImpedance = ...

class FrequencyAnchorMode(Enum):
    DC = ...
    Periodic = ...
    Averaged = ...
    Auto = ...

class FrequencySweepScale(Enum):
    Logarithmic = ...
    Linear = ...

class FrequencyMetricUndefinedReason(Enum):
    None_ = ...
    NotTransferMode = ...
    NoGainCrossover = ...
    NoPhaseCrossover = ...

class AveragedConverterTopology(Enum):
    Buck = ...
    Boost = ...
    BuckBoost = ...

class AveragedOperatingMode(Enum):
    CCM = ...
    DCM = ...
    Auto = ...

class AveragedEnvelopePolicy(Enum):
    Strict = ...
    Warn = ...

# =============================================================================
# Device Classes
# =============================================================================

class Circuit:
    def __init__(self) -> None: ...
    def add_node(self, name: str) -> int: ...
    def get_node(self, name: str) -> int: ...
    @staticmethod
    def ground() -> int: ...
    def num_nodes(self) -> int: ...
    def num_branches(self) -> int: ...
    def system_size(self) -> int: ...
    def node_name(self, index: int) -> str: ...
    def node_names(self) -> List[str]: ...
    def signal_names(self) -> List[str]: ...
    def initial_state(self) -> List[float]: ...
    def num_devices(self) -> int: ...
    def add_virtual_component(
        self,
        type: str,
        name: str,
        nodes: List[int],
        numeric_params: Dict[str, float] = ...,
        metadata: Dict[str, str] = ...,
    ) -> None: ...
    def num_virtual_components(self) -> int: ...
    def virtual_components(self) -> List["VirtualComponent"]: ...
    def virtual_component_names(self) -> List[str]: ...
    def virtual_channel_metadata(self) -> Dict[str, "VirtualChannelMetadata"]: ...
    @staticmethod
    def mixed_domain_phase_order() -> List[str]: ...
    def execute_mixed_domain_step(self, x: List[float], time: float) -> "MixedDomainStepResult": ...
    def evaluate_virtual_signals(self, x: List[float]) -> Dict[str, float]: ...
    def set_timestep(self, dt: float) -> None: ...
    def timestep(self) -> float: ...

class VirtualComponent:
    type: str
    name: str
    nodes: List[int]
    numeric_params: Dict[str, float]
    metadata: Dict[str, str]

    def __init__(self) -> None: ...

class MixedDomainStepResult:
    phase_order: List[str]
    channel_values: Dict[str, float]

    def __init__(self) -> None: ...

class VirtualChannelMetadata:
    component_type: str
    component_name: str
    source_component: str
    domain: str
    unit: str
    nodes: List[int]

    def __init__(self) -> None: ...

class Resistor:
    def __init__(self, resistance: float, name: str = "") -> None: ...
    def resistance(self) -> float: ...
    def name(self) -> str: ...

class Capacitor:
    def __init__(self, capacitance: float, initial_voltage: float = 0.0, name: str = "") -> None: ...
    def capacitance(self) -> float: ...
    def name(self) -> str: ...
    def set_timestep(self, dt: float) -> None: ...

class Inductor:
    def __init__(self, inductance: float, initial_current: float = 0.0, name: str = "") -> None: ...
    def inductance(self) -> float: ...
    def name(self) -> str: ...
    def set_timestep(self, dt: float) -> None: ...

class VoltageSource:
    def __init__(self, voltage: float, name: str = "") -> None: ...
    def voltage(self) -> float: ...
    def name(self) -> str: ...

class CurrentSource:
    def __init__(self, current: float, name: str = "") -> None: ...
    def current(self) -> float: ...
    def name(self) -> str: ...

# =============================================================================
# Solver Configuration
# =============================================================================

class Tolerances:
    voltage_abstol: float
    voltage_reltol: float
    current_abstol: float
    current_reltol: float
    residual_tol: float

    def __init__(self) -> None: ...
    @staticmethod
    def defaults() -> Tolerances: ...

class NewtonOptions:
    max_iterations: int
    initial_damping: float
    min_damping: float
    auto_damping: bool
    track_history: bool
    check_per_variable: bool
    num_nodes: int
    num_branches: int
    tolerances: Tolerances

    def __init__(self) -> None: ...

class NewtonResult:
    solution: List[float]
    status: SolverStatus
    iterations: int
    final_residual: float
    final_weighted_error: float
    error_message: str

    def __init__(self) -> None: ...
    def success(self) -> bool: ...

# =============================================================================
# Convergence Aids
# =============================================================================

class GminConfig:
    initial_gmin: float
    final_gmin: float
    reduction_factor: float
    max_steps: int
    enable_logging: bool

    def __init__(self) -> None: ...
    def required_steps(self) -> int: ...

class SourceSteppingConfig:
    initial_scale: float
    final_scale: float
    initial_step: float
    min_step: float
    max_step: float
    max_steps: int
    max_failures: int
    enable_logging: bool

    def __init__(self) -> None: ...

class PseudoTransientConfig:
    initial_dt: float
    max_dt: float
    min_dt: float
    dt_increase: float
    dt_decrease: float
    convergence_threshold: float
    max_iterations: int
    enable_logging: bool

    def __init__(self) -> None: ...

class InitializationConfig:
    default_voltage: float
    supply_voltage: float
    diode_forward: float
    mosfet_threshold: float
    use_zero_init: bool
    use_warm_start: bool
    max_random_restarts: int
    random_seed: int
    random_voltage_range: float

    def __init__(self) -> None: ...

class DCConvergenceConfig:
    strategy: DCStrategy
    gmin_config: GminConfig
    source_config: SourceSteppingConfig
    pseudo_config: PseudoTransientConfig
    init_config: InitializationConfig
    enable_random_restart: bool
    max_strategy_attempts: int

    def __init__(self) -> None: ...

class DCAnalysisResult:
    newton_result: NewtonResult
    strategy_used: DCStrategy
    random_restarts: int
    total_newton_iterations: int
    success: bool
    message: str

    def __init__(self) -> None: ...

class StiffnessConfig:
    enable: bool
    rejection_streak_threshold: int
    newton_iter_threshold: int
    newton_streak_threshold: int
    cooldown_steps: int
    dt_backoff: float
    max_bdf_order: int
    monitor_conditioning: bool
    conditioning_error_threshold: float
    switch_integrator: bool
    stiff_integrator: Integrator

    def __init__(self) -> None: ...

class PeriodicSteadyStateOptions:
    period: float
    max_iterations: int
    tolerance: float
    relaxation: float
    store_last_transient: bool

    def __init__(self) -> None: ...

class HarmonicBalanceOptions:
    period: float
    num_samples: int
    max_iterations: int
    tolerance: float
    relaxation: float
    initialize_from_transient: bool

    def __init__(self) -> None: ...

class FrequencyAnalysisPort:
    positive_node: str
    negative_node: str

    def __init__(self) -> None: ...

class FrequencyAnalysisOptions:
    enabled: bool
    mode: FrequencyAnalysisMode
    anchor_mode: FrequencyAnchorMode
    sweep_scale: FrequencySweepScale
    f_start_hz: float
    f_stop_hz: float
    points: int
    injection_current_amplitude: float
    perturbation_port: FrequencyAnalysisPort
    output_port: FrequencyAnalysisPort

    def __init__(self) -> None: ...

class AveragedConverterOptions:
    enabled: bool
    topology: AveragedConverterTopology
    operating_mode: AveragedOperatingMode
    envelope_policy: AveragedEnvelopePolicy
    vin_source: str
    inductor: str
    capacitor: str
    load_resistor: str
    output_node: str
    duty: float
    duty_min: float
    duty_max: float
    switching_frequency_hz: float
    initial_inductor_current: float
    initial_output_voltage: float
    ccm_current_threshold: float

    def __init__(self) -> None: ...

class SwitchingEnergy:
    eon: float
    eoff: float
    err: float

    def __init__(self) -> None: ...

class SwitchingEnergySurface3D:
    current_axis: List[float]
    voltage_axis: List[float]
    temperature_axis: List[float]
    eon_table: List[float]
    eoff_table: List[float]
    err_table: List[float]

    def __init__(self) -> None: ...
    def has_eon(self) -> bool: ...
    def has_eoff(self) -> bool: ...
    def has_err(self) -> bool: ...
    def expected_table_size(self) -> int: ...
    def valid_shape(self) -> bool: ...
    def evaluate_eon(self, current: float, voltage: float, temperature: float) -> float: ...
    def evaluate_eoff(self, current: float, voltage: float, temperature: float) -> float: ...
    def evaluate_err(self, current: float, voltage: float, temperature: float) -> float: ...

class LinearSolverKind(Enum):
    SparseLU = ...
    EnhancedSparseLU = ...
    KLU = ...
    GMRES = ...
    BiCGSTAB = ...
    CG = ...

class PreconditionerKind(Enum):
    None_ = ...
    Jacobi = ...
    ILU0 = ...
    ILUT = ...
    AMG = ...

class IterativeSolverConfig:
    max_iterations: int
    tolerance: float
    restart: int
    preconditioner: PreconditionerKind
    enable_scaling: bool
    scaling_floor: float
    ilut_drop_tolerance: float
    ilut_fill_factor: float

    def __init__(self) -> None: ...
    @staticmethod
    def defaults() -> IterativeSolverConfig: ...

class LinearSolverStackConfig:
    order: List[LinearSolverKind]
    fallback_order: List[LinearSolverKind]
    direct_config: LinearSolverConfig
    iterative_config: IterativeSolverConfig
    allow_fallback: bool
    auto_select: bool
    size_threshold: int
    nnz_threshold: int
    diag_min_threshold: float

    def __init__(self) -> None: ...
    @staticmethod
    def defaults() -> LinearSolverStackConfig: ...

class LinearSolverTelemetry:
    total_solve_calls: int
    total_analyze_calls: int
    total_factorize_calls: int
    total_iterations: int
    total_fallbacks: int
    last_iterations: int
    last_error: float
    total_analyze_time_seconds: float
    total_factorize_time_seconds: float
    total_solve_time_seconds: float
    last_analyze_time_seconds: float
    last_factorize_time_seconds: float
    last_solve_time_seconds: float
    last_solver: Optional[LinearSolverKind]
    last_preconditioner: Optional[PreconditionerKind]

    def __init__(self) -> None: ...

class SimulationEventType(Enum):
    SwitchOn = ...
    SwitchOff = ...
    ConvergenceWarning = ...
    TimestepChange = ...

class FallbackReasonCode(Enum):
    NewtonFailure = ...
    LTERejection = ...
    EventSplit = ...
    StiffnessBackoff = ...
    TransientGminEscalation = ...
    MaxRetriesExceeded = ...

class ThermalCouplingPolicy(Enum):
    LossOnly = ...
    LossWithTemperatureScaling = ...

class ThermalNetworkKind(Enum):
    SingleRC = ...
    Foster = ...
    Cauer = ...

class SimulationEvent:
    time: float
    type: SimulationEventType
    component: str
    description: str
    value1: float
    value2: float

    def __init__(self) -> None: ...

class FallbackPolicyOptions:
    trace_retries: bool
    enable_transient_gmin: bool
    gmin_retry_threshold: int
    gmin_initial: float
    gmin_max: float
    gmin_growth: float

    def __init__(self) -> None: ...

class BackendTelemetry:
    requested_backend: str
    selected_backend: str
    solver_family: str
    formulation_mode: str
    function_evaluations: int
    jacobian_evaluations: int
    nonlinear_iterations: int
    nonlinear_convergence_failures: int
    error_test_failures: int
    escalation_count: int
    reinitialization_count: int
    backend_recovery_count: int
    state_space_primary_steps: int
    dae_fallback_steps: int
    segment_non_admissible_steps: int
    segment_model_cache_hits: int
    segment_model_cache_misses: int
    linear_factor_cache_hits: int
    linear_factor_cache_misses: int
    equation_assemble_system_calls: int
    equation_assemble_residual_calls: int
    equation_assemble_system_time_seconds: float
    equation_assemble_residual_time_seconds: float
    failure_reason: str

    def __init__(self) -> None: ...

class FallbackTraceEntry:
    step_index: int
    retry_index: int
    time: float
    dt: float
    reason: FallbackReasonCode
    solver_status: SolverStatus
    action: str

    def __init__(self) -> None: ...

class ThermalCouplingOptions:
    enable: bool
    ambient: float
    policy: ThermalCouplingPolicy
    default_rth: float
    default_cth: float

    def __init__(self) -> None: ...

class ThermalDeviceConfig:
    enabled: bool
    network_kind: ThermalNetworkKind
    rth: float
    cth: float
    stage_rth: List[float]
    stage_cth: List[float]
    temp_init: float
    temp_ref: float
    alpha: float
    shared_sink_id: str
    shared_sink_rth: float
    shared_sink_cth: float

    def __init__(self) -> None: ...

class DeviceThermalTelemetry:
    device_name: str
    enabled: bool
    final_temperature: float
    peak_temperature: float
    average_temperature: float

    def __init__(self) -> None: ...

class ThermalSummary:
    enabled: bool
    ambient: float
    max_temperature: float
    device_temperatures: List[DeviceThermalTelemetry]

    def __init__(self) -> None: ...

class ComponentElectrothermalTelemetry:
    component_name: str
    thermal_enabled: bool
    conduction: float
    turn_on: float
    turn_off: float
    reverse_recovery: float
    total_loss: float
    total_energy: float
    average_power: float
    peak_power: float
    final_temperature: float
    peak_temperature: float
    average_temperature: float

    def __init__(self) -> None: ...

class SystemLossSummary:
    device_losses: Dict[str, object]
    total_loss: float
    total_conduction: float
    total_switching: float
    input_power: float
    output_power: float
    efficiency: float

    def __init__(self) -> None: ...
    def compute_totals(self) -> None: ...

class SimulationOptions:
    tstart: float
    tstop: float
    dt: float
    dt_min: float
    dt_max: float
    newton_options: NewtonOptions
    dc_config: DCConvergenceConfig
    linear_solver: LinearSolverStackConfig
    adaptive_timestep: bool
    step_mode: StepMode
    control_mode: ControlUpdateMode
    control_sample_time: float
    formulation_mode: FormulationMode
    direct_formulation_fallback: bool
    timestep_config: AdvancedTimestepConfig
    lte_config: RichardsonLTEConfig
    integrator: Integrator
    enable_bdf_order_control: bool
    bdf_config: BDFOrderConfig
    stiffness_config: StiffnessConfig
    enable_periodic_shooting: bool
    periodic_options: PeriodicSteadyStateOptions
    enable_harmonic_balance: bool
    harmonic_balance: HarmonicBalanceOptions
    frequency_analysis: FrequencyAnalysisOptions
    averaged_converter: AveragedConverterOptions
    enable_events: bool
    enable_losses: bool
    switching_energy: Dict[str, SwitchingEnergy]
    switching_energy_surfaces: Dict[str, SwitchingEnergySurface3D]
    thermal: ThermalCouplingOptions
    thermal_devices: Dict[str, ThermalDeviceConfig]
    gmin_fallback: GminConfig
    max_step_retries: int
    fallback_policy: FallbackPolicyOptions

    def __init__(self) -> None: ...

class SimulationResult:
    time: List[float]
    states: List[List[float]]
    events: List[SimulationEvent]
    mixed_domain_phase_order: List[str]
    virtual_channels: Dict[str, List[float]]
    virtual_channel_metadata: Dict[str, VirtualChannelMetadata]
    success: bool
    final_status: SolverStatus
    message: str
    total_steps: int
    newton_iterations_total: int
    timestep_rejections: int
    total_time_seconds: float
    linear_solver_telemetry: LinearSolverTelemetry
    fallback_trace: List[FallbackTraceEntry]
    backend_telemetry: BackendTelemetry
    loss_summary: SystemLossSummary
    thermal_summary: ThermalSummary
    component_electrothermal: List[ComponentElectrothermalTelemetry]
    data: List[List[float]]

    def __init__(self) -> None: ...

class PeriodicSteadyStateResult:
    success: bool
    iterations: int
    residual_norm: float
    steady_state: List[float]
    last_cycle: SimulationResult
    message: str

    def __init__(self) -> None: ...

class HarmonicBalanceResult:
    success: bool
    iterations: int
    residual_norm: float
    solution: List[float]
    sample_times: List[float]
    message: str

    def __init__(self) -> None: ...

class FrequencyAnalysisResult:
    success: bool
    diagnostic: object
    message: str
    mode: FrequencyAnalysisMode
    anchor_mode_selected: FrequencyAnchorMode
    failed_point_index: int
    failed_frequency_hz: float
    frequency_hz: List[float]
    response_real: List[float]
    response_imag: List[float]
    magnitude: List[float]
    magnitude_db: List[float]
    phase_deg: List[float]
    channel_metadata: Dict[str, VirtualChannelMetadata]
    gain_crossover_hz: float
    phase_crossover_hz: float
    phase_margin_deg: float
    gain_margin_db: float
    gain_crossover_reason: FrequencyMetricUndefinedReason
    phase_crossover_reason: FrequencyMetricUndefinedReason
    phase_margin_reason: FrequencyMetricUndefinedReason
    gain_margin_reason: FrequencyMetricUndefinedReason

    def __init__(self) -> None: ...

class FrequencyAnalysisError(RuntimeError):
    diagnostic: object
    reason_code: str
    message: str
    failed_point_index: int
    failed_frequency_hz: float
    mode: FrequencyAnalysisMode
    anchor_mode_selected: FrequencyAnchorMode

class Simulator:
    options: SimulationOptions

    def __init__(self, circuit: Circuit, options: SimulationOptions = ...) -> None: ...
    def dc_operating_point(self) -> DCAnalysisResult: ...
    def run_transient(self, x0: Optional[List[float]] = ...) -> SimulationResult: ...
    def run_periodic_shooting(
        self,
        x0_or_options: Optional[object] = ...,
        options: Optional[PeriodicSteadyStateOptions] = ...,
    ) -> PeriodicSteadyStateResult: ...
    def run_harmonic_balance(
        self,
        x0_or_options: Optional[object] = ...,
        options: Optional[HarmonicBalanceOptions] = ...,
    ) -> HarmonicBalanceResult: ...
    def run_frequency_analysis(self, options: Optional[FrequencyAnalysisOptions] = ...) -> FrequencyAnalysisResult: ...
    def set_switching_energy(self, device_name: str, energy: SwitchingEnergy) -> None: ...
    def set_switching_energy_surface(self, device_name: str, surface: SwitchingEnergySurface3D) -> None: ...

class YamlParserOptions:
    strict: bool
    validate_nodes: bool

    def __init__(self) -> None: ...

class YamlParser:
    errors: List[str]
    warnings: List[str]

    def __init__(self, options: YamlParserOptions = ...) -> None: ...
    def load(self, path: str) -> Tuple[Circuit, SimulationOptions]: ...
    def load_string(self, content: str) -> Tuple[Circuit, SimulationOptions]: ...

# =============================================================================
# C-Block API
# =============================================================================

class CBlockCompileError(RuntimeError):
    source: str
    stderr_output: str
    compiler_path: Optional[str]

class CBlockABIError(RuntimeError):
    expected_version: int
    found_version: Optional[int]

class CBlockRuntimeError(RuntimeError):
    return_code: int
    t: float
    step_index: int

def detect_compiler() -> Optional[str]: ...
def compile_cblock(
    source: str | Path,
    *,
    output_dir: str | Path | None = ...,
    name: str = ...,
    extra_cflags: List[str] | None = ...,
    compiler: str | None = ...,
) -> Path: ...

class CBlockLibrary:
    @property
    def n_inputs(self) -> int: ...
    @property
    def n_outputs(self) -> int: ...
    @property
    def name(self) -> str: ...

    def __init__(
        self,
        lib_path: str | Path,
        n_inputs: int = ...,
        n_outputs: int = ...,
        name: str = ...,
    ) -> None: ...
    def step(self, t: float, dt: float, inputs: Sequence[float]) -> List[float]: ...
    def reset(self) -> None: ...
    def __enter__(self) -> "CBlockLibrary": ...
    def __exit__(self, *args: Any) -> None: ...

class PythonCBlock:
    @property
    def n_inputs(self) -> int: ...
    @property
    def n_outputs(self) -> int: ...
    @property
    def name(self) -> str: ...

    def __init__(
        self,
        fn: Callable[..., Any],
        n_inputs: int = ...,
        n_outputs: int = ...,
        name: str = ...,
    ) -> None: ...
    def step(self, t: float, dt: float, inputs: Sequence[float]) -> List[float]: ...
    def reset(self) -> None: ...

capabilities: Dict[str, bool]

# =============================================================================
# Validation Framework
# =============================================================================

class RCAnalytical:
    def __init__(self, R: float, C: float, V_initial: float, V_final: float) -> None: ...
    def tau(self) -> float: ...
    def voltage(self, t: float) -> float: ...
    def current(self, t: float) -> float: ...
    def waveform(self, t_start: float, t_end: float, dt: float) -> List[float]: ...

class RLAnalytical:
    def __init__(self, R: float, L: float, V_source: float, I_initial: float) -> None: ...
    def tau(self) -> float: ...
    def I_final(self) -> float: ...
    def current(self, t: float) -> float: ...
    def voltage_R(self, t: float) -> float: ...
    def voltage_L(self, t: float) -> float: ...
    def waveform(self, t_start: float, t_end: float, dt: float) -> List[float]: ...

class RLCAnalytical:
    def __init__(self, R: float, L: float, C: float, V_source: float, V_initial: float, I_initial: float) -> None: ...
    def omega_0(self) -> float: ...
    def zeta(self) -> float: ...
    def alpha(self) -> float: ...
    def damping_type(self) -> RLCDamping: ...
    def voltage(self, t: float) -> float: ...
    def current(self, t: float) -> float: ...
    def waveform(self, t_start: float, t_end: float, dt: float) -> List[float]: ...

class ValidationResult_v2:
    test_name: str
    passed: bool
    max_error: float
    rms_error: float
    max_relative_error: float
    mean_error: float
    num_points: int
    error_threshold: float

    def __init__(self) -> None: ...
    def to_string(self) -> str: ...

def compare_waveforms(name: str, simulated: List[tuple[float, float]], analytical: List[tuple[float, float]], threshold: float = 0.001) -> ValidationResult_v2: ...
def export_validation_csv(results: List[ValidationResult_v2]) -> str: ...
def export_validation_json(results: List[ValidationResult_v2]) -> str: ...

# =============================================================================
# Benchmark Framework
# =============================================================================

class BenchmarkTiming:
    name: str
    iterations: int

    def __init__(self) -> None: ...
    def average_ms(self) -> float: ...
    def min_ms(self) -> float: ...
    def max_ms(self) -> float: ...
    def total_ms(self) -> float: ...

class BenchmarkResult:
    circuit_name: str
    num_nodes: int
    num_devices: int
    num_timesteps: int
    timing: BenchmarkTiming
    simulation_time: float

    def __init__(self) -> None: ...
    def timesteps_per_second(self) -> float: ...
    def to_string(self) -> str: ...

def export_benchmark_csv(results: List[BenchmarkResult]) -> str: ...
def export_benchmark_json(results: List[BenchmarkResult]) -> str: ...

# =============================================================================
# Integration Methods
# =============================================================================

class BDFOrderConfig:
    min_order: int
    max_order: int
    initial_order: int
    order_increase_threshold: float
    order_decrease_threshold: float
    steps_before_increase: int
    enable_auto_order: bool

    def __init__(self) -> None: ...

class TimestepConfig:
    dt_min: float
    dt_max: float
    dt_initial: float
    safety_factor: float
    error_tolerance: float
    growth_factor: float
    shrink_factor: float
    max_rejections: int
    k_p: float
    k_i: float

    def __init__(self) -> None: ...
    @staticmethod
    def defaults() -> TimestepConfig: ...
    @staticmethod
    def conservative() -> TimestepConfig: ...
    @staticmethod
    def aggressive() -> TimestepConfig: ...

class AdvancedTimestepConfig(TimestepConfig):
    target_newton_iterations: int
    min_newton_iterations: int
    max_newton_iterations: int
    newton_feedback_gain: float
    max_growth_rate: float
    max_shrink_rate: float
    enable_smoothing: bool
    lte_weight: float
    newton_weight: float

    def __init__(self) -> None: ...
    @staticmethod
    def defaults() -> AdvancedTimestepConfig: ...
    @staticmethod
    def for_switching() -> AdvancedTimestepConfig: ...
    @staticmethod
    def for_power_electronics() -> AdvancedTimestepConfig: ...

class RichardsonLTEConfig:
    method: TimestepMethod
    extrapolation_order: int
    voltage_tolerance: float
    current_tolerance: float
    use_weighted_norm: bool
    history_depth: int

    def __init__(self) -> None: ...
    @staticmethod
    def defaults() -> RichardsonLTEConfig: ...
    @staticmethod
    def step_doubling() -> RichardsonLTEConfig: ...

# =============================================================================
# High-Performance Features
# =============================================================================

class LinearSolverConfig:
    pivot_tolerance: float
    reuse_symbolic: bool
    detect_pattern_change: bool
    deterministic_pivoting: bool

    def __init__(self) -> None: ...

def detect_simd_level() -> SIMDLevel: ...
def simd_vector_width() -> int: ...
def backend_capabilities() -> Dict[str, bool]: ...
def solver_status_to_string(status: SolverStatus) -> str: ...
def collect_result_channels(
    circuit: Circuit,
    result: SimulationResult,
    *,
    include_states: bool = ...,
    include_virtual: bool = ...,
) -> Tuple[Dict[str, List[float]], Dict[str, VirtualChannelMetadata]]: ...
def run_frequency_analysis(
    circuit: Circuit,
    options: FrequencyAnalysisOptions,
    *,
    raise_on_failure: bool = ...,
) -> FrequencyAnalysisResult: ...
