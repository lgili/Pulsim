# Pulsim Python API Inventory

## Table of Contents
- [Refresh](#refresh)
- [Summary](#summary)
- [Enums](#enums)
- [Classes](#classes)
- [Top-Level Functions](#top-level-functions)
- [Export Gaps](#export-gaps)

## Refresh

Generated on: `2026-02-21T18:33:44.587200+00:00`
- Export source: `/Users/lgili/Documents/01 - Codes/01 - Github/PulsimCore/python/pulsim/__init__.py`
- Type surface source: `/Users/lgili/Documents/01 - Codes/01 - Github/PulsimCore/python/pulsim/__init__.pyi`

Regenerate with:

```bash
python3 skills/pulsim-library-expert/scripts/build_api_inventory.py
```

## Summary

- Exported symbols (`__all__`): **125**
- Enums in stub: **11**
- Classes in stub: **44**
- Top-level functions in stub: **8**
- Top-level variables in stub: **1**

## Enums

### DCStrategy

- `Direct`
- `GminStepping`
- `SourceStepping`
- `PseudoTransient`
- `Auto`

### DeviceHint

- `None_`
- `DiodeAnode`
- `DiodeCathode`
- `MOSFETGate`
- `MOSFETDrain`
- `MOSFETSource`
- `BJTBase`
- `BJTCollector`
- `BJTEmitter`
- `SupplyPositive`
- `SupplyNegative`
- `Ground`

### DeviceType

- `Resistor`
- `Capacitor`
- `Inductor`
- `VoltageSource`
- `CurrentSource`
- `Diode`
- `Switch`
- `MOSFET`
- `IGBT`
- `Transformer`

### FallbackReasonCode

- `NewtonFailure`
- `LTERejection`
- `EventSplit`
- `StiffnessBackoff`
- `TransientGminEscalation`
- `MaxRetriesExceeded`

### Integrator

- `Trapezoidal`
- `BDF1`
- `BDF2`
- `BDF3`
- `BDF4`
- `BDF5`
- `Gear`
- `TRBDF2`
- `RosenbrockW`
- `SDIRK2`

### RLCDamping

- `Underdamped`
- `Critical`
- `Overdamped`

### SIMDLevel

- `None_`
- `SSE2`
- `SSE4`
- `AVX`
- `AVX2`
- `AVX512`
- `NEON`

### SimulationEventType

- `SwitchOn`
- `SwitchOff`
- `ConvergenceWarning`
- `TimestepChange`

### SolverStatus

- `Success`
- `MaxIterationsReached`
- `SingularMatrix`
- `NumericalError`
- `ConvergenceStall`
- `Diverging`

### ThermalCouplingPolicy

- `LossOnly`
- `LossWithTemperatureScaling`

### TimestepMethod

- `StepDoubling`
- `Richardson`

## Classes

### AdvancedTimestepConfig (bases: TimestepConfig)

Attributes:
- `target_newton_iterations: int`
- `min_newton_iterations: int`
- `max_newton_iterations: int`
- `newton_feedback_gain: float`
- `max_growth_rate: float`
- `max_shrink_rate: float`
- `enable_smoothing: bool`
- `lte_weight: float`
- `newton_weight: float`

Methods:
- `def __init__(self) -> None`
- `def defaults() -> AdvancedTimestepConfig`
- `def for_switching() -> AdvancedTimestepConfig`
- `def for_power_electronics() -> AdvancedTimestepConfig`

### BDFOrderConfig

Attributes:
- `min_order: int`
- `max_order: int`
- `initial_order: int`
- `order_increase_threshold: float`
- `order_decrease_threshold: float`
- `steps_before_increase: int`
- `enable_auto_order: bool`

Methods:
- `def __init__(self) -> None`

### BenchmarkResult

Attributes:
- `circuit_name: str`
- `num_nodes: int`
- `num_devices: int`
- `num_timesteps: int`
- `timing: BenchmarkTiming`
- `simulation_time: float`

Methods:
- `def __init__(self) -> None`
- `def timesteps_per_second(self) -> float`
- `def to_string(self) -> str`

### BenchmarkTiming

Attributes:
- `name: str`
- `iterations: int`

Methods:
- `def __init__(self) -> None`
- `def average_ms(self) -> float`
- `def min_ms(self) -> float`
- `def max_ms(self) -> float`
- `def total_ms(self) -> float`

### Capacitor

Methods:
- `def __init__(self, capacitance: float, initial_voltage: float = 0.0, name: str = "") -> None`
- `def capacitance(self) -> float`
- `def name(self) -> str`
- `def set_timestep(self, dt: float) -> None`

### CurrentSource

Methods:
- `def __init__(self, current: float, name: str = "") -> None`
- `def current(self) -> float`
- `def name(self) -> str`

### DCAnalysisResult

Attributes:
- `newton_result: NewtonResult`
- `strategy_used: DCStrategy`
- `random_restarts: int`
- `total_newton_iterations: int`
- `success: bool`
- `message: str`

Methods:
- `def __init__(self) -> None`

### DCConvergenceConfig

Attributes:
- `strategy: DCStrategy`
- `gmin_config: GminConfig`
- `source_config: SourceSteppingConfig`
- `pseudo_config: PseudoTransientConfig`
- `init_config: InitializationConfig`
- `enable_random_restart: bool`
- `max_strategy_attempts: int`

Methods:
- `def __init__(self) -> None`

### DeviceThermalTelemetry

Attributes:
- `device_name: str`
- `enabled: bool`
- `final_temperature: float`
- `peak_temperature: float`
- `average_temperature: float`

Methods:
- `def __init__(self) -> None`

### FallbackPolicyOptions

Attributes:
- `trace_retries: bool`
- `enable_transient_gmin: bool`
- `gmin_retry_threshold: int`
- `gmin_initial: float`
- `gmin_max: float`
- `gmin_growth: float`

Methods:
- `def __init__(self) -> None`

### FallbackTraceEntry

Attributes:
- `step_index: int`
- `retry_index: int`
- `time: float`
- `dt: float`
- `reason: FallbackReasonCode`
- `solver_status: SolverStatus`
- `action: str`

Methods:
- `def __init__(self) -> None`

### GminConfig

Attributes:
- `initial_gmin: float`
- `final_gmin: float`
- `reduction_factor: float`
- `max_steps: int`
- `enable_logging: bool`

Methods:
- `def __init__(self) -> None`
- `def required_steps(self) -> int`

### HarmonicBalanceOptions

Attributes:
- `period: float`
- `num_samples: int`
- `max_iterations: int`
- `tolerance: float`
- `relaxation: float`
- `initialize_from_transient: bool`

Methods:
- `def __init__(self) -> None`

### HarmonicBalanceResult

Attributes:
- `success: bool`
- `iterations: int`
- `residual_norm: float`
- `solution: List[float]`
- `sample_times: List[float]`
- `message: str`

Methods:
- `def __init__(self) -> None`

### Inductor

Methods:
- `def __init__(self, inductance: float, initial_current: float = 0.0, name: str = "") -> None`
- `def inductance(self) -> float`
- `def name(self) -> str`
- `def set_timestep(self, dt: float) -> None`

### InitializationConfig

Attributes:
- `default_voltage: float`
- `supply_voltage: float`
- `diode_forward: float`
- `mosfet_threshold: float`
- `use_zero_init: bool`
- `use_warm_start: bool`
- `max_random_restarts: int`
- `random_seed: int`
- `random_voltage_range: float`

Methods:
- `def __init__(self) -> None`

### LinearSolverConfig

Attributes:
- `pivot_tolerance: float`
- `reuse_symbolic: bool`
- `detect_pattern_change: bool`
- `deterministic_pivoting: bool`

Methods:
- `def __init__(self) -> None`

### LinearSolverTelemetry

Attributes:
- `total_solve_calls: int`
- `total_iterations: int`
- `total_fallbacks: int`
- `last_iterations: int`
- `last_error: float`
- `last_solver: Optional[LinearSolverKind]`
- `last_preconditioner: Optional[PreconditionerKind]`

Methods:
- `def __init__(self) -> None`

### NewtonOptions

Attributes:
- `max_iterations: int`
- `initial_damping: float`
- `min_damping: float`
- `auto_damping: bool`
- `track_history: bool`
- `check_per_variable: bool`
- `num_nodes: int`
- `num_branches: int`
- `tolerances: Tolerances`

Methods:
- `def __init__(self) -> None`

### NewtonResult

Attributes:
- `solution: List[float]`
- `status: SolverStatus`
- `iterations: int`
- `final_residual: float`
- `final_weighted_error: float`
- `error_message: str`

Methods:
- `def __init__(self) -> None`
- `def success(self) -> bool`

### PeriodicSteadyStateOptions

Attributes:
- `period: float`
- `max_iterations: int`
- `tolerance: float`
- `relaxation: float`
- `store_last_transient: bool`

Methods:
- `def __init__(self) -> None`

### PeriodicSteadyStateResult

Attributes:
- `success: bool`
- `iterations: int`
- `residual_norm: float`
- `steady_state: List[float]`
- `last_cycle: SimulationResult`
- `message: str`

Methods:
- `def __init__(self) -> None`

### PseudoTransientConfig

Attributes:
- `initial_dt: float`
- `max_dt: float`
- `min_dt: float`
- `dt_increase: float`
- `dt_decrease: float`
- `convergence_threshold: float`
- `max_iterations: int`
- `enable_logging: bool`

Methods:
- `def __init__(self) -> None`

### RCAnalytical

Methods:
- `def __init__(self, R: float, C: float, V_initial: float, V_final: float) -> None`
- `def tau(self) -> float`
- `def voltage(self, t: float) -> float`
- `def current(self, t: float) -> float`
- `def waveform(self, t_start: float, t_end: float, dt: float) -> List[float]`

### RLAnalytical

Methods:
- `def __init__(self, R: float, L: float, V_source: float, I_initial: float) -> None`
- `def tau(self) -> float`
- `def I_final(self) -> float`
- `def current(self, t: float) -> float`
- `def voltage_R(self, t: float) -> float`
- `def voltage_L(self, t: float) -> float`
- `def waveform(self, t_start: float, t_end: float, dt: float) -> List[float]`

### RLCAnalytical

Methods:
- `def __init__(self, R: float, L: float, C: float, V_source: float, V_initial: float, I_initial: float) -> None`
- `def omega_0(self) -> float`
- `def zeta(self) -> float`
- `def alpha(self) -> float`
- `def damping_type(self) -> RLCDamping`
- `def voltage(self, t: float) -> float`
- `def current(self, t: float) -> float`
- `def waveform(self, t_start: float, t_end: float, dt: float) -> List[float]`

### Resistor

Methods:
- `def __init__(self, resistance: float, name: str = "") -> None`
- `def resistance(self) -> float`
- `def name(self) -> str`

### RichardsonLTEConfig

Attributes:
- `method: TimestepMethod`
- `extrapolation_order: int`
- `voltage_tolerance: float`
- `current_tolerance: float`
- `use_weighted_norm: bool`
- `history_depth: int`

Methods:
- `def __init__(self) -> None`
- `def defaults() -> RichardsonLTEConfig`
- `def step_doubling() -> RichardsonLTEConfig`

### SimulationEvent

Attributes:
- `time: float`
- `type: SimulationEventType`
- `component: str`
- `description: str`
- `value1: float`
- `value2: float`

Methods:
- `def __init__(self) -> None`

### SimulationOptions

Attributes:
- `tstart: float`
- `tstop: float`
- `dt: float`
- `dt_min: float`
- `dt_max: float`
- `newton_options: NewtonOptions`
- `dc_config: DCConvergenceConfig`
- `linear_solver: LinearSolverStackConfig`
- `adaptive_timestep: bool`
- `timestep_config: AdvancedTimestepConfig`
- `lte_config: RichardsonLTEConfig`
- `integrator: Integrator`
- `enable_bdf_order_control: bool`
- `bdf_config: BDFOrderConfig`
- `stiffness_config: StiffnessConfig`
- `enable_periodic_shooting: bool`
- `periodic_options: PeriodicSteadyStateOptions`
- `enable_harmonic_balance: bool`
- `harmonic_balance: HarmonicBalanceOptions`
- `enable_events: bool`
- `enable_losses: bool`
- `switching_energy: Dict[str, SwitchingEnergy]`
- `thermal: ThermalCouplingOptions`
- `thermal_devices: Dict[str, ThermalDeviceConfig]`
- `gmin_fallback: GminConfig`
- `max_step_retries: int`
- `fallback_policy: FallbackPolicyOptions`

Methods:
- `def __init__(self) -> None`

### SimulationResult

Attributes:
- `time: List[float]`
- `states: List[List[float]]`
- `events: List[SimulationEvent]`
- `success: bool`
- `final_status: SolverStatus`
- `message: str`
- `total_steps: int`
- `newton_iterations_total: int`
- `timestep_rejections: int`
- `total_time_seconds: float`
- `linear_solver_telemetry: LinearSolverTelemetry`
- `fallback_trace: List[FallbackTraceEntry]`
- `loss_summary: SystemLossSummary`
- `thermal_summary: ThermalSummary`
- `data: List[List[float]]`

Methods:
- `def __init__(self) -> None`

### Simulator

Attributes:
- `options: SimulationOptions`

Methods:
- `def __init__(self, circuit: Circuit, options: SimulationOptions = ...) -> None`
- `def dc_operating_point(self) -> DCAnalysisResult`
- `def run_transient(self, x0: Optional[List[float]] = ...) -> SimulationResult`
- `def run_periodic_shooting( self, x0_or_options: Optional[object] = ..., options: Optional[PeriodicSteadyStateOptions] = ..., ) -> PeriodicSteadyStateResult`
- `def run_harmonic_balance( self, x0_or_options: Optional[object] = ..., options: Optional[HarmonicBalanceOptions] = ..., ) -> HarmonicBalanceResult`
- `def set_switching_energy(self, device_name: str, energy: SwitchingEnergy) -> None`

### SourceSteppingConfig

Attributes:
- `initial_scale: float`
- `final_scale: float`
- `initial_step: float`
- `min_step: float`
- `max_step: float`
- `max_steps: int`
- `max_failures: int`
- `enable_logging: bool`

Methods:
- `def __init__(self) -> None`

### StiffnessConfig

Attributes:
- `enable: bool`
- `rejection_streak_threshold: int`
- `newton_iter_threshold: int`
- `newton_streak_threshold: int`
- `cooldown_steps: int`
- `dt_backoff: float`
- `max_bdf_order: int`
- `monitor_conditioning: bool`
- `conditioning_error_threshold: float`
- `switch_integrator: bool`
- `stiff_integrator: Integrator`

Methods:
- `def __init__(self) -> None`

### SwitchingEnergy

Attributes:
- `eon: float`
- `eoff: float`
- `err: float`

Methods:
- `def __init__(self) -> None`

### ThermalCouplingOptions

Attributes:
- `enable: bool`
- `ambient: float`
- `policy: ThermalCouplingPolicy`
- `default_rth: float`
- `default_cth: float`

Methods:
- `def __init__(self) -> None`

### ThermalDeviceConfig

Attributes:
- `enabled: bool`
- `rth: float`
- `cth: float`
- `temp_init: float`
- `temp_ref: float`
- `alpha: float`

Methods:
- `def __init__(self) -> None`

### ThermalSummary

Attributes:
- `enabled: bool`
- `ambient: float`
- `max_temperature: float`
- `device_temperatures: List[DeviceThermalTelemetry]`

Methods:
- `def __init__(self) -> None`

### TimestepConfig

Attributes:
- `dt_min: float`
- `dt_max: float`
- `dt_initial: float`
- `safety_factor: float`
- `error_tolerance: float`
- `growth_factor: float`
- `shrink_factor: float`
- `max_rejections: int`
- `k_p: float`
- `k_i: float`

Methods:
- `def __init__(self) -> None`
- `def defaults() -> TimestepConfig`
- `def conservative() -> TimestepConfig`
- `def aggressive() -> TimestepConfig`

### Tolerances

Attributes:
- `voltage_abstol: float`
- `voltage_reltol: float`
- `current_abstol: float`
- `current_reltol: float`
- `residual_tol: float`

Methods:
- `def __init__(self) -> None`
- `def defaults() -> Tolerances`

### ValidationResult_v2

Attributes:
- `test_name: str`
- `passed: bool`
- `max_error: float`
- `rms_error: float`
- `max_relative_error: float`
- `mean_error: float`
- `num_points: int`
- `error_threshold: float`

Methods:
- `def __init__(self) -> None`
- `def to_string(self) -> str`

### VoltageSource

Methods:
- `def __init__(self, voltage: float, name: str = "") -> None`
- `def voltage(self) -> float`
- `def name(self) -> str`

### YamlParser

Attributes:
- `errors: List[str]`
- `warnings: List[str]`

Methods:
- `def __init__(self, options: YamlParserOptions = ...) -> None`
- `def load(self, path: str) -> Tuple[Circuit, SimulationOptions]`
- `def load_string(self, content: str) -> Tuple[Circuit, SimulationOptions]`

### YamlParserOptions

Attributes:
- `strict: bool`
- `validate_nodes: bool`

Methods:
- `def __init__(self) -> None`

## Top-Level Functions

- `def compare_waveforms(name: str, simulated: List[tuple[float, float]], analytical: List[tuple[float, float]], threshold: float = 0.001) -> ValidationResult_v2`
- `def export_validation_csv(results: List[ValidationResult_v2]) -> str`
- `def export_validation_json(results: List[ValidationResult_v2]) -> str`
- `def export_benchmark_csv(results: List[BenchmarkResult]) -> str`
- `def export_benchmark_json(results: List[BenchmarkResult]) -> str`
- `def detect_simd_level() -> SIMDLevel`
- `def simd_vector_width() -> int`
- `def solver_status_to_string(status: SolverStatus) -> str`

## Export Gaps

Symbols exported in `__all__` but not typed in `__init__.pyi`:
- `CauerNetwork`
- `CauerStage`
- `Circuit`
- `Comparator`
- `ConductionLoss`
- `ConvergenceHistory`
- `DiodeLossParams`
- `EfficiencyCalculator`
- `FosterNetwork`
- `FosterStage`
- `HysteresisController`
- `IGBT`
- `IGBTLossParams`
- `IGBTParams`
- `IdealDiode`
- `IdealSwitch`
- `IterationRecord`
- `IterativeSolverConfig`
- `LinearSolverKind`
- `LinearSolverStackConfig`
- `LookupTable1D`
- `LossAccumulator`
- `LossBreakdown`
- `LossResult`
- `MOSFET`
- `MOSFETLossParams`
- `MOSFETParams`
- `MovingAverageFilter`
- `NetlistParseError`
- `NetlistWarning`
- `PIController`
- `PIDController`
- `PWMParams`
- `PWMVoltageSource`
- `ParsedNetlist`
- `PerVariableConvergence`
- `PreconditionerKind`
- `PulseParams`
- `PulseVoltageSource`
- `RampGenerator`
- `RampParams`
- `RateLimiter`
- `SampleHold`
- `SineParams`
- `SineVoltageSource`
- `SwitchingLoss`
- `SystemLossSummary`
- `ThermalLimitMonitor`
- `ThermalResult`
- `ThermalSimulator`
- `ValidationResult`
- `VariableConvergence`
- `create_from_datasheet_4param`
- `create_mosfet_thermal_model`
- `create_simple_thermal_model`
- `dc_operating_point`
- `parse_netlist`
- `parse_netlist_verbose`
- `parse_value`
- `run_transient`
- `run_transient_streaming`
- `solve_dc`

Symbols present in stub but not exported in `__all__`:
- `ValidationResult_v2`

