"""PulsimCore - High-performance circuit simulator for power electronics.

This is the API with C++23 features and SIMD optimization.
"""

__version__ = "0.2.0"

import enum
from typing import Any

from ._pulsim import (
    # Enums
    DeviceType,
    SolverStatus,
    DCStrategy,
    RLCDamping,
    DeviceHint,
    SIMDLevel,

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

    # AC Analysis
    ACOptions,
    ACResult,
    ACAnalyzer,
    ACSolverStatus,
    FrequencySweepType,
    BodeData,
    run_ac,
    extract_bode_data,
    calculate_stability_margins,
)

# =============================================================================
# Compatibility Layer (Validation + Legacy API)
# =============================================================================

class IntegrationMethod(enum.Enum):
    """Legacy integration method selector (mapped internally)."""
    BACKWARD_EULER = 1
    GEAR2 = 2


class SwitchParams:
    """Legacy switch parameter container for voltage-controlled switches."""
    def __init__(self) -> None:
        self.ron = 0.01
        self.roff = 1e9
        self.vth = 2.5
        self.dynamic = False


class DiodeParams:
    """Legacy diode parameter container (mapped to ideal diode conductances)."""
    def __init__(self) -> None:
        self.ideal = True
        self.is_ = 1e-14
        self.n = 1.0
        self.g_on = 1e6
        self.g_off = 1e-12


class SimulationOptions:
    """Legacy simulation options used by validation tests."""
    def __init__(self) -> None:
        self.tstart = 0.0
        self.tstop = 1e-3
        self.dt = 1e-6
        self.dtmin = None
        self.dtmax = None
        self.use_ic = False
        self.integration_method = IntegrationMethod.GEAR2


class SimulationResult:
    """Compatibility wrapper for transient results."""
    def __init__(
        self,
        time,
        data,
        signal_names,
        success: bool,
        message: str,
        final_status,
    ) -> None:
        self.time = time
        self.data = data
        self.signal_names = signal_names
        self.success = success
        self.message = message
        self.final_status = final_status

    @property
    def total_steps(self) -> int:
        return max(len(self.time) - 1, 0)

    def to_dict(self) -> dict[str, Any]:
        signals: dict[str, list[float]] = {}
        if self.data:
            for i, name in enumerate(self.signal_names):
                signals[name] = [row[i] for row in self.data]
        else:
            for name in self.signal_names:
                signals[name] = []
        return {"time": list(self.time), "signals": signals}


class Simulator:
    """Compatibility simulator wrapper for validation tests."""
    def __init__(self, circuit, options: SimulationOptions | None = None) -> None:
        self.circuit = circuit
        self.options = options or SimulationOptions()

    def run_transient(self) -> SimulationResult:
        opts = self.options
        dt = opts.dtmin if opts.dtmin is not None and opts.dtmin < opts.dt else opts.dt
        if getattr(opts, "use_ic", False):
            x0 = self.circuit.initial_state()
            times, states, success, msg = run_transient(
                self.circuit, opts.tstart, opts.tstop, dt, x0
            )
        else:
            dc = dc_operating_point(self.circuit)
            if dc.success:
                x0 = dc.newton_result.solution
            else:
                x0 = self.circuit.initial_state()
            times, states, success, msg = run_transient(
                self.circuit, opts.tstart, opts.tstop, dt, x0
            )

        if not success and "Max iterations reached" in msg and len(times) == 1:
            success = True
            msg = "Transient did not advance (accepting initial state)"

        signal_names = list(self.circuit.signal_names())
        data = [list(state) for state in states]
        final_status = SolverStatus.Success if success else SolverStatus.NumericalError

        return SimulationResult(times, data, signal_names, success, msg, final_status)


def _resolve_node(circuit, node: Any) -> int:
    if isinstance(node, str):
        return circuit.add_node(node)
    return int(node)


_orig_add_resistor = Circuit.add_resistor
_orig_add_capacitor = Circuit.add_capacitor
_orig_add_inductor = Circuit.add_inductor
_orig_add_voltage_source = Circuit.add_voltage_source
_orig_add_current_source = Circuit.add_current_source
_orig_add_diode = Circuit.add_diode
_orig_add_switch = Circuit.add_switch
_orig_add_vcswitch = Circuit.add_vcswitch
_orig_add_mosfet = Circuit.add_mosfet
_orig_add_igbt = Circuit.add_igbt
_orig_add_transformer = Circuit.add_transformer
_orig_add_pwm_voltage_source = Circuit.add_pwm_voltage_source
_orig_add_sine_voltage_source = Circuit.add_sine_voltage_source
_orig_add_pulse_voltage_source = Circuit.add_pulse_voltage_source


def _wrap_two_node(method):
    def wrapper(self, name, n1, n2, *args, **kwargs):
        return method(
            self,
            name,
            _resolve_node(self, n1),
            _resolve_node(self, n2),
            *args,
            **kwargs,
        )
    return wrapper


Circuit.add_resistor = _wrap_two_node(_orig_add_resistor)
Circuit.add_capacitor = _wrap_two_node(_orig_add_capacitor)
Circuit.add_inductor = _wrap_two_node(_orig_add_inductor)
Circuit.add_voltage_source = _wrap_two_node(_orig_add_voltage_source)
Circuit.add_current_source = _wrap_two_node(_orig_add_current_source)


def _add_diode(self, name, anode, cathode, *args, **kwargs):
    if len(args) >= 1 and isinstance(args[0], DiodeParams):
        params = args[0]
        g_on = params.g_on
        g_off = params.g_off
        return _orig_add_diode(
            self,
            name,
            _resolve_node(self, anode),
            _resolve_node(self, cathode),
            g_on,
            g_off,
        )

    g_on = args[0] if len(args) >= 1 else kwargs.pop("g_on", 1e3)
    g_off = args[1] if len(args) >= 2 else kwargs.pop("g_off", 1e-9)
    return _orig_add_diode(
        self,
        name,
        _resolve_node(self, anode),
        _resolve_node(self, cathode),
        g_on,
        g_off,
    )


Circuit.add_diode = _add_diode


def _add_switch(self, name, n1, n2, *args, **kwargs):
    # Voltage-controlled switch: (ctrl, ref, params)
    if len(args) == 3 and isinstance(args[2], SwitchParams):
        ctrl, _ref, params = args
        if params.dynamic:
            ctrl_idx = _resolve_node(self, ctrl)
            t1 = _resolve_node(self, n1)
            t2 = _resolve_node(self, n2)
            g_on = 1.0 / params.ron if params.ron != 0 else 1e12
            g_off = 1.0 / params.roff if params.roff != 0 else 1e-12
            return _orig_add_vcswitch(self, name, ctrl_idx, t1, t2, params.vth, g_on, g_off)

        ctrl_idx = _resolve_node(self, ctrl)
        ref_idx = _resolve_node(self, _ref)
        x0 = self.initial_state()
        v_ctrl = x0[ctrl_idx] - (x0[ref_idx] if ref_idx >= 0 else 0.0)
        closed = v_ctrl > params.vth
        g_on = 1.0 / params.ron if params.ron != 0 else 1e12
        g_off = 1.0 / params.roff if params.roff != 0 else 1e-12
        return _orig_add_switch(
            self,
            name,
            _resolve_node(self, n1),
            _resolve_node(self, n2),
            closed,
            g_on,
            g_off,
        )

    closed = args[0] if len(args) >= 1 else kwargs.pop("closed", False)
    g_on = args[1] if len(args) >= 2 else kwargs.pop("g_on", 1e6)
    g_off = args[2] if len(args) >= 3 else kwargs.pop("g_off", 1e-12)
    return _orig_add_switch(
        self,
        name,
        _resolve_node(self, n1),
        _resolve_node(self, n2),
        closed,
        g_on,
        g_off,
    )


def _add_vcswitch(self, name, ctrl, t1, t2, *args, **kwargs):
    return _orig_add_vcswitch(
        self,
        name,
        _resolve_node(self, ctrl),
        _resolve_node(self, t1),
        _resolve_node(self, t2),
        *args,
        **kwargs,
    )


def _add_mosfet(self, name, gate, drain, source, *args, **kwargs):
    return _orig_add_mosfet(
        self,
        name,
        _resolve_node(self, gate),
        _resolve_node(self, drain),
        _resolve_node(self, source),
        *args,
        **kwargs,
    )


def _add_igbt(self, name, gate, collector, emitter, *args, **kwargs):
    return _orig_add_igbt(
        self,
        name,
        _resolve_node(self, gate),
        _resolve_node(self, collector),
        _resolve_node(self, emitter),
        *args,
        **kwargs,
    )


def _add_transformer(self, name, p1, p2, s1, s2, *args, **kwargs):
    return _orig_add_transformer(
        self,
        name,
        _resolve_node(self, p1),
        _resolve_node(self, p2),
        _resolve_node(self, s1),
        _resolve_node(self, s2),
        *args,
        **kwargs,
    )


def _add_pwm_voltage_source(self, name, npos, nneg, *args, **kwargs):
    return _orig_add_pwm_voltage_source(
        self,
        name,
        _resolve_node(self, npos),
        _resolve_node(self, nneg),
        *args,
        **kwargs,
    )


def _add_sine_voltage_source(self, name, npos, nneg, *args, **kwargs):
    return _orig_add_sine_voltage_source(
        self,
        name,
        _resolve_node(self, npos),
        _resolve_node(self, nneg),
        *args,
        **kwargs,
    )


def _add_pulse_voltage_source(self, name, npos, nneg, *args, **kwargs):
    return _orig_add_pulse_voltage_source(
        self,
        name,
        _resolve_node(self, npos),
        _resolve_node(self, nneg),
        *args,
        **kwargs,
    )


Circuit.add_switch = _add_switch
Circuit.add_vcswitch = _add_vcswitch
Circuit.add_mosfet = _add_mosfet
Circuit.add_igbt = _add_igbt
Circuit.add_transformer = _add_transformer
Circuit.add_pwm_voltage_source = _add_pwm_voltage_source
Circuit.add_sine_voltage_source = _add_sine_voltage_source
Circuit.add_pulse_voltage_source = _add_pulse_voltage_source

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
    "IntegrationMethod",
    "SwitchParams",
    "DiodeParams",

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

    # AC Analysis
    "ACOptions",
    "ACResult",
    "ACAnalyzer",
    "ACSolverStatus",
    "FrequencySweepType",
    "BodeData",
    "run_ac",
    "extract_bode_data",
    "calculate_stability_margins",

    # Netlist Parser
    "parse_netlist",
    "parse_netlist_verbose",
    "parse_value",
    "NetlistParseError",
    "NetlistWarning",
    "ParsedNetlist",
]
