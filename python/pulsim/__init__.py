"""Pulsim - High-performance circuit simulator for power electronics."""

__version__ = "0.1.0"

# v2 API (high-performance, C++23)
# Import as: from pulsim import v2
# or: import pulsim.v2 as pv2
try:
    from . import _pulsim_v2 as v2
except ImportError:
    v2 = None  # v2 not available (build without v2 support)

from ._pulsim import (
    # Enums
    ComponentType,
    SolverStatus,
    MOSFETType,
    ThermalNetworkType,
    SimulationState,
    IntegrationMethod,
    DiagnosticSeverity,
    DiagnosticCode,
    ParameterType,
    SimulationEventType,

    # Waveforms
    DCWaveform,
    PulseWaveform,
    SineWaveform,
    PWLWaveform,
    PWMWaveform,

    # Component Parameters
    DiodeParams,
    SwitchParams,
    MOSFETParams,
    IGBTParams,
    TransformerParams,

    # Simulation
    SimulationOptions,
    SimulationResult,
    Circuit,
    Simulator,
    PowerLosses,
    SwitchEvent,
    simulate,

    # Simulation Control (GUI integration)
    SimulationController,
    SimulationProgress,
    ProgressCallbackConfig,
    StreamingConfig,

    # Enhanced Result Types
    SignalInfo,
    SolverInfo,
    SimulationEvent,

    # Component Metadata (GUI integration)
    ParameterMetadata,
    PinMetadata,
    ComponentMetadata,
    ComponentRegistry,

    # Schematic Position (GUI integration)
    SchematicPosition,

    # Validation (GUI integration)
    Diagnostic,
    ValidationResult,
    validate_circuit,
    diagnostic_code_description,

    # Parsing
    parse_netlist_file,
    parse_netlist_string,
    circuit_to_json,

    # Thermal
    ThermalRCStage,
    FosterNetwork,
    ThermalModel,
    ThermalState,
    ThermalWarning,
    ThermalSimulator,
    create_mosfet_thermal,
    fit_foster_network,

    # Device Library
    devices,
)

__all__ = [
    # Enums
    "ComponentType",
    "SolverStatus",
    "MOSFETType",
    "ThermalNetworkType",
    "SimulationState",
    "IntegrationMethod",
    "DiagnosticSeverity",
    "DiagnosticCode",
    "ParameterType",
    "SimulationEventType",

    # Waveforms
    "DCWaveform",
    "PulseWaveform",
    "SineWaveform",
    "PWLWaveform",
    "PWMWaveform",

    # Component Parameters
    "DiodeParams",
    "SwitchParams",
    "MOSFETParams",
    "IGBTParams",
    "TransformerParams",

    # Simulation
    "SimulationOptions",
    "SimulationResult",
    "Circuit",
    "Simulator",
    "PowerLosses",
    "SwitchEvent",
    "simulate",

    # Simulation Control (GUI integration)
    "SimulationController",
    "SimulationProgress",
    "ProgressCallbackConfig",
    "StreamingConfig",

    # Enhanced Result Types
    "SignalInfo",
    "SolverInfo",
    "SimulationEvent",

    # Component Metadata (GUI integration)
    "ParameterMetadata",
    "PinMetadata",
    "ComponentMetadata",
    "ComponentRegistry",

    # Schematic Position (GUI integration)
    "SchematicPosition",

    # Validation (GUI integration)
    "Diagnostic",
    "ValidationResult",
    "validate_circuit",
    "diagnostic_code_description",

    # Parsing
    "parse_netlist_file",
    "parse_netlist_string",
    "circuit_to_json",

    # Thermal
    "ThermalRCStage",
    "FosterNetwork",
    "ThermalModel",
    "ThermalState",
    "ThermalWarning",
    "ThermalSimulator",
    "create_mosfet_thermal",
    "fit_foster_network",

    # Device Library
    "devices",

    # v2 High-Performance API
    "v2",

    # Version
    "__version__",
]
