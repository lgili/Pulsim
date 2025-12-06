"""SpiceLab - High-performance circuit simulator for power electronics."""

from ._spicelab import (
    # Enums
    ComponentType,
    SolverStatus,
    MOSFETType,
    ThermalNetworkType,

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
    TransformerParams,

    # Simulation
    SimulationOptions,
    SimulationResult,
    Circuit,
    Simulator,
    PowerLosses,
    SwitchEvent,
    simulate,

    # Parsing
    parse_netlist_file,
    parse_netlist_string,

    # Thermal
    ThermalRCStage,
    FosterNetwork,
    ThermalModel,
    ThermalState,
    ThermalWarning,
    ThermalSimulator,
    create_mosfet_thermal,
    fit_foster_network,

    # Version
    __version__,
)

__all__ = [
    # Enums
    "ComponentType",
    "SolverStatus",
    "MOSFETType",
    "ThermalNetworkType",

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
    "TransformerParams",

    # Simulation
    "SimulationOptions",
    "SimulationResult",
    "Circuit",
    "Simulator",
    "PowerLosses",
    "SwitchEvent",
    "simulate",

    # Parsing
    "parse_netlist_file",
    "parse_netlist_string",

    # Thermal
    "ThermalRCStage",
    "FosterNetwork",
    "ThermalModel",
    "ThermalState",
    "ThermalWarning",
    "ThermalSimulator",
    "create_mosfet_thermal",
    "fit_foster_network",

    # Version
    "__version__",
]
