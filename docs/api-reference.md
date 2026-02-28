# API Reference

This page is generated from the `pulsim` package surface (stubs + public exports).

!!! tip "Canonical usage"
    For new integrations, prefer `YamlParser` + `SimulationOptions` + `Simulator`.

## Circuit Runtime

::: pulsim.Circuit

::: pulsim.VirtualComponent

::: pulsim.MixedDomainStepResult

::: pulsim.VirtualChannelMetadata

## Linear Devices

::: pulsim.Resistor

::: pulsim.Capacitor

::: pulsim.Inductor

::: pulsim.VoltageSource

::: pulsim.CurrentSource

## Nonlinear and Switching Devices

::: pulsim.IdealDiode

::: pulsim.IdealSwitch

::: pulsim.MOSFETParams

::: pulsim.MOSFET

::: pulsim.IGBTParams

::: pulsim.IGBT

## Time-Varying Sources

::: pulsim.PWMParams

::: pulsim.PWMVoltageSource

::: pulsim.SineParams

::: pulsim.SineVoltageSource

::: pulsim.RampParams

::: pulsim.RampGenerator

::: pulsim.PulseParams

::: pulsim.PulseVoltageSource

## Control and Signal Blocks

::: pulsim.PIController

::: pulsim.PIDController

::: pulsim.Comparator

::: pulsim.SampleHold

::: pulsim.RateLimiter

::: pulsim.MovingAverageFilter

::: pulsim.HysteresisController

::: pulsim.LookupTable1D

## Parser and Simulation Entry Points

::: pulsim.YamlParserOptions

::: pulsim.YamlParser

::: pulsim.Simulator

::: pulsim.SimulationOptions

::: pulsim.SimulationResult

## Solver Configuration

::: pulsim.Tolerances

::: pulsim.NewtonOptions

::: pulsim.NewtonResult

::: pulsim.LinearSolverKind

::: pulsim.PreconditionerKind

::: pulsim.IterativeSolverConfig

::: pulsim.LinearSolverStackConfig

::: pulsim.LinearSolverConfig

::: pulsim.LinearSolverTelemetry

## Integration and Timestep

::: pulsim.Integrator

::: pulsim.StepMode

::: pulsim.TimestepMethod

::: pulsim.TimestepConfig

::: pulsim.AdvancedTimestepConfig

::: pulsim.RichardsonLTEConfig

::: pulsim.BDFOrderConfig

::: pulsim.StiffnessConfig

## DC and Convergence

::: pulsim.DCStrategy

::: pulsim.GminConfig

::: pulsim.SourceSteppingConfig

::: pulsim.PseudoTransientConfig

::: pulsim.InitializationConfig

::: pulsim.DCConvergenceConfig

::: pulsim.DCAnalysisResult

## Periodic and Harmonic Analysis

::: pulsim.PeriodicSteadyStateOptions

::: pulsim.PeriodicSteadyStateResult

::: pulsim.HarmonicBalanceOptions

::: pulsim.HarmonicBalanceResult

## Thermal and Loss Modeling

::: pulsim.ThermalCouplingPolicy

::: pulsim.ThermalCouplingOptions

::: pulsim.ThermalDeviceConfig

::: pulsim.DeviceThermalTelemetry

::: pulsim.ThermalSummary

::: pulsim.SwitchingEnergy

::: pulsim.MOSFETLossParams

::: pulsim.IGBTLossParams

::: pulsim.DiodeLossParams

::: pulsim.ConductionLoss

::: pulsim.SwitchingLoss

::: pulsim.LossBreakdown

::: pulsim.LossAccumulator

::: pulsim.EfficiencyCalculator

::: pulsim.LossResult

::: pulsim.SystemLossSummary

::: pulsim.FosterStage

::: pulsim.FosterNetwork

::: pulsim.CauerStage

::: pulsim.CauerNetwork

::: pulsim.ThermalSimulator

::: pulsim.ThermalLimitMonitor

::: pulsim.ThermalResult

## Events and Backend Telemetry

::: pulsim.SimulationEventType

::: pulsim.SimulationEvent

::: pulsim.FallbackReasonCode

::: pulsim.FallbackPolicyOptions

::: pulsim.FallbackTraceEntry

::: pulsim.BackendTelemetry

## Convergence Monitoring

::: pulsim.IterationRecord

::: pulsim.ConvergenceHistory

::: pulsim.VariableConvergence

::: pulsim.PerVariableConvergence

## Validation and Benchmarks

::: pulsim.RCAnalytical

::: pulsim.RLAnalytical

::: pulsim.RLCAnalytical

::: pulsim.RLCDamping

::: pulsim.ValidationResult

::: pulsim.compare_waveforms

::: pulsim.export_validation_csv

::: pulsim.export_validation_json

::: pulsim.BenchmarkTiming

::: pulsim.BenchmarkResult

::: pulsim.export_benchmark_csv

::: pulsim.export_benchmark_json

## Pure-Python Utilities

::: pulsim.ParsedNetlist

::: pulsim.NetlistParseError

::: pulsim.NetlistWarning

::: pulsim.parse_netlist

::: pulsim.parse_netlist_verbose

::: pulsim.parse_value

::: pulsim.SignalEvaluator

::: pulsim.AlgebraicLoopError

## Performance Utilities and Enums

::: pulsim.SIMDLevel

::: pulsim.DeviceType

::: pulsim.DeviceHint

::: pulsim.SolverStatus

::: pulsim.detect_simd_level

::: pulsim.simd_vector_width

::: pulsim.backend_capabilities

::: pulsim.solver_status_to_string
