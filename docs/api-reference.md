# API Reference

Documentação gerada automaticamente a partir dos type stubs do módulo `pulsim`.

!!! tip "Uso canônico"
    A superfície suportada é `import pulsim as ps`. Use `YamlParser` + `SimulationOptions` + `Simulator` para novas integrações.

## Circuit Builder

Construtores e tipos para montar o circuito antes da simulação.

::: pulsim.Circuit

::: pulsim.VirtualComponent

::: pulsim.MixedDomainStepResult

::: pulsim.VirtualChannelMetadata

---

## Dispositivos Lineares

::: pulsim.Resistor

::: pulsim.Capacitor

::: pulsim.Inductor

::: pulsim.VoltageSource

::: pulsim.CurrentSource

---

## Dispositivos Não-Lineares

::: pulsim.IdealDiode

::: pulsim.IdealSwitch

::: pulsim.MOSFETParams

::: pulsim.MOSFET

::: pulsim.IGBTParams

::: pulsim.IGBT

---

## Fontes Variáveis no Tempo

::: pulsim.PWMParams

::: pulsim.PWMVoltageSource

::: pulsim.SineParams

::: pulsim.SineVoltageSource

::: pulsim.RampParams

::: pulsim.RampGenerator

::: pulsim.PulseParams

::: pulsim.PulseVoltageSource

---

## Blocos de Controle

::: pulsim.PIController

::: pulsim.PIDController

::: pulsim.Comparator

::: pulsim.SampleHold

::: pulsim.RateLimiter

::: pulsim.MovingAverageFilter

::: pulsim.HysteresisController

::: pulsim.LookupTable1D

---

## Simulação Principal

### YAML Parser

::: pulsim.YamlParserOptions

::: pulsim.YamlParser

### Simulador

::: pulsim.Simulator

::: pulsim.SimulationOptions

::: pulsim.SimulationResult

---

## Configuração do Solver

::: pulsim.Tolerances

::: pulsim.NewtonOptions

::: pulsim.NewtonResult

::: pulsim.LinearSolverKind

::: pulsim.PreconditionerKind

::: pulsim.IterativeSolverConfig

::: pulsim.LinearSolverStackConfig

::: pulsim.LinearSolverConfig

::: pulsim.LinearSolverTelemetry

---

## Integração e Timestep

::: pulsim.Integrator

::: pulsim.StepMode

::: pulsim.TimestepMethod

::: pulsim.TimestepConfig

::: pulsim.AdvancedTimestepConfig

::: pulsim.RichardsonLTEConfig

::: pulsim.BDFOrderConfig

::: pulsim.StiffnessConfig

---

## DC e Convergência

::: pulsim.DCStrategy

::: pulsim.GminConfig

::: pulsim.SourceSteppingConfig

::: pulsim.PseudoTransientConfig

::: pulsim.InitializationConfig

::: pulsim.DCConvergenceConfig

::: pulsim.DCAnalysisResult

---

## Análise Periódica e Harmônica

::: pulsim.PeriodicSteadyStateOptions

::: pulsim.PeriodicSteadyStateResult

::: pulsim.HarmonicBalanceOptions

::: pulsim.HarmonicBalanceResult

---

## Térmica e Perdas

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

---

## Telemetria e Eventos

::: pulsim.SimulationEventType

::: pulsim.SimulationEvent

::: pulsim.FallbackReasonCode

::: pulsim.FallbackPolicyOptions

::: pulsim.FallbackTraceEntry

::: pulsim.BackendTelemetry

---

## Convergência — Monitoramento

::: pulsim.IterationRecord

::: pulsim.ConvergenceHistory

::: pulsim.VariableConvergence

::: pulsim.PerVariableConvergence

---

## Validação e Benchmark

### Soluções Analíticas

::: pulsim.RCAnalytical

::: pulsim.RLAnalytical

::: pulsim.RLCAnalytical

::: pulsim.RLCDamping

### Comparação de Formas de Onda

::: pulsim.ValidationResult

::: pulsim.compare_waveforms

::: pulsim.export_validation_csv

::: pulsim.export_validation_json

### Benchmark

::: pulsim.BenchmarkTiming

::: pulsim.BenchmarkResult

::: pulsim.export_benchmark_csv

::: pulsim.export_benchmark_json

---

## Netlist Parser (Python puro)

::: pulsim.ParsedNetlist

::: pulsim.NetlistParseError

::: pulsim.NetlistWarning

::: pulsim.parse_netlist

::: pulsim.parse_netlist_verbose

::: pulsim.parse_value

---

## Utilitários de Alto Desempenho

::: pulsim.SIMDLevel

::: pulsim.DeviceType

::: pulsim.DeviceHint

::: pulsim.SolverStatus

::: pulsim.detect_simd_level

::: pulsim.simd_vector_width

::: pulsim.backend_capabilities

::: pulsim.solver_status_to_string

---

## run_transient (atalho com retry automático)

::: pulsim.run_transient
