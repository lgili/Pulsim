## ADDED Requirements

### Requirement: Motor and Grid-Tied Benchmark Coverage Set

The `benchmarks/circuits/` directory SHALL ship seven YAML benchmarks exercising the consolidated motor and three-phase device API, with frozen baselines under `benchmarks/baselines/`:

- `motor_dc_brush_step_load.yaml` (DC motor, load step)
- `motor_pmsm_dq_open_loop.yaml` (PMSM dq frame, open loop)
- `motor_bldc_six_step.yaml` (BLDC, six-step commutation)
- `motor_induction_locked_rotor.yaml` (induction, locked rotor)
- `three_phase_inverter_svpwm.yaml` (6-switch sine-PWM, wye RL load)
- `grid_tied_single_phase_pll.yaml` (single-phase grid-tied with PLL)
- `back_to_back_rectifier_inverter.yaml` (AC-DC-AC with shared DC link)

Each benchmark SHALL invoke a registered device-variant motor or magnetics component when one exists; it SHALL NOT rebuild a motor from primitive macros if the corresponding `BldcMotorDevice`, `InductionMotorDevice`, `PmsmDevice`, `DcMotorDevice`, or `SaturableTransformerDevice` is available.

#### Scenario: BLDC benchmark uses the device class
- **GIVEN** the consolidated codebase
- **WHEN** `benchmarks/circuits/motor_bldc_six_step.yaml` is loaded
- **THEN** the netlist references `type: bldc_motor` (or equivalent device-variant entry)
- **AND** the simulation runtime stamps the `BldcMotorDevice` directly, not a primitive macro composed of voltage sources and inductors

#### Scenario: Induction benchmark uses the device class
- **GIVEN** the consolidated codebase
- **WHEN** `benchmarks/circuits/motor_induction_locked_rotor.yaml` is loaded
- **THEN** the netlist references `type: induction_motor`
- **AND** the simulation stamps the `InductionMotorDevice` directly

#### Scenario: KPI baselines frozen
- **GIVEN** the consolidated benchmark set
- **WHEN** `benchmarks/freeze_kpi_baseline.py` is invoked over the motor and grid-tied scenarios
- **THEN** each scenario writes a baseline file under `benchmarks/baselines/`
- **AND** subsequent CI runs detect regression deltas above the KPI thresholds in `benchmarks/kpi_thresholds.yaml`
