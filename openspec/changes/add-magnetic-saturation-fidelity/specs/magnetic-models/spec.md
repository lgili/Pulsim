## ADDED Requirements

### Requirement: Piecewise-Linear B-H Saturation Curve
The `saturable_inductor` component SHALL accept a piecewise-linear B-H curve specification in YAML and derive effective permeability per operating point during simulation.

#### Scenario: Saturable inductor exhibits a current knee
- **GIVEN** a saturable inductor declares `bh_curve: [{B: 0, H: 0}, {B: 0.3, H: 100}, {B: 0.4, H: 1000}, {B: 0.42, H: 10000}]`
- **WHEN** the inductor is driven by a current-source ramp from 0 to 200 A
- **THEN** the recovered effective inductance L_eff = N·dΦ/dI matches the curve's slope (μ_eff·A·N²/l) to within 5 % at each point on the curve
- **AND** the i_L vs t trace exhibits the documented "knee" at the saturation current

### Requirement: Multi-Winding Transformer Support
The magnetic-component family SHALL provide either an extended `coupled_inductor` with N ≥ 3 windings or a new `multi_winding_transformer` primitive that shares a single magnetizing branch.

#### Scenario: Center-tapped transformer in a full-wave rectifier
- **WHEN** simulating `center_tapped_full_wave_rectifier` with a 3-winding transformer (primary + two secondaries)
- **THEN** the primary current equals the (turns-ratio-adjusted) sum of the two secondary currents within 2 % at every instant
- **AND** the rectified DC output matches the analytical prediction `V_pk · turns_ratio · 2/π` within 5 %

### Requirement: Core Loss KPI Extraction
The KPI layer SHALL compute average core loss density using the Steinmetz model from a recorded B(t) waveform.

#### Scenario: Sinusoidally-excited inductor reports Steinmetz core loss
- **WHEN** running `core_loss_steinmetz` with a sinewave drive and declared Steinmetz coefficients `k, α, β` in YAML
- **THEN** the KPI emits `kpi__core_loss_w_per_kg`
- **AND** the measured value matches the analytical Steinmetz formula `k · f^α · B^β` within 10 %
