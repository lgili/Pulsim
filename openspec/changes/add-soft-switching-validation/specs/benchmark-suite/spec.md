## ADDED Requirements

### Requirement: ZVS / ZCS Detection KPIs
The KPI layer SHALL provide `compute_zvs_fraction`, `compute_zcs_fraction`, and `compute_switching_loss` helpers that operate on captured switch-state and V_DS / I_D waveform data to quantify soft-switching effectiveness.

#### Scenario: LLC tank achieves ZVS
- **WHEN** running `llc_half_bridge_zvs` to steady-state
- **THEN** the KPI `kpi__zvs_fraction` is at least 95 % for both half-bridge switches
- **AND** the KPI `kpi__avg_switching_loss_w` is below 1 % of the rated output power

#### Scenario: Hard-switched buck reference shows no ZVS
- **WHEN** running `buck_hard_switching_loss_reference`
- **THEN** the KPI `kpi__zvs_fraction` is below 5 %
- **AND** the KPI `kpi__avg_switching_loss_w` reports a documented non-zero loss

### Requirement: Soft-Switching Benchmark Coverage
The benchmark suite SHALL include at least two soft-switching benchmarks (LLC and PSFB) that exercise the ZVS detection KPI and validate the ZVS fraction against the topology's design target.

#### Scenario: PSFB lagging-leg ZVS at full load
- **WHEN** running `psfb_full_bridge_zvs` at the documented full-load operating point
- **THEN** both leading-leg switches achieve ≥ 95 % ZVS fraction
- **AND** both lagging-leg switches achieve ≥ 95 % ZVS fraction
