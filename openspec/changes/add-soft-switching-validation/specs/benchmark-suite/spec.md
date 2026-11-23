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

### Requirement: Soft-Switching Transitions Against Independent Truths
The test suite SHALL check dead-time commutations with device output capacitances against truths that do not come from a Pulsim run (closed forms in charge and energy, mpmath quadrature or ODE integration, or `benchmarks/tools/exact_truth.py`), on every engine that admits the circuit, each through explicit engine options, and SHALL park a disagreement that is a new defect as a strict xfail named in the change's FINDINGS.md.

#### Scenario: Current-driven transition with a nonlinear C_oss
- **WHEN** a half-bridge leg with C(v) = C0/(1+v/V0)^m across each switch turns its high side off into a constant current I
- **THEN** the time the node reaches each level v equals t_off + ([Q(v_0) − Q(v)] + [Q(V_dc − v) − Q(V_dc − v_0)]) / I within 0.05 ns on the second-order and error-controlled engines

#### Scenario: Partial ZVS from too little inductive energy
- **WHEN** the inductor energy ½·L·i_off² is below ∫₀^{V_dc} C_tot(v)·(V_m − v) dv
- **THEN** the node turns around at the v_min given by the energy balance, within 0.5 V

#### Scenario: LLC half-bridge against the exact truth
- **WHEN** `llc_half_bridge_zvs` runs five switching periods
- **THEN** the tank current at every turn-off and every sample inside each transition agree with `exact_truth.py` within 2.5e-4 A and 0.5 V, and the ZVS fraction is 100 % on both switches

#### Scenario: PSFB against the exact truth
- **WHEN** `psfb_full_bridge_zvs` runs three switching periods
- **THEN** the leakage current at every gate-off agrees with `exact_truth.py` within 5e-3 A, every sample inside each transition within 0.5 V, the output voltage and inductor current averaged over the last period (time weighted) within 1e-3 relative, and the ZVS fraction is 100 % on all four switches

#### Scenario: The truth tool settles a transformer-coupled rectifier
- **WHEN** `exact_truth.py` solves a half-bridge or full bridge feeding an ideal transformer and a diode bridge through a gate edge
- **THEN** it does not fail to find a consistent diode mode, its own diode law holds on every segment, and a dead-time interval agrees with a hand-derived matrix-exponential closed form within 1e-9 relative
