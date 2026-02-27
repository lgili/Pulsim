## ADDED Requirements
### Requirement: Stress Testing Scenarios
The benchmark suite SHALL include complex topologies that stress the non-linear, highly oscillatory, and switching behavior of the runtime solvers.

#### Scenario: LLC Resonant Converter
- **WHEN** simulating `ll11_llc_resonant_converter.yaml`
- **THEN** the solver reliably resolves resonant tank oscillations and soft-switching transitions

#### Scenario: PFC Boost Converter
- **WHEN** simulating `ll12_pfc_boost_continuous.yaml`
- **THEN** the solver efficiently handles continuous conduction mode with active power factor correction switching patterns

#### Scenario: Active Clamp Forward Converter
- **WHEN** simulating `ll13_active_clamp_forward.yaml`
- **THEN** the solver correctly models transformer magnetizing reset and secondary side synchronous rectification
