## ADDED Requirements

### Requirement: Single Robustness Policy Owner
The kernel SHALL provide `RobustnessProfile` as the single source of truth for robust default configuration of Newton, linear solver, integrator, recovery, and fallback knobs.

#### Scenario: Single declaration site
- **WHEN** the codebase is grepped for "robust default" / `apply_robust_*` / `_tune_*_for_robust`
- **THEN** only one definition site (in `robustness_profile.hpp/cpp`) is found
- **AND** all callers route through this single owner

#### Scenario: Tier resolution
- **GIVEN** `RobustnessProfile::for_circuit(circuit, RobustnessTier::Aggressive)`
- **WHEN** the factory runs
- **THEN** the resulting profile has knobs derived from circuit analysis (switching count, nonlinear count) and the tier
- **AND** identical inputs produce identical profile (deterministic)

### Requirement: Robustness Profile Telemetry
`BackendTelemetry` SHALL include `robustness_profile` reflecting the resolved tier, key knob values, and a reproducibility hash.

#### Scenario: Telemetry capture
- **WHEN** a simulation completes
- **THEN** `BackendTelemetry.robustness_profile` exposes `tier`, `newton_max_iter`, `linear_solver_order`, `integrator`, `max_step_retries`, `gmin_initial`, `gmin_max`
- **AND** a hash combining these into a single identifier appears

#### Scenario: Profile diff in verbose mode
- **GIVEN** verbosity-enabled output and a non-default profile
- **WHEN** the result message is composed
- **THEN** the diff vs the default profile is included as a structured list
