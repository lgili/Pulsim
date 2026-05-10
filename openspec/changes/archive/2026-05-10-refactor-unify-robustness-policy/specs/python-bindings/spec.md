## ADDED Requirements

### Requirement: Robustness Tier Surface in Python
Python bindings SHALL expose `pulsim.RobustnessTier` enum and `SimulationOptions.robustness` field.

#### Scenario: Set tier from Python
- **WHEN** Python code does `options.robustness = pulsim.RobustnessTier.Aggressive`
- **THEN** the simulation resolves the same profile as YAML-driven `simulation.robustness: aggressive`
- **AND** both invocations produce identical `BackendTelemetry.robustness_profile`

### Requirement: Deprecation of Python Wrapper Retry Layer
The Python wrapper retry layer in `pulsim/__init__.py:run_transient` (auto-bleeders, dt-halving retries) SHALL log a deprecation warning when triggered, with guidance pointing to the kernel-level robustness profile.

#### Scenario: Retry deprecation warning
- **GIVEN** legacy retry path is reached (e.g., behind `PULSIM_LEGACY_RETRY_FALLBACK=1`)
- **WHEN** the retry executes
- **THEN** a one-time deprecation warning is emitted
- **AND** the warning message names the replacement mechanism (`SimulationOptions.robustness`)

#### Scenario: Default path bypasses wrapper retry
- **GIVEN** PWL engine is resolved as the default for the circuit (post `refactor-pwl-switching-engine`)
- **WHEN** `run_transient` is called
- **THEN** the wrapper does not invoke retry/auto-bleeder logic
- **AND** the kernel handles recovery via the resolved robustness profile

### Requirement: Profile Inspection from Python
Python bindings SHALL expose `Simulator.options.robustness_profile` as a read-only structured object for debugging.

#### Scenario: Inspect resolved profile
- **WHEN** Python reads `sim.options.robustness_profile.newton_max_iter`
- **THEN** the value reflects the kernel-resolved profile
- **AND** Python users can compare profiles between runs deterministically
