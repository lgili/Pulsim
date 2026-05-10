## ADDED Requirements

### Requirement: Property-Based Testing Harness
The benchmark/test infrastructure SHALL include a property-based testing harness checking physical invariants across randomly-generated circuits.

#### Scenario: Property suite runs in CI
- **WHEN** the standard CI pipeline runs
- **THEN** the property-based suite executes within a ≤30 s budget
- **AND** failures block the merge

#### Scenario: Failure produces minimal repro
- **GIVEN** a property test that fails on a generated circuit
- **WHEN** Hypothesis shrinks the failure
- **THEN** the minimal repro circuit is emitted as YAML
- **AND** added to the regression corpus

### Requirement: KCL/KVL Per-Step Invariant
Property tests SHALL assert KCL (current balance at every node) and KVL (voltage balance around every loop) at every accepted simulation step within numerical tolerance.

#### Scenario: KCL on accepted step
- **WHEN** any simulation step is accepted
- **THEN** for every non-ground node, the absolute sum of incident currents is below 1e-6 relative to total source magnitude

#### Scenario: KVL on independent loop
- **WHEN** any simulation step is accepted
- **THEN** for every independent loop in the circuit, the sum of branch voltages is below 1e-6 relative tolerance

### Requirement: Tellegen and Energy Invariants
Property tests SHALL assert Tellegen's theorem and lossless energy conservation in compatible test circuits.

#### Scenario: Tellegen invariant
- **WHEN** any simulation step is accepted
- **THEN** `Σ_branches (v_k · i_k)` for compatible v, i is below 1e-6 absolute (with appropriate sign convention)

#### Scenario: Lossless energy conservation
- **GIVEN** a lossless RLC test circuit (R = 0)
- **WHEN** simulation runs across a full transient
- **THEN** `stored_energy(t) - integral(P_src dt)` is below 1e-6 relative tolerance over the transient

### Requirement: Passivity Invariant
Property tests SHALL assert per-element passivity: resistors dissipate (`v·i ≥ 0`), capacitors and inductors balance (cycle-averaged `v·i = 0`).

#### Scenario: Resistor passivity
- **WHEN** a resistor is part of any random circuit
- **THEN** at every step, `v_R · i_R` ≥ 0 within numerical noise

#### Scenario: Capacitor cycle balance
- **GIVEN** a capacitor in a periodic steady-state circuit
- **WHEN** integration over one period completes
- **THEN** `integral(v_C · i_C dt)` over the period is below 1e-6 relative

### Requirement: Periodicity Invariant
Property tests SHALL assert periodicity in periodic-steady-state results: `x(t+T) ≈ x(t)`.

#### Scenario: PWM steady-state periodicity
- **GIVEN** a PWM-driven circuit operating in periodic steady state
- **WHEN** simulation captures two consecutive periods
- **THEN** the state vectors at corresponding phase points match within 1e-4 relative

### Requirement: PWL Mode No-Newton Property
Property tests SHALL assert that in PWL mode within a stable topology window, Newton iteration count is zero.

#### Scenario: PWL stable window
- **GIVEN** a converter operating in PWL mode with no event in the current step
- **WHEN** the step is accepted
- **THEN** `BackendTelemetry.nonlinear_iterations` for that step equals 0

### Requirement: C++ Property Tests
The C++ test suite SHALL include RapidCheck-based property tests for invariants verifiable at MNA stamp level (Tellegen, KCL, KVL).

#### Scenario: C++ property test
- **WHEN** `ctest` runs
- **THEN** the RapidCheck-based property tests execute
- **AND** failures produce shrunken minimal-repro inputs
