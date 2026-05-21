## ADDED Requirements

### Requirement: Saturable Inductor Device Model

The system SHALL provide a `SaturableInductor` device model whose effective inductance smoothly decreases as the magnetising current approaches and exceeds the saturation current. The model SHALL be a nonlinear branch device with a Newton-refresh stamping pass.

#### Scenario: L decreases above I_sat
- **WHEN** a saturable inductor with `L_0 = 100 µH, I_sat = 5 A` carries current `i = 10 A`
- **THEN** the effective inductance `L(i) = L_0 · g(i, I_sat)` returned by the model SHALL be at most `0.5 · L_0`
- **AND** the derivative `∂L/∂i` SHALL be continuous (C¹ smooth) for Newton stability

#### Scenario: L equals L_0 below I_sat
- **WHEN** a saturable inductor with `I_sat = 5 A` carries current `i = 1 A`
- **THEN** the effective inductance `L(i)` SHALL be within 5 %% of `L_0`

#### Scenario: Newton converges in a saturating RL circuit
- **WHEN** a saturable inductor is excited by a step voltage source that drives `i_L` past `I_sat`
- **THEN** `run_transient` SHALL converge at every timestep with the default Newton tolerances

### Requirement: Multi-Winding Transformer Device Model

The system SHALL provide a `MultiWindingTransformer` device that couples N (2 ≤ N ≤ 6) inductor branches via a mutual-inductance matrix `M_ij = k_ij · √(L_i · L_j)`, with optional shared saturable core that ties all windings to a single magnetising current.

#### Scenario: 2-winding case matches existing TwoWindingTransformer
- **WHEN** a multi-winding transformer is configured with `N = 2`, primary `L_p` and secondary `L_s`, coupling `k`
- **THEN** its behaviour SHALL be numerically equivalent to the existing `TwoWindingTransformer` device (within 1e-9 relative error in mutual inductance)

#### Scenario: 3-winding flyback w/ aux bias
- **WHEN** a flyback YAML defines a 3-winding transformer (primary, main secondary, aux bias)
- **THEN** all three outputs SHALL settle to the analytical values predicted by the turns ratios
- **AND** the simulation SHALL complete in ≤ 5 ms wall-clock for a 5 ms transient at dt = 0.1 µs

### Requirement: Core-Loss Estimator

The system SHALL provide a `CoreLossEstimator` post-process pass that integrates instantaneous core loss density using a Steinmetz law `P_v = K · f^α · B^β` over the recorded simulation history, returning total core loss in watts.

#### Scenario: Steinmetz on a sinusoidal flux history
- **WHEN** the core-loss estimator is invoked on a transformer flux history that contains a pure sinusoidal `B(t) = B_pk · sin(2π·f·t)`
- **THEN** the returned total core loss SHALL be within 5 %% of the analytical Steinmetz prediction `P_v = K · f^α · B_pk^β · Volume`

### Requirement: YAML Schema for Saturable Magnetics

The YAML loader SHALL parse `type: saturable_inductor` and `type: multi_winding_transformer` device entries, mapping their parameters to the corresponding `CircuitBuilder` method calls.

#### Scenario: Round-trip parsing
- **WHEN** a YAML file declares a `saturable_inductor` with `L_0`, `I_sat`, and optional smoothing
- **THEN** `yaml::load_string()` SHALL return a builder containing exactly one branch of `StoredKind::SaturableInductor`
- **AND** the stored parameters SHALL match the YAML values within floating-point exactness

### Requirement: Python Bindings for Saturable Magnetics

The Python `pulsim.v2` module SHALL expose `CircuitBuilder.add_saturable_inductor(...)` and `CircuitBuilder.add_multi_winding_transformer(...)` with keyword arguments matching the C++ method signatures.

#### Scenario: Build flyback from Python
- **WHEN** a user constructs a flyback circuit in Python using `add_multi_winding_transformer` with N=3
- **THEN** the resulting circuit graph SHALL have one nonlinear branch (saturable core), three inductor branches (windings), and `run_transient` SHALL converge

### Requirement: Showcase YAMLs

The repository SHALL include at least two showcase YAML files demonstrating saturable magnetics: a 2-winding flyback with saturating primary and a 3-winding flyback with auxiliary bias supply.

#### Scenario: Showcase test passes
- **WHEN** the showcase test suite runs the `flyback_saturated.yaml` and `flyback_3winding.yaml` examples
- **THEN** both tests SHALL pass with no Newton convergence failures
- **AND** the measured outputs SHALL match the analytical predictions within 10 %%
