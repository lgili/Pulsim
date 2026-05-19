## MODIFIED Requirements

### Requirement: Model Regularization Defaults

`SimulationOptions::ModelRegularizationOptions::enable_auto` SHALL
default to `true` and `apply_only_in_recovery` SHALL default to
`false`, so per-device-class `g_off_min` floors apply at the first
Newton step (not only on convergence retry). The previous behaviour
(off-by-default until recovery) is opt-in via explicit user
override.

#### Scenario: MOSFET with extreme g_off in a freshly-built sim uses the regularization floor

- **GIVEN** a `MOSFETParams` with `g_off = 1e-15` (an aggressive user setting)
- **AND** the user does NOT explicitly configure `SimulationOptions::model_regularization`
- **WHEN** the `Simulator` is constructed and the first Newton step is solved
- **THEN** the effective `g_off` actually stamped SHALL be `max(1e-15, 1e-7) = 1e-7` (the MOSFET floor in `ModelRegularizationOptions::mosfet_g_off_min`)
- **AND** the linear system SHALL be well-conditioned even if the gate node is otherwise unanchored
- **AND** `result.backend_telemetry.model_regularization_events` SHALL be ≥ 1.

#### Scenario: SPICE-parity test mode restores the legacy 1e-12 g_off floor

- **GIVEN** a benchmark test that needs exact `g_off = 1e-12` for SPICE-parity comparison
- **WHEN** the test explicitly sets `opts.model_regularization.enable_auto = false`
- **THEN** the simulator SHALL NOT apply the regularization floors
- **AND** the effective `g_off` stamped SHALL be the user's exact `1e-12` value.
