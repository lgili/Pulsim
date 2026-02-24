## 1. Direct backend core
- [ ] 1.1 Add SUNDIALS formulation mode in runtime options (`projected_wrapper`, `direct`) with backward-compatible defaults.
- [ ] 1.2 Implement direct IDA residual/Jacobian callbacks using runtime circuit assembly (no `project_rhs`).
- [ ] 1.3 Implement direct CVODE/ARKODE RHS/Jacobian callbacks consistent with runtime MNA state updates.
- [ ] 1.4 Keep projected-wrapper path available as explicit compatibility mode.

## 2. Event/reinit and consistency
- [ ] 2.1 Ensure PWM/switch event segmentation and reinitialization preserve direct formulation state consistency.
- [ ] 2.2 Add consistent-initial-condition handling for direct IDA startup and warm-start transitions.
- [ ] 2.3 Add direct-path failure mapping and deterministic fallback from direct SUNDIALS to wrapper/native when configured.

## 3. Telemetry and configuration surfaces
- [ ] 3.1 Expose formulation mode and detailed SUNDIALS counters in `SimulationResult.backend_telemetry`.
- [ ] 3.2 Extend YAML parser and Python bindings for formulation selection and direct-path knobs.
- [ ] 3.3 Keep behavior-compatible defaults for users who do not configure SUNDIALS explicitly.

## 4. Validation and parity gates
- [ ] 4.1 Add C++ tests for direct backend convergence and deterministic behavior on stiff switched converters.
- [ ] 4.2 Add Python tests for config, telemetry, and fallback behavior of direct mode.
- [ ] 4.3 Add benchmark parity checks against native and LTspice references with documented thresholds.
- [ ] 4.4 Update docs/notebooks with guidance on when to use native vs direct SUNDIALS modes.
