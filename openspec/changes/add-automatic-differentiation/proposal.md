## Why

Every device in `core/include/pulsim/v1/components/` derives its Jacobian by hand. A reviewer must verify chain-rule sign, transposes, and per-region derivatives manually for each new device. This is:

- **Error-prone**: subtle sign mistakes degrade convergence silently (the simulation still completes — just with worse Newton convergence rate). Suspect mismatches already exist; verifying requires algebraic re-derivation per device.
- **A growth bottleneck**: each new device costs an analytical pass + careful review. PSIM/PLECS ship hundreds of device variants; Pulsim has fewer than 12.
- **A barrier to runtime extensibility**: users cannot add custom devices in Python without dropping into C++ to write the Jacobian.

Automatic differentiation (AD) eliminates manual Jacobian derivation. Devices declare only the residual `F(x, t, dt)`; AD produces `J = ∂F/∂x` automatically. This is a well-supported technique in numerical C++ (Eigen has `Eigen::AutoDiffScalar`; Enzyme provides LLVM-level reverse-mode AD; CppAD is the reference for many ODE simulators).

## What Changes

### Forward-mode AD via Eigen::AutoDiff
- Introduce `pulsim::v1::ad::ADReal = Eigen::AutoDiffScalar<Eigen::VectorXd>` for terminal-voltage scalars.
- Each device implements `template <typename Scalar> Scalar residual_at(...)` rather than `stamp_jacobian_impl`.
- Kernel evaluates `residual_at<ADReal>` once per step; AD machinery extracts the Jacobian.

### Hybrid Stamping Strategy
- Linear devices (`Resistor`, ideal `Capacitor`, ideal `Inductor`, `VoltageSource`) keep their direct stamps — AD overhead is unjustified for trivially constant Jacobians.
- Non-linear devices (`MOSFET`, `IGBT`, `IdealDiode` in `Behavioral` mode) switch to AD-derived Jacobians.
- PWL devices (`Ideal` mode) stamp constants per topology; AD not needed.

### Custom Device Registration via Python
- Expose `pulsim.register_device(name, residual_fn, num_pins, params_schema)` where `residual_fn(x_terminals, t, dt, params) -> residual_vector` is a Python callable.
- pybind11 wraps the callable; AD propagates through Python via Eigen ND-array bridge.
- Performance fallback: if AD-through-Python is slow, allow user-supplied Jacobian function.

### Numerical Validation Layer
- Add `--validate-jacobians` mode that compares AD result against finite-difference at startup for each device. Mismatch ⇒ deterministic diagnostic.
- Use this to retire any incorrect manual stamps discovered.

### Deprecation Path
- Manual `stamp_jacobian_impl` retained for one release cycle behind `PULSIM_LEGACY_MANUAL_JACOBIAN`. Deprecation warning when used.
- Tests assert AD path produces convergence rate ≥ manual path.

## Impact

- **Affected specs**: `device-models` (new AD-based device contract), `python-bindings` (custom device registration).
- **Affected code**: every file in `core/include/pulsim/v1/components/`, `core/include/pulsim/v1/device_base.hpp`, `python/bindings.cpp`, plus a new module `core/include/pulsim/v1/ad/` for the AD scalar bridge.
- **Performance**: AD adds ~10–30% overhead per nonlinear stamp vs hand-derived. Mitigated by sparse-pattern caching and forward-mode compilation. Net negative only on tight inner loops; PWL path (most production usage post-Phase-0) bypasses AD entirely.

## Success Criteria

1. **Correctness**: AD-derived Jacobian for every existing device matches manual stamp within 1e-12 (FD validation) on representative operating points.
2. **Convergence**: switching converter benchmarks show equal or better Newton convergence rate vs manual stamps.
3. **Custom devices**: Python user can register a non-trivial nonlinear device (e.g., simple JFET) with ~30 lines of Python and run a transient.
4. **Build**: total compile time impact ≤15% on `pulsim_core` target.
5. **Test parity**: every existing C++ and Python test passes unmodified after default switch to AD path.
