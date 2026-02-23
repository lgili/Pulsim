## Why
Current SUNDIALS integration uses a projected-wrapper formulation (`project_rhs`) around the native Newton step. This path improves convergence in some difficult cases, but for switched converters it is often slower and less accurate than the native transient solver. We need a true direct SUNDIALS formulation to make SUNDIALS a production-grade backend rather than only a fallback tool.

## What Changes
- Add a direct DAE/ODE SUNDIALS execution path that assembles residual/Jacobian directly from the runtime circuit state (without the projected-wrapper approximation).
- Introduce runtime formulation selection (`projected_wrapper` vs `direct`) and make `Auto` prefer native first, then direct SUNDIALS escalation.
- Add direct-backend Jacobian and consistent-initial-condition handling for IDA, and aligned direct callbacks for CVODE/ARKODE.
- Add telemetry for formulation mode and SUNDIALS internal counters (nfe, nje, nni, nonlinear fails, error-test fails).
- Add parity benchmarks/tests against native and LTspice references with acceptance thresholds for switched converter accuracy and runtime.

## Impact
- Affected specs:
  - `kernel-v1-core`
  - `python-bindings`
- Affected code:
  - `core/src/v1/sundials_backend.cpp`
  - `core/src/v1/simulation.cpp`
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/src/v1/yaml_parser.cpp`
  - `core/tests/test_v1_kernel.cpp`
  - `python/bindings.cpp`
  - `python/pulsim/__init__.py` and stubs
  - notebook/benchmark assets for parity validation
