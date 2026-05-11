## Why
PLECS RT-Box is sold to controls engineers because the same model that runs in simulation can be exported as C code, an FMU, or HIL hardware. Pulsim already has `python/pulsim/codegen/` and `python/pulsim/fmu/` directories — the infrastructure exists. What's missing is a **validation loop** that proves a generated artifact reproduces the in-simulator result.

If we can show "Pulsim generates a C controller, gcc compiles it, the controller runs in a Python harness, and the closed-loop output matches the all-in-simulator version bit-for-bit", that's commercial-grade co-simulation.

## What Changes
- Add an end-to-end test category `cosim_*`:
  - `cosim_pi_buck_c_codegen` — take an existing `cl_buck_pi` setup, export the PI controller as C code, compile with the host's CC, load it via cffi, and use it in a Python loop driving the Pulsim plant. Validate the closed-loop output matches the all-in-simulator baseline within 1 %.
  - `cosim_pi_buck_fmu_export` — same circuit, but export the controller as an FMU (Functional Mock-up Interface) and consume it from FMPy or an equivalent runner.
  - `cosim_python_co_simulation` — driver script demonstrating Python-side controller tuning (PI gains modified at runtime) while Pulsim simulates the plant.
- Audit the existing `codegen` and `fmu` Python modules; document what they currently produce and any gaps.
- Add CI hooks (or at least a documented manual workflow) for the co-sim tests so they run on a CI/CD pipeline where a C compiler is available.

## Impact
- Affected specs: `code-generation`, `fmi-export`.
- Affected code: new test scripts in `python/tests/`, possibly extensions to the codegen module for missing pieces, and a small C harness for the generated controller.
- Requires a C compiler in the test environment (typically gcc or clang). Document the dependency.
