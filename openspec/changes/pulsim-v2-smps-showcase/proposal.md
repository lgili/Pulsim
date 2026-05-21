## Why

V0 through V8 built every layer of v2 — kernel, builder,
Python bindings, MOSFET helpers, transformer, YAML loader.
Each layer has dedicated unit + integration tests. But the
END-TO-END story — "load a YAML, run a PWM-driven buck,
verify steady-state V_out" — isn't validated as a single
flow yet.

V9 ships **SMPS showcases**: integration tests that compose
every layer into a realistic SMPS workload. Specifically,
this OpenSpec ships:

- An **open-loop buck converter** test: load
  `examples/v2/buck.yaml`, drive Q1 with a 100 kHz PWM
  switch_fn at 50 % duty cycle, simulate 5 ms (~500 PWM
  periods), verify steady-state V_out ≈ V_in · D = 12 V.
- A **Python runner script** demonstrating the same flow
  from Python — what a real SMPS engineer would write to
  iterate on a design.
- A **C++ integration test** that automates the same
  validation for CI.

This is the "v2 is production-grade" milestone: every layer
exercised, every API used as designed, against an
analytically-verifiable target.

## What Changes

**Scope decision — Layer 9 V0** (SMPS showcase):

- New `examples/v2/buck.yaml` already exists from Layer 8
  (no changes here).

- New C++ integration test
  `core/tests/v2/showcases/test_buck_open_loop.cpp`:
  - Loads `examples/v2/buck.yaml` via `pulsim::v2::yaml::load_file`.
  - Constructs `PwlStateSpaceCache`, runs `run_transient`
    with a 100 kHz / 50 % PWM `switch_fn`.
  - Verifies steady-state mean V_out is within 0.5 V of
    `V_in · D = 12 V` (small loss from MOSFET R_on +
    inductor IR drops is expected).
  - Verifies output voltage ripple is bounded by ~1 V
    p-p (LC filter performance check).

- New Python runner script
  `examples/v2/scripts/run_buck.py`:
  - Loads `examples/v2/buck.yaml`.
  - Generates PWM switch_fn from Python.
  - Runs `run_transient`, prints steady-state stats.
  - Optionally plots V_out vs time (if matplotlib
    available — graceful no-op otherwise).

- New showcase test infrastructure:
  `pulsim_v2_showcase_tests` Catch2 target.

- Documentation: `docs/pulsim-v2/layer9-smps-showcase.md`
  explains the steady-state math + how to extend the
  pattern to boost, flyback, half-bridge, etc.

## Impact

- **Affected specs**: ADDED requirement on
  `kernel-v2-solver` for end-to-end SMPS validation.
- **Affected code** (~250 LOC):
  - NEW `core/tests/v2/showcases/test_main.cpp`
  - NEW `core/tests/v2/showcases/test_buck_open_loop.cpp`
  - MODIFIED `core/CMakeLists.txt` (showcase target)
  - NEW `examples/v2/scripts/run_buck.py`
  - NEW `docs/pulsim-v2/layer9-smps-showcase.md`
- **Migration**: zero. Pure additive validation.
- **Risk**: low. The underlying machinery is already
  tested; this is composition validation.
