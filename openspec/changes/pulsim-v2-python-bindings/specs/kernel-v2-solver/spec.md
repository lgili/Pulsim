## ADDED Requirements

### Requirement: pulsim.v2 Python module

A `pulsim.v2` Python module SHALL expose the v2 kernel
through pybind11. The module MUST re-export the following
symbols from the C++ extension `pulsim._pulsim.v2_kernel`:

- `CircuitBuilder` (Layer 6's builder)
- `Graph`, `DevicePool` (opaque handles created by the
  builder)
- `SwitchStateMask` (constructible from int)
- `PwlStateSpaceCache` (constructible from graph + pool)
- `SimulationOptions` (with positional and keyword
  constructors)
- `SimulationResult`, `CommutationEvent` (read-only)
- `run_transient` (full signature with `switch_fn`,
  `b_extra_fn`, `start_from_dc_op`)
- `IdealDiodeParams` (for nonlinear diodes)

Vector data (`SimulationResult.states[k]`,
`b_extra_fn(t)`) MUST be exchanged as `numpy.ndarray` via
the standard `pybind11/eigen.h` interop.

#### Scenario: Importing pulsim.v2 yields the expected symbols

- **GIVEN** a Python interpreter with `PYTHONPATH` pointing
  at the built `pulsim` package
- **WHEN** the user runs `import pulsim.v2 as p`
- **THEN** the `p` module SHALL have all of
  `CircuitBuilder`, `Graph`, `DevicePool`,
  `SwitchStateMask`, `PwlStateSpaceCache`,
  `SimulationOptions`, `SimulationResult`,
  `CommutationEvent`, `run_transient`, `IdealDiodeParams`
  as attributes.

#### Scenario: V_dc circuit solves correctly via Python

- **GIVEN** a Python script that builds V=5V → R(1Ω) → gnd
  via `CircuitBuilder` and runs `run_transient`
- **WHEN** the simulation completes
- **THEN** `result.states[k][0]` (the v_n0 node voltage)
  SHALL equal 5.0 within 1e-9 for every recorded sample.

#### Scenario: Half-wave rectifier runs entirely from Python

- **GIVEN** a Python script that constructs a half-wave
  rectifier (V_sine=10V at 60Hz → switched diode →
  R_load=10Ω) via `CircuitBuilder` and runs `run_transient`
  with a Python `b_extra_fn` that returns numpy arrays
- **WHEN** the simulation completes over 2 cycles at
  dt=100µs
- **THEN** > 90 % of positive-half samples (V_sine > 0.5V)
  SHALL track V_sine within 0.5 V
- **AND** > 90 % of negative-half samples (V_sine < -0.5V)
  SHALL be within 0.1 V of zero.
