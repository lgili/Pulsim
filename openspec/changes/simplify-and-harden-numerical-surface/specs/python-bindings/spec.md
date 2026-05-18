## ADDED Requirements

### Requirement: Python Preset Enum and Factory

The Python module SHALL expose `pulsim.Preset` (mirror of
`pulsim::v1::Preset`) with the four values `Auto`, `Fast`,
`Robust`, `HighFidelity`, and SHALL expose
`pulsim.SimulationOptions.from_preset(preset, dt, tstop)` as a
classmethod factory.

`pulsim.Preset` SHALL appear in `pulsim.__all__` and be re-exported
from the top-level package.

#### Scenario: Python user constructs options from a preset

- **GIVEN** a Python user writes
  `opts = pulsim.SimulationOptions.from_preset(
      pulsim.Preset.Robust, 1e-6, 1e-3)`
- **WHEN** they pass `opts` to `pulsim.Simulator(circuit, opts)` and
  call `run_transient()`
- **THEN** the simulation runs with the Robust preset's tuning
- **AND** no other field on `opts` needs to be set

#### Scenario: Preset is iterable from Python

- **GIVEN** a Python user writes `list(pulsim.Preset)`
- **WHEN** they print the result
- **THEN** the list contains exactly four entries:
  `[Preset.Auto, Preset.Fast, Preset.Robust, Preset.HighFidelity]`

### Requirement: AdvancedOptions Python Namespace

The Python module SHALL expose `opts.advanced` as a nested namespace
on `SimulationOptions`, with sub-objects `newton`, `timestep`, `lte`,
`bdf_order`, `dc`, `stiffness`, `fallback`, `formulation`, and
`linear_solver`.

Existing top-level field accessors (e.g. `opts.newton_options`,
`opts.timestep_config`) SHALL continue to work for one release as
`@property` shims that forward to `opts.advanced.*` and emit a
`DeprecationWarning` on first access per process.

#### Scenario: Advanced namespace is the canonical path

- **GIVEN** a Python user writes
  `opts.advanced.newton.max_iterations = 100`
- **WHEN** the simulator runs
- **THEN** the underlying `NewtonOptions::max_iterations` is set to
  100

#### Scenario: Legacy alias still works

- **GIVEN** legacy code that writes `opts.newton_options.max_iterations
  = 100`
- **WHEN** the field is set
- **THEN** the value is forwarded to `opts.advanced.newton.max_iterations`
- **AND** a `DeprecationWarning` is raised on first access per process

## MODIFIED Requirements

### Requirement: SimulationOptions Python Surface

The Python `SimulationOptions` class SHALL expose the following
top-level fields and no others:

- `tstart`, `tstop`, `dt`, `dt_min`, `dt_max` (essential timing)
- `step_mode` (`StepMode.Fixed | StepMode.Variable`)
- `switching_mode` (`SwitchingMode.Auto | Ideal | Behavioral`)
- `integrator` (`Integrator` enum, narrowed to the curated 5-value set)
- `linear_solver_kind` (`LinearSolverKind.Auto | Direct | Iterative`)
- `enable_losses`, `enable_events`, `enable_bdf_order_control`
  (existing feature toggles)
- `advanced` (the `AdvancedOptions` namespace defined in the ADDED
  requirement above)

Deprecated top-level fields (`newton_options`, `timestep_config`,
`lte_config`, `bdf_config`, `dc_config`, `stiffness_config`,
`fallback_policy`, `formulation_mode`, `linear_solver`,
`adaptive_timestep`, `direct_formulation_fallback`) SHALL remain
accessible via `@property` shims for one release with
`DeprecationWarning` emitted on first access.

#### Scenario: Top-level dir() shows the slim surface

- **GIVEN** a Python user calls `dir(opts)` on a fresh
  `SimulationOptions`
- **WHEN** they grep for fields starting with letters (not
  underscores)
- **THEN** they see at most 15 top-level fields
- **AND** `advanced` appears as a sub-namespace handle

#### Scenario: Removed integrator values raise from Python

- **GIVEN** a Python user writes
  `opts.integrator = pulsim.Integrator.BDF3` in v0.11
- **WHEN** the assignment runs
- **THEN** a `DeprecationWarning` is raised

#### Scenario: Removed integrator values fail hard in v0.12

- **GIVEN** a Python user writes
  `pulsim.Integrator.BDF3` in v0.12 after the removal cycle
- **WHEN** the attribute is accessed
- **THEN** an `AttributeError` is raised because the enum value no
  longer exists

### Requirement: YAML / Python Surface Parity

The Python `SimulationOptions` API SHALL stay in surface parity with
the YAML `simulation:` block: every field the YAML parser recognises
SHALL have a Python equivalent on either `SimulationOptions` or
`SimulationOptions.advanced.*`, and every Python field SHALL be
expressible in YAML.

#### Scenario: YAML preset round-trips through Python

- **GIVEN** a YAML file with `simulation.preset: robust`
- **WHEN** the YAML parser loads it and exposes the result to Python
- **THEN** the resulting `SimulationOptions` is equivalent to
  `SimulationOptions.from_preset(pulsim.Preset.Robust, dt, tstop)`

#### Scenario: Python advanced field round-trips through YAML

- **GIVEN** Python code that sets
  `opts.advanced.newton.max_iterations = 100`
- **WHEN** the options are serialized to YAML via the parser's
  reverse path
- **THEN** the YAML contains
  `simulation.advanced.newton.max_iterations: 100`
