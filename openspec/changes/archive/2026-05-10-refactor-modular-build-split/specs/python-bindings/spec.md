## ADDED Requirements

### Requirement: Modular pybind11 Binding Layout
The Python bindings SHALL be split across multiple translation units under `python/bindings/`, each ≤500 lines, with a single `main.cpp` orchestrator file containing the `PYBIND11_MODULE` entry.

#### Scenario: Modular layout
- **WHEN** the project is built
- **THEN** `python/bindings/` contains per-domain files (`devices.cpp`, `control.cpp`, `simulation.cpp`, `parser.cpp`, `solver.cpp`, `thermal.cpp`, `loss.cpp`, `analysis.cpp`, `main.cpp`)
- **AND** each non-main file ≤500 lines
- **AND** `main.cpp` calls `register_<domain>(m)` from each module

#### Scenario: Editing one domain
- **GIVEN** a one-line edit in `bindings/devices.cpp`
- **WHEN** an incremental build runs
- **THEN** only `bindings/devices.cpp` recompiles
- **AND** the link step pulls cached objects for unchanged domains

### Requirement: Build-Time Performance Targets
The build SHALL meet target metrics for clean and incremental builds:

#### Scenario: Clean build target
- **WHEN** a clean build runs on a CI baseline machine
- **THEN** total wallclock is at most 75% of the pre-refactor baseline
- **AND** the metric is recorded in CI artifacts

#### Scenario: Incremental build target
- **GIVEN** a one-line edit in any single source file
- **WHEN** an incremental build runs
- **THEN** wallclock is ≤10% of clean-build wallclock
- **AND** only the touched object plus the link step are recompiled

### Requirement: Public API Stability During Refactor
The bindings refactor SHALL preserve the existing Python public API and ABI bit-for-bit.

#### Scenario: Existing user notebooks
- **GIVEN** any existing example notebook in `examples/notebooks/`
- **WHEN** executed against the refactored bindings
- **THEN** results are bit-identical to pre-refactor
- **AND** no user-facing import or call requires modification
