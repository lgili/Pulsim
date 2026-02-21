## ADDED Requirements

### Requirement: Python-Only Supported Runtime Surface
Python bindings SHALL be the only supported user-facing runtime interface for simulation workflows.

#### Scenario: User follows supported workflow
- **WHEN** a user executes simulation workflows documented as supported
- **THEN** all workflows are available through the Python package interface
- **AND** documentation does not require direct C++ or legacy CLI usage

### Requirement: Full v1 Configuration Exposure
Python bindings SHALL expose all v1 solver, integrator, periodic, and thermal configuration required by declared converter workflows.

#### Scenario: Configure advanced converter run
- **WHEN** Python code configures declared v1 runtime options for a converter case
- **THEN** equivalent options are available through bindings without undocumented C++-only fallback

### Requirement: Converter Component and Thermal API Coverage
Python bindings SHALL expose APIs to build and run declared converter component sets with associated thermal and loss models.

#### Scenario: Build electro-thermal converter in Python
- **WHEN** a benchmark converter case uses declared electrical and thermal components
- **THEN** the case can be constructed and executed from Python without legacy adapters

### Requirement: Deprecated Surface Retirement Policy
Deprecated Python entrypoints SHALL include a migration path and versioned removal policy.

#### Scenario: Deprecated entrypoint present
- **WHEN** an entrypoint is marked for removal
- **THEN** bindings and docs provide a supported replacement and removal version
- **AND** CI includes migration coverage during the deprecation window
