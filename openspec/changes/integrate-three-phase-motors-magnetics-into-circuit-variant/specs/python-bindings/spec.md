## ADDED Requirements

### Requirement: pybind11 exposure of three-phase grid devices
`python/bindings.cpp` SHALL expose `grid::ThreePhaseSource`,
`grid::ProgrammableThreePhaseSource`, and the `grid::PhaseSequence` enum. The
exposure SHALL include a matching `Circuit.add_three_phase_source(...)` method
binding.

#### Scenario: Python smoke test
- **WHEN** a Python user constructs a `pulsim.Circuit`, calls
  `circuit.add_three_phase_source("Vgrid", ["a","b","c"], pulsim.ThreePhaseSource())`,
  and runs a transient
- **THEN** the result exposes three named signals (one per phase) with the expected
  120° phase separation

### Requirement: pybind11 exposure of motors and mechanical devices
`python/bindings.cpp` SHALL expose `motors::PMSM`, `PMSM_FOC`, `DC_Motor`, and
`Mechanical`, plus their associated parameter / state structs. The exposure SHALL
include matching `Circuit.add_pmsm`, `add_pmsm_foc`, `add_dc_motor`, and
`add_mechanical` method bindings.

#### Scenario: Python PMSM-FOC roundtrip
- **WHEN** a Python user instantiates `pulsim.PMSM_FOC(params)`, adds it to a Circuit,
  and runs a transient
- **THEN** the rotor-speed signal is reachable via the same result API used for
  electrical probes

### Requirement: pybind11 exposure of saturable magnetics + BH curve catalog
`python/bindings.cpp` SHALL expose `magnetic::SaturableTransformer`,
`HysteresisInductor`, `BHCurve`, and `CoreCatalog` (with its entries). The exposure
SHALL include matching `Circuit.add_saturable_transformer` and `add_hysteresis_inductor`
method bindings.

#### Scenario: Python catalog access
- **WHEN** a Python user looks up `pulsim.CoreCatalog.entries()`
- **THEN** the returned sequence lists each known core material with its associated
  BHCurve
- **AND** the user can construct a `SaturableTransformer` initialized from one of
  the catalog entries

### Requirement: Module-level re-exports
`python/pulsim/__init__.py` SHALL re-export the new classes in `__all__` so they
appear under `pulsim.<ClassName>` for users.

#### Scenario: Top-level imports succeed
- **WHEN** a Python user runs `from pulsim import ThreePhaseSource, PMSM, SaturableTransformer`
- **THEN** the imports succeed without `ImportError`
