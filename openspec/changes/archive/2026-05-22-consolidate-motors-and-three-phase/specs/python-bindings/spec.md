## ADDED Requirements

### Requirement: Python Exposure for the Motor and Magnetics Family

The Python module `pulsim` SHALL expose `Circuit.add_*` methods for every motor and magnetics device registered in `DeviceVariant`, plus the per-device parameter dataclasses. The exposure SHALL stay aligned with the C++ surface — adding or removing a device class in `components/` requires a matching update in `python/bindings.cpp` and `python/pulsim/__init__.py::__all__`.

#### Scenario: Full motor family on Python
- **GIVEN** the consolidated codebase
- **WHEN** a user imports `pulsim` and constructs a `Circuit`
- **THEN** the methods `add_dc_motor`, `add_pmsm`, `add_pmsm_foc`, `add_bldc_motor`, `add_induction_motor`, and `add_mechanical` are present
- **AND** each method accepts a corresponding `*Params` dataclass exposed from the same module

#### Scenario: Full magnetics family on Python
- **GIVEN** the consolidated codebase
- **WHEN** a user imports `pulsim` and constructs a `Circuit`
- **THEN** the methods `add_saturable_inductor`, `add_saturable_transformer`, `add_hysteresis_inductor` are present
- **AND** the `__all__` list of `python/pulsim/__init__.py` enumerates the matching `*Params` types

### Requirement: PmsmSteadyStateParams Removed From the Python Surface

After the consolidation lands, the Python `pulsim` module SHALL NOT expose `PmsmSteadyStateParams`, `Circuit.add_pmsm_steady_state`, or any equivalent helper. Steady-state operating-point semantics for PMSM are achieved through `Circuit.add_pmsm` with pinned mechanical state.

#### Scenario: Removed binding
- **GIVEN** the consolidated `python/pulsim/__init__.py`
- **WHEN** a user imports `pulsim`
- **THEN** `pulsim.PmsmSteadyStateParams` raises `AttributeError`
- **AND** `Circuit.add_pmsm_steady_state` raises `AttributeError`
