## ADDED Requirements

### Requirement: Typed Stub Distribution

The `pulsim` package SHALL ship a PEP 561 marker file `py.typed` and a
type stub `_pulsim.pyi` for the native module `pulsim._pulsim`, in the
wheel and in the `build/python/pulsim` build-tree copy. The stub SHALL
be generated from the built extension by `scripts/generate_stubs.py`
using a pinned `pybind11-stubgen` version, SHALL be committed to the
repository, and SHALL declare the members the Python layer attaches to
native classes (at least `SimulationResult.v`, `SimulationResult.i`,
`SimulationResult.power`, `SimulationResult.engine_used` and
`SimulationResult.engine_route_reason`). The stub SHALL NOT import
`pybind11_stubgen` or any module that is not a runtime dependency of
`pulsim`.

#### Scenario: Wheel carries the typing files
- **WHEN** a wheel is built and its file list is inspected
- **THEN** it contains `pulsim/py.typed` and `pulsim/_pulsim.pyi`
- **AND** the cibuildwheel smoke test asserts that both files exist next
  to the installed `pulsim/__init__.py`

#### Scenario: Type checker rejects a guessed argument name
- **GIVEN** an environment with the wheel and mypy installed
- **WHEN** mypy checks `b.add_sine_voltage_source("V", "a", "gnd",
  amplitude=10.0, frequency=50.0)`
- **THEN** mypy reports an unexpected keyword argument `amplitude` on
  `CircuitBuilder.add_sine_voltage_source`

#### Scenario: Python-attached members are visible to the type checker
- **WHEN** mypy checks `res = p.simulate(b, t_end=1e-3);
  print(res.engine_used, res.v("vc"))`
- **THEN** mypy reports no error on `engine_used` or `v`

#### Scenario: Stub type-checks cleanly
- **WHEN** `mypy --follow-imports=silent python/pulsim/_pulsim.pyi`
  runs
- **THEN** it reports zero errors

### Requirement: Stub Freshness Verification

The test suite SHALL fail when `_pulsim.pyi` no longer matches the
compiled extension. A dependency-free test SHALL run on every CI leg and
check the following:

- the stub parses with `ast.parse`;
- the stub contains no object-address reprs (`0x…`), no C++ qualified
  names (`::`) and no `pybind11_stubgen` import;
- every public class, function, method and property of
  `pulsim._pulsim` is declared in the stub;
- for each native callable, the parameter names, their order and
  whether each has a default are the same in the stub as in the
  signature on the first line of the callable's `__doc__`.

In addition, one designated CI job SHALL run
`scripts/generate_stubs.py --check`, which regenerates the stub and
fails on any textual difference or on any stubgen error.

#### Scenario: New binding without a stub update
- **GIVEN** a commit that adds a method `CircuitBuilder.add_foo` to the
  bindings but does not regenerate the stub
- **WHEN** the Python test suite runs
- **THEN** the stub freshness test fails, naming `CircuitBuilder.add_foo`
- **AND** the failure message gives the regeneration command
  `python scripts/generate_stubs.py`

#### Scenario: Reordered parameters
- **GIVEN** a binding change that swaps two parameters of an existing
  method
- **WHEN** the stub freshness test runs against the stale stub
- **THEN** it fails and reports both parameter orders

#### Scenario: Exact regeneration check
- **WHEN** the designated CI job runs `scripts/generate_stubs.py --check`
  on an up-to-date tree
- **THEN** it exits 0 and reports no difference

### Requirement: Binding Signatures Render as Valid Python

Every pybind11 signature in `pulsim._pulsim` SHALL render, on the
first line of its `__doc__`, as a valid Python function signature. In
particular:

- no parameter or return type renders as a C++ name; classes used as
  property or parameter types SHALL be registered (or forward-declared
  with `py::class_`) before the members that use them;
- no default value renders as an object address; non-literal defaults
  SHALL use `py::arg_v` with a Python-readable description;
- no required parameter follows a defaulted one unless it is
  keyword-only (`py::kw_only()`);
- a `None` default SHALL be typed as optional.

These fixes SHALL NOT change kernel behaviour and SHALL NOT break any
positional call used in the repository.

#### Scenario: Property types render as Python classes
- **WHEN** Python reads `pulsim.CircuitBuilder.graph.fget.__doc__`
- **THEN** the return type is `pulsim._pulsim.Graph`, not
  `pulsim::topology::Graph`

#### Scenario: Defaults render readably
- **WHEN** Python reads the first line of
  `pulsim.CircuitBuilder.run_preflight.__doc__`
- **THEN** the default of `options` reads `PreflightOptions()` and
  contains no `0x` address

#### Scenario: Stub generation reports no errors
- **WHEN** `scripts/generate_stubs.py` runs `pybind11-stubgen` with
  `--exit-code`
- **THEN** stubgen reports zero errors and writes the stub
