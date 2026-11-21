## ADDED Requirements

### Requirement: Lazy Import Of The Scope GUI Stack

`import pulsim` SHALL NOT import `pulsim.scope`, pyqtgraph or a Qt
binding. `pulsim.LiveScope` SHALL be resolved on first attribute
access; when pyqtgraph is not installed that access SHALL raise an
`AttributeError` whose message names the optional dependency, so that
`hasattr(pulsim, "LiveScope")` is `False` as it is today. `pulsim.scope` SHALL
remain the plotting function before and after `LiveScope` is resolved,
and `from pulsim.scope import LiveScope` SHALL keep working.

#### Scenario: GUI stack not loaded at import
- **GIVEN** a fresh interpreter with pyqtgraph installed
- **WHEN** `import pulsim` completes
- **THEN** neither `"pyqtgraph"` nor `"pulsim.scope"` is in
  `sys.modules`

#### Scenario: LiveScope on demand
- **WHEN** code reads `pulsim.LiveScope`
- **THEN** the class is returned and `"pyqtgraph"` is now in
  `sys.modules`
- **AND** `callable(pulsim.scope)` is still true

#### Scenario: Missing optional dependency
- **GIVEN** an environment without pyqtgraph
- **WHEN** code reads `pulsim.LiveScope`
- **THEN** an `AttributeError` naming pyqtgraph is raised,
  `hasattr(pulsim, "LiveScope")` is `False`, and every other `pulsim`
  attribute works

### Requirement: Step-Loop Callback Report

Every result returned by `simulate()` SHALL carry
`step_loop_python_callbacks`, a tuple of the names of the Python
callables that were installed in the engine's step loop for that run.
The tuple SHALL be empty when the step loop was fully native, and SHALL
be `("unknown",)` when the engine branch cannot determine it.

#### Scenario: Native run
- **GIVEN** a circuit driven by a `SwitchSchedule` and no observer
- **WHEN** `simulate()` returns
- **THEN** `res.step_loop_python_callbacks == ()`

#### Scenario: Python switch function
- **GIVEN** `switch_fn=lambda t: mask`
- **WHEN** `simulate()` returns
- **THEN** `"switch_fn"` is in `res.step_loop_python_callbacks`

#### Scenario: Every engine reports
- **WHEN** the same native circuit is run with each engine that
  `simulate()` can dispatch to
- **THEN** each result has the attribute
