## ADDED Requirements

### Requirement: Python Template Builder API
Python bindings SHALL expose `pulsim.templates.<name>(...)` builder functions that return ready-to-simulate `Circuit` objects.

#### Scenario: Buck builder from Python
- **WHEN** Python code calls `pulsim.templates.buck(vin=48, vout=12, pout=240, fsw=100e3)`
- **THEN** a `Circuit` is returned with all template wiring expanded
- **AND** the circuit can be used with `Simulator(circuit, options)` directly

#### Scenario: Builder reports parameters used
- **GIVEN** a builder call with partial parameters relying on auto-design
- **WHEN** `circuit.template_metadata()` is called
- **THEN** the returned dict includes the resolved parameter set including auto-designed values

### Requirement: Template Listing and Introspection
Python bindings SHALL expose `pulsim.list_templates()` and `pulsim.describe_template(name)` for runtime discovery.

#### Scenario: List templates
- **WHEN** Python calls `pulsim.list_templates()`
- **THEN** a list of `(name, version, description)` tuples is returned
- **AND** the list is stable across runs

#### Scenario: Describe template
- **WHEN** Python calls `pulsim.describe_template("buck_template")`
- **THEN** a structured object is returned with parameter schema, defaults, and links to docs

### Requirement: Template Use with Parameter Sweep
Templates SHALL be composable with the parameter-sweep API (post-`add-monte-carlo-parameter-sweep` landing).

#### Scenario: Buck parameter sweep
- **GIVEN** Python code that creates a template via builder and varies `Lout`
- **WHEN** the parameter sweep runs
- **THEN** each instance is a fresh template expansion
- **AND** results aggregate per parameter setting
