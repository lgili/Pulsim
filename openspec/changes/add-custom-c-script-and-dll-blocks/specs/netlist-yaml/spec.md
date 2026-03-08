## ADDED Requirements

### Requirement: C_BLOCK YAML Component Type
The YAML netlist parser SHALL accept `type: C_BLOCK` as a valid signal-domain
component type.  A valid `C_BLOCK` component MUST declare `n_inputs` (integer ≥ 1)
and `n_outputs` (integer ≥ 1) under `parameters`.  It MUST specify exactly one
source for the implementation: `lib_path` (path to a pre-compiled shared library)
or `source` (path to a `.c` file to be compiled at load time).  Specifying both
or neither SHALL produce a validation diagnostic (ERROR level for both-specified;
INFO level for neither-specified, which is valid for Python-only usage).
An optional `extra_cflags` list of strings may be included.

The component pins MUST be declared in `pins` with:
- Indices 0 through `n_inputs - 1` as input pins (name convention: `IN0..IN{n-1}`)
- Indices `n_inputs` through `n_inputs + n_outputs - 1` as output pins
  (name convention: `OUT0..OUT{m-1}`; single-output blocks may use `OUT`)

#### Scenario: Valid C_BLOCK with lib_path parses without errors
- **GIVEN** a YAML circuit containing:
  ```yaml
  - id: cb1
    type: C_BLOCK
    parameters:
      n_inputs: 1
      n_outputs: 1
      lib_path: controllers/gain.so
    pins:
      - { index: 0, name: IN0 }
      - { index: 1, name: OUT }
  ```
- **WHEN** the YAML is parsed
- **THEN** no ERROR-level diagnostics are emitted
- **AND** the component appears in the parsed circuit with correct `n_inputs=1`, `n_outputs=1`

#### Scenario: Valid C_BLOCK with source path parses without errors
- **GIVEN** a YAML circuit containing a `C_BLOCK` with `source: controllers/gain.c`
  and no `lib_path`
- **WHEN** the YAML is parsed
- **THEN** no ERROR-level diagnostics are emitted

#### Scenario: Both lib_path and source specified produces error diagnostic
- **GIVEN** a YAML circuit containing a `C_BLOCK` that specifies both `lib_path`
  and `source`
- **WHEN** the YAML is parsed
- **THEN** an ERROR-level `ParseDiagnostic` is emitted
- **AND** the diagnostic message SHALL mention that `lib_path` and `source` are
  mutually exclusive

#### Scenario: Missing n_inputs produces error diagnostic
- **GIVEN** a YAML `C_BLOCK` component with no `n_inputs` parameter
- **WHEN** the YAML is parsed
- **THEN** an ERROR-level `ParseDiagnostic` is emitted naming the missing field

#### Scenario: Neither lib_path nor source produces INFO diagnostic only
- **GIVEN** a YAML `C_BLOCK` that specifies neither `lib_path` nor `source`
- **WHEN** the YAML is parsed
- **THEN** an INFO-level (not ERROR-level) `ParseDiagnostic` is emitted
- **AND** the circuit is still usable for Python-only simulation workflows

#### Scenario: n_inputs less than 1 is rejected
- **GIVEN** a YAML `C_BLOCK` with `n_inputs: 0`
- **WHEN** the YAML is parsed
- **THEN** an ERROR-level `ParseDiagnostic` is emitted requiring a positive integer

#### Scenario: extra_cflags forwarded to compiler
- **GIVEN** a YAML `C_BLOCK` with `source: block.c` and `extra_cflags: ["-DDEBUG=1"]`
- **WHEN** the circuit is loaded and `SignalEvaluator` compiles the block
- **THEN** the `-DDEBUG=1` flag is passed to the compiler

### Requirement: C_BLOCK YAML Round-trip Serialisation
A `Circuit` object constructed to include a `C_BLOCK` component MUST serialise
back to YAML and re-parse to an equivalent circuit dict.  The round-trip MUST
preserve `n_inputs`, `n_outputs`, `lib_path` (or `source`), and pin definitions.

#### Scenario: Serialise and re-parse C_BLOCK circuit
- **GIVEN** a `Circuit` with a `C_BLOCK`, a `VOLTAGE_PROBE`, and a `PWM_GENERATOR`
  wired in sequence
- **WHEN** the circuit is serialised to YAML and immediately re-parsed
- **THEN** the resulting circuit dict is structurally identical to the original
- **AND** `n_inputs` and `n_outputs` are preserved exactly
