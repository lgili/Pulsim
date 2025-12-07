## ADDED Requirements

### Requirement: CLI Application
The system SHALL provide a command-line interface for simulation.

#### Scenario: Basic invocation
- **WHEN** `pulsim run circuit.json` is executed
- **THEN** the circuit is simulated
- **AND** results are written to stdout or file

#### Scenario: Help display
- **WHEN** `pulsim --help` is executed
- **THEN** usage information is displayed
- **AND** all commands and options are listed

#### Scenario: Version display
- **WHEN** `pulsim --version` is executed
- **THEN** version and build info is shown
- **AND** includes kernel version

### Requirement: Run Command
The CLI SHALL provide a `run` command for simulation.

#### Scenario: Run with defaults
- **WHEN** `pulsim run circuit.json` is executed
- **THEN** simulation runs with default options
- **AND** options from the model file are used

#### Scenario: Run with options
- **WHEN** `pulsim run circuit.json --tstop 0.01 --dt 1e-6` is executed
- **THEN** command-line options override model defaults
- **AND** simulation runs with specified parameters

#### Scenario: Output file
- **WHEN** `pulsim run circuit.json -o result.csv` is executed
- **THEN** results are written to the specified file
- **AND** format is inferred from extension

#### Scenario: Output format
- **WHEN** `--format csv|hdf5|parquet` is specified
- **THEN** results are written in that format
- **AND** overrides extension inference

#### Scenario: Signal selection
- **WHEN** `--signals V(out),I(L1)` is specified
- **THEN** only those signals are output
- **AND** reduces file size

### Requirement: Validate Command
The CLI SHALL provide a `validate` command for model checking.

#### Scenario: Validate model
- **WHEN** `pulsim validate circuit.json` is executed
- **THEN** the model is parsed and validated
- **AND** errors are reported if any

#### Scenario: Validate with schema
- **WHEN** `--schema path/to/schema.json` is specified
- **THEN** model is validated against custom schema
- **AND** additional checks are applied

### Requirement: Convert Command
The CLI SHALL provide a `convert` command for format conversion.

#### Scenario: Convert netlist format
- **WHEN** `pulsim convert circuit.json -o circuit.yaml` is executed
- **THEN** the model is converted between formats
- **AND** semantic content is preserved

#### Scenario: Convert results format
- **WHEN** `pulsim convert result.csv -o result.hdf5` is executed
- **THEN** results are converted between formats
- **AND** data integrity is maintained

### Requirement: Info Command
The CLI SHALL provide an `info` command for inspection.

#### Scenario: Model info
- **WHEN** `pulsim info circuit.json` is executed
- **THEN** summary is displayed: components, nodes, parameters
- **AND** potential issues are warned

#### Scenario: Device info
- **WHEN** `pulsim info --device MOSFET` is executed
- **THEN** device model documentation is shown
- **AND** includes parameters and defaults

#### Scenario: Library info
- **WHEN** `pulsim info --library` is executed
- **THEN** available device models are listed
- **AND** with brief descriptions

### Requirement: Sweep Command
The CLI SHALL provide a `sweep` command for parameter sweeps.

#### Scenario: Single parameter sweep
- **WHEN** `pulsim sweep circuit.json --param R1=100:1000:10` is executed
- **THEN** simulation runs for each value
- **AND** results are aggregated

#### Scenario: Output directory
- **WHEN** `--outdir results/` is specified
- **THEN** each run's results go to a separate file
- **AND** named by parameter values

#### Scenario: Parallel execution
- **WHEN** `--parallel 4` is specified
- **THEN** up to 4 simulations run concurrently
- **AND** uses available CPU cores

### Requirement: Serve Command
The CLI SHALL provide a `serve` command for API server.

#### Scenario: Start server
- **WHEN** `pulsim serve` is executed
- **THEN** the gRPC server starts
- **AND** listens on default port 50051

#### Scenario: Server options
- **WHEN** `--port 8080 --workers 4` is specified
- **THEN** server uses custom port and worker count
- **AND** logs connection info

#### Scenario: Server with REST
- **WHEN** `--rest-gateway` is specified
- **THEN** REST gateway is also started
- **AND** on a separate port

### Requirement: Output Formatting
The CLI SHALL support various output modes.

#### Scenario: Quiet mode
- **WHEN** `--quiet` is specified
- **THEN** only errors are printed
- **AND** progress is suppressed

#### Scenario: Verbose mode
- **WHEN** `--verbose` is specified
- **THEN** detailed progress is shown
- **AND** includes solver iterations

#### Scenario: JSON output
- **WHEN** `--json` is specified for info commands
- **THEN** output is machine-readable JSON
- **AND** suitable for scripting

### Requirement: Progress Reporting
The CLI SHALL report simulation progress.

#### Scenario: Progress bar
- **WHEN** running interactively
- **THEN** a progress bar shows simulation advancement
- **AND** includes time and ETA

#### Scenario: Non-interactive progress
- **WHEN** running non-interactively
- **THEN** periodic status lines are printed
- **AND** at configurable intervals

### Requirement: Error Reporting
The CLI SHALL provide clear error messages.

#### Scenario: Parse error
- **WHEN** a model has syntax errors
- **THEN** the error location is shown
- **AND** with context lines

#### Scenario: Convergence failure
- **WHEN** simulation fails to converge
- **THEN** diagnostic info is provided
- **AND** suggestions for resolution

#### Scenario: Exit codes
- **WHEN** CLI completes
- **THEN** exit code indicates status
- **AND** 0=success, 1=error, 2=validation failure

### Requirement: Configuration File
The CLI SHALL support configuration files.

#### Scenario: Default config
- **WHEN** `~/.pulsim/config.yaml` exists
- **THEN** settings are loaded as defaults
- **AND** command-line overrides config

#### Scenario: Project config
- **WHEN** `.pulsim.yaml` exists in current directory
- **THEN** it overrides user config
- **AND** project-specific settings apply

#### Scenario: Config command
- **WHEN** `pulsim config --list` is executed
- **THEN** current configuration is displayed
- **AND** shows source of each setting
