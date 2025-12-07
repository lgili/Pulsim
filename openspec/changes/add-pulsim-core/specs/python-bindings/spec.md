## ADDED Requirements

### Requirement: Native Python Bindings
The library SHALL provide native Python bindings via pybind11.

#### Scenario: Import module
- **WHEN** `import pulsim` is executed
- **THEN** the module loads without error
- **AND** exposes core classes and functions

#### Scenario: Direct simulation
- **WHEN** `pulsim.simulate(model, options)` is called
- **THEN** the kernel runs in-process
- **AND** returns results as numpy arrays

#### Scenario: Low latency
- **WHEN** using native bindings
- **THEN** overhead is minimal (no serialization)
- **AND** suitable for real-time control loops

### Requirement: gRPC Python Client
The library SHALL provide a gRPC client for remote simulation.

#### Scenario: Connect to server
- **WHEN** `Client.connect(url)` is called
- **THEN** a gRPC connection is established
- **AND** server health is verified

#### Scenario: Remote simulation
- **WHEN** `client.simulate(model, options)` is called
- **THEN** the model is sent to the server
- **AND** results are streamed back

#### Scenario: Connection pooling
- **WHEN** multiple simulations are requested
- **THEN** connections are reused
- **AND** configurable pool size

### Requirement: Unified API
The library SHALL provide a unified API for both modes.

#### Scenario: Mode selection
- **WHEN** Pulsim is instantiated
- **THEN** mode is selectable: 'native' or 'remote'
- **AND** default is 'native' if kernel is available

#### Scenario: API consistency
- **WHEN** switching between modes
- **THEN** the same methods work
- **AND** only connection setup differs

### Requirement: Model Definition
The library SHALL provide Pythonic model definition.

#### Scenario: Programmatic model
- **WHEN** a model is built with Python code
- **THEN** components are added with methods: `circuit.add_resistor(...)`
- **AND** connections are made by node names

#### Scenario: Model from file
- **WHEN** a model file path is provided
- **THEN** the file is loaded and parsed
- **AND** supports JSON, YAML formats

#### Scenario: Model from string
- **WHEN** a model string is provided
- **THEN** it is parsed directly
- **AND** format is auto-detected

#### Scenario: Model validation
- **WHEN** a model is constructed
- **THEN** it can be validated before simulation
- **AND** errors are reported as exceptions

### Requirement: Simulation Options
The library SHALL accept simulation options in Pythonic form.

#### Scenario: Basic options
- **WHEN** options are provided
- **THEN** they include: tstop, dt, reltol, abstol
- **AND** as keyword arguments or dict

#### Scenario: Analysis type
- **WHEN** analysis type is specified
- **THEN** supported types: 'tran', 'dc', 'ac'
- **AND** each has type-specific options

#### Scenario: Solver selection
- **WHEN** a solver is specified
- **THEN** it overrides the default
- **AND** options: 'be', 'trap', 'gear', 'sundials'

### Requirement: Result Handling
The library SHALL provide convenient result access.

#### Scenario: Numpy arrays
- **WHEN** simulation completes
- **THEN** results are numpy arrays
- **AND** with named columns for signals

#### Scenario: Pandas DataFrame
- **WHEN** `results.to_dataframe()` is called
- **THEN** a pandas DataFrame is returned
- **AND** with time as index

#### Scenario: Xarray Dataset
- **WHEN** `results.to_xarray()` is called
- **THEN** an xarray Dataset is returned
- **AND** with proper dimensions and coordinates

#### Scenario: Signal access
- **WHEN** a specific signal is accessed
- **THEN** `results['V(out)']` returns that waveform
- **AND** time is `results.time`

### Requirement: Streaming Interface
The library SHALL support streaming results during simulation.

#### Scenario: Callback interface
- **WHEN** a callback is registered
- **THEN** it is called with each sample
- **AND** `on_sample(time, values)`

#### Scenario: Iterator interface
- **WHEN** iterating over simulation
- **THEN** samples are yielded as computed
- **AND** `for sample in sim.run(): ...`

#### Scenario: Async interface
- **WHEN** using asyncio
- **THEN** `async for sample in sim.run_async(): ...`
- **AND** non-blocking iteration

### Requirement: Jupyter Integration
The library SHALL integrate with Jupyter notebooks.

#### Scenario: Rich display
- **WHEN** results are displayed in Jupyter
- **THEN** a summary table is shown
- **AND** with basic statistics

#### Scenario: Interactive plotting
- **WHEN** `results.plot()` is called
- **THEN** an interactive plot is displayed
- **AND** using plotly or bokeh

#### Scenario: Live streaming plot
- **WHEN** streaming with plotting enabled
- **THEN** the plot updates in real-time
- **AND** using ipywidgets

#### Scenario: Progress bar
- **WHEN** simulation runs in notebook
- **THEN** a progress bar shows advancement
- **AND** using tqdm or similar

### Requirement: Error Handling
The library SHALL provide clear error handling.

#### Scenario: Convergence failure
- **WHEN** Newton iteration fails
- **THEN** `ConvergenceError` is raised
- **AND** includes iteration history and last residual

#### Scenario: Model error
- **WHEN** model is invalid
- **THEN** `ModelError` is raised
- **AND** includes validation details

#### Scenario: Connection error
- **WHEN** remote server is unreachable
- **THEN** `ConnectionError` is raised
- **AND** with retry suggestion

### Requirement: Parameter Sweeps
The library SHALL provide convenient sweep interfaces.

#### Scenario: Single parameter sweep
- **WHEN** `pulsim.sweep(model, 'R1', [100, 200, 500])` is called
- **THEN** multiple simulations run
- **AND** results are organized by parameter

#### Scenario: Multi-parameter sweep
- **WHEN** multiple parameters are swept
- **THEN** all combinations are simulated
- **AND** results are a multi-dimensional array

#### Scenario: Parallel execution
- **WHEN** parallel=True is specified
- **THEN** sweeps run concurrently
- **AND** using multiprocessing or remote workers

### Requirement: Loss and Thermal Access
The library SHALL provide access to power electronics quantities.

#### Scenario: Loss summary
- **WHEN** `results.losses` is accessed
- **THEN** per-device loss breakdown is available
- **AND** includes conduction, switching, total

#### Scenario: Efficiency
- **WHEN** `results.efficiency(input_port, output_port)` is called
- **THEN** efficiency is computed
- **AND** based on average power

#### Scenario: Thermal results
- **WHEN** thermal simulation was enabled
- **THEN** `results.temperatures` has Tj waveforms
- **AND** peak temperatures are summarized

### Requirement: Export Capabilities
The library SHALL export results in various formats.

#### Scenario: CSV export
- **WHEN** `results.to_csv(path)` is called
- **THEN** results are saved as CSV
- **AND** with header row

#### Scenario: HDF5 export
- **WHEN** `results.to_hdf5(path)` is called
- **THEN** results are saved in HDF5 format
- **AND** with metadata attributes

#### Scenario: Parquet export
- **WHEN** `results.to_parquet(path)` is called
- **THEN** results are saved as Parquet
- **AND** efficient for large datasets
