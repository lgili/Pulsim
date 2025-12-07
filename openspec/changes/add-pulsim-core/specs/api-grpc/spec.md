## ADDED Requirements

### Requirement: gRPC Service Definition
The API SHALL expose simulation capabilities via gRPC.

#### Scenario: Service initialization
- **WHEN** the API server starts
- **THEN** it listens on a configurable port (default 50051)
- **AND** serves the SimulatorService

#### Scenario: Health check
- **WHEN** a health check is requested
- **THEN** the service returns its status
- **AND** includes version information

### Requirement: Session Management
The API SHALL manage simulation sessions.

#### Scenario: Create session
- **WHEN** CreateSession is called with a model
- **THEN** a unique session_id is returned
- **AND** the model is validated and stored

#### Scenario: Session lifecycle
- **WHEN** a session is created
- **THEN** it can be: configured, started, paused, resumed, stopped
- **AND** session state is queryable

#### Scenario: Session cleanup
- **WHEN** a session is stopped or times out
- **THEN** resources are released
- **AND** results remain available for a configurable period

#### Scenario: Session listing
- **WHEN** ListSessions is called
- **THEN** all active sessions are returned
- **AND** with their current status

### Requirement: Simulation Control
The API SHALL provide simulation control operations.

#### Scenario: Start simulation
- **WHEN** StartSimulation is called with options
- **THEN** the transient simulation begins
- **AND** returns immediately (async operation)

#### Scenario: Simulation options
- **WHEN** options are provided
- **THEN** they include: tstart, tstop, dt, solver settings
- **AND** override defaults from the model

#### Scenario: Pause simulation
- **WHEN** PauseSimulation is called
- **THEN** the simulation pauses at the next safe point
- **AND** state is preserved

#### Scenario: Resume simulation
- **WHEN** ResumeSimulation is called
- **THEN** simulation continues from paused state
- **AND** optionally with updated parameters

#### Scenario: Stop simulation
- **WHEN** StopSimulation is called
- **THEN** simulation terminates gracefully
- **AND** partial results are available

### Requirement: Waveform Streaming
The API SHALL stream simulation results in real-time.

#### Scenario: Stream waveforms
- **WHEN** StreamWaveforms is called
- **THEN** a gRPC stream returns samples as computed
- **AND** each sample includes: time, values, signal names

#### Scenario: Selective streaming
- **WHEN** signal names are specified
- **THEN** only those signals are streamed
- **AND** reduces bandwidth

#### Scenario: Downsampling
- **WHEN** a decimation factor is specified
- **THEN** only every Nth sample is streamed
- **AND** full data is still stored

#### Scenario: Stream backpressure
- **WHEN** the client cannot keep up
- **THEN** the stream buffers up to a limit
- **AND** then drops samples or pauses simulation

### Requirement: Result Retrieval
The API SHALL provide access to completed results.

#### Scenario: Get result metadata
- **WHEN** GetResult is called
- **THEN** metadata is returned: signals, time range, sample count
- **AND** without transferring all data

#### Scenario: Download results
- **WHEN** DownloadResult is called with format
- **THEN** results are returned in the specified format
- **AND** supported formats: CSV, HDF5, Parquet

#### Scenario: Partial result access
- **WHEN** a time range is specified
- **THEN** only that portion of results is returned
- **AND** useful for large datasets

### Requirement: Model Management
The API SHALL manage circuit models.

#### Scenario: Upload model
- **WHEN** UploadModel is called with model data
- **THEN** the model is validated and stored
- **AND** a model_id is returned

#### Scenario: List models
- **WHEN** ListModels is called
- **THEN** all stored models are listed
- **AND** with metadata: name, date, description

#### Scenario: Get model
- **WHEN** GetModel is called with model_id
- **THEN** the model definition is returned
- **AND** in the original format

#### Scenario: Delete model
- **WHEN** DeleteModel is called
- **THEN** the model is removed
- **AND** associated sessions become invalid

### Requirement: Parameter Sweeps
The API SHALL support parameter sweep operations.

#### Scenario: Define sweep
- **WHEN** CreateSweep is called with parameters and ranges
- **THEN** a sweep job is created
- **AND** returns sweep_id

#### Scenario: Run sweep
- **WHEN** RunSweep is called
- **THEN** multiple simulations run (optionally in parallel)
- **AND** progress is reportable

#### Scenario: Sweep results
- **WHEN** GetSweepResults is called
- **THEN** aggregated results are returned
- **AND** organized by parameter values

### Requirement: Error Handling
The API SHALL provide meaningful error responses.

#### Scenario: Validation error
- **WHEN** an invalid model is submitted
- **THEN** a detailed error is returned
- **AND** includes location and suggestion

#### Scenario: Simulation error
- **WHEN** simulation fails (non-convergence, etc.)
- **THEN** an error is returned with diagnostics
- **AND** partial results may be available

#### Scenario: Resource limit
- **WHEN** resource limits are exceeded
- **THEN** a resource exhausted error is returned
- **AND** includes which limit was hit

### Requirement: REST Gateway
The API SHALL optionally expose REST endpoints.

#### Scenario: REST fallback
- **WHEN** a client cannot use gRPC
- **THEN** REST endpoints mirror gRPC methods
- **AND** generated via gRPC-gateway

#### Scenario: OpenAPI documentation
- **WHEN** REST is enabled
- **THEN** OpenAPI/Swagger spec is available
- **AND** at /swagger endpoint

### Requirement: Authentication and Authorization
The API SHALL support authentication.

#### Scenario: Token authentication
- **WHEN** auth is enabled
- **THEN** requests require a JWT token
- **AND** token is validated on each request

#### Scenario: Per-user quotas
- **WHEN** quotas are configured
- **THEN** usage is tracked per user
- **AND** limits are enforced (concurrent sessions, CPU time)

### Requirement: Observability
The API SHALL expose metrics and traces.

#### Scenario: Prometheus metrics
- **WHEN** metrics endpoint is accessed
- **THEN** Prometheus-format metrics are returned
- **AND** include: request count, latency, active sessions

#### Scenario: Request tracing
- **WHEN** tracing is enabled
- **THEN** OpenTelemetry traces are emitted
- **AND** span context is propagated
