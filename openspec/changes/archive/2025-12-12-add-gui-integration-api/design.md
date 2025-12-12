# GUI Integration API Design

## Context

PulsimGui requires a rich API from the Pulsim core library to provide a professional-grade user experience. The current API is focused on batch simulation but lacks features needed for interactive GUI operation:

- Real-time progress feedback with detailed metrics
- Pause/resume capability for long simulations
- Component metadata for building UI editors
- Circuit validation with detailed error reporting
- Position storage for schematic layout persistence

This design document captures the technical decisions for implementing these features.

## Goals

1. **Non-breaking changes** - All new APIs are additions; existing code continues to work
2. **Thread-safe** - GUI runs simulation in background thread; APIs must be safe
3. **Efficient** - Progress callbacks should not significantly impact simulation performance
4. **Python-friendly** - All features must be accessible from Python bindings

## Non-Goals

1. Real-time parameter modification during simulation (out of scope)
2. Distributed simulation support
3. GUI-specific rendering code in core library

## Key Decisions

### 1. Simulation State Machine

**Decision**: Extend `SimulationControl` to a full state machine.

Current state:
```cpp
class SimulationControl {
    virtual bool should_stop() const = 0;
    virtual bool should_pause() const = 0;
    virtual void wait_until_resumed() = 0;
};
```

New design:
```cpp
enum class SimulationState {
    Idle,       // Not started
    Running,    // Actively simulating
    Paused,     // Paused, can resume
    Stopping,   // Stop requested, finishing current step
    Completed,  // Finished successfully
    Error       // Terminated with error
};

class SimulationController {
public:
    // State queries (thread-safe)
    SimulationState state() const;
    bool is_running() const { return state() == SimulationState::Running; }
    bool is_paused() const { return state() == SimulationState::Paused; }

    // Control commands (thread-safe)
    void request_pause();
    void request_resume();
    void request_stop();

    // Wait for state change (for synchronization)
    void wait_for_state(SimulationState target, int timeout_ms = -1);

    // Internal: called by simulator
    void set_state(SimulationState state);
    bool check_and_handle_pause();  // Returns false if should stop

private:
    std::atomic<SimulationState> state_{SimulationState::Idle};
    std::mutex mutex_;
    std::condition_variable cv_;
};
```

**Rationale**:
- `std::atomic` for lock-free state reads (frequent in simulation loop)
- Condition variable for efficient waiting
- Clear state transitions prevent race conditions

### 2. Progress Callback Structure

**Decision**: Create a rich progress struct with all GUI-needed information.

```cpp
struct SimulationProgress {
    // Time progress
    Real current_time;          // Current simulation time (s)
    Real total_time;            // Target end time (s)
    double progress_percent;    // 0.0 to 100.0

    // Step statistics
    int64_t steps_completed;
    int64_t total_steps_estimate;  // -1 if unknown

    // Current step info
    int newton_iterations;      // Iterations for current step
    bool convergence_warning;   // True if >10 iterations

    // Wall-clock timing
    double elapsed_seconds;
    double estimated_remaining_seconds;  // -1 if unknown

    // Memory usage (optional, may be expensive)
    int64_t memory_bytes;       // -1 if not tracked
};

// Callback type
using ProgressCallback = std::function<void(const SimulationProgress& progress)>;

// Configuration
struct ProgressCallbackConfig {
    ProgressCallback callback;
    double min_interval_ms = 100.0;  // Minimum time between callbacks
    int min_steps = 100;             // Minimum steps between callbacks
    bool include_memory = false;     // Track memory usage (slower)
};
```

**Rationale**:
- Configurable frequency prevents GUI from being overwhelmed
- Estimated remaining time helps users decide whether to wait or cancel
- Memory tracking is optional due to overhead

### 3. Component Metadata System

**Decision**: Static metadata registry with compile-time registration.

```cpp
// Parameter types
enum class ParameterType {
    Real,       // Floating point number
    Integer,    // Integer
    Boolean,    // True/false
    Enum,       // One of several choices
    String      // Text (e.g., model name)
};

struct ParameterMetadata {
    std::string name;           // Internal name (e.g., "resistance")
    std::string display_name;   // GUI display (e.g., "Resistance")
    std::string description;    // Help text
    ParameterType type;

    // For Real/Integer
    std::optional<double> default_value;
    std::optional<double> min_value;
    std::optional<double> max_value;
    std::string unit;           // e.g., "ohm", "F", "H"

    // For Enum
    std::vector<std::string> enum_values;

    // Validation
    bool required = true;
};

struct PinMetadata {
    std::string name;           // e.g., "anode", "drain"
    std::string description;    // e.g., "Positive terminal"
};

struct ComponentMetadata {
    ComponentType type;
    std::string name;           // Internal name
    std::string display_name;   // GUI display
    std::string description;    // Help text
    std::string category;       // e.g., "Passive", "Semiconductor"

    std::vector<PinMetadata> pins;
    std::vector<ParameterMetadata> parameters;

    // GUI hints
    std::string symbol_id;      // Reference for symbol rendering
    bool has_loss_model;        // Supports power loss calculation
    bool has_thermal_model;     // Supports thermal simulation
};

// Registry interface
class ComponentRegistry {
public:
    static const ComponentRegistry& instance();

    const ComponentMetadata& get(ComponentType type) const;
    std::vector<ComponentType> all_types() const;
    std::vector<ComponentType> types_in_category(const std::string& category) const;

private:
    std::unordered_map<ComponentType, ComponentMetadata> metadata_;
};
```

**Rationale**:
- Singleton registry initialized once at startup
- Metadata is read-only after initialization
- Categories enable hierarchical library display

### 4. Schematic Position Storage

**Decision**: Add optional position data to Circuit, stored alongside components.

```cpp
struct SchematicPosition {
    double x = 0.0;
    double y = 0.0;
    int orientation = 0;        // 0, 90, 180, 270 degrees
    bool mirrored = false;      // Horizontal mirror
};

// In Circuit class:
class Circuit {
public:
    // Position management
    void set_position(const std::string& component_name, const SchematicPosition& pos);
    std::optional<SchematicPosition> get_position(const std::string& component_name) const;
    bool has_position(const std::string& component_name) const;

    // Bulk operations
    std::unordered_map<std::string, SchematicPosition> all_positions() const;
    void set_all_positions(const std::unordered_map<std::string, SchematicPosition>& positions);

private:
    std::unordered_map<std::string, SchematicPosition> positions_;
};
```

**JSON format extension**:
```json
{
  "components": [
    {
      "name": "R1",
      "type": "resistor",
      "nodes": ["in", "out"],
      "params": {"resistance": 1000},
      "position": {"x": 100, "y": 200, "orientation": 0, "mirrored": false}
    }
  ]
}
```

**Rationale**:
- Positions are optional (simulator ignores them)
- JSON extension is backward-compatible
- Bulk operations for efficient load/save

### 5. Validation API

**Decision**: Comprehensive validation with categorized diagnostics.

```cpp
enum class DiagnosticSeverity {
    Error,      // Must be fixed before simulation
    Warning,    // May cause issues but simulation can proceed
    Info        // Informational message
};

enum class DiagnosticCode {
    // Errors
    E_NO_GROUND,            // No ground reference node
    E_VOLTAGE_SOURCE_LOOP,  // Voltage sources form a loop
    E_INDUCTOR_LOOP,        // Inductors form a loop with V sources
    E_NO_DC_PATH,           // Node has no DC path to ground
    E_INVALID_PARAMETER,    // Parameter out of valid range
    E_UNKNOWN_NODE,         // Referenced node doesn't exist
    E_DUPLICATE_NAME,       // Component name already used

    // Warnings
    W_FLOATING_NODE,        // Node with single connection
    W_SHORT_CIRCUIT,        // Very low impedance path
    W_HIGH_VOLTAGE,         // Unusually high voltage expected
    W_MISSING_IC,           // No initial condition specified

    // Info
    I_IDEAL_SWITCH,         // Using ideal switch model
    I_NO_LOSS_MODEL,        // Loss calculation not available
};

struct Diagnostic {
    DiagnosticSeverity severity;
    DiagnosticCode code;
    std::string message;

    // Location info
    std::string component_name;  // Empty if circuit-level
    std::string node_name;       // Empty if component-level
    std::string parameter_name;  // Empty if not parameter-related

    // Structured data for GUI
    std::vector<std::string> related_components;
};

struct ValidationResult {
    bool is_valid;              // True if no errors (warnings OK)
    std::vector<Diagnostic> diagnostics;

    // Convenience
    bool has_errors() const;
    bool has_warnings() const;
    std::vector<Diagnostic> errors() const;
    std::vector<Diagnostic> warnings() const;
};

// In Circuit class:
ValidationResult validate_detailed() const;

// Parameter validation
ValidationResult validate_parameter(
    ComponentType type,
    const std::string& param_name,
    double value
);
```

**Rationale**:
- Structured diagnostics enable GUI highlighting
- Diagnostic codes allow localization
- Severity levels let user decide whether to proceed

### 6. Result Streaming Configuration

**Decision**: Add streaming options to SimulationOptions.

```cpp
struct StreamingConfig {
    // Decimation
    int decimation_factor = 1;      // Store every Nth point

    // Rolling buffer
    bool use_rolling_buffer = false;
    int64_t max_points = 100000;    // Max points in buffer

    // Callback configuration
    double callback_interval_ms = 0;  // 0 = every stored point
};

// Extended SimulationOptions
struct SimulationOptions {
    // ... existing fields ...

    // Streaming configuration
    StreamingConfig streaming;

    // Progress callback
    ProgressCallbackConfig progress;
};
```

**Rationale**:
- Decimation reduces memory for long simulations
- Rolling buffer enables indefinite streaming
- Separate from progress callback (different use case)

### 7. Enhanced SimulationResult

**Decision**: Add comprehensive metadata to results.

```cpp
struct SignalInfo {
    std::string name;           // e.g., "V(out)"
    std::string type;           // "voltage", "current", "power"
    std::string unit;           // "V", "A", "W"
    std::string component;      // Associated component (if any)
    std::vector<std::string> nodes;  // Related nodes
};

struct SolverInfo {
    IntegrationMethod method;
    double abstol;
    double reltol;
    bool adaptive_timestep;
};

struct SimulationResult {
    // ... existing fields ...

    // Enhanced metadata
    std::vector<SignalInfo> signal_info;
    SolverInfo solver_info;

    // Performance metrics
    double average_newton_iterations;
    int convergence_failures;       // Steps that needed damping
    int timestep_reductions;        // Adaptive timestep reductions
    double peak_memory_bytes;

    // Event log
    std::vector<SwitchEvent> events;
};
```

## Migration Plan

All changes are additive:
1. New enums and structs can be added without breaking ABI
2. New methods on existing classes extend the API
3. Existing code continues to work unchanged
4. Python bindings add new exports incrementally

## Testing Strategy

1. **Unit tests**: Each new class/struct tested in isolation
2. **Integration tests**: Full simulation with GUI-like control flow
3. **Thread safety tests**: Concurrent access from multiple threads
4. **Performance tests**: Ensure callbacks don't slow simulation >5%

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Callback overhead | Performance degradation | Configurable frequency, benchmark |
| Thread safety bugs | Crashes, data corruption | Careful mutex design, TSAN testing |
| API surface growth | Maintenance burden | Minimal API, good documentation |
| Python binding complexity | Harder to use | Pythonic wrappers, examples |

## Open Questions

1. **Should validation run automatically before simulation?**
   - *Recommendation*: No, let GUI control when to validate

2. **Should we support cancellation mid-Newton-iteration?**
   - *Recommendation*: No, only at timestep boundaries (simpler)

3. **How to handle very large result sets in Python?**
   - *Recommendation*: Streaming API with numpy array chunks
