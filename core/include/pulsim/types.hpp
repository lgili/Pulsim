#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulsim {

// Basic numeric types
using Real = double;
using Index = std::int32_t;

// Sparse matrix types (CSC format for efficiency with Eigen)
using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::ColMajor>;
using Triplet = Eigen::Triplet<Real>;

// Dense vector/matrix types
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// Node identifier
using NodeId = std::string;
constexpr const char* GROUND_NODE = "0";

// Component types
enum class ComponentType {
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    VCVS,  // Voltage-Controlled Voltage Source
    VCCS,  // Voltage-Controlled Current Source
    CCVS,  // Current-Controlled Voltage Source
    CCCS,  // Current-Controlled Current Source
    Diode,
    Switch,
    MOSFET,
    IGBT,
    Transformer,
};

// Analysis types
enum class AnalysisType {
    DC,       // DC operating point
    Transient, // Time-domain transient
    AC,       // Small-signal AC
};

// Solver status
enum class SolverStatus {
    Success,
    MaxIterationsReached,
    SingularMatrix,
    NumericalError,
};

// Simulation state machine (for GUI integration)
enum class SimulationState {
    Idle,       // Not started
    Running,    // Actively simulating
    Paused,     // Paused, can resume
    Stopping,   // Stop requested, finishing current step
    Completed,  // Finished successfully
    Error       // Terminated with error
};

// Simulation result for a single timestep
struct TimePoint {
    Real time;
    Vector state;  // Node voltages and branch currents
};

// Integration methods for transient analysis
enum class IntegrationMethod {
    BackwardEuler,    // First-order implicit (default), O(dt)
    Trapezoidal,      // Second-order implicit, O(dt^2)
    BDF2,             // Second-order BDF, O(dt^2), more stable than Trap
    GEAR2,            // Alias for Trapezoidal
};

// Streaming configuration for result storage (defined in simulation_control.hpp)
// Forward declare here for use in SimulationOptions
struct StreamingConfig;

// Simulation options
struct SimulationOptions {
    // Time parameters
    Real tstart = 0.0;
    Real tstop = 1.0;
    Real dt = 1e-6;
    Real dtmin = 1e-15;
    Real dtmax = 1e-3;

    // Solver tolerances
    Real abstol = 1e-12;
    Real reltol = 1e-3;
    int max_newton_iterations = 50;
    Real damping_factor = 1.0;

    // Initial conditions
    bool use_ic = false;  // If true, skip DC operating point

    // Integration method
    IntegrationMethod integration_method = IntegrationMethod::BackwardEuler;

    // Adaptive timestep control
    bool adaptive_timestep = false;  // Enable error-based adaptive stepping
    Real lte_rtol = 1e-3;            // Local truncation error relative tolerance
    Real lte_atol = 1e-9;            // Local truncation error absolute tolerance

    // Output options
    std::vector<std::string> output_signals;

    // Streaming configuration (for GUI result storage)
    // decimation_factor: store every Nth point (1 = all points)
    int streaming_decimation = 1;

    // Rolling buffer: if true, keep only the last N points
    bool streaming_rolling_buffer = false;
    int64_t streaming_max_points = 100000;

    // Progress callback throttling (used with run_transient_with_progress)
    // The actual callback is passed to run_transient_with_progress()
    double progress_min_interval_ms = 100.0;  // Minimum ms between callbacks
    int progress_min_steps = 100;             // Minimum steps between callbacks
    bool progress_include_memory = false;     // Track memory usage (slower)
};

// Signal metadata for result interpretation
struct SignalInfo {
    std::string name;           // e.g., "V(out)"
    std::string type;           // "voltage", "current", "power"
    std::string unit;           // "V", "A", "W"
    std::string component;      // Associated component (if any)
    std::vector<std::string> nodes;  // Related nodes
};

// Solver configuration info
struct SolverInfo {
    IntegrationMethod method = IntegrationMethod::BackwardEuler;
    Real abstol = 1e-12;
    Real reltol = 1e-3;
    bool adaptive_timestep = false;
};

// Simulation event types
enum class SimulationEventType {
    SwitchClose,    // Switch closed
    SwitchOpen,     // Switch opened
    Convergence,    // Convergence issue
    TimestepChange  // Timestep was adjusted
};

// Simulation event for GUI timeline display
struct SimulationEvent {
    Real time = 0.0;
    SimulationEventType type = SimulationEventType::SwitchClose;
    std::string component;         // Component name (for switch events)
    std::string description;       // Human-readable description
    Real value1 = 0.0;             // Context-dependent value (e.g., voltage)
    Real value2 = 0.0;             // Context-dependent value (e.g., current)
};

// Result container
struct SimulationResult {
    std::vector<Real> time;
    std::vector<std::string> signal_names;
    std::vector<Vector> data;  // Each vector is one timestep

    // Metadata
    Real total_time_seconds = 0.0;
    int total_steps = 0;
    int newton_iterations_total = 0;
    SolverStatus final_status = SolverStatus::Success;
    std::string error_message;

    // Enhanced metadata (for GUI)
    std::vector<SignalInfo> signal_info;
    SolverInfo solver_info;

    // Performance metrics
    double average_newton_iterations = 0.0;
    int convergence_failures = 0;       // Steps that needed damping
    int timestep_reductions = 0;        // Adaptive timestep reductions
    int64_t peak_memory_bytes = -1;     // -1 if not tracked

    // Events (for GUI timeline display)
    std::vector<SimulationEvent> events;

    // Convenience methods
    size_t num_signals() const { return signal_names.size(); }
    size_t num_points() const { return time.size(); }
    size_t num_events() const { return events.size(); }
};

}  // namespace pulsim
