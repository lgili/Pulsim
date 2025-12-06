#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace spicelab {

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

// Simulation result for a single timestep
struct TimePoint {
    Real time;
    Vector state;  // Node voltages and branch currents
};

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

    // Output options
    std::vector<std::string> output_signals;
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
};

}  // namespace spicelab
