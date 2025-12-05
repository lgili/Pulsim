#pragma once

#include "spicelab/circuit.hpp"
#include "spicelab/mna.hpp"
#include "spicelab/solver.hpp"
#include "spicelab/types.hpp"
#include <functional>

namespace spicelab {

// Callback for streaming results during simulation
using SimulationCallback = std::function<void(Real time, const Vector& state)>;

// Event callback for switch state changes
struct SwitchEvent {
    std::string switch_name;
    Real time;
    bool new_state;  // true = closed, false = open
    Real voltage;    // Voltage across switch at event time
    Real current;    // Current through switch at event time
};
using EventCallback = std::function<void(const SwitchEvent& event)>;

// Power loss accumulator
struct PowerLosses {
    Real conduction_loss = 0.0;   // Energy lost to conduction (J)
    Real switching_loss = 0.0;    // Energy lost to switching (J)
    Real total_loss() const { return conduction_loss + switching_loss; }
};

// Main simulation engine
class Simulator {
public:
    explicit Simulator(const Circuit& circuit, const SimulationOptions& options = {});

    // Run DC operating point analysis
    NewtonResult dc_operating_point();

    // Run transient simulation
    SimulationResult run_transient();

    // Run transient with streaming callback
    SimulationResult run_transient(SimulationCallback callback);

    // Run transient with event callback
    SimulationResult run_transient(SimulationCallback callback, EventCallback event_callback);

    // Access the circuit
    const Circuit& circuit() const { return circuit_; }

    // Access options
    const SimulationOptions& options() const { return options_; }
    void set_options(const SimulationOptions& options) { options_ = options; }

    // Access MNA assembler (for switch states)
    const MNAAssembler& assembler() const { return assembler_; }
    MNAAssembler& assembler() { return assembler_; }

    // Get accumulated power losses
    const PowerLosses& power_losses() const { return power_losses_; }

private:
    // Single timestep of transient simulation
    NewtonResult step(Real time, Real dt, const Vector& x_prev);

    // Build system function for Newton solver
    void build_system(const Vector& x, Vector& f, SparseMatrix& J,
                     Real time, Real dt, const Vector& x_prev);

    // Detect and handle switch events using bisection
    bool find_event_time(Real t_start, Real t_end, const Vector& x_start,
                        Real& t_event, Vector& x_event);

    // Calculate switching losses at an event
    Real calculate_switching_loss(const Component& comp, const SwitchState& state,
                                  Real voltage, Real current, bool turning_on);

    // Accumulate conduction losses
    void accumulate_conduction_losses(const Vector& x, Real dt);

    const Circuit& circuit_;
    SimulationOptions options_;
    MNAAssembler assembler_;
    NewtonSolver newton_solver_;

    // Cached matrices for reuse
    SparseMatrix G_;  // Conductance matrix
    Vector b_;        // RHS vector

    // Power loss tracking
    PowerLosses power_losses_;
};

// Convenience function for quick simulation
SimulationResult simulate(const Circuit& circuit, const SimulationOptions& options = {});

}  // namespace spicelab
