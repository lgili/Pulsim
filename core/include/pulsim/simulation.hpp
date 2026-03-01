#pragma once

#include "pulsim/circuit.hpp"
#include "pulsim/mna.hpp"
#include "pulsim/simulation_control.hpp"
#include "pulsim/solver.hpp"
#include "pulsim/types.hpp"
#include <functional>

namespace pulsim {

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

// Lookup table for switching energy interpolation
struct SwitchingLossTable {
    std::vector<Real> voltages;   // Voltage breakpoints (V)
    std::vector<Real> currents;   // Current breakpoints (A)
    std::vector<Real> eon_data;   // Turn-on energy (J), size = voltages.size() * currents.size()
    std::vector<Real> eoff_data;  // Turn-off energy (J), size = voltages.size() * currents.size()
    std::vector<Real> err_data;   // Reverse recovery energy (J), for diodes

    // 2D interpolation for switching energy
    Real interpolate_eon(Real voltage, Real current) const;
    Real interpolate_eoff(Real voltage, Real current) const;
    Real interpolate_err(Real voltage, Real current) const;

    // Check if table has data
    bool has_eon() const { return !eon_data.empty(); }
    bool has_eoff() const { return !eoff_data.empty(); }
    bool has_err() const { return !err_data.empty(); }
};

// Power loss accumulator with detailed breakdown
struct PowerLosses {
    Real conduction_loss = 0.0;      // Energy lost to conduction (J)
    Real turn_on_loss = 0.0;         // Energy lost to turn-on switching (J)
    Real turn_off_loss = 0.0;        // Energy lost to turn-off switching (J)
    Real reverse_recovery_loss = 0.0; // Diode reverse recovery energy (J)

    // Convenience accessors
    Real switching_loss() const { return turn_on_loss + turn_off_loss + reverse_recovery_loss; }
    Real total_loss() const { return conduction_loss + switching_loss(); }

    // Per-device loss breakdown (device name -> energy in J)
    std::unordered_map<std::string, Real> device_conduction_loss;
    std::unordered_map<std::string, Real> device_switching_loss;
};

// Efficiency calculation result
struct EfficiencyResult {
    Real input_energy = 0.0;      // Total energy from sources (J)
    Real output_energy = 0.0;     // Energy delivered to load (J)
    Real loss_energy = 0.0;       // Total losses (J)
    Real efficiency = 0.0;        // Output / Input (0 to 1)
    Real average_input_power = 0.0;   // Average input power (W)
    Real average_output_power = 0.0;  // Average output power (W)
    Real average_loss_power = 0.0;    // Average loss power (W)
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
    SimulationResult run_transient(SimulationCallback callback, EventCallback event_callback,
                                   SimulationControl* control = nullptr);

    // Run transient with progress callback (for GUI integration)
    SimulationResult run_transient_with_progress(
        SimulationCallback callback,
        EventCallback event_callback,
        SimulationControl* control,
        const ProgressCallbackConfig& progress_config);

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

    // Register switching loss lookup table for a device
    void set_switching_loss_table(const std::string& device_name, const SwitchingLossTable& table);

    // Calculate efficiency (call after simulation)
    // load_nodes: list of node names where load power is dissipated
    // source_names: list of source component names to measure input power
    EfficiencyResult calculate_efficiency(const SimulationResult& result,
                                          const std::vector<std::string>& load_nodes,
                                          const std::vector<std::string>& source_names) const;

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

    // Calculate diode reverse recovery loss
    Real calculate_reverse_recovery_loss(const Component& diode_comp, Real current, Real di_dt);

    // Accumulate conduction losses
    void accumulate_conduction_losses(const Vector& x, Real dt);

    // Track diode states for reverse recovery detection
    void update_diode_states(const Vector& x, const Vector& x_prev, Real dt);

    const Circuit& circuit_;
    SimulationOptions options_;
    MNAAssembler assembler_;
    NewtonSolver newton_solver_;

    // Cached matrices for reuse
    SparseMatrix G_;  // Conductance matrix
    Vector b_;        // RHS vector

    // Power loss tracking
    PowerLosses power_losses_;

    // Switching loss lookup tables (device name -> table)
    std::unordered_map<std::string, SwitchingLossTable> switching_loss_tables_;

    // Diode state tracking for reverse recovery detection
    struct DiodeState {
        Real prev_current = 0.0;
        bool was_conducting = false;
    };
    std::unordered_map<std::string, DiodeState> diode_states_;
};

// Convenience function for quick simulation
SimulationResult simulate(const Circuit& circuit, const SimulationOptions& options = {});

}  // namespace pulsim
