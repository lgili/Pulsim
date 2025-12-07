#pragma once

#include "pulsim/circuit.hpp"
#include "pulsim/types.hpp"

namespace pulsim {

// Switch state information
struct SwitchState {
    std::string name;
    bool is_closed;
    Real last_control_voltage;
    Real turn_on_time;   // Time of last turn-on event
    Real turn_off_time;  // Time of last turn-off event
};

// Dynamic element history for multi-step methods (BDF2, etc.)
struct DynamicHistory {
    Vector x_prev;      // Previous state x_{n-1}
    Vector x_prev2;     // Second previous state x_{n-2} (for BDF2)
    Vector i_prev;      // Previous currents (for Trapezoidal inductor)
    Real dt_prev = 0;   // Previous timestep (for variable step BDF2)
    bool has_prev2 = false;  // True if x_prev2 is valid
};

// MNA (Modified Nodal Analysis) matrix assembler
class MNAAssembler {
public:
    explicit MNAAssembler(const Circuit& circuit);

    // Assemble the DC (time-independent) part of the matrix
    void assemble_dc(SparseMatrix& G, Vector& b);

    // Assemble companion models for dynamic elements (capacitors, inductors)
    // using Backward Euler integration (default)
    void assemble_transient(SparseMatrix& G, Vector& b,
                           const Vector& x_prev, Real dt);

    // Assemble companion models with specified integration method
    void assemble_transient(SparseMatrix& G, Vector& b,
                           const DynamicHistory& history, Real dt,
                           IntegrationMethod method);

    // Update matrix for nonlinear elements (diodes, etc.)
    // Returns the Jacobian contributions
    void assemble_nonlinear(SparseMatrix& J, Vector& f,
                           const Vector& x);

    // Evaluate source values at a given time
    void evaluate_sources(Vector& b, Real time);

    // Get the total number of variables (nodes + branch currents)
    Index variable_count() const { return next_branch_idx_; }

    // Check if circuit has nonlinear elements
    bool has_nonlinear() const { return has_nonlinear_; }

    // Switch state management
    const std::vector<SwitchState>& switch_states() const { return switch_states_; }
    void update_switch_states(const Vector& x, Real time);
    bool check_switch_events(const Vector& x) const;

    // Get switch state by name
    SwitchState* find_switch_state(const std::string& name);
    const SwitchState* find_switch_state(const std::string& name) const;

private:
    // Stamp functions for each component type
    void stamp_resistor(std::vector<Triplet>& triplets, Vector& b,
                       const Component& comp);
    void stamp_capacitor_dc(std::vector<Triplet>& triplets, Vector& b,
                           const Component& comp);
    void stamp_capacitor_transient(std::vector<Triplet>& triplets, Vector& b,
                                   const Component& comp,
                                   const Vector& x_prev, Real dt);
    // Trapezoidal/BDF2 versions
    void stamp_capacitor_trapezoidal(std::vector<Triplet>& triplets, Vector& b,
                                     const Component& comp,
                                     const DynamicHistory& history, Real dt);
    void stamp_capacitor_bdf2(std::vector<Triplet>& triplets, Vector& b,
                              const Component& comp,
                              const DynamicHistory& history, Real dt);

    void stamp_inductor_dc(std::vector<Triplet>& triplets, Vector& b,
                          const Component& comp, Index branch_idx);
    void stamp_inductor_transient(std::vector<Triplet>& triplets, Vector& b,
                                  const Component& comp, Index branch_idx,
                                  const Vector& x_prev, Real dt);
    // Trapezoidal/BDF2 versions
    void stamp_inductor_trapezoidal(std::vector<Triplet>& triplets, Vector& b,
                                    const Component& comp, Index branch_idx,
                                    const DynamicHistory& history, Real dt);
    void stamp_inductor_bdf2(std::vector<Triplet>& triplets, Vector& b,
                             const Component& comp, Index branch_idx,
                             const DynamicHistory& history, Real dt);
    void stamp_voltage_source(std::vector<Triplet>& triplets, Vector& b,
                             const Component& comp, Index branch_idx, Real time);
    void stamp_current_source(Vector& b, const Component& comp, Real time);
    void stamp_diode(std::vector<Triplet>& triplets, Vector& f,
                    const Component& comp, const Vector& x);
    void stamp_diode_capacitance(std::vector<Triplet>& triplets, Vector& b,
                                 const Component& comp, const Vector& x,
                                 const Vector& x_prev, Real dt);
    void stamp_switch(std::vector<Triplet>& triplets, Vector& b,
                     const Component& comp, const SwitchState& state);
    void stamp_mosfet(std::vector<Triplet>& triplets, Vector& f,
                     const Component& comp, const Vector& x);
    void stamp_mosfet_capacitances(std::vector<Triplet>& triplets, Vector& b,
                                   const Component& comp, const Vector& x_prev, Real dt);
    void stamp_igbt(std::vector<Triplet>& triplets, Vector& f,
                   const Component& comp, const Vector& x);
    void stamp_transformer_dc(std::vector<Triplet>& triplets, Vector& b,
                             const Component& comp, Index branch_idx_p, Index branch_idx_s);
    void stamp_transformer_transient(std::vector<Triplet>& triplets, Vector& b,
                                    const Component& comp, Index branch_idx_p, Index branch_idx_s,
                                    const Vector& x_prev, Real dt);

    // Evaluate waveform at time t
    Real evaluate_waveform(const Waveform& waveform, Real time);

    const Circuit& circuit_;
    bool has_nonlinear_ = false;

    // Branch current indices for voltage sources and inductors
    std::unordered_map<std::string, Index> branch_indices_;
    Index next_branch_idx_ = 0;

    // Switch states
    std::vector<SwitchState> switch_states_;
};

}  // namespace pulsim
