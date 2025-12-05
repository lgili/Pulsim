#pragma once

#include "spicelab/circuit.hpp"
#include "spicelab/types.hpp"

namespace spicelab {

// MNA (Modified Nodal Analysis) matrix assembler
class MNAAssembler {
public:
    explicit MNAAssembler(const Circuit& circuit);

    // Assemble the DC (time-independent) part of the matrix
    void assemble_dc(SparseMatrix& G, Vector& b);

    // Assemble companion models for dynamic elements (capacitors, inductors)
    // using Backward Euler integration
    void assemble_transient(SparseMatrix& G, Vector& b,
                           const Vector& x_prev, Real dt);

    // Update matrix for nonlinear elements (diodes, etc.)
    // Returns the Jacobian contributions
    void assemble_nonlinear(SparseMatrix& J, Vector& f,
                           const Vector& x);

    // Evaluate source values at a given time
    void evaluate_sources(Vector& b, Real time);

    // Get the total number of variables (nodes + branch currents)
    Index variable_count() const { return circuit_.total_variables(); }

    // Check if circuit has nonlinear elements
    bool has_nonlinear() const { return has_nonlinear_; }

private:
    // Stamp functions for each component type
    void stamp_resistor(std::vector<Triplet>& triplets, Vector& b,
                       const Component& comp);
    void stamp_capacitor_dc(std::vector<Triplet>& triplets, Vector& b,
                           const Component& comp);
    void stamp_capacitor_transient(std::vector<Triplet>& triplets, Vector& b,
                                   const Component& comp,
                                   const Vector& x_prev, Real dt);
    void stamp_inductor_dc(std::vector<Triplet>& triplets, Vector& b,
                          const Component& comp, Index branch_idx);
    void stamp_inductor_transient(std::vector<Triplet>& triplets, Vector& b,
                                  const Component& comp, Index branch_idx,
                                  const Vector& x_prev, Real dt);
    void stamp_voltage_source(std::vector<Triplet>& triplets, Vector& b,
                             const Component& comp, Index branch_idx, Real time);
    void stamp_current_source(Vector& b, const Component& comp, Real time);
    void stamp_diode(std::vector<Triplet>& triplets, Vector& f,
                    const Component& comp, const Vector& x);

    // Evaluate waveform at time t
    Real evaluate_waveform(const Waveform& waveform, Real time);

    const Circuit& circuit_;
    bool has_nonlinear_ = false;

    // Branch current indices for voltage sources and inductors
    std::unordered_map<std::string, Index> branch_indices_;
    Index next_branch_idx_ = 0;
};

}  // namespace spicelab
