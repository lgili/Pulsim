#pragma once

#include "pulsim/types.hpp"
#include "pulsim/circuit.hpp"
#include "pulsim/mna.hpp"
#include <complex>
#include <vector>
#include <functional>

namespace pulsim {

// Complex number type for AC analysis
using Complex = std::complex<Real>;
using ComplexVector = Eigen::VectorXcd;
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexSparseMatrix = Eigen::SparseMatrix<Complex, Eigen::ColMajor>;

// =============================================================================
// AC Analysis Result Types
// =============================================================================

// Single frequency point result
struct ACPoint {
    Real frequency;         // Frequency in Hz
    ComplexVector response; // Complex response at each node/branch
};

// Frequency sweep result
struct ACResult {
    std::vector<Real> frequencies;          // Frequency points (Hz)
    std::vector<std::string> signal_names;  // Names of signals
    std::vector<ComplexVector> data;        // Complex response at each frequency

    // Convenience accessors
    size_t num_frequencies() const { return frequencies.size(); }
    size_t num_signals() const { return signal_names.size(); }

    // Get magnitude at frequency index for signal index
    Real magnitude(size_t freq_idx, size_t signal_idx) const;

    // Get phase in degrees at frequency index for signal index
    Real phase_deg(size_t freq_idx, size_t signal_idx) const;

    // Get magnitude in dB at frequency index for signal index
    Real magnitude_db(size_t freq_idx, size_t signal_idx) const;

    // Get transfer function (output/input) magnitude and phase
    // output_idx, input_idx are signal indices
    Real transfer_magnitude_db(size_t freq_idx, size_t output_idx, size_t input_idx) const;
    Real transfer_phase_deg(size_t freq_idx, size_t output_idx, size_t input_idx) const;

    // Status
    SolverStatus status = SolverStatus::Success;
    std::string error_message;
};

// =============================================================================
// Frequency Sweep Options
// =============================================================================

enum class FrequencySweepType {
    Linear,     // Linear spacing: f = fstart + i * (fstop - fstart) / (npoints - 1)
    Decade,     // Logarithmic by decades: npoints per decade
    Octave,     // Logarithmic by octaves: npoints per octave
    List,       // Explicit list of frequencies
};

struct ACOptions {
    FrequencySweepType sweep_type;
    Real fstart;            // Start frequency (Hz)
    Real fstop;             // Stop frequency (Hz)
    int npoints;            // Number of points (or points per decade/octave)
    std::vector<Real> frequency_list;  // For List sweep type

    ACOptions()
        : sweep_type(FrequencySweepType::Decade)
        , fstart(1.0)
        , fstop(1e6)
        , npoints(10) {}

    // Generate frequency points based on options
    std::vector<Real> generate_frequencies() const;
};

// =============================================================================
// AC Small-Signal Analyzer
// =============================================================================

class ACAnalyzer {
public:
    explicit ACAnalyzer(const Circuit& circuit);

    // Set operating point for linearization
    // This should be the DC solution from DC operating point analysis
    void set_operating_point(const Vector& x_op);

    // Run AC analysis with given options
    ACResult analyze(const ACOptions& options);

    // Analyze at a single frequency
    ACPoint analyze_at_frequency(Real frequency);

    // Build linearized small-signal model at operating point
    // Returns (G_lin, C_lin) where Y(s) = G_lin + s*C_lin
    void build_linearized_model(SparseMatrix& G_lin, SparseMatrix& C_lin);

    // Build complex admittance matrix at frequency: Y(jw) = G + jw*C
    void build_admittance_matrix(ComplexSparseMatrix& Y, Real frequency);

    // Get the input source index (for transfer function calculations)
    Index find_source_index(const std::string& source_name) const;

    const Circuit& circuit() const { return circuit_; }

private:
    // Linearize a nonlinear device at operating point
    void linearize_diode(const Component& comp, std::vector<Triplet>& G_triplets,
                        std::vector<Triplet>& C_triplets);
    void linearize_mosfet(const Component& comp, std::vector<Triplet>& G_triplets,
                         std::vector<Triplet>& C_triplets);
    void linearize_igbt(const Component& comp, std::vector<Triplet>& G_triplets,
                       std::vector<Triplet>& C_triplets);

    // Add capacitor stamp to C matrix
    void stamp_capacitor_ac(const Component& comp, std::vector<Triplet>& C_triplets);

    // Add inductor stamp to G and C matrices
    void stamp_inductor_ac(const Component& comp, std::vector<Triplet>& G_triplets,
                          std::vector<Triplet>& C_triplets);

    const Circuit& circuit_;
    Vector x_op_;  // Operating point
    bool has_operating_point_ = false;

    // Cached linearized model
    SparseMatrix G_lin_;
    SparseMatrix C_lin_;
    bool model_valid_ = false;
};

// =============================================================================
// Bode Plot Data Generator
// =============================================================================

struct BodeData {
    std::vector<Real> frequencies;      // Hz
    std::vector<Real> magnitude_db;     // dB
    std::vector<Real> phase_deg;        // degrees

    // Stability margins (computed if phase crosses -180° or magnitude crosses 0dB)
    Real gain_margin_db = std::numeric_limits<Real>::quiet_NaN();
    Real phase_margin_deg = std::numeric_limits<Real>::quiet_NaN();
    Real gain_crossover_freq = std::numeric_limits<Real>::quiet_NaN();
    Real phase_crossover_freq = std::numeric_limits<Real>::quiet_NaN();

    bool has_gain_margin() const { return !std::isnan(gain_margin_db); }
    bool has_phase_margin() const { return !std::isnan(phase_margin_deg); }
};

// Extract Bode plot data from AC result
// output_signal: Name or index of output signal
// input_signal: Name or index of input signal (typically AC source)
BodeData extract_bode_data(const ACResult& result, size_t output_idx, size_t input_idx);
BodeData extract_bode_data(const ACResult& result, const std::string& output_signal,
                           const std::string& input_signal);

// Calculate stability margins from Bode data
void calculate_stability_margins(BodeData& bode);

// =============================================================================
// Convenience Functions
// =============================================================================

// Run AC analysis on a circuit (convenience function)
ACResult ac_analysis(const Circuit& circuit, const ACOptions& options,
                     const Vector& operating_point = Vector());

}  // namespace pulsim
