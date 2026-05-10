#pragma once

#include "pulsim/v1/runtime_circuit.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace pulsim::v1 {

// =============================================================================
// add-frequency-domain-analysis — public types
// =============================================================================
//
// Extracted from `simulation.hpp` so `SimulationOptions` can hold
// `std::vector<AcSweepOptions>` / `std::vector<FraOptions>` (Phase 7's YAML
// `analysis:` array) without requiring forward declarations of complete
// types. The Simulator methods that consume them (`linearize_around`,
// `run_ac_sweep`, `run_fra`) are still declared on the `Simulator` class
// in `simulation.hpp`.

/// Phase 1 of `add-frequency-domain-analysis`: a small-signal linearization
/// around an operating point, in descriptor (DAE) form `E·dx/dt = A·x + B·u`
/// with output `y = C·x + D·u`.
struct LinearSystem {
    SparseMatrix E;
    SparseMatrix A;
    SparseMatrix B;
    SparseMatrix C;
    SparseMatrix D;

    Index state_size = 0;
    Index input_size = 0;
    Index output_size = 0;

    Real t_linearization = 0.0;
    Vector x_linearization;

    std::string method;
    std::string failure_reason;

    [[nodiscard]] bool ok() const { return failure_reason.empty(); }
};

enum class AcSweepScale : std::uint8_t {
    Logarithmic,
    Linear,
};

/// Phase 2: AC small-signal sweep options.
struct AcSweepOptions {
    Real f_start = 1.0;
    Real f_stop  = 1e6;
    int  points_per_decade = 20;
    int  num_points        = 0;
    AcSweepScale scale = AcSweepScale::Logarithmic;

    /// Single perturbation source. Used when `perturbation_sources` is empty.
    std::string perturbation_source;

    /// Phase 4 of `add-frequency-domain-analysis`: multi-input
    /// transfer-function matrix. When non-empty, each source name produces
    /// its own B-column and the AC sweep returns one `AcMeasurement` per
    /// `(source, node)` pair (so the result is an `N_inputs × N_outputs`
    /// matrix flattened into a list, with both labels carried on each
    /// measurement). Useful for state-space identification and MIMO
    /// control design where you want `H[i,j](ω)` for several
    /// inputs / outputs at once.
    std::vector<std::string> perturbation_sources;

    std::vector<std::string> measurement_nodes;

    Real t_op = 0.0;
    bool use_dc_op = true;
    Vector x_op;

    /// Phase 7 of `add-frequency-domain-analysis`: optional human-readable
    /// label populated from the YAML `name:` field. Empty for sweeps
    /// constructed in code.
    std::string label;
};

struct AcMeasurement {
    std::string node;
    Index       state_index = -1;
    /// Phase 4 of `add-frequency-domain-analysis`: multi-input matrix.
    /// Empty for single-source sweeps (Phases 2 / 3 backward compat);
    /// populated with the source name when `perturbation_sources` was
    /// used so the consumer can group `H[i,j]` entries by either axis.
    std::string perturbation_source;
    std::vector<Real> magnitude_db;
    std::vector<Real> phase_deg;
    std::vector<Real> real_part;
    std::vector<Real> imag_part;
};

struct AcSweepResult {
    bool success = false;
    std::string failure_reason;
    std::vector<Real> frequencies;
    std::vector<AcMeasurement> measurements;
    int    total_factorizations = 0;
    int    total_solves         = 0;
    double wall_seconds         = 0.0;
};

/// Phase 3: Frequency Response Analysis options.
struct FraOptions {
    Real f_start = 1.0;
    Real f_stop  = 1e6;
    int  points_per_decade = 5;
    AcSweepScale scale = AcSweepScale::Logarithmic;

    std::string perturbation_source;
    Real perturbation_amplitude = 1e-2;
    Real perturbation_phase     = 0.0;

    std::vector<std::string> measurement_nodes;

    int n_cycles       = 6;
    int discard_cycles = 2;
    int samples_per_cycle = 32;

    /// Phase 7 of `add-frequency-domain-analysis`: optional human-readable
    /// label populated from the YAML `name:` field.
    std::string label;
};

struct FraMeasurement {
    std::string node;
    Index       state_index = -1;
    std::vector<Real> magnitude_db;
    std::vector<Real> phase_deg;
    std::vector<Real> real_part;
    std::vector<Real> imag_part;
};

struct FraResult {
    bool success = false;
    std::string failure_reason;
    std::vector<Real> frequencies;
    std::vector<FraMeasurement> measurements;
    int    total_transient_steps = 0;
    double wall_seconds          = 0.0;
};

}  // namespace pulsim::v1
