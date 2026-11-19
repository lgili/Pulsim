// SPDX-License-Identifier: MIT
//
// Pulsim — Phase A.3: MNA-linearised AC sweep.
//
// For LTI (or linearised-at-x_op) plants, the small-signal transfer
// function H(jω) = C·(jωI − A)⁻¹·B can be evaluated directly from the
// cached MNA matrix without running a transient per frequency. This
// header provides the C++ kernel implementation:
//
//   1. Refuse any kind with no small-signal model (it would be dropped,
//      not linearised).
//   2. Take (G, C) from the transient assembler `assemble_segment_split`
//      at the caller's switch mask; the mass matrix is E = C/2 (the
//      split stores the trapezoidal 2/dt coefficients), so capacitors,
//      inductors and coupled windings are the transient's own. A Newton
//      device adds ∂f/∂x, ∂q/∂x of its law at the DC operating point
//      (`analysis/linearize.hpp`).
//   3. For each test frequency ω_k = 2π·f_k, build the complex matrix
//      `M_k = jω_k·E − A`, solve `M_k · X = B` (with B sourced from
//      the unit-impulse on the input branch), and read off the output
//      node voltage as H(jω_k).
//
// The complex sparse path uses the in-house PulsimSparseLuSolver
// templated on `std::complex<Real>` (per
// `openspec/changes/add-pulsim-complex-sparse-lu`, v1.4.0+). Each
// frequency is a fresh `analyze + factorize + solve` cycle. For
// typical SMPS state sizes (≤ 32 states) the per-frequency cost is
// ~10 µs, making a 100-point sweep finish in a few milliseconds.
// That's ~200× faster than the existing swept-sine path. With this
// migration `Eigen::SparseLU<std::complex<Real>>` is no longer pulled
// in by production code — it remains available only as a paper-
// comparison baseline (`Backend::Eigen`).
//
// Header-only, C++20.

#pragma once

#include "pulsim/pwl/branch_structure.hpp"
#include "pulsim/analysis/cancellation.hpp"
#include "pulsim/analysis/linearize.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/dc_operating_point.hpp"
#include "pulsim/pwl/external_device_history.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <Eigen/Sparse>
#include <complex>
#include <format>
#include <stdexcept>
#include <vector>

namespace pulsim::analysis {

using Complex = std::complex<Real>;
// `VectorT` lives in `pulsim::` (`pulsim/numeric/dense.hpp`);
// `MatrixT` lives in `pulsim::sparse::` (`pulsim/sparse/matrix.hpp`).
// Both are templated on Scalar per `add-pulsim-complex-sparse-lu`.
using ComplexVector = pulsim::VectorT<Complex>;
// ColMajor (matches `sparse::MatrixT<Complex>`) so the in-house
// `PulsimSparseLuSolverT<Complex>` can consume it without transpose-
// and-copy. v1.3.0 used RowMajor here as an Eigen::SparseLU
// convention — Eigen accepted it via an internal transpose; the
// in-house solver requires ColMajor (CSC). Triplet-based assembly
// elsewhere in this header is layout-independent.
using ComplexSparseMatrix = sparse::MatrixT<Complex>;

/// Result of a single-input / single-output MNA AC sweep.
///
/// Parallel arrays of length `freqs.size()`. `H[k]` is the complex
/// transfer function value at `freqs[k]` (Hz).
struct MnaSweepResult {
    std::vector<Real> freqs;
    std::vector<Complex> H;
};

namespace detail {

/// Refuse a kind the sweep has no small-signal model for (the
/// `kKindCaps` ac_sweep column): `linearize` stamps the linear elements,
/// switches at the fixed mask, and the Newton kinds whose law it
/// linearises at the DC point; any other kind would be dropped, giving the
/// transfer function of another circuit.
inline void require_linear_for_sweep(const topology::Graph& graph,
                                      const pwl::DevicePool& pool) {
    require_branch_form_supported(graph, pool, pwl::Engine::AcSweep);
    const auto bad = find_unsupported(graph, pool, pwl::Engine::AcSweep);
    if (!bad) return;
    const Index bid = *bad;
    throw unsupported_device(graph, pool, pwl::Engine::AcSweep, bid,
        std::format(
            "run_mna_sweep: branch '{}' is a {} (a Newton device); the "
            "kernel AC sweep has no small-signal model for this kind (it "
            "linearises only the kinds marked ac_sweep in "
            "pulsim.engine_capabilities()), so it would return the transfer "
            "function of a different circuit. "
            "Use a time-domain routine instead: pulsim.run_ac_sweep, "
            "pulsim.fra.run_fra, "
            "pulsim.frequency_response, or run_mna_sweep(..., "
            "use_kernel=False, dt=..., t_end=...) (impulse-FFT).",
            graph.branch_name(bid),
            pwl::stored_kind_name(pool.kind_of(bid))));
}

/// The linear part as (G, E), the mass matrix E = C_split/2.
struct SweepMatrices {
    sparse::Matrix G;
    sparse::Matrix E;
};

[[nodiscard]] inline SweepMatrices
assemble_sweep_matrices(const topology::Graph& graph,
                         const pwl::DevicePool& pool,
                         const topology::SwitchStateMask& mask) {
    Linearization m = linear_part(graph, pool, mask);
    return {std::move(m.G), std::move(m.C)};
}

/// The small-signal model the sweep solves: the linear part alone for a
/// linear circuit (the assembly it always used, operation for operation),
/// else `linearize` at the DC operating point of `mask` — the external
/// devices from their declared Jacobians, the built-in Newton kinds from
/// their `Device` laws, the DC point solved through those laws
/// (`dc_device_refresh` on the interface path).
[[nodiscard]] inline Linearization sweep_model(const topology::Graph& graph,
                                               const pwl::DevicePool& pool,
                                               const topology::SwitchStateMask& mask) {
    if (!pool.requires_newton()) return linear_part(graph, pool, mask);
    require_linearisable(graph, pool);
    pwl::DCOperatingPointOptions dc_opts;
    const pwl::NonlinearRefreshFn refresh = has_builtin_newton(graph, pool)
        ? pwl::dc_device_refresh(pool, true, device::DevicePath::Interface)
        : pwl::NonlinearRefreshFn{};
    const auto op = pwl::compute_dc_operating_point(
        graph, pool, mask, refresh, dc_opts, nullptr, "run_mna_sweep", "ac_sweep");
    return linearize(graph, pool, mask, op.x, dc_opts.t_eval);
}

[[nodiscard]] inline ComplexSparseMatrix to_complex(const sparse::Matrix& A) {
    ComplexSparseMatrix out(A.rows(), A.cols());
    std::vector<Eigen::Triplet<Complex, Index>> triplets;
    triplets.reserve(static_cast<std::size_t>(A.nonZeros()));
    for (Index k = 0; k < A.outerSize(); ++k) {
        for (sparse::Matrix::InnerIterator it(A, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(),
                                   Complex{it.value(), Real{0}});
        }
    }
    out.setFromTriplets(triplets.begin(), triplets.end());
    out.makeCompressed();
    return out;
}

}  // namespace detail

/// Compute H(jω) at each frequency in `freqs` by linearising the cached
/// MNA matrix at the supplied operating-point switch mask.
///
/// Parameters
/// ----------
/// graph, pool, mask
///     The circuit description + switch state at the operating point.
/// freqs
///     Test frequencies in Hz.
/// input_state_idx
///     Index of the state-vector row where the input excitation is
///     injected (typically the source-branch constraint row, obtained
///     via `pool.branch_var_id_for_source(...)`).
/// output_node_idx
///     Index of the state-vector row whose voltage is the desired
///     output (typically `graph.node_id_of("vout")`).
///
/// Returns
/// -------
/// MnaSweepResult
///     freqs[k], H[k] for k = 0..freqs.size()-1.
[[nodiscard]] inline MnaSweepResult run_mna_sweep(
    const topology::Graph& graph,
    const pwl::DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const std::vector<Real>& freqs,
    Size input_state_idx,
    Size output_node_idx,
    const ShouldContinueFn& should_continue = {}) {

    // 1. Newton devices are refused, not dropped; an ideal branch-form
    //    element that leaves a node floating or closes a voltage loop at
    //    this mask is refused by the transient's names.
    detail::require_linear_for_sweep(graph, pool);
    if (pwl::BranchStructureChecker chk{graph, pool, "ac_sweep"}; chk.active()) {
        // an ON ideal diode closing a loop: the sweep poses the mode it is
        // given (no event iteration), so it refuses it as the dsed cache does
        if (const auto& ms = chk.check(mask, true); !ms.loop_diodes.empty()) {
            const std::string d{graph.branch_name(ms.loop_diodes.front())};
            throw UnsupportedError(
                "ac_sweep", "ideal_voltage_loop", d, "Diode",
                "run_mna_sweep: the ideal diode '" + d + "' (R_on = 0), ON in "
                "the mask, closes a loop of voltage sources and closed ideal "
                "elements (loop: " + ms.loop_texts.front() + "): its current "
                "is undefined. Open it in the mask or give it R_on > 0.");
        }
    }

    // 2. (G, E): the descriptor form reads E·dx/dt + G·x + b = 0;
    //    substituting x -> x_op + x̂·e^{jωt} gives (jω·E + G)·x̂ = −b̂, read
    //    at output_node_idx for b̂ = e_input. Rows past the circuit's are
    //    device-local charges (linearize.hpp).
    const Linearization sm = detail::sweep_model(graph, pool, mask);

    const Index n = sm.G.rows();
    if (sm.n_circuit <= 0) {
        throw std::runtime_error(
            "run_mna_sweep: empty MNA matrix");
    }
    if (static_cast<Index>(input_state_idx) >= sm.n_circuit ||
        static_cast<Index>(output_node_idx) >= sm.n_circuit) {
        throw std::runtime_error(
            "run_mna_sweep: input/output index out of range");
    }
    const ComplexSparseMatrix J_c = detail::to_complex(sm.G);
    const ComplexSparseMatrix E_c = detail::to_complex(sm.C);

    // 4. Per-frequency factor + solve.
    MnaSweepResult result;
    result.freqs = freqs;
    result.H.resize(freqs.size());

    // Source-constraint row uses convention `V_branch − V_source = 0`,
    // so a positive δV_source perturbation enters b as `−δV_source`.
    // For a unit-input H(jω) = x_out / u_in, we want `b̂_input = −1`
    // (equivalent to δV_source = +1), and solve `J · x̂ = −b̂ = +1`.
    // The matrix here is `M = jωE + J`, so `M · x̂ = +1`.
    ComplexVector rhs = ComplexVector::Zero(n);
    rhs[static_cast<Index>(input_state_idx)] = Complex{Real{1},
                                                          Real{0}};

    for (std::size_t k = 0; k < freqs.size(); ++k) {
        check_cancellation(should_continue, "run_mna_sweep",
                            static_cast<long>(k));
        const Real omega = Real{2} * Real{3.141592653589793238462643}
                              * freqs[k];
        const Complex j_omega{Real{0}, omega};

        ComplexSparseMatrix M = J_c + j_omega * E_c;
        M.makeCompressed();

        // v1.4.0: in-house complex sparse LU (no third-party LU on the
        // production AC-sweep path). Lifecycle matches the v1.3.0 real
        // solver — `analyze + factorize + solve`. `analyze` returns
        // `false` on a malformed pattern; `factorize` returns `false`
        // on numerical singularity (zero pivot), accessible via
        // `numeric_singular()`.
        sparse::PulsimComplexSparseLuSolver solver;
        if (!solver.analyze(M)) {
            throw std::runtime_error(
                "run_mna_sweep: complex symbolic analysis (Pulsim) "
                "failed at f=" + std::to_string(freqs[k]));
        }
        if (!solver.factorize(M)) {
            throw std::runtime_error(
                "run_mna_sweep: complex numeric factorisation (Pulsim) "
                "failed at f=" + std::to_string(freqs[k]) +
                (solver.numeric_singular()
                     ? " — matrix is numerically singular"
                     : ""));
        }
        ComplexVector x;
        solver.solve(rhs, x);
        result.H[k] = x[static_cast<Index>(output_node_idx)];
    }
    return result;
}

}  // namespace pulsim::analysis
