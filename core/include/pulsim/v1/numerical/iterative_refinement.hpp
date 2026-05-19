#pragma once

// simplify-and-harden-numerical-surface — Phase 6.1.
//
// Canonical location for the iterative-refinement-on-direct-solve hook
// that fires automatically after every successful KLU / SparseLU /
// EnhancedSparseLU back-substitution when the relative residual
// `||b − A·x|| / ||b||` exceeds `10·ε_machine`.
//
// The actual implementation lives inline in
// `RuntimeLinearSolver::solve()` (in `high_performance.hpp` →
// `numerical/linear_solver.hpp`). Extracting into a free function
// `refine_if_needed(A, x, b, threshold)` would require either:
//   - re-passing the factorization to enable the cheap re-solve, OR
//   - re-factorizing inside the helper (which defeats the cheap-refine
//     contract).
//
// Either path adds overhead with no clarity gain — extraction is
// deferred to a Phase 3-style linear-solver refactor.
//
// Default policy:
//   - Direct solvers: residual check + at most ONE refinement round
//     per back-solve. Skipped silently when residual is within bound.
//   - Iterative solvers (GMRES, BiCGSTAB, CG): refinement always
//     skipped — iterative algorithms apply equivalent in-loop refinement
//     internally.
//
// Telemetry:
//   `result.linear_solver_telemetry.linear_refinement_steps`
//
// Reference test: `core/tests/test_iterative_refinement.cpp`
// (4 cases, 12 assertions).

#include "pulsim/v1/numerical/linear_solver.hpp"
