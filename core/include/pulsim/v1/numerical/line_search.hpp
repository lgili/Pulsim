#pragma once

// simplify-and-harden-numerical-surface — Phase 4.1.
//
// Canonical location for the Armijo backtracking line-search machinery.
//
// The actual implementation lives inline in
// `NewtonRaphsonSolver::line_search()` (in `solver.hpp`) — extracting the
// function body into a free function here would require threading the
// Newton state (Jacobian, residual norm tracking, telemetry counter) as
// explicit parameters, which buys nothing for the kernel and would force
// every call site to plumb the same arguments. The Phase 1 reorg
// convention is: new code SHOULD include THIS path; the canonical entry
// point is `pulsim::v1::NewtonOptions::armijo_line_search` and
// `armijo_sigma` for tuning, plus
// `NewtonResult::telemetry.line_search_backtracks` for observability.
//
// Tuning surface (post-Phase 3):
//
//   opts.advanced().newton.armijo_line_search = true;   // default
//   opts.advanced().newton.armijo_sigma        = 1e-4;  // default
//   opts.advanced().newton.min_damping         = 0.01;  // backtrack floor
//
// Telemetry:
//
//   result.newton_result.telemetry.line_search_backtracks
//
// Reference test: `core/tests/test_armijo_line_search.cpp` (5 cases,
// 14 assertions).

#include "pulsim/v1/numerical/newton.hpp"  // pulls in NewtonOptions
                                            // + NewtonRaphsonSolver
