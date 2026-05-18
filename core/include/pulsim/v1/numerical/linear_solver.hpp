#pragma once

// simplify-and-harden-numerical-surface — Phase 1 reorganization.
//
// New canonical location for `LinearSolverKind`, `LinearSolverStackConfig`,
// `LinearSolverTelemetry`, and the solver factory. Re-exports from the
// legacy `high_performance.hpp` path so this is purely additive in the
// first release.

#include "pulsim/v1/high_performance.hpp"
